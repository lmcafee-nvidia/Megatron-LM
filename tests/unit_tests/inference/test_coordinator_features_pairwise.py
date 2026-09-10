# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real coordinator routing crossed with model topology and feature execution."""

import asyncio
import copy
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _instrument_nccl_dispatch_runtime,
    _instrument_parallel_runtime,
    _instrument_scenario_runtime,
)


def _active_target_ids(harness):
    context = harness.engine.context
    if getattr(context, "_bookkeeping_no_real_work", False):
        return ()
    ids = context.request_ids[context.paused_request_count : context.total_request_count].tolist()
    return tuple(request_id for request_id in ids if request_id in harness.engine.requests)


def _install_target_attribution(harness, runtime):
    """Attribute completed runtime deltas only to a real coordinator request."""
    target = Counter()
    model = harness.engine.controller.inference_wrapped_model.model
    forward = model.forward

    def attributed_forward(*args, **kwargs):
        ids = _active_target_ids(harness)
        before = runtime.copy()
        result = forward(*args, **kwargs)
        if ids:
            target["model-forward"] += 1
            target["prefill-forward"] += int(harness.engine.context.num_prefill_requests > 0)
            target["decode-forward"] += int(harness.engine.context.num_decode_requests > 0)
            for request_id in ids:
                target[("request", request_id)] += 1
            for key, count in runtime.items():
                target[key] += count - before[key]
        return result

    model.forward = attributed_forward
    return target


def _assert_output_matches_direct(output, direct):
    for key in ("status", "generated_tokens", "prompt_tokens", "generated_text"):
        assert output[key] == direct[key], key


def _assert_every_owner_executed(harness, target):
    target_ids = {key[1] for key in target if isinstance(key, tuple) and key[0] == "request"}
    assert target["model-forward"] > 0
    assert target["prefill-forward"] > 0
    assert target["decode-forward"] > 0
    assert len(target_ids) == 1, "Each rank must execute its sole routed target"

    groups = harness.engine.pg_collection
    record = {
        "rank": harness.rank,
        "dp": torch.distributed.get_rank(groups.dp),
        "mp": torch.distributed.get_rank(groups.mp),
        "tp": torch.distributed.get_rank(groups.tp),
        "pp": torch.distributed.get_rank(groups.pp),
        "ids": sorted(target_ids),
        "forwards": target["model-forward"],
    }
    records = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(records, record)
    if harness.rank != 0:
        return

    assert all(item["forwards"] > 0 and len(item["ids"]) == 1 for item in records)
    assert {item["dp"] for item in records} == set(range(harness.dp_size))
    routed_ids = set()
    for owner in range(harness.dp_size):
        owner_records = [item for item in records if item["dp"] == owner]
        assert {item["mp"] for item in owner_records} == set(range(len(owner_records)))
        assert {(item["tp"], item["pp"]) for item in owner_records} == {
            (tp, pp)
            for tp in range(harness.config.tensor_model_parallel_size)
            for pp in range(harness.config.pipeline_model_parallel_size)
        }
        owner_ids = {item["ids"][0] for item in owner_records}
        assert len(owner_ids) == 1
        routed_ids.update(owner_ids)
    assert len(routed_ids) == harness.dp_size


async def _exercise_all_owners(harness, prompt, params, direct, runtime=None, install=None):
    runtime = Counter() if runtime is None else runtime
    await harness.start()
    target = _install_target_attribution(harness, runtime)
    if install is not None:
        install(target)
    await harness.pause()
    pending = []
    if harness.rank == 0:
        for client in harness.clients:
            request_id, future = client.add_request_with_id(prompt, copy.deepcopy(params))
            assert request_id == 0
            pending.append(future)
        await until(lambda: len(harness.service.coordinator.request_id_to_rank) == harness.dp_size)
        assert sorted(harness.service.coordinator.request_id_to_rank.values()) == sorted(
            harness.service.coordinator.identities_of_data_parallel_ranks
        )
    await harness.barrier()
    await harness.unpause()
    if harness.rank == 0:
        outputs = await asyncio.wait_for(asyncio.gather(*pending), timeout=120)
        for output in outputs:
            _assert_output_matches_direct(output, direct)
    await harness.barrier()
    await harness.pause()
    _assert_every_owner_executed(harness, target)
    harness.assert_retired()
    return target


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"tensor_model_parallel_size": 2}, id="tp2-dp4"),
        pytest.param({"pipeline_model_parallel_size": 2}, id="pp2-dp4"),
        pytest.param(
            {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 2,
                "sequence_parallel": True,
            },
            id="tp2-pp2-sp-dp2",
        ),
    ],
)
async def test_routed_topology_matches_direct_on_every_owner(monkeypatch, overrides):
    """TP, PP and the full eight-rank TP2xPP2xDP2 layout execute routed targets."""
    prompt = list(range(4, 16))
    params = greedy_params(num_tokens_to_generate=4, return_prompt_tokens=True)
    runtime = Counter()
    async with routed_model(monkeypatch, transformer_impl="local", **overrides) as harness:
        direct = await harness.direct(prompt, params)

        def install(_target):
            if harness.config.sequence_parallel:
                from megatron.core.tensor_parallel import layers as tp_layers

                all_gather = tp_layers.dist_all_gather_func

                def traced_all_gather(output, input_, *args, **kwargs):
                    result = all_gather(output, input_, *args, **kwargs)
                    runtime["tp-collective:gather_from_sequence_parallel_region"] += 1
                    runtime["tp-sp-gather-dimensions"] += int(
                        output.shape[0]
                        == input_.shape[0] * harness.config.tensor_model_parallel_size
                    )
                    return result

                monkeypatch.setattr(tp_layers, "dist_all_gather_func", traced_all_gather)
            for module in harness.engine.controller.inference_wrapped_model.model.modules():
                if "ParallelLinear" in type(module).__name__:
                    _instrument_parallel_runtime(module, runtime)

        target = await _exercise_all_owners(
            harness, prompt, params, direct, runtime=runtime, install=install
        )
        if harness.config.tensor_model_parallel_size > 1:
            assert runtime["tp-column-partitions-installed"] > 0
            assert runtime["tp-row-partitions-installed"] > 0
            if harness.config.sequence_parallel:
                assert target["tp-collective:gather_from_sequence_parallel_region"] > 0
                assert target["tp-collective:reduce_scatter_to_sequence_parallel_region"] > 0
                assert target["tp-sp-gather-dimensions"] > 0
                assert target["tp-sp-reduce-scatter-dimensions"] > 0
            else:
                assert target["tp-collective:reduce_from_tensor_model_parallel_region"] > 0


@pytest.mark.internal
@pytest.mark.asyncio
async def test_routed_cuda_graph_replays_for_every_target(monkeypatch):
    """Completed routed decode forwards invoke the production CUDA replay node."""
    prompt = list(range(4, 16))
    params = greedy_params(num_tokens_to_generate=4, return_prompt_tokens=True)
    async with routed_model(
        monkeypatch,
        num_cuda_graphs=4,
        force_build_cuda_graphs=True,
        use_cuda_graphs_for_non_decode_steps=False,
        cuda_graph_max_tokens=16,
    ) as harness:
        direct = await harness.direct(prompt, params)

        def install(target):
            replay = _CudagraphReplayNode.forward

            def attributed_replay(*args, **kwargs):
                result = replay(*args, **kwargs)
                if _active_target_ids(harness):
                    target.update(("model-forward", "decode-forward", "cuda-graph-replay"))
                return result

            monkeypatch.setattr(_CudagraphReplayNode, "forward", staticmethod(attributed_replay))

        target = await _exercise_all_owners(harness, prompt, params, direct, install=install)
        assert target["cuda-graph-replay"] > 0


@pytest.mark.internal
@pytest.mark.asyncio
async def test_routed_mtp_executes_inner_steps_on_every_target(monkeypatch):
    """Coordinator-owned speculative decoding calls the real inner MTP model step."""
    prompt = list(range(4, 16))
    params = greedy_params(num_tokens_to_generate=6, return_prompt_tokens=True)
    async with routed_model(monkeypatch, num_speculative_tokens=1) as harness:
        direct = await harness.direct(prompt, params)

        def install(target):
            model = harness.engine.controller.inference_wrapped_model.model
            compute = model.compute_mtp_single_step

            def traced_compute(*args, **kwargs):
                result = compute(*args, **kwargs)
                if _active_target_ids(harness):
                    target["mtp-inner-step"] += 1
                return result

            model.compute_mtp_single_step = traced_compute

        target = await _exercise_all_owners(harness, prompt, params, direct, install=install)
        assert target["mtp-inner-step"] > 0
        assert harness.engine._spec_steps > 0
        assert int(harness.engine._spec_tokens_proposed_per_pos.sum()) > 0


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("mixer", ["mamba", "gdp", "gdn"])
async def test_routed_hybrid_updates_owned_recurrent_state(monkeypatch, mixer):
    """Completed target decode updates the target's mapped recurrent state slot."""
    prompt = list(range(4, 16))
    params = greedy_params(num_tokens_to_generate=4, return_prompt_tokens=True)
    async with routed_model(monkeypatch, model_provider="hybrid", ssm_mixer=mixer) as harness:
        direct = await harness.direct(prompt, params)

        def install(target):
            model = harness.engine.controller.inference_wrapped_model.model
            context = harness.engine.context
            expected = {
                "mamba": "MambaMixer",
                "gdp": "GatedDeltaProductMixer",
                "gdn": "GatedDeltaNet",
            }
            selected = [
                module for module in model.modules() if type(module).__name__ == expected[mixer]
            ]
            assert selected, expected[mixer]
            for module in selected:
                module.register_forward_hook(
                    lambda *_: target.update(
                        {"mixer-forward": int(bool(_active_target_ids(harness)))}
                    )
                )
            forward = model.forward

            def traced_forward(*args, **kwargs):
                snapshots = []
                if context.is_decode_only():
                    for request_id in _active_target_ids(harness):
                        row = int(
                            (
                                context.request_ids[: context.total_request_count] == request_id
                            ).nonzero()[0]
                        )
                        slot = int(context.mamba_metadata.request_to_mamba_state_idx[row])
                        assert 0 <= slot < context.max_requests
                        snapshots.append(
                            (
                                slot,
                                context.mamba_conv_states[:, slot].clone(),
                                context.mamba_ssm_states[:, slot].clone(),
                            )
                        )
                result = forward(*args, **kwargs)
                for slot, conv, ssm in snapshots:
                    assert not torch.equal(conv, context.mamba_conv_states[:, slot])
                    assert not torch.equal(ssm, context.mamba_ssm_states[:, slot])
                    target["owned-state-updates"] += 1
                return result

            model.forward = traced_forward

        target = await _exercise_all_owners(harness, prompt, params, direct, install=install)
        assert harness.engine.context.is_hybrid_model
        assert target["mixer-forward"] > 0
        assert target["owned-state-updates"] > 0


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,signals,required",
    [
        ({"fp8": True, "hidden_size": 128}, ("fp8",), ("fp8-quantized-forwards",)),
        (
            {
                "hidden_size": 64,
                "position_embedding_type": "rope",
                "use_flashinfer_fused_rope": True,
            },
            ("fused-rope",),
            ("fused-rope-kernel",),
        ),
        (
            {"window_size": (4, 0), "softmax_type": "off-by-one"},
            ("softmax-sink",),
            ("swa-kernel-calls", "sink-correction-calls"),
        ),
        (
            {"window_size": (4, 0), "window_attn_skip_freq": 2, "softmax_type": "learnable"},
            ("softmax-sink",),
            ("swa-kernel-calls", "full-attention-kernel-calls", "sink-correction-calls"),
        ),
    ],
    ids=["fp8", "fused-rope", "swa-sink", "alternating-swa-sink"],
)
async def test_routed_model_kernels_match_direct(monkeypatch, options, signals, required):
    prompt = list(range(4, 16))
    params = greedy_params(num_tokens_to_generate=4, return_prompt_tokens=True)
    async with routed_model(monkeypatch, **options) as harness:
        direct = await harness.direct(prompt, params)
        runtime = Counter()
        _instrument_scenario_runtime(
            SimpleNamespace(engine=harness.engine), SimpleNamespace(signals=signals), runtime
        )
        target = await _exercise_all_owners(harness, prompt, params, direct, runtime=runtime)
        for counter in required:
            assert target[counter] > 0, counter


@pytest.mark.internal
@pytest.mark.asyncio
async def test_routed_moe_participates_in_ep_collectives(monkeypatch):
    """Every coordinator target dispatches and combines through its EP collective."""
    prompt = list(range(4, 16))
    params = greedy_params(num_tokens_to_generate=4, return_prompt_tokens=True)
    runtime = Counter()
    async with routed_model(
        monkeypatch,
        expert_model_parallel_size=2,
        use_moe_layer_spec=True,
        inference_moe_token_dispatcher_type="nccl",
        transformer_impl="inference_optimized",
    ) as harness:
        direct = await harness.direct(prompt, params)

        def install(_target):
            for module in harness.engine.controller.inference_wrapped_model.model.modules():
                if type(module).__name__ == "MoELayer":
                    _instrument_nccl_dispatch_runtime(module, runtime)

        target = await _exercise_all_owners(
            harness, prompt, params, direct, runtime=runtime, install=install
        )
        assert torch.distributed.get_world_size(harness.engine.pg_collection.ep) == 2
        assert runtime["nccl-dispatchers-installed"] > 0
        assert target["nccl-token-dispatches"] > 0
        assert target["nccl-token-dispatches"] == target["nccl-token-combines"]
        assert target["nccl-combine-before-dispatch"] == 0
        assert target["nccl-dispatch-inflight"] == 0
