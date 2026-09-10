# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Target-owned MoE invariance across real NVLS rank/token participation.

Run each MCORE_BI_TEST_COLLECTIVE=ordered/multimem selection in a fresh EP4
interpreter. Native TE additionally requires CUBLASLT_WORKSPACE_SIZE=0 before
interpreter startup. These layer tests complement, not replace, dynamic GPT
and existing NVLS graph tests.
"""

import os

import pytest
import torch

from megatron.core.inference.moe import batch_invariant
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.parallel_state import get_expert_model_parallel_group
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe import experts, token_dispatcher_inference
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from tests.unit_tests.inference.test_moe_dispatching_and_routing import _make_base_config
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def nvls_runtime():
    assert Utils.world_size == 4, "this manifest requires four real NVLink EP ranks"
    backend = os.environ.get("MCORE_BI_TEST_BACKEND", "te_native")
    collective = os.environ.get("MCORE_BI_TEST_COLLECTIVE", "ordered")
    if backend == "te_native":
        assert os.environ.get("CUBLASLT_WORKSPACE_SIZE") == "0"
    assert not bik.is_batch_invariant_mode_enabled(), "inherited a live BI backend"
    # Explicit global selection precedes model construction and the first GEMM.
    bik.enable_batch_invariant_mode(backend=backend, collective=collective)
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=4)
    try:
        assert bik.get_batch_invariant_backend() == backend
        assert bik.get_batch_invariant_collective() == collective
        yield backend, collective
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        InferenceMode.unset_active()
        bik.disable_batch_invariant_mode()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("grouped_backend", ["vllm", "torch"])
@pytest.mark.parametrize("peers", ["idle", "uneven"])
@torch.inference_mode()
def test_nvls_target_batch_invariance(nvls_runtime, monkeypatch, grouped_backend, peers):
    backend, collective = nvls_runtime
    rank = torch.distributed.get_rank()
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
    config = _make_base_config(
        expert_model_parallel_size=4,
        inference_grouped_gemm_backend=grouped_backend,
        inference_moe_token_dispatcher_type="nvls",
        batch_invariant_mode=True,
        batch_invariant_backend=backend,
        batch_invariant_collective=collective,
        attention_backend=AttnBackend.flash,
        flash_attention_version=3,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        moe_shared_expert_intermediate_size=None,
    )
    NVLSAllGatherVDispatcher.allocate_buffers(
        per_rank_worst_case_token_count=128,
        topk=config.moe_router_topk,
        hidden_size=config.hidden_size,
        ep_group=get_expert_model_parallel_group(),
    )
    layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    dispatcher = layer._inference_token_dispatcher
    assert isinstance(dispatcher, NVLSAllGatherVDispatcher)
    target = (
        torch.sin(torch.arange(8 * config.hidden_size, device="cuda") / 29)
        .reshape(8, 1, config.hidden_size)
        .bfloat16()
    )
    phase, evidence = {}, []
    original_gate = layer.router.gating

    def gate(inputs):
        result = original_gate(inputs)
        phase["physical"] = inputs.shape[0]
        if rank == 0:
            phase["gate"] = result[phase["target_row"] : phase["target_row"] + 8].clone()
        return result

    monkeypatch.setattr(layer.router, "gating", gate)
    expert_name = "vllm_fused_moe" if grouped_backend == "vllm" else "mcore_fused_moe"
    original_experts = getattr(experts, expert_name)

    def expert_call(*args, **kwargs):
        valid = int(kwargs["valid_tokens"].item())
        assert valid == sum(phase["counts"])
        routing = kwargs["routing_map"][:valid]
        start = phase["target_row"]  # target is on EP rank zero, the first AGV segment
        phase["target_experts"] = routing[start : start + 8].clone()
        phase["loads"] = torch.bincount(routing.flatten(), minlength=config.num_moe_experts)
        assert kwargs["out"] is NVLSAllGatherVDispatcher._get_rsv_tensor()
        phase["experts"] = phase.get("experts", 0) + 1
        return original_experts(*args, **kwargs)

    monkeypatch.setattr(experts, expert_name, expert_call)
    for owner, name in (
        (batch_invariant, "ordered_reduce_scatter_v"),
        (token_dispatcher_inference, "multimem_reduce_scatter_v"),
    ):
        original = getattr(owner, name)

        def reduce(*args, original=original, name=name, **kwargs):
            assert args[0].shape[0] == phase["counts"][rank]
            phase.setdefault("collectives", []).append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, reduce)

    def run(counts, target_row):
        phase.clear()
        phase.update(counts=counts, target_row=target_row)
        # Repeating a real target input deliberately changes the load on its
        # actual routed experts; no synthetic routing map replaces the router.
        inputs = target[:1].repeat(counts[rank], 1, 1)
        if rank == 0:
            inputs[target_row : target_row + 8] = target
        with InferenceMode.active():
            output, bias = layer(inputs)
        assert bias is None and output.shape == inputs.shape
        assert phase["physical"] == counts[rank]
        assert phase["experts"] == 1
        expected_collective = (
            "ordered_reduce_scatter_v" if collective == "ordered" else "multimem_reduce_scatter_v"
        )
        assert phase["collectives"] == [expected_collective]
        # Independent oracle for the real target partials written by each rank's
        # experts into symmetric memory; NCCL here only gathers oracle inputs.
        partial = NVLSAllGatherVDispatcher._get_rsv_tensor()[target_row : target_row + 8].clone()
        partials = [torch.empty_like(partial) for _ in range(4)]
        torch.distributed.all_gather(partials, partial)
        if collective == "multimem":
            expected = torch.stack(partials).double().sum(dim=0).float()
        else:
            expected = torch.zeros_like(partial)
            for contribution in partials:
                expected = expected + contribution
        if rank == 0:
            selected = output[target_row : target_row + 8, 0].clone()
            assert selected.abs().max() > 0
            assert torch.equal(selected, expected.bfloat16()), "actual NVLS combine changed target"
            phase["output"] = selected
        evidence.append(dict(phase))

    run([64, 0, 0, 0], 0)
    wide = [128, 0, 0, 0] if peers == "idle" else [128, 32, 64, 96]
    run(wide, 120)
    run(wide, 0)
    # Return to the smaller shape with idle peers: stale expert/symmetric-buffer
    # data from the preceding wide execution must not affect the target.
    run([64, 0, 0, 0], 0)
    for actual in evidence[1:]:
        assert torch.equal(actual["target_experts"], evidence[0]["target_experts"])
        if rank == 0:
            assert torch.equal(actual["gate"], evidence[0]["gate"])
            assert torch.equal(actual["output"], evidence[0]["output"])
    selected_experts = evidence[0]["target_experts"][0]
    assert torch.all(
        evidence[1]["loads"][selected_experts] > evidence[0]["loads"][selected_experts]
    )
    assert not torch.equal(evidence[1]["loads"].min(), evidence[1]["loads"].max())
    if rank == 0:
        assert {row["physical"] for row in evidence} == {64, 128}
    else:
        assert evidence[0]["physical"] == evidence[-1]["physical"] == 0
    print(
        "BI_NVLS_WITNESS",
        rank,
        backend,
        collective,
        grouped_backend,
        peers,
        "physical",
        [row["physical"] for row in evidence],
        "target_rows",
        [row["target_row"] for row in evidence],
        "expert_loads",
        [row["loads"].tolist() for row in evidence],
    )
