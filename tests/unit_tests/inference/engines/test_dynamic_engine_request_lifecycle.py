# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Request-correlated differential coverage for checkpoint and memory residency.

Prefix-cache #39/#40 and async-scheduling #43 remain prerequisite test owners;
cluster validation selects ``PREREQUISITE_NODE_IDS`` without copying their tests.
"""

import gc
import os
from collections import Counter
from dataclasses import dataclass, fields
from unittest import mock

import msgpack
import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.data_parallel_inference_coordinator.handlers import handle_engine_reply
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
    unwrap_serialized_tensors,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.unified_memory import prefetch_managed_tensor
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder as _set_rounder
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _ASYNC_PARALLEL_SCENARIOS,
    _BASE_PAIR_CONFIG,
    _assert_request_parity,
    _AsyncPairScenario,
    _check_scenario_prerequisite,
    _instrument_scenario_runtime,
    _make_scenario_requests,
    _snapshot_requests,
)
from tests.unit_tests.inference.test_dynamic_prefix_caching_coordinator import (
    make_coordinator_direct,
)
from tests.unit_tests.test_utilities import Utils

# Policy values describe checkpoint/merge ownership.  The exhaustiveness test
# intentionally fails whenever either public dataclass gains a field without an
# explicit lifecycle decision.
REQUEST_FIELD_POLICY = {
    "request_id": "checkpoint:preserve / merge:first",
    "prompt": "checkpoint:reset / merge:first",
    "sampling_params": "checkpoint:copy-and-reduce-budget / merge:first",
    "inference_parameters": "checkpoint:drop-deprecated-alias / merge:drop",
    "prompt_tokens": "checkpoint:append-output / merge:first / wire:opt-in",
    "prompt_length": "checkpoint:reset / merge:reset / wire:derive",
    "arrival_time": "checkpoint:reset / merge:reset",
    "status": "checkpoint:preserve / merge:last",
    "encoder_prompt": "checkpoint:drop / merge:drop",
    "generated_text": "checkpoint:reset / merge:concatenate",
    "segments": "checkpoint:drop / merge:drop-unsupported",
    "generated_segments": "checkpoint:drop / merge:drop-unsupported",
    "generated_sequence_lengths": "checkpoint:drop / merge:drop-unsupported",
    "generated_tokens": "checkpoint:reset / merge:concatenate",
    "prompt_log_probs": "checkpoint:reset / merge:most-complete-original-prompt",
    "generated_log_probs": "checkpoint:reset / merge:concatenate",
    "prompt_top_n_logprobs": "checkpoint:reset / merge:most-complete-original-prompt",
    "generated_top_n_logprobs": "checkpoint:reset / merge:concatenate",
    "generated_length": "checkpoint:reset / merge:derive-from-output",
    "tpot": "checkpoint:reset / merge:concatenate",
    "remaining_prompt_tokens": "checkpoint:cumulative-prompt / merge:reinitialize",
    "policy_epoch": "checkpoint:deep-copy / merge:last-deep-copy",
    "kv_cache_epoch": "checkpoint:reset-for-recompute / merge:last-deep-copy",
    "latency": "checkpoint:record-owned / merge:record",
    "routing_indices": "checkpoint:reset / merge:concatenate",
    "finished_chunk_token_count": "checkpoint:reset / merge:reset",
    "stop_word_ids": "checkpoint:deep-copy / merge:drop-terminal-only",
    "cg_wait_iters": "checkpoint:reset-admission-local / merge:reset",
    "block_size_tokens": "checkpoint:preserve / merge:first",
    "enable_prefix_caching": "checkpoint:preserve / merge:first",
    "num_cached_tokens": "checkpoint:reset / merge:first-observation",
    "precomputed_block_hashes": "checkpoint:recompute / merge:first",
    "ttft": "checkpoint:reset / merge:first-populated",
    "events": "checkpoint:new-segment / merge:concatenate",
    "event_add_engine": "checkpoint:preserve-original / merge:drop / wire:drop",
}

SAMPLING_FIELD_POLICY = {
    "temperature": "preserve",
    "top_k": "preserve",
    "top_p": "preserve",
    "return_log_probs": "preserve",
    "skip_prompt_log_probs": "preserve",
    "return_segments": "preserve",
    "num_tokens_to_generate": "subtract-generated",
    "num_tokens_total": "clear-after-budget-conversion",
    "termination_id": "preserve",
    "top_n_logprobs": "preserve",
    "return_prompt_top_n_logprobs": "preserve-derived-policy",
    "add_BOS": "preserve",
    "stop_words": "deep-copy-preserve",
    "detokenize_stop_sequence": "preserve",
    "return_prompt_tokens": "preserve",
    "streaming": "preserve",
    "streaming_interval": "preserve",
}


def test_checkpoint_field_policy_tables_are_exhaustive():
    """Every dataclass field has an explicit checkpoint/merge policy."""
    request_fields = {field.name for field in fields(DynamicInferenceRequest)}
    sampling_fields = {field.name for field in fields(SamplingParams)}
    assert request_fields == set(REQUEST_FIELD_POLICY), (
        f"unclassified={sorted(request_fields - REQUEST_FIELD_POLICY)}, "
        f"stale={sorted(REQUEST_FIELD_POLICY - request_fields)}"
    )
    assert sampling_fields == set(SAMPLING_FIELD_POLICY), (
        f"unclassified={sorted(sampling_fields - SAMPLING_FIELD_POLICY)}, "
        f"stale={sorted(SAMPLING_FIELD_POLICY - sampling_fields)}"
    )


def test_checkpoint_policy_documents_current_unsupported_output_fields():
    """Sentinels prevent the policy table from claiming fields that merge drops."""
    params = SamplingParams(num_tokens_to_generate=2, termination_id=-1)
    request = DynamicInferenceRequest(
        request_id=23,
        prompt="original",
        prompt_tokens=torch.tensor([1, 2], dtype=torch.int64),
        sampling_params=params,
        encoder_prompt="encoder",
        segments=["prompt"],
        generated_segments=["answer"],
        generated_sequence_lengths=[1],
        generated_tokens=[3],
        stop_word_ids=[[3, 4]],
    )
    record = DynamicInferenceRequestRecord.from_request(request)

    record.checkpoint()
    checkpoint = record[-1]
    merged = record.merge()

    assert checkpoint.prompt is None
    assert checkpoint.encoder_prompt is None
    assert checkpoint.segments is None
    assert checkpoint.generated_segments is None
    assert checkpoint.generated_sequence_lengths is None
    assert checkpoint.stop_word_ids == [[3, 4]]
    assert merged.prompt == "original"
    assert merged.encoder_prompt is None
    assert merged.segments is None
    assert merged.generated_segments is None
    assert merged.generated_sequence_lengths is None
    assert merged.stop_word_ids is None


@dataclass(frozen=True)
class _MatrixRow:
    """One designed request-lifecycle interaction and its current owner."""

    name: str
    interaction: str
    owner: str


REQUEST_LIFECYCLE_MATRIX = (
    _MatrixRow(
        "chunked-partial-recompute-api-coordinator",
        "partial chunk + RECOMPUTE + prompt/logprob/top-N wire result",
        "local:test_chunked_partial_recompute_api_coordinator",
    ),
    _MatrixRow(
        "chunked-companion-zero-budget-evict",
        "partial chunk + boundary-crossing companion + paused_limit=0 eviction",
        "local:test_chunked_companion_zero_budget_evict",
    ),
    _MatrixRow(
        "retained-pause-ep2",
        "retained pause + EP2 NCCL dispatch before and after pressure",
        "local:test_retained_pause_ep2",
    ),
    _MatrixRow(
        "single-evict-hybrid-mamba-stop-keep",
        "one eviction + Mamba slot reacquire + retained stop sequence",
        "local:test_single_evict_hybrid_mamba_stop_keep",
    ),
    _MatrixRow(
        "repeated-evict-mtp2-stop-strip",
        "two evictions + MTP depth two + stripped stop sequence",
        "local:test_repeated_evict_mtp2_stop_strip",
    ),
    _MatrixRow(
        "persist-te-swa-stochastic",
        "PERSIST + Transformer Engine + SWA/sink + seeded sampling",
        "local:test_persist_te_swa_stochastic",
    ),
    _MatrixRow(
        "offload-dynamic-fp8",
        "OFFLOAD/dynamic KV + FP8 deallocate and restore",
        "local:test_offload_dynamic_fp8",
    ),
    _MatrixRow(
        "uvm-offload-static-managed-capacity",
        "UVM OFFLOAD + managed capacity beyond the CUDA-only budget",
        "local:test_uvm_offload_static_managed_capacity",
    ),
)

PREREQUISITE_NODE_IDS = (
    "tests/unit_tests/inference/contexts/test_dynamic_prefix_caching.py::"
    "TestPerBlockRouting::test_finished_checkpointed_request_reconstructs_full_routing",
    "tests/unit_tests/inference/contexts/test_dynamic_prefix_caching.py::"
    "TestPrefixCacheRealEngineMatrix::test_real_cuda_graph_suspend_resume_lifecycle",
    "tests/unit_tests/inference/contexts/test_dynamic_prefix_caching.py::"
    "TestPrefixCacheRealEngineMatrix::test_static_uvm_preserves_hybrid_state_cache_and_graphs",
    "tests/unit_tests/inference/contexts/test_dynamic_prefix_caching.py::"
    "TestPrefixCacheRealEngineMatrix::test_persist_tp2_pp2_replays_same_graphs_after_resume",
    "tests/unit_tests/inference/engines/test_dynamic_engine_async_sched.py::"
    "test_async_reset_clears_pending_logits",
    "tests/unit_tests/inference/engines/test_dynamic_engine_async_sched.py::"
    "test_async_suspend_pending_logits_lifecycle",
)


def test_matrix_manifest_has_one_runtime_owner_per_row():
    """Every designed row names its request-correlated runtime owner."""
    assert len(REQUEST_LIFECYCLE_MATRIX) == 8
    assert len({row.name for row in REQUEST_LIFECYCLE_MATRIX}) == 8
    assert all(row.owner.startswith("local:") for row in REQUEST_LIFECYCLE_MATRIX)
    owner_names = [row.owner.removeprefix("local:") for row in REQUEST_LIFECYCLE_MATRIX]
    assert len(set(owner_names)) == len(owner_names)
    assert all(
        hasattr(TestRequestLifecyclePairwise, owner)
        or hasattr(TestRequestLifecyclePairwiseEP2, owner)
        for owner in owner_names
    )
    assert len(PREREQUISITE_NODE_IDS) == len(set(PREREQUISITE_NODE_IDS))


_CHUNKED_RECOMPUTE = _AsyncPairScenario(
    name="chunked-partial-recompute-api-coordinator",
    pairs=("prefill:chunked", "kv:recompute", "api:coordinator"),
    config={
        "enable_chunked_prefill": True,
        "context_max_tokens": 8,
        "kv_cache_management_mode": "recompute",
        "static_kv_memory_pointers": False,
        "materialize_only_last_token_logits": False,
        "return_log_probs": True,
    },
    sampling=(
        {
            "return_log_probs": True,
            "skip_prompt_log_probs": False,
            "top_n_logprobs": 3,
            "return_prompt_tokens": True,
        },
    ),
    signals=("chunked", "logprobs", "top-n"),
)

_PERSIST_TE_SWA = _AsyncPairScenario(
    name="persist-te-swa-stochastic",
    pairs=("kv:persist", "implementation:transformer-engine", "attention:swa-sink"),
    config={
        "kv_cache_management_mode": "persist",
        "static_kv_memory_pointers": True,
        "transformer_impl": "transformer_engine",
        "window_size": (4, 0),
        "window_attn_skip_freq": 2,
        "softmax_type": "learnable",
    },
    sampling=({"temperature": 0.8, "top_k": 8},),
    signals=(
        "transformer-engine",
        "swa-alternating",
        "softmax-sink",
        "sampled",
        "temperature-filter",
        "top-k-filter",
    ),
    prerequisite="transformer-engine",
    parity="reproducible",
)

_OFFLOAD_FP8 = _AsyncPairScenario(
    name="offload-dynamic-fp8",
    pairs=("kv:offload", "precision:fp8"),
    config={"kv_cache_management_mode": "offload", "static_kv_memory_pointers": False, "fp8": True},
    signals=("fp8",),
    prerequisite="fp8",
    atol=5.0e-3,
)

_UVM_OFFLOAD = _AsyncPairScenario(
    name="uvm-offload-static-managed-capacity",
    pairs=("kv:offload", "memory:uvm-capacity"),
    config={
        "kv_cache_management_mode": "offload",
        "static_kv_memory_pointers": True,
        "unified_memory_level": 1,
        "context_buffer_size_gb": 0.002,
        "context_paused_buffer_size_gb": 0.002,
    },
    signals=("gpt",),
)

_HYBRID_MAMBA_STOP_KEEP = _AsyncPairScenario(
    name="single-evict-hybrid-mamba-stop-keep",
    pairs=("lifecycle:evict", "model:hybrid-mamba", "termination:stop-keep"),
    config={"model_provider": "hybrid"},
    signals=("hybrid",),
    prerequisite="mamba",
)

_EP2_NCCL = next(case for case in _ASYNC_PARALLEL_SCENARIOS if case.name == "moe-ep2-nccl")


@dataclass
class _RunResult:
    requests: list[DynamicInferenceRequest]
    record_lengths: dict[int, int]
    runtime: Counter
    witness: dict[str, object] | None


def _configure_test_tokenizer(engine):
    detokenize = lambda tokens, **_kwargs: "".join(f"<{token}>" for token in tokens)
    engine.controller.tokenizer.bos = None
    engine.controller.tokenizer.tokenize = lambda text: [int(token) for token in text.split()]
    engine.controller.tokenizer.detokenize = detokenize
    engine.controller.detokenize = lambda _tokenizer, tokens, **kwargs: detokenize(tokens, **kwargs)


def _make_manual_request(
    context,
    request_id,
    prompt_tokens,
    num_tokens_to_generate,
    *,
    full_logprobs=False,
    stop_words=None,
    keep_stop=False,
):
    return DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(
            num_tokens_to_generate=num_tokens_to_generate,
            termination_id=-1,
            top_k=1,
            return_log_probs=True,
            skip_prompt_log_probs=not full_logprobs,
            top_n_logprobs=3,
            stop_words=stop_words,
            detokenize_stop_sequence=keep_stop,
        ),
        block_size_tokens=context.block_size_tokens,
        enable_prefix_caching=context.enable_prefix_caching,
    )


def _active_request_ids(context):
    return [
        request_id
        for request_id in context.request_ids[
            context.paused_request_count : context.total_request_count
        ].tolist()
        if request_id >= 0
    ]


def _install_incrementing_logits(env, runtime, modulo):
    engine = env.engine
    context = engine.context
    model = engine.controller.inference_wrapped_model.model
    original = model.forward

    def deterministic_forward(*args, **kwargs):
        request_ids = _active_request_ids(context)
        phase = engine.evicted_request_count
        logits = original(*args, **kwargs)
        tokens = kwargs.get("tokens", kwargs.get("input_ids", args[0] if args else None))
        assert tokens is not None
        if logits.shape[1] != tokens.shape[1]:
            tokens = tokens[:, context.active_logit_idxs[: logits.shape[1]]]
        next_tokens = torch.remainder(tokens + 1, modulo)
        logits.zero_()
        logits.scatter_(-1, next_tokens.unsqueeze(-1), 100.0)
        for request_id in request_ids:
            runtime[("model-forward", phase, request_id)] += 1
        return logits

    model.forward = deterministic_forward
    return model


def _install_real_mtp_logits(env, runtime, modulo):
    model = _install_incrementing_logits(env, runtime, modulo)
    engine = env.engine
    context = engine.context
    original = model.compute_mtp_single_step

    def deterministic_mtp(
        hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
    ):
        request_ids = _active_request_ids(context)
        phase = engine.evicted_request_count
        hidden_states, logits = original(
            hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
        )
        predicted = torch.remainder(next_token_ids + 1, modulo)
        logits.zero_()
        logits.scatter_(-1, predicted.transpose(0, 1).unsqueeze(-1), 100.0)
        for request_id in request_ids:
            runtime[("mtp-depth", phase, request_id, depth)] += 1
        return hidden_states, logits

    model.compute_mtp_single_step = deterministic_mtp
    return model


def _event_count(request, event_type):
    return sum(event.type == event_type for event in request.events)


def _collect_finished(result, completed, record_lengths):
    for record in result["finished_request_records"]:
        merged = record.merge()
        completed[merged.request_id] = merged
        record_lengths[merged.request_id] = len(record.requests)


def _assert_engine_drained(engine, futures, completed, request_ids):
    assert set(completed) == set(request_ids)
    assert all(future.done() for future in futures)
    assert not engine.requests
    assert not engine.waiting_request_ids
    assert not engine.failed_request_ids
    assert engine.context.total_request_count == 0
    assert engine.context.paused_request_count == 0


def _snapshot_run(result):
    snapshots = _snapshot_requests(result.requests)
    return {
        "requests": {
            request.request_id: snapshot for request, snapshot in zip(result.requests, snapshots)
        },
        "text": {request.request_id: request.generated_text for request in result.requests},
    }


def _assert_run_parity(actual, expected, atol=1.0e-3):
    _assert_request_parity(
        actual.requests,
        [expected["requests"][request.request_id] for request in actual.requests],
        atol,
        exact_numerics=True,
        exact_top_n=True,
    )
    assert {request.request_id: request.generated_text for request in actual.requests} == expected[
        "text"
    ]


def _run_treatment_pair(run, cleanup):
    baseline = run(treatment=False)
    expected = _snapshot_run(baseline)
    del baseline
    cleanup()
    treatment = run(treatment=True)
    _assert_run_parity(treatment, expected)
    return treatment


def _allocate_leaving(allocator, remaining):
    count = allocator.get_allocatable_count() - remaining
    assert count >= 0
    if count == 0:
        return None
    blocks = allocator.allocate_memory_blocks(count)
    assert blocks is not None and allocator.get_allocatable_count() == remaining
    return blocks.clone()


def _release_filler(allocator, blocks):
    if blocks is not None:
        allocator.release_memory_blocks(blocks)


def _install_mamba_slot_witnesses(env, runtime):
    context = env.engine.context
    metadata = context.mamba_metadata
    original_add = context.add_request

    def traced_add(request, *args, **kwargs):
        result = original_add(request, *args, **kwargs)
        row = _active_request_row(context, request.request_id)
        slot = int(metadata.request_to_mamba_state_idx[row].item())
        assert slot >= 0
        runtime[("mamba-slot-acquire", request.request_id)] += 1
        runtime[("mamba-slot-value", request.request_id, slot)] += 1
        return result

    context.add_request = traced_add
    original_free = metadata.free_slots

    def traced_free(request_indices):
        request_ids = context.request_ids[request_indices].tolist()
        for request_id in request_ids:
            if request_id >= 0:
                runtime[("mamba-slot-free", request_id)] += 1
        return original_free(request_indices)

    metadata.free_slots = traced_free
    installed = 0
    for module in env.engine.controller.inference_wrapped_model.model.modules():
        if type(module).__name__ != "MambaMixer":
            continue
        original_forward = module.forward

        def traced_mamba(*args, _original=original_forward, **kwargs):
            request_ids = _active_request_ids(context)
            phase = env.engine.evicted_request_count
            output = _original(*args, **kwargs)
            for request_id in request_ids:
                runtime[("mamba-forward", phase, request_id)] += 1
            return output

        module.forward = traced_mamba
        installed += 1
    assert installed > 0


def _install_nccl_request_witnesses(env, runtime):
    context = env.engine.context
    installed = 0
    for module in env.engine.controller.inference_wrapped_model.model.modules():
        dispatcher = getattr(module, "_inference_token_dispatcher", None)
        if dispatcher is None or type(dispatcher).__name__ != "NCCLAllGatherDispatcher":
            continue
        original = dispatcher.token_dispatch

        def traced_dispatch(*args, _original=original, **kwargs):
            request_ids = _active_request_ids(context)
            result = _original(*args, **kwargs)
            for request_id in request_ids:
                runtime[("nccl-dispatch", request_id)] += 1
            return result

        dispatcher.token_dispatch = traced_dispatch
        installed += 1
    assert installed > 0


def _feature_keys(scenario):
    """Return the production counters which prove a scenario's companion behavior."""
    if scenario is _CHUNKED_RECOMPUTE:
        return ("log-probs-calculations",)
    if scenario is _PERSIST_TE_SWA:
        return (
            "module-forward:transformer-engine",
            "swa-kernel-calls",
            "full-attention-kernel-calls",
            "sink-correction-calls",
            "temperature-filter",
            "top-k-filter",
        )
    if scenario is _OFFLOAD_FP8:
        return (
            "module-forward:fp8",
            "fp8-context-forwards",
            "fp8-quantized-forwards",
            "fp8-recipe-forwards",
        )
    assert scenario is _UVM_OFFLOAD
    return ("module-forward:gpt",)


def _request_feature_key(request_id, feature):
    """Counter key for a feature invocation whose batch contained request_id."""
    return ("request-feature", request_id, feature)


def _instrument_request_correlated_runtime(env, scenario, runtime):
    """Tag production feature deltas with every real request in that invocation."""
    _instrument_scenario_runtime(env, scenario, runtime)
    controller = env.engine.controller
    context = env.engine.context
    feature_keys = set(_feature_keys(scenario))

    def active_request_ids():
        return context.request_ids[
            context.paused_request_count : context.total_request_count
        ].tolist()

    def instrument_call(owner, method_name, owned_keys):
        keys = feature_keys & set(owned_keys)
        if not keys:
            return
        original = getattr(owner, method_name)

        def correlated(*args, **kwargs):
            request_ids = active_request_ids()
            before = {key: runtime[key] for key in keys}
            result = original(*args, **kwargs)
            for key in keys:
                delta = runtime[key] - before[key]
                if delta > 0:
                    for request_id in request_ids:
                        runtime[_request_feature_key(request_id, key)] += delta
            return result

        setattr(owner, method_name, correlated)

    instrument_call(
        controller,
        "_dynamic_step_forward_logits",
        {
            "module-forward:transformer-engine",
            "swa-kernel-calls",
            "full-attention-kernel-calls",
            "sink-correction-calls",
            "module-forward:fp8",
            "fp8-context-forwards",
            "fp8-quantized-forwards",
            "fp8-recipe-forwards",
            "module-forward:gpt",
        },
    )
    instrument_call(controller._sampling, "sample_kernel", {"temperature-filter", "top-k-filter"})
    instrument_call(
        context, "calculate_log_probs_tensors", {"log-probs-calculations", "log-probs-kernel"}
    )


def _active_request_row(context, request_id):
    """Resolve one live request's bookkeeping row without assuming its order."""
    rows = torch.nonzero(
        context.request_ids[: context.total_request_count] == request_id, as_tuple=False
    ).flatten()
    assert rows.numel() == 1, (request_id, context.request_ids[: context.total_request_count])
    return int(rows.item())


def _request_kv_snapshot(context, request_id):
    """Copy only the KV blocks currently owned by one correlated request ID."""
    row = _active_request_row(context, request_id)
    block_ids = context.request_to_kv_block_ids[row]
    block_ids = torch.unique(block_ids[block_ids >= 0]).to(
        device=context.memory_buffer.device, dtype=torch.long
    )
    assert block_ids.numel() > 0
    return row, block_ids, context.memory_buffer.index_select(2, block_ids).clone()


def _coordinator_projection(request):
    """Round-trip a merged result through msgpack and the real reply handler."""
    coordinator = make_coordinator_direct(data_parallel_size=1, enable_prefix_caching=False)
    rank = coordinator.identities_of_data_parallel_ranks[0]
    request_id = request.request_id
    client = b"request-lifecycle-client"
    client_request_id = 1000 + request_id
    coordinator.router_socket = mock.Mock()
    coordinator.request_id_to_client_id = {request_id: client}
    coordinator.request_id_to_client_request_id = {request_id: client_request_id}
    coordinator.client_request_to_request_id = {(client, client_request_id): request_id}
    coordinator.request_id_to_rank = {request_id: rank}
    coordinator._pending_counts[coordinator.identity_to_rank_index[rank]] = 1

    engine_payload = msgpack.packb(
        [Headers.ENGINE_REPLY.value, [request.serialize()]], use_bin_type=True
    )
    handle_engine_reply(coordinator, rank, msgpack.unpackb(engine_payload, raw=False))

    frames = coordinator.router_socket.send_multipart.call_args.args[0]
    assert frames[0] == client
    header, returned_id, returned = msgpack.unpackb(frames[1], raw=False)
    assert header == Headers.ENGINE_REPLY.value
    assert returned_id == client_request_id
    assert request_id not in coordinator.request_id_to_client_id
    return DynamicInferenceRequest.deserialize(returned), unwrap_serialized_tensors(returned)


def _assert_coordinator_parity(actual, expected):
    """Compare stable client-visible fields; timing/event traces may differ."""
    exact_fields = (
        "request_id",
        "status",
        "prompt",
        "prompt_tokens",
        "remaining_prompt_tokens",
        "prompt_length",
        "generated_tokens",
        "generated_text",
        "generated_length",
        "sampling_params",
    )
    for field_name in exact_fields:
        actual_value = getattr(actual, field_name)
        expected_value = getattr(expected, field_name)
        if isinstance(actual_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value)
        elif field_name == "sampling_params":
            assert actual_value.serialize() == expected_value.serialize()
        else:
            assert actual_value == expected_value
    for field_name in ("prompt_log_probs", "generated_log_probs"):
        assert getattr(actual, field_name) == pytest.approx(
            getattr(expected, field_name), rel=0, abs=1.0e-3
        )
    for field_name in ("prompt_top_n_logprobs", "generated_top_n_logprobs"):
        actual_values = getattr(actual, field_name)
        expected_values = getattr(expected, field_name)
        assert len(actual_values) == len(expected_values)
        for actual_row, expected_row in zip(actual_values, expected_values):
            assert actual_row.keys() == expected_row.keys()
            assert list(actual_row.values()) == pytest.approx(
                list(expected_row.values()), rel=0, abs=1.0e-3
            )


def _assert_coordinator_result_contract(request, endpoint_result):
    """Check client-visible tensor and score lengths, not only pair parity."""
    prompt_length = len(request.prompt_tokens)
    generated_length = len(request.generated_tokens)
    assert request.prompt_length == prompt_length
    assert torch.equal(request.remaining_prompt_tokens, request.prompt_tokens)
    assert endpoint_result["prompt_tokens"] == request.prompt_tokens.tolist()
    assert endpoint_result["remaining_prompt_tokens"] == request.prompt_tokens.tolist()
    assert len(request.prompt_log_probs) == max(0, prompt_length - 1)
    assert len(request.prompt_top_n_logprobs) == max(0, prompt_length - 1)
    assert len(request.generated_log_probs) == generated_length
    assert len(request.generated_top_n_logprobs) == generated_length
    assert request.generated_length == generated_length


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRequestLifecyclePairwise(_DynamicInferenceEngineTestBase):
    """Run one seeded baseline and one request-correlated lifecycle treatment."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        _set_rounder(64)
        Utils.destroy_model_parallel()

    @staticmethod
    def _cleanup():
        gc.collect()
        delete_cuda_graphs()
        torch.cuda.empty_cache()

    @staticmethod
    def _feature_keys(scenario):
        return _feature_keys(scenario)

    @classmethod
    def _intervene(cls, scenario, engine, target_id, runtime):
        context = engine.context
        request = engine.get_request(target_id)
        feature_keys = cls._feature_keys(scenario)
        counts_before = {key: runtime[_request_feature_key(target_id, key)] for key in feature_keys}
        assert all(count > 0 for count in counts_before.values())

        if scenario is _CHUNKED_RECOMPUTE:
            assert context.chunked_prefill_request_id == target_id
            assert request.finished_chunk_token_count > 0
            assert 0 < len(request.remaining_prompt_tokens) < len(request.prompt_tokens)
            assert request.generated_tokens == []
            witness = {
                "request_id": target_id,
                "finished_chunk_token_count": request.finished_chunk_token_count,
                "remaining_prompt_length": len(request.remaining_prompt_tokens),
            }
            engine.suspend()
            assert engine.context.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE
            assert len(engine.requests[target_id].record.requests) == 1
            engine.resume()
            resumed = engine.get_request(target_id)
            assert resumed.finished_chunk_token_count == 0
            assert torch.equal(resumed.remaining_prompt_tokens, resumed.prompt_tokens)
            witness["record_segments_after_resume"] = len(
                engine.requests[target_id].record.requests
            )
        else:
            row, block_ids, kv_before = _request_kv_snapshot(context, target_id)
            pointer_before = context.memory_buffer.data_ptr()
            storage_bytes_before = context.memory_buffer.untyped_storage().nbytes()
            witness = {
                "request_id": target_id,
                "row": row,
                "block_ids": block_ids.tolist(),
                "pointer_before": pointer_before,
                "storage_bytes_before": storage_bytes_before,
            }
            engine.suspend()
            assert not context.is_tensor_state_allocated
            if scenario is _PERSIST_TE_SWA:
                assert context.kv_cache_management_mode == KVCacheManagementMode.PERSIST
                assert context.static_kv_memory_pointers
                assert context.memory_buffer.data_ptr() == pointer_before
                assert torch.equal(context.memory_buffer.index_select(2, block_ids), kv_before)
            elif scenario is _UVM_OFFLOAD:
                allocator = context.kv_block_allocator
                cuda_only_usable_blocks = allocator.pool_size - allocator.paused_limit - 1
                assert context.unified_memory_level == 1
                assert context.unified_memory_mempool is not None
                assert context.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
                assert context.static_kv_memory_pointers
                assert any(block_id >= cuda_only_usable_blocks for block_id in block_ids.tolist())
                prefetch_managed_tensor(context.memory_buffer, device=-1)
                torch.cuda.current_stream().synchronize()
                prefetch_managed_tensor(context.memory_buffer, device=torch.cuda.current_device())
                torch.cuda.current_stream().synchronize()
                assert context.memory_buffer.data_ptr() == pointer_before
                assert torch.equal(context.memory_buffer.index_select(2, block_ids), kv_before)
                witness["cuda_only_usable_blocks"] = cuda_only_usable_blocks
                witness["prefetch_succeeded"] = True
            else:
                assert scenario is _OFFLOAD_FP8
                assert context.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
                assert not context.static_kv_memory_pointers
                assert context.memory_buffer.untyped_storage().nbytes() == 0
                assert context._offloadable_cpu_backups
            engine.resume()
            assert context.is_tensor_state_allocated
            resumed_row = _active_request_row(context, target_id)
            assert resumed_row == row
            assert torch.equal(context.memory_buffer.index_select(2, block_ids), kv_before)
            if scenario in (_PERSIST_TE_SWA, _UVM_OFFLOAD):
                assert context.memory_buffer.data_ptr() == pointer_before
            else:
                assert context.memory_buffer.untyped_storage().nbytes() == storage_bytes_before

        witness["feature_counts_before"] = counts_before
        return witness

    @classmethod
    def _run_once(cls, scenario, *, treatment):
        config_values = dict(_BASE_PAIR_CONFIG)
        config_values.update(scenario.config)
        config_values.update(num_requests=0, async_sched_mode=AsyncScheduleMode.LEGACY)
        test_config = _DynamicEngineTestConfig(**config_values)
        if test_config.unified_memory_level:
            test_config.inference_config_overrides["unified_memory_level"] = (
                test_config.unified_memory_level
            )
        test_config.use_flashinfer_fused_rope = False
        env = cls._build_test_env(test_config)
        engine = env.engine
        if scenario is _UVM_OFFLOAD:
            assert engine.context.unified_memory_level == 1, (
                "the designated UVM owner must use managed allocation; "
                "a fallback run receives no pairwise credit"
            )
        detokenize = lambda tokens, **_kwargs: "".join(f"<{token}>" for token in tokens)
        engine.controller.tokenizer.detokenize = detokenize
        engine.controller.detokenize = lambda _tokenizer, tokens, **kwargs: detokenize(
            tokens, **kwargs
        )
        requests = _make_scenario_requests(env, scenario)[:3]
        env.requests = requests
        target_id = requests[0].request_id
        futures = [engine._add_request(request) for request in requests]
        runtime = Counter()
        _instrument_request_correlated_runtime(env, scenario, runtime)
        feature_keys = cls._feature_keys(scenario)
        intervention = None
        completed = {}
        record_lengths = {}

        for step in range(128):
            result = engine.step_modern()
            runtime["steps"] += 1
            for record in result["finished_request_records"]:
                merged = record.merge()
                completed[merged.request_id] = merged
                record_lengths[merged.request_id] = len(record.requests)

            if treatment and intervention is None and target_id in engine.requests:
                target = engine.get_request(target_id)
                if scenario is _CHUNKED_RECOMPUTE:
                    ready = (
                        target.finished_chunk_token_count > 0
                        and len(target.remaining_prompt_tokens) < len(target.prompt_tokens)
                        and not target.generated_tokens
                    )
                elif scenario is _PERSIST_TE_SWA:
                    ready = len(target.generated_tokens) >= 2
                else:
                    ready = len(target.generated_tokens) >= 1
                if ready:
                    intervention = cls._intervene(scenario, engine, target_id, runtime)

            if not engine.has_unfinished_requests():
                break
        else:
            pytest.fail(f"{scenario.name} did not drain within 128 steps")

        assert len(completed) == len(requests)
        assert set(completed) == {request.request_id for request in requests}
        assert all(future.done() for future in futures)
        assert not engine.requests
        assert not engine.waiting_request_ids
        assert not engine.failed_request_ids
        assert engine.context.total_request_count == 0
        assert engine.context.paused_request_count == 0
        if treatment:
            assert intervention is not None
            assert intervention["request_id"] == target_id
            for key in feature_keys:
                assert (
                    runtime[_request_feature_key(target_id, key)]
                    > intervention["feature_counts_before"][key]
                )

        finished = [completed[request_id] for request_id in sorted(completed)]
        assert all(request.status == Status.COMPLETED for request in finished)
        if scenario is _CHUNKED_RECOMPUTE:
            assert runtime["log-probs-calculations"] > 0
            assert all(request.prompt_log_probs is not None for request in finished)
            assert all(request.generated_log_probs is not None for request in finished)
            assert all(request.prompt_top_n_logprobs for request in finished)
            assert all(request.generated_top_n_logprobs for request in finished)
        return _RunResult(finished, record_lengths, runtime, intervention)

    @classmethod
    def _assert_pair(cls, scenario, *, coordinator=False):
        _check_scenario_prerequisite(scenario)
        baseline = cls._run_once(scenario, treatment=False)
        baseline_by_id = {request.request_id: request for request in baseline.requests}
        baseline_snapshot_by_id = {
            request.request_id: snapshot
            for request, snapshot in zip(baseline.requests, _snapshot_requests(baseline.requests))
        }
        baseline_text_by_id = {
            request.request_id: request.generated_text for request in baseline.requests
        }
        baseline_wire = (
            {
                request_id: _coordinator_projection(request)
                for request_id, request in baseline_by_id.items()
            }
            if coordinator
            else None
        )
        del baseline
        cls._cleanup()

        treatment = cls._run_once(scenario, treatment=True)
        assert {request.request_id for request in treatment.requests} == set(
            baseline_snapshot_by_id
        )
        _assert_request_parity(
            treatment.requests,
            [baseline_snapshot_by_id[request.request_id] for request in treatment.requests],
            scenario.atol,
            exact_numerics=True,
            exact_top_n=True,
        )
        assert {
            request.request_id: request.generated_text for request in treatment.requests
        } == baseline_text_by_id
        if coordinator:
            treatment_wire = {
                request.request_id: _coordinator_projection(request)
                for request in treatment.requests
            }
            assert set(treatment_wire) == set(baseline_wire)
            for request_id, (actual, actual_endpoint) in treatment_wire.items():
                expected, _ = baseline_wire[request_id]
                _assert_coordinator_parity(actual, expected)
                _assert_coordinator_result_contract(actual, actual_endpoint)
        assert treatment.witness is not None
        return treatment

    @classmethod
    def _manual_env(cls, **config_values):
        vocab_size = config_values.pop("vocab_size", None)
        values = {
            "context_max_requests": 4,
            "context_buffer_size_gb": 0.01,
            "context_paused_buffer_size_gb": 0.0,
            "track_generated_token_events": True,
            "inference_config_overrides": {"track_paused_request_events": True},
        }
        values.update(config_values)
        config = _DynamicEngineTestConfig(num_requests=0, **values)
        if vocab_size is not None:
            config.vocab_size = vocab_size
        config.use_flashinfer_fused_rope = False
        env = cls._build_test_env(config)
        _configure_test_tokenizer(env.engine)
        return config, env

    @classmethod
    def _run_chunked_companion_evict(cls, *, treatment):
        config, env = cls._manual_env(
            max_sequence_length=544,
            context_max_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
            materialize_only_last_token_logits=True,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        runtime = Counter()
        _install_incrementing_logits(env, runtime, config.vocab_size)
        companion_id, chunk_id = 41, 73
        companion = _make_manual_request(
            context, companion_id, torch.arange(255, device="cuda", dtype=torch.int64) % 97, 6
        )
        chunk = _make_manual_request(
            context, chunk_id, (torch.arange(513, device="cuda", dtype=torch.int64) + 17) % 97, 4
        )
        head_filler = _allocate_leaving(allocator, 4) if treatment else None
        futures = [engine._add_request(companion)]
        completed, record_lengths = {}, {}
        _collect_finished(engine.step_modern(), completed, record_lengths)
        assert (
            companion.generated_tokens
            and context.request_last_kv_block_offset[
                _active_request_row(context, companion_id)
            ].item()
            == 255
        )
        target_forwards_before = runtime[("model-forward", 0, companion_id)]

        futures.append(engine._add_request(chunk))
        engine.schedule_waiting_requests()
        assert context.chunked_prefill_request_id == chunk_id
        assert chunk.finished_chunk_token_count == 255
        assert len(chunk.remaining_prompt_tokens) == 258
        tail_filler = _allocate_leaving(allocator, 0) if treatment else None
        evictions_before = engine.evicted_request_count
        _collect_finished(engine.step_modern(), completed, record_lengths)
        witness = None
        if treatment:
            record = engine.requests[companion_id].record
            merged = record.merge()
            assert engine.evicted_request_count == evictions_before + 1
            assert len(record.requests) == 2
            assert _event_count(merged, DynamicInferenceEventType.PAUSE) == 1
            assert _event_count(merged, DynamicInferenceEventType.EVICT) == 1
            assert context.chunked_prefill_request_id == chunk_id
            assert chunk.finished_chunk_token_count == 255
            assert _event_count(chunk, DynamicInferenceEventType.EVICT) == 0
            assert runtime[("model-forward", 0, companion_id)] > target_forwards_before
            assert runtime[("model-forward", 0, chunk_id)] > 0
            witness = {
                "request_id": companion_id,
                "chunk_id": chunk_id,
                "partial_chunk_tokens": chunk.finished_chunk_token_count,
            }
        _release_filler(allocator, tail_filler)
        _release_filler(allocator, head_filler)

        for _ in range(128):
            if not engine.has_unfinished_requests():
                break
            _collect_finished(engine.step_modern(), completed, record_lengths)
        else:
            pytest.fail("chunked companion eviction did not drain")
        _assert_engine_drained(engine, futures, completed, (companion_id, chunk_id))
        requests = [completed[request_id] for request_id in (companion_id, chunk_id)]
        if treatment:
            assert record_lengths == {companion_id: 2, chunk_id: 1}
        return _RunResult(requests, record_lengths, runtime, witness)

    @classmethod
    def _run_hybrid_mamba_stop_keep(cls, *, treatment):
        config, env = cls._manual_env(
            model_provider="hybrid",
            max_sequence_length=320,
            context_max_tokens=320,
            materialize_only_last_token_logits=False,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        assert context.mamba_slot_allocator is None
        runtime = Counter()
        _install_incrementing_logits(env, runtime, config.vocab_size)
        _install_mamba_slot_witnesses(env, runtime)
        request_id = 61
        prompt = torch.zeros(255, device="cuda", dtype=torch.int64)
        prompt[-1] = 10
        request = _make_manual_request(
            context, request_id, prompt, 8, full_logprobs=True, stop_words=["12 13"], keep_stop=True
        )
        filler = _allocate_leaving(allocator, 1) if treatment else None
        future = engine._add_request(request)
        completed, record_lengths = {}, {}
        _collect_finished(engine.step_modern(), completed, record_lengths)
        assert request.generated_tokens == [11]
        row = _active_request_row(context, request_id)
        slot_before = int(context.mamba_metadata.request_to_mamba_state_idx[row].item())
        _collect_finished(engine.step_modern(), completed, record_lengths)
        witness = None
        if treatment:
            record = engine.requests[request_id].record
            assert engine.evicted_request_count == 1
            assert len(record.requests) == 2
            assert record.merge().generated_tokens == [11, 12]
            assert context.total_request_count == 0
            assert context.mamba_metadata.request_to_mamba_state_idx[0].item() == -1
            assert context.mamba_metadata.mamba_state_free_slot_count == context.max_requests
            assert runtime[("mamba-slot-free", request_id)] == 1
            _release_filler(allocator, filler)
            filler = None
            engine.schedule_waiting_requests()
            row = _active_request_row(context, request_id)
            slot_after = int(context.mamba_metadata.request_to_mamba_state_idx[row].item())
            assert context.mamba_metadata.mamba_state_free_slot_count == context.max_requests - 1
            assert runtime[("mamba-slot-acquire", request_id)] == 2
            witness = {
                "request_id": request_id,
                "slot_before": slot_before,
                "slot_after": slot_after,
            }

        for _ in range(16):
            if not engine.has_unfinished_requests():
                break
            _collect_finished(engine.step_modern(), completed, record_lengths)
        else:
            pytest.fail("hybrid Mamba stop-keep request did not drain")
        _release_filler(allocator, filler)
        _assert_engine_drained(engine, [future], completed, (request_id,))
        merged = completed[request_id]
        assert merged.generated_tokens == [11, 12, 13]
        score_count = len(merged.generated_tokens)
        assert (
            len(merged.generated_log_probs) == len(merged.generated_top_n_logprobs) == score_count
        )
        if treatment:
            assert record_lengths[request_id] == 2
            assert _event_count(merged, DynamicInferenceEventType.EVICT) == 1
            assert runtime[("mamba-forward", 0, request_id)] > 0
            assert runtime[("mamba-forward", 1, request_id)] > 0
        return _RunResult([merged], record_lengths, runtime, witness)

    @classmethod
    def _run_repeated_mtp_stop_strip(cls, *, treatment):
        config, env = cls._manual_env(
            max_sequence_length=600,
            context_max_tokens=640,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            position_embedding_type="none",
            vocab_size=512,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        runtime = Counter()
        _install_real_mtp_logits(env, runtime, config.vocab_size)
        request_id = 87
        prompt = torch.remainder(torch.arange(255, device="cuda") + 268, config.vocab_size)
        request = _make_manual_request(
            context,
            request_id,
            prompt,
            300,
            full_logprobs=True,
            stop_words=["267 268"],
            keep_stop=False,
        )
        first_filler = _allocate_leaving(allocator, 1) if treatment else None
        future = engine._add_request(request)
        completed, record_lengths = {}, {}
        _collect_finished(engine.step_modern(), completed, record_lengths)
        if treatment:
            assert engine.evicted_request_count == 1
            assert len(engine.requests[request_id].record.requests) == 2
            assert engine.requests[request_id].record.merge().generated_tokens == [11]
            _release_filler(allocator, first_filler)
            first_filler = None

        second_filler = None
        second_boundary = None
        for _ in range(160):
            if not engine.has_unfinished_requests():
                break
            live_ids = context.request_ids[: context.total_request_count].tolist()
            if (
                treatment
                and engine.evicted_request_count == 1
                and second_filler is None
                and request_id in live_ids
                and len(engine.requests[request_id].record.requests) == 2
            ):
                row = _active_request_row(context, request_id)
                block_count = int(context.request_kv_block_counts[row].item())
                offset = int(context.request_last_kv_block_offset[row].item())
                if block_count == 2 and offset >= 253:
                    second_filler = _allocate_leaving(allocator, 0)
                    assert second_filler is not None
            _collect_finished(engine.step_modern(), completed, record_lengths)
            if treatment and engine.evicted_request_count == 2 and second_boundary is None:
                partial = engine.requests[request_id].record.merge()
                second_boundary = (len(partial.generated_tokens), partial.generated_tokens[-1])
                _release_filler(allocator, second_filler)
                second_filler = None
        else:
            pytest.fail("repeated MTP eviction request did not drain")
        _release_filler(allocator, first_filler)
        _release_filler(allocator, second_filler)
        _assert_engine_drained(engine, [future], completed, (request_id,))
        merged = completed[request_id]
        assert merged.generated_tokens == list(range(11, 267))
        score_count = len(merged.generated_tokens)
        assert (
            len(merged.generated_log_probs) == len(merged.generated_top_n_logprobs) == score_count
        )
        assert _event_count(merged, DynamicInferenceEventType.GENERATED_TOKEN) == len(
            merged.generated_tokens
        )
        witness = None
        if treatment:
            assert second_boundary == (257, 267)
            assert engine.evicted_request_count == 2
            assert record_lengths[request_id] == 3
            assert _event_count(merged, DynamicInferenceEventType.PAUSE) == 2
            assert _event_count(merged, DynamicInferenceEventType.EVICT) == 2
            for phase in range(3):
                for depth in range(2):
                    assert runtime[("mtp-depth", phase, request_id, depth)] > 0
            assert all(int(count) > 0 for count in engine._spec_tokens_proposed_per_pos)
            witness = {
                "request_id": request_id,
                "evictions": engine.evicted_request_count,
                "second_boundary": second_boundary,
            }
        return _RunResult([merged], record_lengths, runtime, witness)

    @torch.inference_mode()
    def test_chunked_companion_zero_budget_evict(self):
        try:
            treatment = _run_treatment_pair(self._run_chunked_companion_evict, self._cleanup)
            assert treatment.witness["partial_chunk_tokens"] == 255
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_single_evict_hybrid_mamba_stop_keep(self):
        _check_scenario_prerequisite(_HYBRID_MAMBA_STOP_KEEP)
        try:
            treatment = _run_treatment_pair(self._run_hybrid_mamba_stop_keep, self._cleanup)
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_repeated_evict_mtp2_stop_strip(self):
        try:
            treatment = _run_treatment_pair(self._run_repeated_mtp_stop_strip, self._cleanup)
            assert treatment.witness["evictions"] == 2
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_generate_drains_real_invalid_and_mixed_admission(self):
        """Direct generation returns real admission failures without leaked work."""
        try:
            env = self._build_test_env(
                _DynamicEngineTestConfig(
                    num_requests=0,
                    max_sequence_length=12,
                    context_max_requests=4,
                    num_tokens_to_generate=2,
                )
            )
            engine = env.engine
            engine.controller.tokenize_prompt = lambda _tokenizer, prompt, _add_bos=False: (
                list(range(20)) if prompt == "invalid" else [1, 2, 3]
            )
            params = SamplingParams(num_tokens_to_generate=2, termination_id=-1)

            invalid_only = engine.generate(["invalid"], params)
            assert [record.merge().status for record in invalid_only] == [Status.FAILED]
            assert not engine.requests and not engine.failed_request_ids

            mixed = engine.generate(["invalid", "valid"], params)
            assert [record.merge().status for record in mixed] == [Status.FAILED, Status.COMPLETED]
            assert not engine.requests
            assert not engine.failed_request_ids
            assert not engine.waiting_request_ids
            assert engine.context.total_request_count == 0
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_chunked_partial_recompute_api_coordinator(self):
        """A partial chunk survives RECOMPUTE with identical API-visible scores."""
        try:
            result = self._assert_pair(_CHUNKED_RECOMPUTE, coordinator=True)
            assert result.witness["finished_chunk_token_count"] > 0
            assert result.witness["record_segments_after_resume"] == 1
            assert result.record_lengths[0] == 1
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_persist_te_swa_stochastic(self):
        """PERSIST keeps one request's KV bytes live across a TE/SWA sampled run."""
        try:
            result = self._assert_pair(_PERSIST_TE_SWA)
            assert result.witness["pointer_before"] > 0
            for key in self._feature_keys(_PERSIST_TE_SWA):
                assert (
                    result.runtime[_request_feature_key(result.witness["request_id"], key)]
                    > result.witness["feature_counts_before"][key]
                    > 0
                )
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_offload_dynamic_fp8(self):
        """OFFLOAD removes and restores one FP8 request's KV bytes without drift."""
        try:
            result = self._assert_pair(_OFFLOAD_FP8)
            assert result.witness["storage_bytes_before"] > 0
            for key in self._feature_keys(_OFFLOAD_FP8):
                assert (
                    result.runtime[_request_feature_key(result.witness["request_id"], key)]
                    > result.witness["feature_counts_before"][key]
                    > 0
                )
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_uvm_offload_static_managed_capacity(self):
        """A managed request uses added capacity and keeps its address through prefetch."""
        try:
            result = self._assert_pair(_UVM_OFFLOAD)
            assert result.witness["prefetch_succeeded"]
            assert max(result.witness["block_ids"]) >= result.witness["cuda_only_usable_blocks"]
            assert (
                result.runtime[
                    _request_feature_key(result.witness["request_id"], "module-forward:gpt")
                ]
                > (result.witness["feature_counts_before"]["module-forward:gpt"])
                > 0
            )
        finally:
            self._cleanup()


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRequestLifecyclePairwiseEP2(_DynamicInferenceEngineTestBase):

    @staticmethod
    def _cleanup():
        gc.collect()
        delete_cuda_graphs()
        torch.cuda.empty_cache()

    @classmethod
    def _run_retained_pause_ep2(cls, *, treatment):
        config, env = TestRequestLifecyclePairwise._manual_env.__func__(
            cls,
            **_EP2_NCCL.config,
            max_sequence_length=320,
            context_max_tokens=320,
            context_paused_buffer_size_gb=0.002,
            materialize_only_last_token_logits=True,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        assert allocator.paused_limit >= 1
        runtime = Counter()
        _instrument_scenario_runtime(env, _EP2_NCCL, runtime)
        _install_incrementing_logits(env, runtime, config.vocab_size)
        _install_nccl_request_witnesses(env, runtime)
        target_id, companion_id = 61, 62
        target = _make_manual_request(
            context, target_id, torch.arange(255, device="cuda", dtype=torch.int64) % 97, 8
        )
        companion = _make_manual_request(
            context, companion_id, torch.arange(8, device="cuda", dtype=torch.int64) + 13, 12
        )
        filler = _allocate_leaving(allocator, 2) if treatment else None
        futures = [engine._add_request(target), engine._add_request(companion)]
        completed, record_lengths = {}, {}
        _collect_finished(engine.step_modern(), completed, record_lengths)
        _collect_finished(engine.step_modern(), completed, record_lengths)
        witness = None
        dispatch_at_pause = None
        if treatment:
            assert context.paused_request_count == 1
            assert context.request_ids[0].item() == target_id
            assert allocator.get_paused_used() == 1
            assert engine.evicted_request_count == 0
            assert len(engine.requests[target_id].record.requests) == 1
            paused = engine.requests[target_id].record.merge()
            assert _event_count(paused, DynamicInferenceEventType.PAUSE) == 1
            assert _event_count(paused, DynamicInferenceEventType.EVICT) == 0
            dispatch_at_pause = runtime[("nccl-dispatch", target_id)]
            assert dispatch_at_pause > 0
            _release_filler(allocator, filler)
            filler = None

        for _ in range(32):
            if not engine.has_unfinished_requests():
                break
            _collect_finished(engine.step_modern(), completed, record_lengths)
        else:
            pytest.fail("EP2 retained-pause requests did not drain")
        _release_filler(allocator, filler)
        _assert_engine_drained(engine, futures, completed, (target_id, companion_id))
        assert runtime["nccl-token-dispatches"] == runtime["nccl-token-combines"] > 0
        assert runtime["nccl-combine-before-dispatch"] == 0
        assert runtime["nccl-dispatch-inflight"] == 0
        if treatment:
            assert runtime[("nccl-dispatch", target_id)] > dispatch_at_pause
            assert engine.evicted_request_count == 0
            assert record_lengths[target_id] == 1
            witness = {
                "request_id": target_id,
                "dispatches_at_pause": dispatch_at_pause,
                "dispatches_total": runtime[("nccl-dispatch", target_id)],
            }
        requests = [completed[request_id] for request_id in (target_id, companion_id)]
        return _RunResult(requests, record_lengths, runtime, witness)

    @torch.inference_mode()
    def test_retained_pause_ep2(self):
        if int(os.environ.get("WORLD_SIZE", "1")) != 2:
            pytest.skip("the retained-pause EP2 owner requires exactly two GPUs")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        )
        try:
            treatment = _run_treatment_pair(self._run_retained_pause_ep2, self._cleanup)
            assert treatment.witness["dispatches_total"] > treatment.witness["dispatches_at_pause"]
        finally:
            self._cleanup()
            _set_rounder(64)
            Utils.destroy_model_parallel()
