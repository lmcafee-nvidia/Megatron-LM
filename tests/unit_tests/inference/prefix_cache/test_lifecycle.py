# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-model request eviction, checkpoint, requeue, and resume stress."""

import gc

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    Status,
    compute_block_hashes_batched,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
from tests.unit_tests.test_utilities import Utils

_EPOCH = 17
_REQUEST_COUNT = 4
_POOL_SIZE = 7  # Six usable blocks and one dummy block.
_POLICIES = (PrefixCachingEvictionPolicy.REF_ZERO, PrefixCachingEvictionPolicy.LRU)


def _prompts(block_size, vocab_size):
    """Create one slow anchor and three near-boundary requests with one shared block."""
    device = torch.cuda.current_device()
    shared = torch.arange(block_size, dtype=torch.int64, device=device).add_(1)
    shared.remainder_(vocab_size - 1)
    anchor = torch.cat((shared, shared.new_tensor([vocab_size - 2])))
    pressure = [
        torch.cat(
            (
                shared,
                (
                    torch.arange(block_size - 1, dtype=torch.int64, device=device)
                    + 13 * request_id
                    + 7
                )
                % (vocab_size - 1),
            )
        )
        for request_id in range(1, _REQUEST_COUNT)
    ]
    return [anchor, *pressure]


def _snapshot(record):
    request = record.merge()
    return {
        "status": request.status,
        "tokens": list(request.generated_tokens),
        "logprobs": [float(value) for value in request.generated_log_probs],
        "segments": len(record.requests),
        "eviction_events": sum(
            event.type == DynamicInferenceEventType.EVICT for event in request.events
        ),
    }


def _assert_output_parity(reference, actual):
    assert set(actual) == set(reference)
    for request_id, expected in reference.items():
        observed = actual[request_id]
        assert expected["status"] == observed["status"] == Status.COMPLETED
        assert observed["tokens"] == expected["tokens"]
        assert len(observed["logprobs"]) == len(observed["tokens"])
        assert len(expected["logprobs"]) == len(expected["tokens"])
        torch.testing.assert_close(
            torch.tensor(observed["logprobs"], dtype=torch.float32),
            torch.tensor(expected["logprobs"], dtype=torch.float32),
            rtol=0.0,
            atol=1.0e-6,
        )


def _install_lifecycle_observers(engine, *, enabled, block_size):
    """Observe the production checkpoint and admission paths without replacing them."""
    context = engine.context
    pending_by_object = {}
    scheduled = {}
    executed = set()
    checkpoints = []
    requeues = []

    original_add_request = context.add_request

    def observed_add_request(request, *args, **kwargs):
        result = original_add_request(request, *args, **kwargs)
        tag = pending_by_object.pop(id(request), None)
        if tag is None:
            return result

        request_idx = context.total_request_count - 1
        computed_tokens = int(context.request_query_lengths[request_idx].item())
        skipped_tokens = len(request.remaining_prompt_tokens) - computed_tokens
        scheduled[tag] = {
            "cached_tokens": request.num_cached_tokens,
            "skipped_tokens": skipped_tokens,
            "mamba_matches": getattr(request, "_mamba_num_matched_blocks", 0),
        }
        return result

    context.add_request = observed_add_request
    original_post_process = engine.post_process_requests

    def observed_post_process(*args, **kwargs):
        request_ids = kwargs.get("request_ids", args[0])
        evict_request_ids = kwargs.get("evict_request_ids", args[2])

        for request_id in request_ids.tolist():
            entry = engine.requests.get(request_id)
            if entry is None:
                continue
            tag = (request_id, len(entry.record.requests) - 1)
            if tag in scheduled:
                executed.add(tag)

        evicted = [] if evict_request_ids is None else evict_request_ids.tolist()
        result = original_post_process(*args, **kwargs)

        for request_id in evicted:
            record = engine.requests[request_id].record
            assert len(record.requests) >= 2
            previous = record[-2]
            checkpointed = record[-1]
            tag = (request_id, len(record.requests) - 1)

            expected_prompt = torch.cat(
                (
                    previous.prompt_tokens,
                    torch.tensor(
                        previous.generated_tokens,
                        dtype=previous.prompt_tokens.dtype,
                        device=previous.prompt_tokens.device,
                    ),
                )
            )
            assert previous.generated_tokens
            assert previous.enable_prefix_caching is enabled
            assert checkpointed.enable_prefix_caching is enabled
            assert checkpointed.block_size_tokens == block_size
            assert checkpointed.prefix_cache_namespace == _EPOCH
            assert torch.equal(checkpointed.prompt_tokens, expected_prompt)
            assert (
                checkpointed.sampling_params.num_tokens_to_generate
                == previous.sampling_params.num_tokens_to_generate - len(previous.generated_tokens)
            )
            expected_hashes = (
                compute_block_hashes_batched(
                    checkpointed.prompt_tokens, block_size, namespace=_EPOCH
                )
                if enabled
                else []
            )
            assert checkpointed.precomputed_block_hashes == expected_hashes
            assert (
                checkpointed.precomputed_block_hashes[: len(previous.precomputed_block_hashes)]
                == previous.precomputed_block_hashes
            )
            assert checkpointed.sampling_params.return_log_probs
            assert checkpointed.sampling_params.skip_prompt_log_probs
            assert checkpointed.events[-1].type == DynamicInferenceEventType.EVICT
            assert list(engine.waiting_request_ids).count(request_id) == 1

            pending_by_object[id(checkpointed)] = tag
            checkpoints.append(tag)
            requeues.append(tag)

        return result

    engine.post_process_requests = observed_post_process
    return pending_by_object, scheduled, executed, checkpoints, requeues


def _run_lifecycle(*, architecture, block_size, enabled, policy):
    generation_tokens = 2 * block_size + 8
    config = _DynamicEngineTestConfig(
        num_requests=0,
        min_prompt_length=block_size + 1,
        max_prompt_length=2 * block_size - 1,
        num_tokens_to_generate=generation_tokens,
        max_sequence_length=4 * block_size + 16,
        context_block_size_tokens=block_size,
        context_max_requests=_REQUEST_COUNT,
        context_max_tokens=6 * block_size,
        context_buffer_size_gb=0.1,
        context_paused_buffer_size_gb=0.0,
        model_provider=architecture,
        materialize_only_last_token_logits=True,
        enable_prefix_caching=enabled,
        prefix_caching_eviction_policy=policy,
        prefix_caching_mamba_gb=(0.01 if enabled and architecture == "hybrid" else None),
        logprobs_mode="raw_logprobs",
        use_flashinfer_fused_rope=False,
        sampling_backend="torch",
    )
    env = TestPrefixCacheRequestLifecycle._build_test_env(config)
    engine = env.engine
    context = engine.context
    previous_allocator = context.kv_block_allocator
    context.kv_block_allocator = KVBlockAllocator(
        context,
        pool_size=_POOL_SIZE,
        paused_limit=0,
        enable_prefix_caching=enabled,
        prefix_caching_eviction_policy=policy,
    )
    if context.mamba_slot_allocator is not None:
        context.kv_block_allocator.on_blocks_deregistered = (
            context.mamba_slot_allocator.on_kv_blocks_deregistered
        )
    del previous_allocator

    engine._apply_generation_epoch(_EPOCH)
    pending, scheduled, executed, checkpoints, requeues = _install_lifecycle_observers(
        engine, enabled=enabled, block_size=block_size
    )

    for request_id, prompt in enumerate(_prompts(block_size, config.vocab_size)):
        engine.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=generation_tokens,
                termination_id=-1,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                top_k=1,
                return_prompt_tokens=True,
            ),
        )

    finished = {}
    step_count = 0
    while engine.has_unfinished_requests():
        result = engine.step_modern()
        for record in result["finished_request_records"]:
            assert record.request_id not in finished
            finished[record.request_id] = _snapshot(record)
        step_count += 1
        assert step_count < 5_000, "request lifecycle stress did not converge"

    assert set(finished) == set(range(_REQUEST_COUNT))
    assert not pending
    assert len(checkpoints) == len(requeues) == engine.evicted_request_count
    assert len(checkpoints) >= 3
    assert set(scheduled) == set(checkpoints)
    assert executed == set(checkpoints)
    for output in finished.values():
        assert output["segments"] == output["eviction_events"] + 1
        assert len(output["tokens"]) == generation_tokens

    metrics = engine.get_prefix_cache_metrics()
    evidence = {
        "evictions": engine.evicted_request_count,
        "resumes": len(executed),
        "hit_resumes": sum(
            tag in executed
            and values["cached_tokens"] >= block_size
            and values["skipped_tokens"] >= block_size
            for tag, values in scheduled.items()
        ),
        "mamba_hit_resumes": sum(
            tag in executed
            and values["skipped_tokens"] >= block_size
            and values["mamba_matches"] >= 1
            for tag, values in scheduled.items()
        ),
        "metrics": metrics,
    }

    del env
    gc.collect()
    torch.cuda.empty_cache()
    return finished, evidence


@pytest.mark.internal
class TestPrefixCacheRequestLifecycle(_DynamicInferenceEngineTestBase):
    """Compare one cache-off reference with both eviction policies per architecture."""

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
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "architecture,block_size", (("gpt", 16), ("hybrid", 64)), ids=("gpt", "hybrid")
    )
    @torch.inference_mode()
    def test_checkpoint_requeue_resume_execution(self, architecture, block_size):
        """Stress every architecture-policy pair against one cache-off execution."""
        if architecture == "hybrid":
            sequence_packing_available, reason = _check_mamba_sequence_packing_support()
            assert sequence_packing_available, reason

        reference, reference_evidence = _run_lifecycle(
            architecture=architecture,
            block_size=block_size,
            enabled=False,
            policy=PrefixCachingEvictionPolicy.REF_ZERO,
        )
        assert reference_evidence["evictions"] >= 3
        assert reference_evidence["resumes"] >= 3
        assert reference_evidence["hit_resumes"] == 0
        assert reference_evidence["metrics"]["enabled"] is False
        assert reference_evidence["metrics"]["hits"] == 0

        for policy in _POLICIES:
            actual, evidence = _run_lifecycle(
                architecture=architecture, block_size=block_size, enabled=True, policy=policy
            )
            _assert_output_parity(reference, actual)
            assert evidence["evictions"] >= 3
            assert evidence["resumes"] >= 3
            assert evidence["hit_resumes"] >= 1
            assert evidence["metrics"]["enabled"] is True
            assert evidence["metrics"]["hits"] >= evidence["hit_resumes"]
            assert evidence["metrics"]["prefill_tokens_skipped"] >= block_size
            if architecture == "hybrid":
                assert evidence["mamba_hit_resumes"] >= 1
                assert evidence["metrics"]["mamba_restore_hits"] >= 1
