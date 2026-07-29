# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# pylint: disable=bad-builtin

import copy
import gc
import hashlib
import io
import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from megatron.training.arguments import parse_and_validate_args

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from examples.inference.utils import (
    Request,
    build_dynamic_engine_setup_prefix,
    build_requests,
    get_curr_time,
    get_global_peak_memory_stats_bytes,
)
from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine, EngineSuspendedError
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
import logging

import megatron
from megatron.core.utils import configure_nvtx_profiling
from megatron.training import get_args, get_tokenizer, initialize_megatron

torch.serialization.add_safe_globals([io.BytesIO])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunState])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunDiagnostic])


def add_gpt_dynamic_inference_args(parser):
    """Add arguments specific to this low-level dynamic-inference example."""
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="Prefix-cache functional comparison")
    group.add_argument(
        "--prefix-cache-compare",
        action="store_true",
        default=False,
        help=(
            "Run one true cache-disabled reference phase followed by cache-enabled "
            "stress phases in the same invocation. Cache-on phases reuse one engine "
            "without resetting it; --inference-repeat-n controls their cycle count."
        ),
    )
    group.add_argument(
        "--prefix-cache-compare-allow-no-prefill-skip",
        action="store_true",
        default=False,
        help=(
            "Allow the cache-on comparison phase to match blocks without skipping "
            "prefill. This is intended for hybrid memory-only prefix caching."
        ),
    )
    group.add_argument(
        "--prefix-cache-stress-groups",
        type=int,
        default=0,
        help=(
            "Replace the normal prompts with this many cache-pressure groups. "
            "Each group gets a distinct long prompt and exact follower copies."
        ),
    )
    group.add_argument(
        "--prefix-cache-stress-copies",
        type=int,
        default=2,
        help="Number of identical requests in each prefix-cache stress group.",
    )
    group.add_argument(
        "--prefix-cache-stress-prompt-tokens",
        type=int,
        default=512,
        help="Minimum prompt length for each generated prefix-cache stress group.",
    )
    group.add_argument(
        "--prefix-cache-stress-staged",
        action="store_true",
        default=False,
        help=(
            "Prefill one complete block for each group's seed request, then add "
            "its identical followers and drain the group. This guarantees that "
            "the followers exercise a materialized cache hit."
        ),
    )
    group.add_argument(
        "--prefix-cache-stress-prompt-logprob-bypass",
        action="store_true",
        default=False,
        help=(
            "Make the final follower in every stress group request prompt "
            "logprobs. Eligible neighbors must still hit while this request "
            "intentionally bypasses prefix skipping and executes its full prompt."
        ),
    )
    group.add_argument(
        "--prefix-cache-generation-epoch-per-cycle",
        action="store_true",
        default=False,
        help=(
            "Advance the generation epoch before every cache-on stress cycle, "
            "invalidating the previous cycle's cache before warming it again."
        ),
    )
    group.add_argument(
        "--prefix-cache-lifecycle-modes",
        nargs="+",
        choices=[mode.value for mode in KVCacheManagementMode],
        default=None,
        help=(
            "Run cache-on stress once with each requested KV lifecycle mode. "
            "Offload uses static cache pointers so persisted CUDA graphs exercise "
            "the address-stable restore path; recompute rebuilds cache state."
        ),
    )
    return parser


def _json_safe(value: Any) -> Any:
    """Convert tensors and array-like values into JSON-compatible objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _get_global_cuda_memory_allocated_bytes() -> int:
    """Return current CUDA tensor allocation, taking the maximum across ranks."""
    allocated = int(torch.cuda.memory_allocated())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        allocated_tensor = torch.tensor([allocated], device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(allocated_tensor, op=torch.distributed.ReduceOp.MAX)
        allocated = int(allocated_tensor.item())
    return allocated


def _build_prefix_cache_stress_requests(
    args, tokenizer, sampling_params: SamplingParams, requests: List[Request]
) -> List[Request]:
    """Build deterministic long-prompt groups that execute real cache hits."""
    if args.prefix_cache_stress_groups == 0:
        return requests
    if args.prefix_cache_stress_groups < 0:
        raise ValueError("--prefix-cache-stress-groups must be non-negative.")
    if args.prefix_cache_stress_copies < 2:
        raise ValueError("--prefix-cache-stress-copies must be at least 2.")
    if args.prefix_cache_stress_prompt_logprob_bypass and args.prefix_cache_stress_copies < 3:
        raise ValueError(
            "--prefix-cache-stress-prompt-logprob-bypass requires at least "
            "three copies: a seed, an eligible follower, and a bypass request."
        )
    if args.prefix_cache_stress_prompt_tokens < 1:
        raise ValueError("--prefix-cache-stress-prompt-tokens must be positive.")

    stress_requests = []
    for group_idx in range(args.prefix_cache_stress_groups):
        group_marker = (
            f"prefix cache stress group {group_idx:04d}; "
            f"this text is intentionally unique to group {group_idx:04d}. "
        )
        repeated_text = group_marker
        while len(tokenizer.tokenize(repeated_text)) < args.prefix_cache_stress_prompt_tokens:
            repeated_text += group_marker

        for copy_idx in range(args.prefix_cache_stress_copies):
            request_sampling_params = copy.deepcopy(sampling_params)
            if (
                args.prefix_cache_stress_prompt_logprob_bypass
                and copy_idx == args.prefix_cache_stress_copies - 1
            ):
                request_sampling_params.skip_prompt_log_probs = False
            request = Request(repeated_text, -1, tokenizer, request_sampling_params)
            request.prefix_cache_stress_group = group_idx
            stress_requests.append(request)

    return stress_requests


class PrefixCacheObservationTracker:
    """Collect cache-state transitions that are observable from the example harness."""

    def __init__(self, engine: DynamicInferenceEngine, requests: List[Request]):
        self.engine = engine
        self._prefix_metrics_baseline = engine.get_prefix_cache_metrics()
        self._request_evictions_baseline = int(engine.evicted_request_count)
        self._stress_group_by_request_id = {
            request_id: getattr(request, "prefix_cache_stress_group", None)
            for request_id, request in enumerate(requests)
        }
        self.cuda_graph_steps = 0
        self.chunk_boundaries_crossed = 0
        self.concurrent_followers = 0
        self._finished_chunk_tokens = {}

    def observe_step(self, result: dict) -> None:
        """Record whether one completed step replayed a CUDA graph."""
        if result.get("cuda_graph_request_count") is not None:
            self.cuda_graph_steps += 1
        group_hit_counts = defaultdict(int)
        for request_id in result["active_request_ids"]:
            request_id = int(request_id)
            stress_group = self._stress_group_by_request_id.get(request_id)
            if stress_group is None or request_id not in self.engine.requests:
                continue
            if self.engine.get_request(request_id).num_cached_tokens > 0:
                group_hit_counts[stress_group] += 1
        self.concurrent_followers = max(
            self.concurrent_followers, max(group_hit_counts.values(), default=0)
        )
        for request_id, entry in self.engine.requests.items():
            finished_chunk_tokens = int(entry.record[-1].finished_chunk_token_count)
            previous = self._finished_chunk_tokens.get(request_id, 0)
            self.chunk_boundaries_crossed += max(
                0,
                finished_chunk_tokens // self.engine.context.block_size_tokens
                - previous // self.engine.context.block_size_tokens,
            )
            self._finished_chunk_tokens[request_id] = finished_chunk_tokens

    def summary(self, requests: List[Request]) -> dict:
        """Return phase-local counters and the final allocator state."""
        engine = self.engine
        context = engine.context
        kv_allocator = context.kv_block_allocator
        mamba_allocator = context.mamba_slot_allocator

        prefix_metrics = engine.get_prefix_cache_metrics()

        def metric_delta(name: str) -> int:
            return int(prefix_metrics[name]) - int(self._prefix_metrics_baseline[name])

        mamba_matched_blocks = [
            int(getattr(request, "mamba_num_matched_blocks", 0)) for request in requests
        ]
        routing_reconstruction_requests = sum(
            getattr(request, "routing_indices", None) is not None
            and int(getattr(request, "num_cached_tokens", 0)) > 0
            for request in requests
        )
        stress_group_sizes = defaultdict(int)
        for request in requests:
            stress_group = getattr(request, "prefix_cache_stress_group", None)
            if stress_group is not None:
                stress_group_sizes[stress_group] += 1

        observations = {
            "prefix_caching_enabled": bool(prefix_metrics["enabled"]),
            "prefix_cache_hits": metric_delta("hits"),
            "prefix_cache_blocks_matched": metric_delta("blocks_matched"),
            "prefill_tokens_computed": metric_delta("prefill_tokens_computed"),
            "prefill_tokens_skipped": metric_delta("prefill_tokens_skipped"),
            "prefix_coordination_waits": metric_delta("coordination_waits"),
            "request_evictions": int(engine.evicted_request_count)
            - self._request_evictions_baseline,
            "kv_physical_block_reuses": metric_delta("kv_physical_reuses"),
            "kv_deregistered_blocks": metric_delta("kv_blocks_deregistered"),
            "kv_lru_evicted_blocks": metric_delta("kv_lru_evictions"),
            "kv_epoch_invalidated_blocks": metric_delta("kv_epoch_invalidations"),
            "kv_cached_block_count": int(prefix_metrics["kv_blocks_cached"]),
            "kv_evictable_block_count": (
                int(kv_allocator.get_evictable_block_count())
                if context.enable_prefix_caching
                else 0
            ),
            "kv_pool_avail": int(kv_allocator.pool_avail),
            "kv_allocatable_block_count": int(kv_allocator.get_allocatable_count()),
            "mamba_matched_blocks": sum(mamba_matched_blocks),
            "mamba_restore_request_count": sum(count > 0 for count in mamba_matched_blocks),
            "mamba_evictions": metric_delta("mamba_evictions"),
            "mamba_restore_hits": metric_delta("mamba_restore_hits"),
            "mamba_restore_misses": metric_delta("mamba_restore_misses"),
            "mamba_commits": metric_delta("mamba_commits"),
            "cuda_graph_steps": self.cuda_graph_steps,
            "routing_reconstruction_requests": routing_reconstruction_requests,
            "producer_follower_cycles": len(stress_group_sizes),
            "concurrent_followers": self.concurrent_followers,
            "chunk_boundaries_crossed": self.chunk_boundaries_crossed,
            "prompt_logprob_bypass_requests": sum(
                not request.sampling_params.skip_prompt_log_probs for request in requests
            ),
            "eligible_cache_followers": sum(
                request.sampling_params.skip_prompt_log_probs
                and int(getattr(request, "num_cached_tokens", 0)) > 0
                for request in requests
            ),
            "generation_epoch_transitions": 0,
            "generation_epoch": _json_safe(getattr(engine, "_generation_epoch", None)),
            "sampling_backend": context.config.sampling_backend,
            "num_speculative_tokens": int(context.num_speculative_tokens),
            "unified_memory_level": int(context.unified_memory_level),
            "kv_cache_management_mode": context.kv_cache_management_mode.value,
            "static_kv_memory_pointers": bool(context.static_kv_memory_pointers),
        }
        if mamba_allocator is not None:
            observations.update(
                {
                    "mamba_cached_hash_count": len(mamba_allocator.hash_to_block_id),
                    "mamba_used_slot_count": int(prefix_metrics["mamba_slots_cached"]),
                    "mamba_slot_count": int(mamba_allocator.max_slots),
                }
            )
        else:
            observations.update(
                {"mamba_cached_hash_count": 0, "mamba_used_slot_count": 0, "mamba_slot_count": 0}
            )
        return observations


def run_inference(
    requests: List[Request],
    engine: DynamicInferenceEngine,
    sampling_params: Optional[SamplingParams] = None,
    observation_tracker: Optional[PrefixCacheObservationTracker] = None,
) -> Dict[str, Any]:
    """Add requests to engine and generate tokens.

    Args:
        requests (List[Request]): Requests that are to be added and processed.
        engine (DynamicInferenceEngine): Inference engine that manages generating tokens.
        sampling_params (SamplingParams): Deprecated as of megatron-core 0.16.
        observation_tracker: Optional functional-test observation collector.

    Return:
        A dictionary of step times with `prefill` and `decode` keys.
    """

    if sampling_params is not None and torch.distributed.get_rank() == 0:
        warnings.warn(
            "The `sampling_params` argument is deprecated. "
            "Sampling parameters are specified per request.",
            DeprecationWarning,
        )

    args = get_args()

    # Parse batch boundaries for batch-drain mode.
    batch_ranges = None
    if args.drain_between_batches and args.batch_boundaries:
        boundaries = [int(x) for x in args.batch_boundaries.split(",")]
        num_requests_total = len(requests)
        batch_ranges = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else num_requests_total
            batch_ranges.append((start, end))

    # Initialize request arrival times.
    base_arrival_time = get_curr_time()
    for request in requests:
        request.time_arrival = request.time_offset + base_arrival_time

    # Add and process requests.
    num_requests_total = len(requests)
    num_requests_added = 0
    num_requests_finished = 0
    step_times = {"prefill": [], "decode": []}
    add_times = []
    output_times = []
    tbar = tqdm(total=num_requests_total)
    total_output_tokens = 0
    attempted_step_count = 0
    suspend_count = 0
    resume_count = 0
    prompt_logprob_cache_immutability_checks = 0
    if args.cuda_graph_impl == "local":
        cuda_graph_request_count_map = {}
    else:
        cuda_graph_request_count_map = None

    def _add_request():
        """Add request to engine.

        *Note: Using `prompt_text` instead of `prompt_tokens` for fair comparison.
        """
        nonlocal num_requests_added
        _request = requests[num_requests_added]
        engine.add_request(num_requests_added, _request.prompt_text, _request.sampling_params)
        _request.time_start = get_curr_time(do_broadcast=False)
        _request.state = "started"
        num_requests_added += 1
        tbar.update(1)

    def _process_step_result(result):
        """Process a single engine step result, updating bookkeeping state."""
        nonlocal total_output_tokens, num_requests_finished

        is_decode_only = engine.is_decode_only

        # Record cuda_graph_request_count.
        cuda_graph_request_count = result["cuda_graph_request_count"]
        if args.cuda_graph_impl == "local" and cuda_graph_request_count is not None:
            cuda_graph_request_count_map[cuda_graph_request_count] = (
                cuda_graph_request_count_map.get(cuda_graph_request_count, 0) + 1
            )

        # Update requests.
        active_request_ids = result["active_request_ids"]
        finished_request_records = result["finished_request_records"]
        step_time = result["step_time"]
        if len(active_request_ids) > 0 or len(finished_request_records) > 0:
            if is_decode_only:
                step_times["decode"].append(step_time)
            else:
                step_times["prefill"].append(step_time)

            # Append output tokens.
            output_start = get_curr_time(do_broadcast=False)
            for finished_request_record in finished_request_records:
                num_cached_tokens = sum(
                    int(record_request.num_cached_tokens)
                    for record_request in finished_request_record.requests
                )
                mamba_num_matched_blocks = sum(
                    int(getattr(record_request, "_mamba_num_matched_blocks", 0))
                    for record_request in finished_request_record.requests
                )
                finished_request = finished_request_record.merge()

                # Update local request object.
                request = requests[finished_request.request_id]
                request.time_end = get_curr_time(do_broadcast=False)
                request.state = "finished"
                request.request_id = finished_request.request_id
                request.events = finished_request.events

                request.ttft = finished_request.ttft

                # Update prompt, in case engine has been suspended and resumed.
                request.prompt_tokens = finished_request.prompt_tokens.tolist()
                request.prompt_text = finished_request.prompt

                # Get output tokens and text.
                request.output_tokens = finished_request.generated_tokens
                request.output_text = finished_request.generated_text
                total_output_tokens += len(request.output_tokens)

                # Log probs.
                if finished_request.sampling_params.return_log_probs:
                    if not finished_request.prompt_log_probs:
                        finished_request.prompt_log_probs = []
                    request.prompt_log_probs = finished_request.prompt_log_probs
                    request.generated_log_probs = finished_request.generated_log_probs
                    request.logprobs = (
                        finished_request.prompt_log_probs + finished_request.generated_log_probs
                    )
                if finished_request.sampling_params.top_n_logprobs > 0:
                    request.generated_top_n_logprobs = finished_request.generated_top_n_logprobs
                if not finished_request.sampling_params.skip_prompt_log_probs:
                    request.prompt_top_n_logprobs = finished_request.prompt_top_n_logprobs
                request.num_cached_tokens = num_cached_tokens
                request.routing_indices = finished_request.routing_indices
                request.policy_epoch = finished_request.policy_epoch
                request.kv_cache_epoch = finished_request.kv_cache_epoch
                request.mamba_num_matched_blocks = mamba_num_matched_blocks
                num_requests_finished += 1
            output_times.append(get_curr_time(do_broadcast=False) - output_start)

        if observation_tracker is not None:
            observation_tracker.observe_step(result)

    def _step_engine():
        """Run one engine step, including the example's suspend/resume controls."""
        nonlocal attempted_step_count, resume_count, suspend_count
        try:
            step_result = engine.step_modern()
        except EngineSuspendedError as error:
            step_result = error
        attempted_step_count += 1

        if args.suspend_resume_interval is not None:
            if attempted_step_count % args.suspend_resume_interval == 0:
                print(
                    "**** step %d/%d ... suspend."
                    % (engine.context.step_count, attempted_step_count)
                )
                was_running = engine.state not in (EngineState.SUSPENDED, EngineState.SUSPENDING)
                engine.suspend()
                suspend_count += int(
                    was_running and engine.state in (EngineState.SUSPENDED, EngineState.SUSPENDING)
                )
            if (
                attempted_step_count > 0
                and (attempted_step_count - args.suspend_resume_interval // 2)
                % args.suspend_resume_interval
                == 0
            ):
                print(
                    "**** step %d/%d ... resume."
                    % (engine.context.step_count, attempted_step_count)
                )
                was_suspended = engine.state in (EngineState.SUSPENDED, EngineState.SUSPENDING)
                engine.resume()
                resume_count += int(
                    was_suspended
                    and engine.state not in (EngineState.SUSPENDED, EngineState.SUSPENDING)
                )

        if not isinstance(step_result, EngineSuspendedError):
            _process_step_result(step_result)
        return step_result

    if args.prefix_cache_stress_staged:
        if args.prefix_cache_stress_groups == 0:
            raise ValueError("--prefix-cache-stress-staged requires --prefix-cache-stress-groups.")
        expected_request_count = args.prefix_cache_stress_groups * args.prefix_cache_stress_copies
        assert len(requests) == expected_request_count

        for group_idx in range(args.prefix_cache_stress_groups):
            group_start = group_idx * args.prefix_cache_stress_copies
            assert num_requests_added == group_start

            # Add one seed and execute at least one complete prompt block before
            # making its exact followers visible to the allocator.
            add_start = get_curr_time(do_broadcast=False)
            _add_request()
            add_times.append(get_curr_time(do_broadcast=False) - add_start)
            seed_request_id = group_start
            target_processed_tokens = min(
                engine.context.block_size_tokens, len(requests[group_start].prompt_tokens) - 1
            )
            while True:
                assert seed_request_id in engine.requests, (
                    f"stress seed {seed_request_id} finished before one complete "
                    "prefix block was materialized"
                )
                seed_request = engine.get_request(seed_request_id)
                processed_tokens = (
                    len(seed_request.prompt_tokens) - seed_request.remaining_prompt_length
                )
                if processed_tokens >= target_processed_tokens:
                    break
                _step_engine()

            if (
                args.prefix_cache_stress_prompt_logprob_bypass
                and engine.context.enable_prefix_caching
            ):
                eligible_request_ids = range(
                    group_start + 1, group_start + args.prefix_cache_stress_copies - 1
                )
                assert eligible_request_ids, (
                    "prompt-logprob bypass stress requires an eligible follower "
                    "beside the seed and bypass request"
                )

                # Admit eligible followers while the seed is still active, then
                # finish them before isolating the bypass request. The seed may
                # legitimately publish more hashes as it advances, so it cannot
                # be part of the bypass immutability window.
                add_start = get_curr_time(do_broadcast=False)
                for _ in eligible_request_ids:
                    _add_request()
                add_times.append(get_curr_time(do_broadcast=False) - add_start)
                while True:
                    eligible_hits = [
                        (
                            request_id in engine.requests
                            and engine.get_request(request_id).num_cached_tokens > 0
                        )
                        or (
                            requests[request_id].state == "finished"
                            and requests[request_id].num_cached_tokens > 0
                        )
                        for request_id in eligible_request_ids
                    ]
                    if all(eligible_hits):
                        break
                    assert not any(
                        requests[request_id].state == "finished" and not hit
                        for request_id, hit in zip(eligible_request_ids, eligible_hits)
                    ), "eligible prompt-logprob neighbor finished without a cache hit"
                    _step_engine()
                    if not all(
                        (
                            request_id in engine.requests
                            and engine.get_request(request_id).num_cached_tokens > 0
                        )
                        or (
                            requests[request_id].state == "finished"
                            and requests[request_id].num_cached_tokens > 0
                        )
                        for request_id in eligible_request_ids
                    ):
                        assert seed_request_id in engine.requests, (
                            "stress seed finished before its eligible followers "
                            "were admitted with cache hits"
                        )
                cached_hash_to_block_before = dict(
                    engine.context.kv_block_allocator.kv_hash_to_block_id
                )
                assert cached_hash_to_block_before, (
                    "prompt-logprob bypass stress reached its immutability check "
                    "without a live cached prefix"
                )
                cached_block_ids = torch.tensor(
                    sorted(set(cached_hash_to_block_before.values())),
                    dtype=torch.int64,
                    device=engine.context.memory_buffer.device,
                )
                kv_block_axis = 1 if engine.context.cache_mla_latent else 2
                cached_kv_before_bypass = engine.context.memory_buffer.index_select(
                    kv_block_axis, cached_block_ids
                ).clone()

                bypass_request_id = group_start + args.prefix_cache_stress_copies - 1
                add_start = get_curr_time(do_broadcast=False)
                _add_request()
                add_times.append(get_curr_time(do_broadcast=False) - add_start)
                _step_engine()

                assert bypass_request_id in engine.requests
                live_bypass_request = engine.get_request(bypass_request_id)
                assert (
                    live_bypass_request.num_cached_tokens == 0
                ), "prompt-logprob bypass request unexpectedly skipped cached prompt tokens"
                current_hash_to_block = engine.context.kv_block_allocator.kv_hash_to_block_id
                assert all(
                    current_hash_to_block.get(block_hash) == block_id
                    for block_hash, block_id in cached_hash_to_block_before.items()
                ), "prompt-logprob bypass changed an existing discoverable prefix mapping"
                assert torch.equal(
                    engine.context.memory_buffer.index_select(kv_block_axis, cached_block_ids),
                    cached_kv_before_bypass,
                ), "prompt-logprob bypass overwrote published KV bytes"
                del cached_kv_before_bypass

                while engine.has_unfinished_requests():
                    _step_engine()

                bypass_request = requests[bypass_request_id]
                assert bypass_request.sampling_params.skip_prompt_log_probs is False
                assert bypass_request.num_cached_tokens == 0, (
                    "prompt-logprob bypass request unexpectedly skipped cached " "prompt tokens"
                )
                prompt_logprob_cache_immutability_checks += 1
            else:
                add_start = get_curr_time(do_broadcast=False)
                while num_requests_added < group_start + args.prefix_cache_stress_copies:
                    _add_request()
                add_times.append(get_curr_time(do_broadcast=False) - add_start)

                # Drain before moving to a distinct prefix. REF_ZERO therefore
                # exercises deregistration/reuse, while LRU accumulates cache
                # pressure and eventually has to evict old prefixes.
                while engine.has_unfinished_requests():
                    _step_engine()
    elif batch_ranges is not None:
        # Batch-drain mode: add all requests in a batch, drain, then next batch.
        for batch_idx, (batch_start, batch_end) in enumerate(batch_ranges):
            # Add all requests in current batch.
            add_start = get_curr_time(do_broadcast=False)
            while num_requests_added < batch_end:
                _add_request()
            add_times.append(get_curr_time(do_broadcast=False) - add_start)

            # Step until all active requests finish (drain).
            while engine.has_unfinished_requests():
                try:
                    result = engine.step_modern()
                except EngineSuspendedError as e:
                    result = e
                attempted_step_count += 1

                if isinstance(result, EngineSuspendedError):
                    continue

                _process_step_result(result)
    else:
        # Original mode: add requests per step based on arrival time or count.
        while True:
            # Add requests.
            add_start = get_curr_time(do_broadcast=False)
            if args.incoming_requests_per_step is None:
                # Add requests with 'earlier' arrival time.
                while num_requests_added < num_requests_total:
                    if requests[num_requests_added].time_arrival > add_start:
                        break
                    _add_request()
            else:
                # Add deterministic number of requests (generally used for debugging).
                for i in range(
                    min(args.incoming_requests_per_step, num_requests_total - num_requests_added)
                ):
                    _add_request()
            add_times.append(get_curr_time(do_broadcast=False) - add_start)

            # Step inference engine (i.e., generate a token for each active request).
            # Before step, we haven't done the scheduling, so we cannot know the is_decode_only
            result = _step_engine()
            if isinstance(result, EngineSuspendedError):
                continue

            # Check if all requests are finished.
            if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
                break

    # Resume engine if the final scheduled transition left it suspended.
    was_suspended = engine.state in (EngineState.SUSPENDED, EngineState.SUSPENDING)
    engine.resume()
    resume_count += int(
        was_suspended and engine.state not in (EngineState.SUSPENDED, EngineState.SUSPENDING)
    )

    return {
        "step_times": step_times,
        "add_times": add_times,
        "output_times": output_times,
        "total_output_tokens": total_output_tokens,
        "cuda_graph_request_count_map": cuda_graph_request_count_map,
        "suspend_count": suspend_count,
        "resume_count": resume_count,
        "concurrent_followers": (
            observation_tracker.concurrent_followers if observation_tracker is not None else 0
        ),
        "prompt_logprob_cache_immutability_checks": (prompt_logprob_cache_immutability_checks),
        "prefix_cache_observations": (
            observation_tracker.summary(requests) if observation_tracker is not None else None
        ),
    }


def _build_dynamic_engine(model, inference_config, tokenizer):
    """Build one low-level dynamic engine and its context."""
    context = DynamicInferenceContext(model.config, inference_config)
    wrapped_model = GPTInferenceWrapper(model, context)
    controller = TextGenerationController(wrapped_model, tokenizer)
    return context, DynamicInferenceEngine(controller, context)


def _validate_request_lengths(args, context, requests: List[Request]) -> None:
    """Validate prompt lengths against the non-chunked token budget."""
    if args.enable_chunked_prefill:
        return
    invalid_prompt_length_map = {
        request_idx: len(request.prompt_tokens)
        for request_idx, request in enumerate(requests)
        if len(request.prompt_tokens) > context.max_tokens
    }
    assert (
        not invalid_prompt_length_map
    ), "request idxs with prompts longer than context.max_tokens: " + ", ".join(
        f"{key}({value})" for key, value in invalid_prompt_length_map.items()
    )


def _run_inference_phase(
    phase_name: str, requests: List[Request], engine: DynamicInferenceEngine
) -> dict:
    """Run one phase and return outputs, observations, timing, and memory."""
    torch.cuda.synchronize()
    start_allocated_bytes = _get_global_cuda_memory_allocated_bytes()
    torch.cuda.reset_peak_memory_stats()
    tracker = PrefixCacheObservationTracker(engine, requests)

    start_time = get_curr_time()
    result = run_inference(requests, engine, observation_tracker=tracker)
    result["prefix_cache_observations"].update(
        {
            "suspend_count": result["suspend_count"],
            "resume_count": result["resume_count"],
            "concurrent_followers": result["concurrent_followers"],
            "prompt_logprob_cache_immutability_checks": result[
                "prompt_logprob_cache_immutability_checks"
            ],
        }
    )
    torch.cuda.synchronize()
    total_time = get_curr_time() - start_time
    end_allocated_bytes = _get_global_cuda_memory_allocated_bytes()
    peak_allocated_bytes = get_global_peak_memory_stats_bytes()["mem-max-allocated-bytes"]

    total_output_tokens = result["total_output_tokens"]
    result.update(
        {
            "phase_name": phase_name,
            "total_time": total_time,
            "throughput": total_output_tokens / total_time,
            "memory": {
                "start_allocated_bytes": start_allocated_bytes,
                "end_allocated_bytes": end_allocated_bytes,
                "peak_allocated_bytes": peak_allocated_bytes,
            },
        }
    )
    return result


def _serialize_request_result(
    request: Request, result: dict, step_count: int, *, scenario_id: str, include_events: bool
) -> dict:
    """Serialize one completed request for golden or direct phase comparison."""
    result_dict = {
        "scenario_id": scenario_id,
        "request_id": request.request_id,
        "input_prompt": request.prompt_text,
        "scenario_prompt": getattr(request, "scenario_prompt_text", request.prompt_text),
        "prompt_token_count": int(
            getattr(request, "scenario_prompt_token_count", len(request.prompt_tokens))
        ),
        "final_prompt_token_count": len(request.prompt_tokens),
        "skip_prompt_log_probs": request.sampling_params.skip_prompt_log_probs,
        "generated_text": request.output_text,
        "generated_tokens": request.output_tokens,
        "output_length": len(request.output_tokens),
        "request_status": request.state,
        "latency": request.time_end - request.time_start,
        "ttft": request.ttft,
        "num_cached_tokens": int(getattr(request, "num_cached_tokens", 0)),
        "cuda_graph_request_count_map": result["cuda_graph_request_count_map"],
        "step_count": step_count,
        "generated_top_n_logprobs": getattr(request, "generated_top_n_logprobs", None),
        "top_n_logprobs": getattr(request, "generated_top_n_logprobs", None),
        "prompt_top_n_logprobs": getattr(request, "prompt_top_n_logprobs", None),
        "routing_indices": getattr(request, "routing_indices", None),
        "policy_epoch": getattr(request, "policy_epoch", None),
        "kv_cache_epoch": getattr(request, "kv_cache_epoch", None),
        "mamba_num_matched_blocks": int(getattr(request, "mamba_num_matched_blocks", 0)),
    }
    if request.sampling_params.return_log_probs:
        prompt_logprobs = getattr(request, "prompt_log_probs", None)
        generated_logprobs = getattr(request, "generated_log_probs", None)
        result_dict["prompt_logprobs"] = prompt_logprobs
        result_dict["generated_logprobs"] = generated_logprobs
        if prompt_logprobs is not None or generated_logprobs is not None:
            result_dict["logprobs"] = (prompt_logprobs or []) + (generated_logprobs or [])
        else:
            result_dict["logprobs"] = None
    if include_events:
        result_dict["events"] = [event.serialize() for event in request.events]
    return _json_safe(result_dict)


def _serialize_phase_requests(
    requests: List[Request], result: dict, step_count: int, include_events: bool
) -> dict[str, dict]:
    """Serialize every request keyed by stable scenario identity."""
    serialized = {}
    for request_idx, request in enumerate(requests):
        scenario_id = str(getattr(request, "scenario_id", request_idx))
        assert scenario_id not in serialized, f"duplicate scenario_id {scenario_id!r}"
        serialized[scenario_id] = _serialize_request_result(
            request, result, step_count, scenario_id=scenario_id, include_events=include_events
        )
    return serialized


def _aggregate_prefix_observations(cycles: list[dict]) -> dict:
    """Aggregate phase-local event counters while retaining final state."""
    if not cycles:
        return {}
    count_fields = {
        "prefix_cache_hits",
        "prefix_cache_blocks_matched",
        "prefill_tokens_computed",
        "prefill_tokens_skipped",
        "prefix_coordination_waits",
        "request_evictions",
        "kv_physical_block_reuses",
        "kv_deregistered_blocks",
        "kv_lru_evicted_blocks",
        "kv_epoch_invalidated_blocks",
        "mamba_matched_blocks",
        "mamba_restore_request_count",
        "mamba_evictions",
        "mamba_restore_hits",
        "mamba_restore_misses",
        "mamba_commits",
        "cuda_graph_steps",
        "routing_reconstruction_requests",
        "suspend_count",
        "resume_count",
        "producer_follower_cycles",
        "chunk_boundaries_crossed",
        "prompt_logprob_bypass_requests",
        "eligible_cache_followers",
        "prompt_logprob_cache_immutability_checks",
        "generation_epoch_transitions",
    }
    observations = {
        key: sum(int(cycle["observations"].get(key, 0)) for cycle in cycles) for key in count_fields
    }
    observations.update(
        {key: value for key, value in cycles[-1]["observations"].items() if key not in count_fields}
    )
    observations["concurrent_followers"] = max(
        int(cycle["observations"].get("concurrent_followers", 0)) for cycle in cycles
    )
    observations["cycle_count"] = len(cycles)
    return observations


def _collect_released_cuda_memory() -> None:
    """Collect Python cycles and return unused CUDA allocations to the driver."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.inference_mode()
def main():
    """Run dynamic inference."""
    # Initialize Megatron.
    args = parse_and_validate_args(
        extra_args_provider=add_gpt_dynamic_inference_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )
    initialize_megatron()

    # Start Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(level=level, force=True)

    configure_nvtx_profiling(True)

    # Build tokenizer
    tokenizer = build_tokenizer(args)

    # Reset peak memory stats so functional tests measure this run and not
    # whatever happened earlier during initialization.
    torch.cuda.reset_peak_memory_stats()

    # Sampling params.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=args.termination_id if args.termination_id is not None else tokenizer.eod,
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
    )

    model = get_model_for_inference()

    # Requests and context configuration.
    requests = build_requests(args, tokenizer, sampling_params)
    requests = _build_prefix_cache_stress_requests(args, tokenizer, sampling_params, requests)
    for request_idx, request in enumerate(requests):
        request.scenario_id = str(request_idx)
        request.scenario_prompt_text = request.prompt_text
        request.scenario_prompt_token_count = len(request.prompt_tokens)
    request_templates = copy.deepcopy(requests)
    inference_config = get_inference_config_from_model_and_args(model, args)

    # Calculate max_sequence_length from requests
    max_gen_length = sampling_params.num_tokens_to_generate
    max_context_length = max(len(r.prompt_tokens) for r in requests)
    inference_config.max_sequence_length = max_context_length + max_gen_length

    comparison_payload = None
    memory_phases = []
    if args.prefix_cache_lifecycle_modes:
        if not args.prefix_cache_compare:
            raise ValueError("--prefix-cache-lifecycle-modes requires --prefix-cache-compare.")
        cache_on_configs = [
            replace(
                inference_config,
                kv_cache_management_mode=KVCacheManagementMode(mode),
                static_kv_memory_pointers=(mode == KVCacheManagementMode.OFFLOAD.value),
            )
            for mode in args.prefix_cache_lifecycle_modes
        ]
    else:
        cache_on_configs = [inference_config]

    if args.prefix_cache_compare:
        if not inference_config.enable_prefix_caching:
            raise ValueError(
                "--prefix-cache-compare requires " "--inference-dynamic-batching-prefix-caching."
            )
        if args.top_k != 1 or args.top_p != 0.0:
            raise ValueError(
                "--prefix-cache-compare currently requires deterministic greedy "
                "sampling (--top_k 1 --top_p 0)."
            )

        # This is a distinct context, not a cold-miss approximation. Removing the
        # Mamba cache budget also ensures the reference excludes prefix-cache-only
        # tensor allocation.
        reference_payloads = {}
        for cache_on_config in cache_on_configs:
            reference_key = (
                cache_on_config.kv_cache_management_mode.value
                if args.prefix_cache_lifecycle_modes
                else "default"
            )
            reference_config = replace(
                cache_on_config, enable_prefix_caching=False, prefix_caching_mamba_gb=None
            )
            reference_context, reference_engine = _build_dynamic_engine(
                model, reference_config, tokenizer
            )
            assert reference_context.enable_prefix_caching is False
            assert reference_engine.get_prefix_cache_metrics()["enabled"] is False
            _validate_request_lengths(args, reference_context, request_templates)
            reference_requests = copy.deepcopy(request_templates)
            reference_engine.reset()
            phase_name = f"cache_off_{reference_key}"
            reference_result = _run_inference_phase(
                phase_name, reference_requests, reference_engine
            )
            for request in reference_requests:
                assert (
                    request.state == "finished"
                ), f"{phase_name} request.state == '{request.state}' != 'finished'."
            reference_payloads[reference_key] = {
                "prefix_caching_enabled": False,
                "requests": _serialize_phase_requests(
                    reference_requests,
                    reference_result,
                    reference_context.step_count,
                    args.output_request_events,
                ),
                "observations": reference_result["prefix_cache_observations"],
                "memory": reference_result["memory"],
            }
            memory_phases.append({"phase": phase_name, **reference_result["memory"]})

            if (
                reference_engine.cuda_graph_impl == "local"
                and reference_context.cuda_graph_batch_dimensions_list
            ):
                delete_cuda_graphs()
            reference_engine.reset()
            del reference_result
            del reference_requests
            del reference_engine
            del reference_context
            _collect_released_cuda_memory()
        reference_payload = next(iter(reference_payloads.values()))

    # Run and time the cache-on workload. In comparison mode the repeats are
    # stress cycles in one engine lifetime, so the cache is not reset between
    # cycles. Non-comparison runs still reset before every repeat.

    throughputs = []
    cache_on_cycles = []
    context = None
    engine = None
    global_cycle_idx = 0
    for config_idx, cache_on_config in enumerate(cache_on_configs):
        if engine is not None:
            if engine.cuda_graph_impl == "local" and context.cuda_graph_batch_dimensions_list:
                delete_cuda_graphs()
            engine.reset()
            del engine
            del context
            _collect_released_cuda_memory()

        context, engine = _build_dynamic_engine(model, cache_on_config, tokenizer)
        if args.prefix_cache_compare:
            assert context.enable_prefix_caching is True
            assert engine.get_prefix_cache_metrics()["enabled"] is True
        _validate_request_lengths(args, context, request_templates)

        if config_idx == 0:
            setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
            print("~~~")
            print(setup_prefix)
            print("~~~")

        for config_cycle_idx in range(args.inference_repeat_n):
            if not args.prefix_cache_compare or config_cycle_idx == 0:
                engine.reset()
            if args.prefix_cache_compare:
                requests = copy.deepcopy(request_templates)
                if args.prefix_cache_generation_epoch_per_cycle:
                    engine._apply_generation_epoch(global_cycle_idx)

            phase_name = (
                f"cache_on_{global_cycle_idx}"
                if args.prefix_cache_compare
                else f"repeat_{global_cycle_idx}"
            )
            result = _run_inference_phase(phase_name, requests, engine)
            if args.prefix_cache_compare and args.prefix_cache_generation_epoch_per_cycle:
                result["prefix_cache_observations"]["generation_epoch_transitions"] = 1
            step_times = result["step_times"]
            add_times = result["add_times"]
            output_times = result["output_times"]
            total_output_tokens = result["total_output_tokens"]
            total_time = result["total_time"]
            throughputs.append(result["throughput"])

            if args.prefix_cache_compare:
                reference_key = (
                    cache_on_config.kv_cache_management_mode.value
                    if args.prefix_cache_lifecycle_modes
                    else "default"
                )
                cycle_payload = {
                    "cycle_index": global_cycle_idx,
                    "reference_key": reference_key,
                    "requests": _serialize_phase_requests(
                        requests, result, context.step_count, args.output_request_events
                    ),
                    "observations": result["prefix_cache_observations"],
                    "memory": result["memory"],
                }
                cache_on_cycles.append(cycle_payload)
                memory_phases.append({"phase": phase_name, **result["memory"]})
            global_cycle_idx += 1

    # Validate all requests finished.
    for request in requests:
        assert request.state == "finished", f"request.state == '{request.state}' != 'finished'."

    if args.prefix_cache_compare:
        comparison_payload = {
            "schema_version": 1,
            "scenario_ids": sorted(reference_payload["requests"]),
            "require_prefill_skip": not args.prefix_cache_compare_allow_no_prefill_skip,
            "cache_off": reference_payload,
            "cache_off_by_lifecycle": reference_payloads,
            "cache_on": {
                "prefix_caching_enabled": True,
                "cycles": cache_on_cycles,
                "aggregate_observations": _aggregate_prefix_observations(cache_on_cycles),
                "lifecycle_modes": (
                    list(args.prefix_cache_lifecycle_modes)
                    if args.prefix_cache_lifecycle_modes
                    else [context.kv_cache_management_mode.value]
                ),
            },
        }
        peak_mem_stats = {
            "mem-max-allocated-bytes": max(
                cycle["memory"]["peak_allocated_bytes"] for cycle in cache_on_cycles
            )
        }
    else:
        peak_mem_stats = get_global_peak_memory_stats_bytes()
    stats = torch.cuda.memory_stats()

    # Print unique prompts + outputs.
    if torch.distributed.get_rank() == 0:

        def escape_str(s):
            return s.replace("\n", "\\n")

        print("~~~~ Unique prompts + outputs. ~~~~")

        # Map requests by their prompt.
        unique_prompt_map = defaultdict(list)
        for request_idx, request in enumerate(requests):
            unique_prompt_map[request.prompt_text].append(request_idx)

        # Print unique prompts + outputs.
        text_hashes = []
        for unique_idx, (prompt_text, request_idxs) in enumerate(unique_prompt_map.items()):

            # ---- Prompt summary line ----
            prompt_len = len(requests[request_idxs[0]].prompt_tokens)
            escaped_prompt_text = escape_str(prompt_text)
            print(
                f"\n{unique_idx+1}/{len(unique_prompt_map)}"
                f"[n {len(request_idxs)}, l {prompt_len}] {escaped_prompt_text}"
            )

            # ---- Group all outputs for this prompt ----
            output_map = defaultdict(list)
            for idx in request_idxs:
                req = requests[idx]
                output_map[req.output_text].append(idx)

            # ---- Print each unique output ----
            for output_text, output_request_idxs in output_map.items():
                evicted = False
                for idx in output_request_idxs:
                    for event in requests[idx].events:
                        if event.type.name == "EVICT":
                            evicted = True
                            break
                if output_text is not None:
                    # Use hash of prompt + generated text in case engine was
                    # suspended and resumed, which misaligns boundary between
                    # prompt and generated tokens.
                    o_hash = hashlib.sha256((prompt_text + output_text).encode()).hexdigest()[:6]
                    o_len = len(requests[output_request_idxs[0]].output_tokens)
                    escaped_output_text = escape_str(output_text)
                else:
                    o_hash = "--"
                    o_len = 0
                    escaped_output_text = "--"
                print(
                    f"  >>>> [n {len(output_request_idxs)}, {o_len} tokens, hash {o_hash}"
                    f"{', <evicted>' if evicted else ''}] {escaped_output_text}"
                )
                text_hashes.append(o_hash)

        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}

            # Write every 'n' requests, plus the final request.
            for i, req in enumerate(requests):
                if i % args.output_every_n_results == 0 or i == len(requests) - 1:
                    print(f' Attributes of request {i}: {req.__dict__}')
                    json_results[req.request_id] = _serialize_request_result(
                        req,
                        result,
                        engine.context.step_count,
                        scenario_id=str(getattr(req, "scenario_id", i)),
                        include_events=args.output_request_events,
                    )

            # Track system-level throughput as a test / debug metric
            if args.record_throughput:
                json_results["throughput"] = throughputs
            # Attach peak memory metrics; the functional test only validates these
            # if the fields exist in the golden values.
            json_results.update(peak_mem_stats)
            json_results["lifetime_prefill_token_count"] = (
                engine.context.lifetime_prefill_token_count
            )
            json_results["async_sched_step_count"] = engine.context.async_sched_step_count
            json_results["async_sched_compaction_step_count"] = (
                engine.context.async_sched_compaction_step_count
            )
            json_results["prefix_cache_observations"] = (
                comparison_payload["cache_on"]["aggregate_observations"]
                if comparison_payload is not None
                else result["prefix_cache_observations"]
            )
            if comparison_payload is not None:
                json_results["prefix_cache_comparison"] = comparison_payload
                json_results["memory_phases"] = memory_phases

            print(f' Saving results to {args.output_path}')
            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp, indent=1)

        # Timing results.
        stats = torch.cuda.memory_stats()
        throughput = total_output_tokens / total_time
        print("~~~")
        peak_alloc_gb = stats["allocated_bytes.all.peak"] / 1024**3
        peak_resvd_gb = stats["reserved_bytes.all.peak"] / 1024**3

        p_times = step_times["prefill"]
        d_times = step_times["decode"]

        p_total = sum(p_times)
        d_total = sum(d_times)

        p_count = len(p_times)
        d_count = len(d_times)

        p_mean = p_total / p_count
        d_mean = d_total / d_count if d_count != 0 else 0.0

        # Commented out for now as the step/add/output times are not calculated correctly.
        # print(
        #     f"{setup_prefix} … "
        #     f"mem {peak_alloc_gb:.1f}/{peak_resvd_gb:.1f} GB … "
        #     f"total time: {step_total:.3f}s … "
        #     f"step time: total {step_total:.3f}s "
        #     f"[ p {p_total:.3f}s, d {d_total:.3f}s ], "
        #     f"mean [ p {p_mean:.3f}s, d {d_mean:.3f}s ], "
        #     f"count [ p {p_count}, d {d_count} ]."
        # )
        capture_str = f"{engine.capture_stats['time']:.2f} sec" if engine.capture_stats else "--"
        print(
            f"{setup_prefix} … " f"throughput: {throughput:.3f} tok/s … ",
            f"total time: {total_time:.3f}s … "
            f"mem {peak_alloc_gb:.1f} allocated/{peak_resvd_gb:.1f} reserved GB … "
            f"steps: {engine.context.step_count:d} … "
            f"capture {capture_str}",
        )
        print("~~~")

    # Stop Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
