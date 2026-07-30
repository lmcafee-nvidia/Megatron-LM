# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy
import gc
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from dataclasses import replace
from typing import List

import torch
import torch.distributed as dist

from examples.inference.advanced.gpt_dynamic_inference import (
    _assert_nested_close,
    build_engine,
    build_prefix_cache_stress_requests,
)
from examples.inference.utils import Request, build_dynamic_engine_setup_prefix, build_requests
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import (
    DynamicInferenceRequestRecord,
    compute_block_hashes_batched,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.moe.router_trace import get_moe_router_tracer, init_moe_router_tracer
from megatron.core.utils import configure_nvtx_profiling
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)
from megatron.training import get_args, get_tokenizer, initialize_megatron
from megatron.training.arguments import parse_and_validate_args

# pylint: disable=line-too-long,protected-access

logging.basicConfig(level=logging.INFO, force=True)


def add_runner_args(parser):
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="Prefix-cache stress")
    group.add_argument("--prefix-cache-compare", action="store_true")
    group.add_argument("--prefix-cache-stress-groups", type=int, default=0)
    group.add_argument("--prefix-cache-stress-copies", type=int, default=2)
    group.add_argument("--prefix-cache-stress-prompt-tokens", type=int, default=512)
    return parser


def _assert_result_parity(reference, actual):
    assert len(reference) == len(actual)
    for idx, (ref_request, request) in enumerate(zip(reference, actual)):
        assert ref_request.generated_tokens == request.generated_tokens
        assert ref_request.generated_text == request.generated_text
        for field in (
            "prompt_log_probs",
            "generated_log_probs",
            "prompt_top_n_logprobs",
            "generated_top_n_logprobs",
        ):
            _assert_nested_close(
                getattr(ref_request, field, None),
                getattr(request, field, None),
                f"request {idx}.{field}",
            )


def _stress_snapshot(engine, first_group_hashes):
    allocator = engine.context.kv_block_allocator
    hashes = allocator.kv_hash_to_block_id
    values = [
        engine._prefix_cache_hits,
        engine._prefix_cache_blocks_matched,
        engine._prefill_tokens_skipped,
        len(hashes),
        int(all(h in hashes for h in first_group_hashes)),
        int(not hashes and allocator.pool_avail == allocator.pool_size - 1),
        int(bool(hashes) and allocator.pool_avail < allocator.pool_size - 1),
    ]
    state = torch.tensor(values, device="cuda", dtype=torch.int64)
    dist.all_reduce(state)
    allocated = torch.tensor([torch.cuda.memory_allocated()], device="cuda", dtype=torch.int64)
    dist.all_reduce(allocated)
    return state.tolist(), int(allocated.item())


async def suspend_resume_cycle(client, engine, args, futures):
    """Wait for all in-flight requests, then suspend/train/resume."""
    await asyncio.gather(*futures)

    client.pause_engines()
    await engine.wait_until(EngineState.PAUSED)
    client.suspend_engines()
    await engine.wait_until(EngineState.SUSPENDED)
    if args.suspend_timeout > 0:
        await asyncio.sleep(args.suspend_timeout)
    client.resume_engines()
    await engine.wait_until(EngineState.RESUMED)
    client.unpause_engines()
    await engine.wait_until(EngineState.RUNNING)


async def main(
    engine: DynamicInferenceEngine,
    requests: List[Request],
    port: int | None = None,
    sampling_params: SamplingParams | None = None,
    stress_cycles: int | None = None,
    write_output: bool = True,
    first_group_hashes=(),
):
    if sampling_params is not None:
        warnings.warn(
            "The `sampling_params` argument is deprecated. "
            "Sampling parameters are specified per request.",
            DeprecationWarning,
        )

    args = get_args()

    dp_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=port,
        launch_inference_coordinator=True,
        coordinator_schedule_output_path=args.coordinator_schedule_output_path,
    )

    num_suspend_resume_cycles = (
        len(requests) // args.suspend_resume_interval if args.suspend_resume_interval else 0
    )

    client = None
    result_cycles = []
    stress_snapshots = []
    results = None
    if dist.get_rank() == 0:
        client = InferenceClient(
            dp_addr, deserialize=True
        )  # submits requests to the inference coordinator
        client.start()
    if stress_cycles is not None:
        for cycle_idx in range(stress_cycles):
            if dist.get_rank() == 0:
                futures = [
                    client.add_request(request.prompt_text, request.sampling_params)
                    for request in copy.deepcopy(requests)
                ]
                results = await asyncio.gather(*futures)
                result_cycles.append(results)
                client.pause_engines()
            await engine.wait_until(EngineState.PAUSED)
            state, allocated = _stress_snapshot(engine, first_group_hashes)
            previous_hits = stress_snapshots[-1][0][0] if stress_snapshots else 0
            stress_snapshots.append((state, allocated, state[0] - previous_hits))
            if cycle_idx + 1 < stress_cycles:
                if dist.get_rank() == 0:
                    client.unpause_engines()
                await engine.wait_until(EngineState.RUNNING)
    elif dist.get_rank() == 0:
        base_arrival_time = time.time_ns() / 10**9
        for request in requests:
            request.time_arrival = request.time_offset + base_arrival_time
        futures = []
        num_requests_total = len(requests)
        num_requests_added = 0
        next_suspend_at = args.suspend_resume_interval or 0
        cycles_done = 0

        while True:
            current_time = time.time_ns() / 10**9
            if args.incoming_requests_per_step is None:
                # Only add requests that have arrived at the current time.
                while (
                    num_requests_added < num_requests_total
                    and requests[num_requests_added].time_arrival <= current_time
                ):
                    request = requests[num_requests_added]
                    # These add-request calls will queue up the request on a zmq socket and return
                    # instantaneously. They will return an asyncio future which can be awaited for
                    # request completion.
                    futures.append(client.add_request(request.prompt_text, request.sampling_params))
                    num_requests_added += 1

                    if (
                        num_requests_added >= next_suspend_at
                        and cycles_done < num_suspend_resume_cycles
                    ):
                        await suspend_resume_cycle(client, engine, args, futures)
                        cycles_done += 1
                        next_suspend_at += args.suspend_resume_interval

            else:
                # Add deterministic number of requests (generally used for debugging).
                for i in range(
                    min(args.incoming_requests_per_step, num_requests_total - num_requests_added)
                ):
                    # Change sampling parameters to force different generation lengths.
                    request = requests[num_requests_added]
                    n = request.sampling_params.num_tokens_to_generate
                    request.sampling_params.num_tokens_to_generate = n + i
                    futures.append(client.add_request(request.prompt_text, request.sampling_params))
                    num_requests_added += 1

                    if (
                        num_requests_added >= next_suspend_at
                        and cycles_done < num_suspend_resume_cycles
                    ):
                        await suspend_resume_cycle(client, engine, args, futures)
                        cycles_done += 1
                        next_suspend_at += args.suspend_resume_interval

            if num_requests_added == num_requests_total:
                break
            # Relinquish control since there are no more requests to add at the moment. This allows the engine to run.
            await asyncio.sleep(0)

        # While we wait for the requests to complete, the engine runs in the background.
        results: List[DynamicInferenceRequestRecord] = await asyncio.gather(*futures)
    else:
        # Non-rank-0: match the suspend/resume cycles that rank 0 drives.
        for _ in range(num_suspend_resume_cycles):
            await engine.wait_until(EngineState.PAUSED)
            await engine.wait_until(EngineState.SUSPENDED)
            await engine.wait_until(EngineState.RESUMED)
            await engine.wait_until(EngineState.RUNNING)

    if dist.get_rank() == 0 and write_output:
        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}
            throughputs = []

            for req in results:
                result_dict = {
                    "input_prompt": req.prompt,
                    "generated_text": req.generated_text.replace("\n", "\\n"),
                    "generated_tokens": req.generated_tokens,
                    "latency": req.latency,  # InferenceClient populates this field in the returned future.
                }
                if req.sampling_params.return_log_probs:
                    result_dict["logprobs"] = req.prompt_log_probs + req.generated_log_probs
                throughput = len(req.generated_tokens) / req.latency
                throughputs.append(throughput)
                if req.routing_indices is not None:
                    result_dict["routing_indices"] = req.routing_indices.tolist()

                json_results[req.request_id] = result_dict
            throughput_dict = {"throughput": throughputs}
            if args.throughput_check_only:
                json_results = throughput_dict
            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp, indent=4)
        else:
            print("Results:")
            unique_prompt_map = defaultdict(list)
            for req in results:
                unique_prompt_map[req.prompt].append(req)
            for idx, (prompt_text, reqs) in enumerate(unique_prompt_map.items()):
                print(
                    f"%d/%d. prompt '%s' ... [%d] output '%s'."
                    % (
                        idx,
                        len(unique_prompt_map),
                        prompt_text.replace("\n", "\\n"),
                        len(reqs),
                        reqs[0].generated_text.replace("\n", "\\n"),
                    )
                )

    if stress_cycles is None and dist.get_rank() == 0:
        # Pause before stopping: STOP requires PAUSED or SUSPENDED state.
        client.pause_engines()

    await engine.wait_until(EngineState.PAUSED)

    if dist.get_rank() == 0:
        client.stop_engines()

    await engine.wait_until(EngineState.STOPPED)

    if dist.get_rank() == 0:
        client.shutdown_coordinator()
        client.stop()
    logging.info(f"Rank: {dist.get_rank()} stopped their engine instance successfully.")
    return result_cycles, stress_snapshots


if __name__ == "__main__":
    # enable inference mode in the very beginning as some fp8 optimizations
    # check for it.
    with torch.inference_mode():
        args = parse_and_validate_args(
            extra_args_provider=add_runner_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )
        initialize_megatron()
        configure_nvtx_profiling(True)

        tokenizer = get_tokenizer()

        # Sampling params.
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_log_probs=args.return_log_probs,
            num_tokens_to_generate=args.num_tokens_to_generate,
            termination_id=(
                args.termination_id if args.termination_id is not None else tokenizer.eod
            ),
        )

        if getattr(args, 'moe_routing_trace_path', None):
            rank = dist.get_rank()
            max_steps = getattr(args, 'moe_routing_trace_max_inference_steps', None) or 10**9
            init_moe_router_tracer(
                output_dir=args.moe_routing_trace_path,
                max_steps=max_steps,
                rank=rank,
                capture_hidden_states=getattr(
                    args, 'moe_routing_trace_capture_hidden_states', False
                ),
                capture_logits=getattr(args, 'moe_routing_trace_capture_logits', False),
                dump_router_weights=getattr(args, 'moe_routing_trace_dump_weights', False),
            )

        model = get_model_for_inference()

        tracer = get_moe_router_tracer()
        if tracer is not None:
            # When router replay is enabled, the in-pipeline recorder (RouterReplay/RoutingMetadata)
            # writes routing indices into a static buffer, and the text generation controller tees
            # that buffer into the tracer once per decode step. If router replay is not on,
            # use the forward hook method which allows for additionally saving hidden states.
            from megatron.core.utils import get_model_config

            if not get_model_config(model).moe_enable_routing_replay:
                tracer.register_hooks(model)

        requests = build_requests(args, tokenizer, sampling_params)
        requests = build_prefix_cache_stress_requests(args, tokenizer, sampling_params, requests)
        inference_config = get_inference_config_from_model_and_args(model, args)
        context, engine = build_engine(model, inference_config, tokenizer)
        first_group_hashes = ()
        if args.prefix_cache_compare:
            assert sampling_params.top_k == 1 and sampling_params.top_p == 0.0
            assert args.prefix_cache_stress_copies > args.data_parallel_size
            group_requests = requests[:: args.prefix_cache_stress_copies]
            distinct_block_demand = sum(
                len(request.prompt_tokens) // context.block_size_tokens
                for request in group_requests
            )
            usable_blocks = (context.kv_block_allocator.pool_size - 1) * args.data_parallel_size
            assert distinct_block_demand > usable_blocks, (distinct_block_demand, usable_blocks)
            first_group_hashes = compute_block_hashes_batched(
                torch.tensor(group_requests[0].prompt_tokens), context.block_size_tokens
            )

        if dist.get_rank() == 0:
            setup_prefix = build_dynamic_engine_setup_prefix(args, model, engine.context, requests)
            print("~~~")
            print(setup_prefix)
            print("~~~")

        # Start Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        if args.prefix_cache_compare:
            assert inference_config.enable_prefix_caching
            delete_cuda_graphs()
            context.deallocate_inference_state_buffers()
            del engine, context
            gc.collect()
            torch.cuda.empty_cache()
            reference_config = replace(inference_config, enable_prefix_caching=False)
            reference_context, reference_engine = build_engine(model, reference_config, tokenizer)
            reference_cycles, _ = asyncio.run(
                main(
                    reference_engine,
                    requests,
                    args.inference_coordinator_port,
                    stress_cycles=1,
                    write_output=False,
                )
            )
            delete_cuda_graphs()
            reference_context.deallocate_inference_state_buffers()
            del reference_engine, reference_context
            gc.collect()
            torch.cuda.empty_cache()
            context, engine = build_engine(model, inference_config, tokenizer)
            cache_cycles, snapshots = asyncio.run(
                main(
                    engine,
                    requests,
                    args.inference_coordinator_port,
                    stress_cycles=args.inference_repeat_n,
                    first_group_hashes=first_group_hashes,
                )
            )
            if dist.get_rank() == 0:
                for cycle in cache_cycles:
                    _assert_result_parity(reference_cycles[0], cycle)
                world_size = dist.get_world_size()
                policy = inference_config.prefix_caching_eviction_policy.value
                assert all(snapshot[2] > 0 for snapshot in snapshots)
                if policy == "ref_zero":
                    assert all(s[0][3] == 0 and s[0][5] == world_size for s in snapshots)
                else:
                    assert all(s[0][6] == world_size and s[0][4] < world_size for s in snapshots)
                assert snapshots[-1][1] <= snapshots[-2][1] + 64 * 1024**2 * world_size
                summary = {
                    "cycles": args.inference_repeat_n,
                    "hits_by_cycle": [snapshot[2] for snapshot in snapshots],
                    "allocated_bytes_by_cycle": [snapshot[1] for snapshot in snapshots],
                }
                print("PREFIX_CACHE_STRESS: " + json.dumps(summary, sort_keys=True))
        else:
            asyncio.run(main(engine, requests, args.inference_coordinator_port))

        # Stop Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStop()
