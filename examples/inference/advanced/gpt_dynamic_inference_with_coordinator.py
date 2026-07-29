# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy
import gc
import json
import logging
import os
import time
import warnings
from dataclasses import replace
from typing import List

import torch
import torch.distributed as dist

from examples.inference.advanced.gpt_dynamic_inference import (
    _build_dynamic_engine,
    _build_prefix_cache_stress_requests,
    _json_safe,
)
from examples.inference.utils import Request, build_dynamic_engine_setup_prefix, build_requests
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.moe.router_trace import get_moe_router_tracer, init_moe_router_tracer
from megatron.core.utils import configure_nvtx_profiling
from megatron.inference.utils import (
    add_inference_args,
    get_dynamic_inference_engine,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)
from megatron.training import get_args, get_tokenizer, initialize_megatron
from megatron.training.arguments import parse_and_validate_args

# pylint: disable=line-too-long

logging.basicConfig(level=logging.INFO, force=True)


def add_coordinator_prefix_cache_args(parser):
    """Add cache-comparison arguments used by coordinator functional tests."""
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="Coordinator prefix-cache comparison")
    group.add_argument("--prefix-cache-compare", action="store_true", default=False)
    group.add_argument("--prefix-cache-stress-groups", type=int, default=0)
    group.add_argument("--prefix-cache-stress-copies", type=int, default=2)
    group.add_argument("--prefix-cache-stress-prompt-tokens", type=int, default=512)
    group.add_argument(
        "--prefix-cache-stress-prompt-logprob-bypass", action="store_true", default=False
    )
    return parser


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


async def _run_standard_coordinator_workload(client, engine, requests, args, is_client_rank):
    """Run the ordinary coordinator example workload, including lifecycle cycles."""
    cycle_count = (
        len(requests) // args.suspend_resume_interval if args.suspend_resume_interval else 0
    )
    if not is_client_rank:
        for _ in range(cycle_count):
            await engine.wait_until(EngineState.PAUSED)
            await engine.wait_until(EngineState.SUSPENDED)
            await engine.wait_until(EngineState.RESUMED)
            await engine.wait_until(EngineState.RUNNING)
        return []

    base_arrival_time = time.time_ns() / 10**9
    for request in requests:
        request.time_arrival = request.time_offset + base_arrival_time
    futures = []
    num_requests_added = 0
    next_suspend_at = args.suspend_resume_interval or 0
    cycles_done = 0
    while num_requests_added < len(requests):
        current_time = time.time_ns() / 10**9
        if args.incoming_requests_per_step is None:
            add_count = 0
            while (
                num_requests_added + add_count < len(requests)
                and requests[num_requests_added + add_count].time_arrival <= current_time
            ):
                add_count += 1
        else:
            add_count = min(args.incoming_requests_per_step, len(requests) - num_requests_added)

        for request_offset in range(add_count):
            request = requests[num_requests_added]
            if args.incoming_requests_per_step is not None:
                request.sampling_params.num_tokens_to_generate += request_offset
            futures.append(client.add_request(request.prompt_text, request.sampling_params))
            num_requests_added += 1
            if num_requests_added >= next_suspend_at and cycles_done < cycle_count:
                await suspend_resume_cycle(client, engine, args, futures)
                cycles_done += 1
                next_suspend_at += args.suspend_resume_interval
        if add_count == 0:
            await asyncio.sleep(0)

    return await asyncio.gather(*futures)


async def main(
    engine: DynamicInferenceEngine,
    requests: List[Request],
    port: int | None = None,
    sampling_params: SamplingParams | None = None,
    cycle_count: int = 1,
    seed_requests: List[Request] | None = None,
    exercise_epoch_removal: bool = False,
    schedule_output_path: str | None = None,
):
    if sampling_params is not None:
        warnings.warn(
            "The `sampling_params` argument is deprecated. "
            "Sampling parameters are specified per request.",
            DeprecationWarning,
        )

    # once you call engine.start_listening_to_data_parallel_coordinator,
    # the engine will start accepting requests from the data parallel coordinator.
    # and processing them in an asyncio coroutine.
    # leaving inference_coordinator_port as None will find a free port automatically.
    args = get_args()

    dp_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=port,
        launch_inference_coordinator=True,
        coordinator_schedule_output_path=schedule_output_path,
    )

    is_client_rank = dist.get_rank() == 0
    client = None
    result_cycles = []
    seed_results = []
    metric_snapshots = []
    removal_snapshot = None
    removal_probe_result = None
    removal_probe_seed_index = None
    if is_client_rank:
        client = InferenceClient(
            dp_addr, deserialize=True
        )  # submits requests to the inference coordinator
        client.start()

    async def pause_and_snapshot():
        if is_client_rank:
            client.pause_engines()
        await engine.wait_until(EngineState.PAUSED)
        return _gather_coordinator_snapshot(engine)

    async def unpause():
        if is_client_rank:
            client.unpause_engines()
        await engine.wait_until(EngineState.RUNNING)

    if not args.prefix_cache_compare:
        results = await _run_standard_coordinator_workload(
            client, engine, requests, args, is_client_rank
        )
        if is_client_rank:
            result_cycles.append(results)
        metric_snapshots.append(await pause_and_snapshot())
    else:
        if seed_requests is not None:
            if is_client_rank:
                seed_futures = [
                    client.add_request(request.prompt_text, request.sampling_params)
                    for request in copy.deepcopy(seed_requests)
                ]
                seed_results = await asyncio.gather(*seed_futures)
            metric_snapshots.append(await pause_and_snapshot())

        for _ in range(cycle_count):
            if engine.state == EngineState.PAUSED:
                await unpause()
            if is_client_rank:
                cycle_requests = copy.deepcopy(requests)
                futures = [
                    client.add_request(request.prompt_text, request.sampling_params)
                    for request in cycle_requests
                ]
                results: List[DynamicInferenceRequestRecord] = await asyncio.gather(*futures)
                result_cycles.append(results)
            metric_snapshots.append(await pause_and_snapshot())

        if exercise_epoch_removal:
            assert seed_requests is not None
            assert engine.state == EngineState.PAUSED
            new_epoch = 1 if engine._generation_epoch is None else engine._generation_epoch + 1
            if is_client_rank:
                client.set_generation_epoch(new_epoch)

            async def wait_for_epoch():
                while engine._generation_epoch != new_epoch:
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(wait_for_epoch(), timeout=30.0)
            removal_snapshot = _gather_coordinator_snapshot(engine)

            # Submit one old-prefix probe while the engines and their confirmed
            # ownership sets are empty. The coordinator must record a zero-depth
            # match before the engines rebuild ownership in the new epoch.
            removal_probe_seed_index = 1 if len(seed_requests) > 1 else 0
            if is_client_rank:
                probe = copy.deepcopy(seed_requests[removal_probe_seed_index])
                removal_probe_future = client.add_request(probe.prompt_text, probe.sampling_params)
            await unpause()
            if is_client_rank:
                removal_probe_result = await removal_probe_future
            await pause_and_snapshot()

    if is_client_rank:
        client.stop_engines()
    await engine.wait_until(EngineState.STOPPED)

    if is_client_rank:
        client.shutdown_coordinator()
        coordinator_process = engine.inference_coordinator_process
        await asyncio.to_thread(coordinator_process.join, 30.0)
        assert not coordinator_process.is_alive(), "inference coordinator did not stop"
        assert coordinator_process.exitcode == 0
        client.stop()
    logging.info(f"Rank: {dist.get_rank()} stopped their engine instance successfully.")
    return {
        "cycles": result_cycles,
        "seed_results": seed_results,
        "metric_snapshots": metric_snapshots,
        "removal_snapshot": removal_snapshot,
        "removal_probe_result": removal_probe_result,
        "removal_probe_seed_index": removal_probe_seed_index,
    }


def _gather_coordinator_snapshot(engine):
    """Gather cache evidence from the one MP-coordinator rank per DP worker."""
    local_snapshot = None
    if engine.is_mp_coordinator:
        local_snapshot = {
            "global_rank": dist.get_rank(),
            "metrics": engine.get_prefix_cache_metrics(),
            "routable_hashes": sorted(engine._get_routable_prefix_hashes()),
        }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_snapshot)
    return [snapshot for snapshot in gathered if snapshot is not None]


def _serialize_coordinator_requests(results, request_templates, scenario_ids=None):
    """Serialize coordinator results by stable workload position."""
    assert len(results) == len(request_templates)
    if scenario_ids is None:
        scenario_ids = range(len(results))
    assert len(results) == len(scenario_ids)
    serialized = {}
    for scenario_id, result, template in zip(scenario_ids, results, request_templates):
        scenario_id = str(scenario_id)
        prompt_logprobs = _json_safe(result.prompt_log_probs)
        generated_logprobs = _json_safe(result.generated_log_probs)
        logprobs = (
            (prompt_logprobs or []) + (generated_logprobs or [])
            if result.sampling_params.return_log_probs
            else None
        )
        serialized[scenario_id] = {
            "scenario_id": scenario_id,
            "request_id": int(result.request_id),
            "input_prompt": result.prompt,
            "scenario_prompt": template.prompt_text,
            "prompt_token_count": len(template.prompt_tokens),
            "final_prompt_token_count": int(result.prompt_length),
            "skip_prompt_log_probs": bool(result.sampling_params.skip_prompt_log_probs),
            "generated_text": result.generated_text,
            "generated_tokens": _json_safe(result.generated_tokens),
            "output_length": len(result.generated_tokens),
            "request_status": result.status.name,
            "latency": float(result.latency),
            "logprobs": logprobs,
            "generated_top_n_logprobs": _json_safe(result.generated_top_n_logprobs),
            "prompt_top_n_logprobs": _json_safe(result.prompt_top_n_logprobs),
            "prompt_logprobs": prompt_logprobs,
            "generated_logprobs": generated_logprobs,
            "routing_indices": _json_safe(result.routing_indices),
            "num_cached_tokens": int(result.num_cached_tokens),
        }
    return serialized


def _coordinator_observations(current_snapshot, args, baseline_snapshot=None, cycle_count=1):
    """Aggregate counter deltas from the one coordinator rank per DP worker."""
    counter_names = {
        "hits": "prefix_cache_hits",
        "blocks_matched": "prefix_cache_blocks_matched",
        "prefill_tokens_computed": "prefill_tokens_computed",
        "prefill_tokens_skipped": "prefill_tokens_skipped",
        "coordination_waits": "prefix_coordination_waits",
        "kv_physical_reuses": "kv_physical_block_reuses",
        "kv_blocks_deregistered": "kv_deregistered_blocks",
        "kv_lru_evictions": "kv_lru_evicted_blocks",
        "kv_epoch_invalidations": "kv_epoch_invalidated_blocks",
        "mamba_evictions": "mamba_evictions",
        "mamba_restore_hits": "mamba_restore_hits",
        "mamba_restore_misses": "mamba_restore_misses",
        "mamba_commits": "mamba_commits",
    }
    current_by_rank = {snapshot["global_rank"]: snapshot for snapshot in current_snapshot}
    baseline_by_rank = {snapshot["global_rank"]: snapshot for snapshot in (baseline_snapshot or [])}

    def counter_delta(snapshot, input_name):
        baseline = baseline_by_rank.get(snapshot["global_rank"])
        baseline_value = 0 if baseline is None else int(baseline["metrics"][input_name])
        return int(snapshot["metrics"][input_name]) - baseline_value

    observations = {
        output_name: sum(
            counter_delta(snapshot, input_name) for snapshot in current_by_rank.values()
        )
        for input_name, output_name in counter_names.items()
    }
    observations.update(
        {
            "prefix_caching_enabled": any(
                bool(snapshot["metrics"]["enabled"]) for snapshot in current_by_rank.values()
            ),
            "ranks_with_prefix_cache_hits": sum(
                counter_delta(snapshot, "hits") > 0 for snapshot in current_by_rank.values()
            ),
            "ranks_with_mamba_restore_hits": sum(
                counter_delta(snapshot, "mamba_restore_hits") > 0
                for snapshot in current_by_rank.values()
            ),
            "coordinator_policy": (
                args.inference_dynamic_batching_prefix_caching_coordinator_policy
            ),
            "dp_rank_count": len(current_by_rank),
            "producer_follower_cycles": cycle_count,
            "cycle_count": cycle_count,
        }
    )
    return observations


def _analyze_coordinator_schedule(
    schedule_path, args, request_count, seed_snapshot, removal_snapshot
):
    """Prove routing used confirmed owners, then stopped using removed owners."""
    with open(schedule_path) as schedule_file:
        schedule = json.load(schedule_file)

    records = schedule["records"]
    group_count = args.prefix_cache_stress_groups
    copies = args.prefix_cache_stress_copies
    cycle_count = args.inference_repeat_n
    expected_records = group_count + cycle_count * request_count + 1
    assert (
        len(records) == expected_records
    ), f"coordinator recorded {len(records)} schedules, expected {expected_records}"
    assert schedule["policy"] == args.inference_dynamic_batching_prefix_caching_coordinator_policy
    assert schedule["data_parallel_size"] == len(removal_snapshot)
    assert group_count == len(
        removal_snapshot
    ), "controlled coordinator stress requires one distinct seed per DP worker"
    assert len(seed_snapshot) == group_count

    seed_records = records[:group_count]
    seed_owners = [record["rank_index"] for record in seed_records]
    assert all(record["matched_prefix_blocks"] == 0 for record in seed_records)
    assert (
        len(set(seed_owners)) == group_count
    ), f"seed phase reached only {len(set(seed_owners))}/{group_count} DP workers"
    seed_ranks_with_routable_prefixes = sum(
        bool(snapshot["routable_hashes"]) for snapshot in seed_snapshot
    )
    assert seed_ranks_with_routable_prefixes == group_count, (
        "seed phase materialized routable prefixes on only "
        f"{seed_ranks_with_routable_prefixes}/{group_count} DP workers"
    )

    cycle_routing = []
    cycle_start = group_count
    for cycle_idx in range(cycle_count):
        cycle_records = records[
            cycle_start + cycle_idx * request_count : cycle_start + (cycle_idx + 1) * request_count
        ]
        owner_matches = 0
        owner_mismatches = 0
        for request_idx, record in enumerate(cycle_records):
            group_idx = request_idx // copies
            if record["rank_index"] == seed_owners[group_idx]:
                owner_matches += 1
            else:
                owner_mismatches += 1

        distinct_ranks = len({record["rank_index"] for record in cycle_records})
        matched_requests = sum(int(record["matched_prefix_blocks"] > 0) for record in cycle_records)
        if schedule["policy"] in {"longest_prefix", "first_prefix_block"}:
            assert owner_matches == request_count, (
                f"{schedule['policy']} cycle {cycle_idx} routed {owner_mismatches} "
                "followers away from their confirmed owners"
            )
            assert matched_requests == request_count, (
                f"{schedule['policy']} cycle {cycle_idx} recorded only "
                f"{matched_requests}/{request_count} cache-affinity matches"
            )
            if schedule["policy"] == "first_prefix_block":
                assert all(record["num_hashes"] == 1 for record in cycle_records), (
                    f"first-prefix-block cycle {cycle_idx} did not route exclusively "
                    "from the first block"
                )
                assert all(record["matched_prefix_blocks"] == 1 for record in cycle_records), (
                    f"first-prefix-block cycle {cycle_idx} did not match the first "
                    "block on every selected owner"
                )
        elif schedule["policy"] == "load_balanced":
            assert distinct_ranks == len(removal_snapshot), (
                f"load-balanced cycle {cycle_idx} reached only "
                f"{distinct_ranks}/{len(removal_snapshot)} DP workers"
            )
            assert (
                owner_mismatches > 0
            ), "load-balanced scheduling accidentally preserved every seed owner"
            assert (
                matched_requests == 0
            ), "load-balanced schedule records must not claim prefix-affinity matches"
        cycle_routing.append(
            {
                "routing_owner_matches": owner_matches,
                "routing_owner_mismatches": owner_mismatches,
                "routing_matched_requests": matched_requests,
                "routing_distinct_ranks": distinct_ranks,
            }
        )

    routing_ranks_cleared = sum(not snapshot["routable_hashes"] for snapshot in removal_snapshot)
    assert (
        routing_ranks_cleared == group_count
    ), "generation-epoch removal left an old routable prefix on a DP worker"
    removal_probe = records[-1]
    assert (
        removal_probe["matched_prefix_blocks"] == 0
    ), "post-removal probe was routed using stale prefix ownership"
    aggregate = {
        "routing_seed_distinct_ranks": len(set(seed_owners)),
        "seed_ranks_with_routable_prefixes": seed_ranks_with_routable_prefixes,
        "routing_owner_matches": sum(cycle["routing_owner_matches"] for cycle in cycle_routing),
        "routing_owner_mismatches": sum(
            cycle["routing_owner_mismatches"] for cycle in cycle_routing
        ),
        "routing_matched_requests": sum(
            cycle["routing_matched_requests"] for cycle in cycle_routing
        ),
        "routing_removal_checks": 1,
        "routing_ranks_cleared": routing_ranks_cleared,
    }
    return cycle_routing, aggregate


if __name__ == "__main__":
    # enable inference mode in the very beginning as some fp8 optimizations
    # check for it.
    with torch.inference_mode():
        args = parse_and_validate_args(
            extra_args_provider=add_coordinator_prefix_cache_args,
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
            skip_prompt_log_probs=args.skip_prompt_log_probs,
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
        requests = _build_prefix_cache_stress_requests(args, tokenizer, sampling_params, requests)

        # Start Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        if args.prefix_cache_compare:
            inference_config = get_inference_config_from_model_and_args(model, args)
            if not inference_config.enable_prefix_caching:
                raise ValueError(
                    "--prefix-cache-compare requires "
                    "--inference-dynamic-batching-prefix-caching."
                )
            if args.inference_repeat_n < 2:
                raise ValueError(
                    "coordinator prefix-cache comparison requires at least two "
                    "cache-on cycles so completed LRU prefixes are reused."
                )

            reference_config = replace(
                inference_config, enable_prefix_caching=False, prefix_caching_mamba_gb=None
            )
            reference_context, reference_engine = _build_dynamic_engine(
                model, reference_config, tokenizer
            )
            reference_run = asyncio.run(main(reference_engine, requests, None, cycle_count=1))
            reference_observations = _coordinator_observations(
                reference_run["metric_snapshots"][0], args
            )

            del reference_engine
            del reference_context
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

            context, engine = _build_dynamic_engine(model, inference_config, tokenizer)
            if dist.get_rank() == 0:
                setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
                print("~~~")
                print(setup_prefix)
                print("~~~")
            seed_scenario_ids = [
                group_idx * args.prefix_cache_stress_copies
                for group_idx in range(args.prefix_cache_stress_groups)
            ]
            seed_requests = [requests[scenario_id] for scenario_id in seed_scenario_ids]
            schedule_output_path = args.coordinator_schedule_output_path
            if schedule_output_path is None:
                if not args.output_path:
                    raise ValueError(
                        "coordinator prefix-cache comparison requires --output-path "
                        "or --coordinator-schedule-output-path"
                    )
                schedule_output_path = f"{args.output_path}.coordinator_schedule.json"
            cache_on_run = asyncio.run(
                main(
                    engine,
                    requests,
                    args.inference_coordinator_port,
                    cycle_count=args.inference_repeat_n,
                    seed_requests=seed_requests,
                    exercise_epoch_removal=True,
                    schedule_output_path=schedule_output_path,
                )
            )

            if dist.get_rank() == 0 and args.output_path:
                baseline_snapshot = cache_on_run["metric_snapshots"][0]
                cycle_routing, routing_aggregate = _analyze_coordinator_schedule(
                    schedule_output_path,
                    args,
                    len(requests),
                    baseline_snapshot,
                    cache_on_run["removal_snapshot"],
                )
                cycle_observations = []
                previous_snapshot = baseline_snapshot
                for cycle_idx, current_snapshot in enumerate(cache_on_run["metric_snapshots"][1:]):
                    observations = _coordinator_observations(
                        current_snapshot, args, baseline_snapshot=previous_snapshot, cycle_count=1
                    )
                    observations.update(cycle_routing[cycle_idx])
                    cycle_observations.append(observations)
                    previous_snapshot = current_snapshot

                cache_on_observations = _coordinator_observations(
                    cache_on_run["removal_snapshot"],
                    args,
                    baseline_snapshot=baseline_snapshot,
                    cycle_count=args.inference_repeat_n,
                )
                cache_on_observations.update(routing_aggregate)
                reference_requests = _serialize_coordinator_requests(
                    reference_run["cycles"][0], requests
                )
                serialized_seed_results = _serialize_coordinator_requests(
                    cache_on_run["seed_results"], seed_requests, seed_scenario_ids
                )
                supplemental_outputs = [
                    {
                        "label": f"seed[{seed_idx}]",
                        "reference_scenario_id": str(scenario_id),
                        "request": serialized_seed_results[str(scenario_id)],
                    }
                    for seed_idx, scenario_id in enumerate(seed_scenario_ids)
                ]
                removal_probe_seed_index = cache_on_run["removal_probe_seed_index"]
                assert removal_probe_seed_index is not None
                removal_probe_scenario_id = seed_scenario_ids[removal_probe_seed_index]
                serialized_removal_probe = _serialize_coordinator_requests(
                    [cache_on_run["removal_probe_result"]],
                    [seed_requests[removal_probe_seed_index]],
                    [removal_probe_scenario_id],
                )[str(removal_probe_scenario_id)]
                supplemental_outputs.append(
                    {
                        "label": "post_epoch_removal_probe",
                        "reference_scenario_id": str(removal_probe_scenario_id),
                        "request": serialized_removal_probe,
                    }
                )
                serialized_cache_on_cycles = []
                for cycle_idx, results in enumerate(cache_on_run["cycles"]):
                    serialized_cache_on_cycles.append(
                        {
                            "cycle_index": cycle_idx,
                            "requests": _serialize_coordinator_requests(results, requests),
                            "observations": cycle_observations[cycle_idx],
                            "memory": {},
                        }
                    )
                comparison = {
                    "schema_version": 1,
                    "scenario_ids": sorted(reference_requests),
                    "require_prefill_skip": True,
                    "cache_off": {
                        "prefix_caching_enabled": False,
                        "generated_output_count": len(reference_run["cycles"][0]),
                        "requests": reference_requests,
                        "observations": reference_observations,
                        "memory": {},
                    },
                    "cache_on": {
                        "prefix_caching_enabled": True,
                        "generated_output_count": (
                            len(cache_on_run["seed_results"])
                            + sum(len(cycle) for cycle in cache_on_run["cycles"])
                            + 1
                        ),
                        "cycles": serialized_cache_on_cycles,
                        "supplemental_outputs": supplemental_outputs,
                        "aggregate_observations": cache_on_observations,
                    },
                }
                with open(args.output_path, "w") as fp:
                    json.dump(
                        {
                            "prefix_cache_comparison": comparison,
                            "prefix_cache_observations": cache_on_observations,
                        },
                        fp,
                        indent=4,
                    )
        else:
            engine = get_dynamic_inference_engine(model=model)
            if dist.get_rank() == 0:
                setup_prefix = build_dynamic_engine_setup_prefix(
                    args, model, engine.context, requests
                )
                print("~~~")
                print(setup_prefix)
                print("~~~")
            run_result = asyncio.run(main(engine, requests, args.inference_coordinator_port))
            if dist.get_rank() == 0 and args.output_path:
                results = run_result["cycles"][0]
                json_results = _serialize_coordinator_requests(results, requests)
                json_results["throughput"] = [
                    len(result.generated_tokens) / result.latency for result in results
                ]
                with open(args.output_path, "w") as output_file:
                    json.dump(json_results, output_file, indent=4)

        # Stop Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStop()
