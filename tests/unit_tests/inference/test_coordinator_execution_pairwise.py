# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import asynccontextmanager

import pytest

from megatron.core.inference.config import CudaGraphSizingDistribution
from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode
from tests.unit_tests.inference import test_coordinator_schedule_pairwise as schedule
from tests.unit_tests.inference.test_coordinator_features_pairwise import _active_target_ids


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("all_prefills", [False, True])
@pytest.mark.parametrize("distribution", list(CudaGraphSizingDistribution))
async def test_routed_prefill_graph_selection(monkeypatch, all_prefills, distribution):
    original = schedule.routed_model

    @asynccontextmanager
    async def instrumented(*args, **kwargs):
        async with original(
            *args,
            **kwargs,
            hidden_size=128,
            num_cuda_graphs=3,
            force_build_cuda_graphs=True,
            use_cuda_graphs_for_non_decode_steps=True,
            cuda_graph_all_prefills=all_prefills,
            cuda_graph_max_tokens=16,
            cuda_graph_mixed_prefill_count=2,
            cuda_graph_sizing_distribution=distribution,
        ) as h:
            replay, observations = _CudagraphReplayNode.forward, []

            def observed(ctx, runner, first, *inputs):
                c = h.engine.context
                ids, padded = _active_target_ids(h), c.padded_batch_dimensions
                counts = (c.active_token_count, c.num_prefill_requests, c.num_decode_requests)
                result = replay(ctx, runner, first, *inputs)
                if h.engine.use_coordinator and any(
                    len(h.engine.get_request(rid).prompt_tokens) == 65 for rid in ids
                ):
                    observations.append((padded, counts, inputs[0].shape[0]))
                return result

            monkeypatch.setattr(_CudagraphReplayNode, "forward", staticmethod(observed))
            yield h
            invalid = any(
                dim not in h.engine.context.cuda_graph_batch_dimensions_list
                or dim.token_count != physical
                for dim, _, physical in observations
            )
            expected = (
                12 if distribution == CudaGraphSizingDistribution.LINEAR else 8 * (1 + all_prefills)
            )
            mixed = any(
                n == 7 and p and d and dim.token_count == expected
                for dim, (n, p, d), _ in observations
            )
            prefill = any(
                p and not d and dim.token_count > 16 for dim, (_, p, d), _ in observations
            )
            assert await h.sync.all_reduce_max(mixed, prefill, invalid) == (1, int(all_prefills), 0)

    monkeypatch.setattr(schedule, "routed_model", instrumented)
    await schedule.test_routed_chunk_partition_beside_decode(monkeypatch)
