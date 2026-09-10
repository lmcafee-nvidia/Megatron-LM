# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Imported prefix reuse and speculative decode continuation on real engines."""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    admit_import,
    assert_import_equal,
    assert_released,
    collocated_reference,
    complete_transfer,
    disagg_config,
    exchange,
    prompt,
    real_engine,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401


@pytest.mark.parametrize("mode", ["cached-prefix", "mtp-depth-two"])
@torch.inference_mode()
def test_cached_prefix_and_mtp_handoff(transport_world, mode):
    cached = mode == "cached-prefix"
    depth = 0 if cached else 2
    config = disagg_config(
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU, num_speculative_tokens=depth
    )
    tokens = prompt(33 if cached else 31)
    weights, expected = collocated_reference(config, tokens, sampling(7))
    source = dist.get_rank() % 2 == 0
    with real_engine(
        config,
        role="prefill" if source else "decode",
        backend="nixl" if cached else "nccl",
        weights=weights,
    ) as engine:
        inner_calls = [0] * depth
        hooks = []
        if depth:
            model = engine.controller.inference_wrapped_model.model
            assert len(model.mtp.layers) == depth
            for index, layer in enumerate(model.mtp.layers):

                def record_inner(module, args, output, index=index):
                    ids = engine.context.request_ids[: engine.context.total_request_count]
                    if 101 in ids.tolist():
                        inner_calls[index] += 1

                hooks.append(layer.mtp_model_layer.register_forward_hook(record_inner))
        try:
            metadata = state = None
            if source:
                request = run_to_completion(
                    engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
                )
                assert request.generated_tokens == expected[:1]
                metadata, state = snapshot_source(engine, request)
                assert len(metadata["kv_meta"]["resume_tokens"]) == depth + 1
            peer = exchange((metadata, state) if source else None, transport_world)
            if not source:
                metadata, state = peer
            for iteration in range(2 if cached else 1):
                agent = engine._kv_transfer_agent
                with mock.patch.object(
                    agent, "begin_pull_blocks", wraps=agent.begin_pull_blocks
                ) as pull:
                    pending, future = complete_transfer(
                        engine, metadata, tokens, sampling(7), transport_world
                    )
                if not source:
                    assert pull.call_count == 1
                    sent_source_blocks, imported_blocks = pull.call_args.args[1:]
                    expected_cached = 2 if cached and iteration else 0
                    assert pending.cached_prefix_block_count == expected_cached
                    assert len(imported_blocks) == len(metadata["block_ids"]) - expected_cached
                    assert sent_source_blocks == metadata["block_ids"][expected_cached:]
                    if expected_cached:
                        assert len(imported_blocks) == 1, "Only the uncached tail should transfer"
                        assert pending.local_blocks[:2] == previously_imported_prefix
                    assert_import_equal(engine, pending, state, len(tokens))
                    previously_imported_prefix = list(pending.local_blocks[:2])
                    transferred_inputs = list(pending.resume_tokens)
                    assert transferred_inputs == metadata["kv_meta"]["resume_tokens"]
                    if depth:
                        assert len(transferred_inputs) == 3
                        assert len(pending.continuation_blocks) == 1
                    admit_import(engine)
                    context = engine.context
                    assert context.request_query_lengths[0].item() == depth + 1
                    assert context.token_to_input_ids[: depth + 1].tolist() == transferred_inputs
                    assert context.token_to_pos_ids[: depth + 1].tolist() == list(
                        range(len(tokens), len(tokens) + depth + 1)
                    )
                    with ForwardWitness(engine) as witness:
                        actual = run_to_completion(engine, future)
                    assert actual.generated_tokens == expected
                    assert witness.steps and all(not step[2] for step in witness.steps)
                    assert witness.steps[0][0] == len(tokens)
                    assert_released(engine)
                dist.barrier(group=transport_world)
            if depth:
                assert all(inner_calls), "Every inner MTP layer must execute for the handoff target"
            if source:
                engine._poll_pending_kv_pushes()
                engine.release_handoff_blocks(101)
                assert_released(engine)
        finally:
            for hook in hooks:
                hook.remove()
