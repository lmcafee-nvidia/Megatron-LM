# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from contextlib import ExitStack
from functools import partial
from unittest import mock

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from tests.unit_tests.inference.engines import disagg_test_utils as dtu
from tests.unit_tests.inference.engines import test_disagg_topology_pairwise as topology
from tests.unit_tests.inference.engines import test_dynamic_engine as dynamic_tests
from tests.unit_tests.test_utilities import Utils

MOE_CONFIG = partial(dynamic_tests.TransformerConfig, moe_token_dispatcher_type="alltoall")
MOE_SPEC = partial(dynamic_tests.get_gpt_layer_with_transformer_engine_spec, moe_grouped_gemm=True)


def moe_state(engine):
    model = engine.controller.inference_wrapped_model.model
    layers = [layer.mlp for layer in model.decoder.layers]
    assert layers and all(type(layer.experts).__name__ == "TEGroupedMLP" for layer in layers)
    weights = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value) and ".experts." in name
    }
    assert weights
    return layers, weights


@mock.patch.object(dynamic_tests, "TransformerConfig", MOE_CONFIG)
@mock.patch.object(dynamic_tests, "get_gpt_layer_with_transformer_engine_spec", MOE_SPEC)
@torch.inference_mode()
def test_real_moe_ep2_handoff(tmp_path):
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    control = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    checkpoint = topology.gather(tmp_path / "moe-weights" if rank == 0 else None, control)[0]
    if rank == 0:
        checkpoint.mkdir()
    dist.barrier(group=control)
    config = dtu.disagg_config(expert_model_parallel_size=2, use_moe_layer_spec=True)
    tokens = dtu.prompt(33)
    try:
        with dtu.real_engine(config) as source:
            model = source.controller.inference_wrapped_model.model
            dist_checkpointing.save(model.sharded_state_dict(), checkpoint)
            topology.load_canonical_model(model, checkpoint)
            source_group = topology.gather(
                dist.get_process_group_ranks(source.pg_collection.ep), control
            )[0]
            source_layers, source_weights = moe_state(source)
            owners = topology.gather(tuple(source_layers[0].local_expert_indices), control)
            assert sorted(owners[member][0] for member in source_group) == [0, 1]
            saved_weights = topology.gather(
                source_weights if rank in source_group else None, control
            )
            expected = dtu.run_to_completion(
                source, source.add_request(101, tokens, dtu.sampling())
            ).generated_tokens
            assert all(result == expected for result in topology.gather(expected, control))
            source.setup_kv_transfer("prefill", backend="nccl")
            with dtu.ForwardWitness(source) as source_witness:
                request = dtu.run_to_completion(
                    source, source.add_request(101, tokens, dtu.sampling(1, do_kv_handoff=True))
                )
            assert request.generated_tokens == expected[:1]
            assert source_witness.steps and all(step[2] for step in source_witness.steps)
            blocks = request.disaggregated_params["block_ids"]
            local_kv = source.context.memory_buffer[:, :, blocks].cpu().clone()
            source_kv = topology.gather(local_kv if rank in source_group else None, control)
            reference_kv = source_kv[source_group[0]]
            for index in range(len(blocks)):
                count = min(16, len(tokens) - 16 * index)
                assert torch.equal(
                    reference_kv[:, :, index, :count],
                    source_kv[source_group[1]][:, :, index, :count],
                )
            handoff = topology.gather(
                request.disaggregated_params if rank == source_group[0] else None, control
            )[source_group[0]]
            with dtu.real_engine(config, role="decode", backend="nccl") as target:
                target_model = target.controller.inference_wrapped_model.model
                topology.load_canonical_model(target_model, checkpoint)
                target_groups = topology.gather(
                    dist.get_process_group_ranks(target.pg_collection.ep), control
                )
                target_group = next(g for g in target_groups if not set(g) & set(source_group))
                assert len(source_group) == len(target_group) == 2
                target_layers, target_weights = moe_state(target)
                weights_match = rank not in target_group
                if rank in target_group:
                    source_rank = source_group[dist.get_rank(target.pg_collection.ep)]
                    weights_match = all(
                        torch.equal(value, saved_weights[source_rank][name])
                        for name, value in target_weights.items()
                    )
                assert all(topology.gather(weights_match, control))
                future = pending = None
                if rank in target_group:
                    future = dtu._enqueue_decode_handoff(target, handoff, tokens, dtu.sampling())
                    pending = target._pending_kv_imports[0]
                peers = topology.gather(
                    dtu.decode_peer_meta(target, pending) if pending is not None else None, control
                )
                if rank == source_group[0]:
                    source.push_handoff_kv(101, [peers[member] for member in target_group])
                    for handle in source._pending_kv_pushes[0][1]:
                        handle.wait()
                    assert source._poll_pending_kv_pushes() == 1
                if rank in target_group:
                    for handle in target._pending_transfer_handles(pending):
                        handle.wait()
                dist.barrier(group=control)
                local_tokens = local_calls = 0
                if rank in target_group:
                    dtu.assert_import_equal(target, pending, {"kv": reference_kv}, len(tokens))
                    dtu.admit_import(target)
                    with ExitStack() as stack, dtu.ForwardWitness(target) as decode_witness:
                        probes = []
                        for layer in target_layers:
                            for owner, method in (
                                (layer.token_dispatcher, "token_dispatch"),
                                (layer.token_dispatcher, "token_combine"),
                                (layer.experts, "forward"),
                            ):
                                original = getattr(owner, method)
                                probe = mock.patch.object(owner, method, wraps=original)
                                probes.append(stack.enter_context(probe))
                        actual = dtu.run_to_completion(target, future)
                    assert actual.generated_tokens == expected
                    assert all(not step[2] for step in decode_witness.steps)
                    assert decode_witness.steps[0][0] == len(tokens)
                    assert all(probe.call_count for probe in probes)
                    local_calls = sum(probe.call_count for probe in probes)
                    local_tokens = sum(call.args[0].shape[0] for call in probes[2].call_args_list)
                else:
                    for _ in expected:
                        assert not target.step_modern()["finished_request_records"]
                witnesses = topology.gather((rank, local_calls, local_tokens), control)
                assert all(witnesses[member][1] > 0 for member in target_group)
                assert sum(witnesses[member][2] for member in target_group) > 0
                source.release_handoff_blocks(101)
                dtu.assert_released(source)
                dtu.assert_released(target)
    finally:
        dist.destroy_process_group(control)
        Utils.destroy_model_parallel()
