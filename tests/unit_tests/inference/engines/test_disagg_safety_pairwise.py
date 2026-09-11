# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native source-lifetime safety at destructive engine boundaries."""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt import GPTModel
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tokenizers.text.libraries.null_tokenizer import NullTokenizer
from tests.unit_tests.inference.engines import disagg_test_utils as dtu
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401

_PIN_ERROR = "Cannot suspend while handoff state remains pinned; wait for RELEASE_KV"


def _error(call):
    try:
        call()
    except Exception as error:  # noqa: BLE001 - exact type and text are asserted below
        return type(error).__name__, str(error)
    return None


@pytest.mark.parametrize(
    ("model", "backend"),
    [pytest.param("gpt", "nixl", id="kv-nixl"), pytest.param("hybrid", "nccl", id="ssm-nccl")],
)
@pytest.mark.parametrize("stop_keep", [None, False, True], ids=["eos", "stop-strip", "stop-keep"])
@torch.inference_mode()
@mock.patch.object(GPTModel, "forward", autospec=True, side_effect=GPTModel.forward)
@mock.patch.object(MambaMixer, "forward", autospec=True, side_effect=MambaMixer.forward)
def test_nonpersist_suspend_rejects_owned_source_state(
    mixer, gpt, transport_world, model, backend, stop_keep
):
    """Owned KV/SSM publications reject suspend before its first mutation."""

    rank = dist.get_rank()
    assert dist.get_world_size() == 4, "safety pair requires two explicit source/decode pairs"
    source = rank % 2 == 0
    tokens = dtu.prompt(33)
    config = dtu.disagg_config(
        model_provider=model,
        kv_cache_management_mode=KVCacheManagementMode.RECOMPUTE,
        static_kv_memory_pointers=False,
    )
    with dtu.real_engine(config, role="prefill" if source else "decode", backend=backend) as engine:
        engine.controller.tokenizer.tokenize = NullTokenizer(config.vocab_size).text_to_ids
        engine.controller.tokenizer.bos = None
        metadata = state = guard = publication = None
        if source:
            request = dtu.run_to_completion(
                engine, engine.add_request(101, tokens, dtu.sampling(1, do_kv_handoff=True))
            )
            metadata, state = dtu.snapshot_source(engine, request)
            blocks = metadata["block_ids"]
            allocator = engine.context.kv_block_allocator
            refs = allocator.block_ref_counts[blocks].clone()
            ssm_slot = engine._pinned_handoff_ssm_slots.get(101)
            buffer = engine.context.memory_buffer
            sentinel = object()
            engine._vision_embedding_cache["suspend-guard"] = sentinel
            engine._vision_embedding_cache_bytes = 1
            with mock.patch.object(
                engine.context,
                "deallocate_inference_state_buffers",
                side_effect=RuntimeError("test blocked destructive deallocation"),
            ) as destructive:
                idempotent = []
                for idle_state in (EngineState.SUSPENDED, EngineState.SUSPENDING):
                    engine.state = idle_state
                    idempotent.append(_error(engine.suspend))
                engine.state = EngineState.RUNNING
                suspend_error = _error(engine.suspend)
            guard = (
                idempotent,
                suspend_error,
                InferenceMode.is_active(),
                engine._vision_embedding_cache.get("suspend-guard") is sentinel,
                engine.context.memory_buffer is buffer,
                destructive.call_count,
                engine.state.name,
                torch.equal(allocator.block_ref_counts[blocks], refs),
            )
            publication = (
                tuple(blocks),
                tuple(refs.tolist()),
                ssm_slot,
                backend != "nixl" or metadata["kv_meta"]["base_addr"] == buffer.data_ptr(),
                mixer.call_count,
                gpt.call_count,
                request.generated_tokens[0],
            )

        peer = dtu.exchange(
            (metadata, state, guard, publication) if source else None, transport_world
        )
        if not source:
            metadata, state, guard, publication = peer
        assert metadata["kv_meta"]["resume_tokens"] == [publication[6]]
        assert engine.controller.tokenizer.tokenize(str(publication[6])) == [publication[6]]
        params = dtu.sampling(7)
        params.termination_id = publication[6] if stop_keep is None else -1
        params.stop_words = [] if stop_keep is None else [str(publication[6])]
        params.detokenize_stop_sequence = bool(stop_keep)
        expected_tokens = [] if stop_keep is False else [publication[6]]
        pending, future = dtu.complete_transfer(engine, metadata, tokens, params, transport_world)

        destination_release = None
        if not source:
            assert pending.sampling_params.num_tokens_to_generate == 7
            dtu.assert_import_equal(engine, pending, state, len(tokens))
            owned = pending.local_blocks + pending.continuation_blocks
            refs_before = engine.context.kv_block_allocator.block_ref_counts[owned].clone()
            with mock.patch.object(
                engine, "_release_pending_kv_import", wraps=engine._release_pending_kv_import
            ) as release:
                dtu.admit_import(engine)
                second_poll = engine._poll_pending_kv_imports()
            result = dtu.run_to_completion(engine, future)
            destination_release = (
                release.call_count,
                second_poll,
                bool((refs_before > 0).all()),
                bool((engine.context.kv_block_allocator.block_ref_counts[owned] == 0).all()),
                (result.generated_tokens, gpt.call_count, mixer.call_count),
            )
        dist.barrier(group=transport_world)

        source_release = None
        if source:
            blocks = metadata["block_ids"]
            refs_before = engine.context.kv_block_allocator.block_ref_counts[blocks].clone()
            with (
                mock.patch.object(
                    engine,
                    "_release_pinned_handoff_blocks",
                    wraps=engine._release_pinned_handoff_blocks,
                ) as release_blocks,
                mock.patch.object(
                    engine,
                    "_release_pinned_handoff_ssm_slot",
                    wraps=engine._release_pinned_handoff_ssm_slot,
                ) as release_ssm,
            ):
                engine.release_handoff_blocks(101)
                engine.release_handoff_blocks(101)
            source_release = (
                release_blocks.call_count,
                release_ssm.call_count,
                bool((refs_before > 0).all()),
                bool((engine.context.kv_block_allocator.block_ref_counts[blocks] == 0).all()),
            )
        dtu.assert_released(engine)
        peer_release = dtu.exchange((source_release, destination_release), transport_world)
        observed_source = source_release if source else peer_release[0]
        observed_destination = destination_release if not source else peer_release[1]
        assert publication[0] and all(ref > 0 for ref in publication[1])
        assert (publication[2] is not None) == (model == "hybrid")
        assert publication[3] and publication[4 if model == "hybrid" else 5] > 0
        assert observed_source == (1, 1, True, True)
        assert observed_destination == (1, 0, True, True, (expected_tokens, 0, 0))
        assert guard[:2] == ([None, None], ("RuntimeError", _PIN_ERROR))
        assert guard[2:] == (True, True, True, 0, "RUNNING", True)
