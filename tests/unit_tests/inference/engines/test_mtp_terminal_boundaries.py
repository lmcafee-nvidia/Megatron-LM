# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import RequestEntry
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
    set_rounder,
)
from tests.unit_tests.test_utilities import Utils


def _make_postprocess_engine(request, num_speculative_tokens=2, track_generated_token_events=False):
    """Build the smallest faithful owner of post-process state."""
    record = DynamicInferenceRequestRecord.from_request(request)
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(
        chunked_prefill_request_id=-1,
        kv_block_allocator=SimpleNamespace(pool_size=8, pool_avail=8, enable_prefix_caching=False),
        remove_vlm_request_data=mock.Mock(),
    )
    engine.controller = SimpleNamespace(
        tokenizer=SimpleNamespace(detokenize=lambda token_ids: str(token_ids[0]))
    )
    engine.requests = {request.request_id: RequestEntry(record=record, future=mock.Mock())}
    engine.finished_request_count = 0
    engine.evicted_request_count = 0
    engine.track_generated_token_events = track_generated_token_events
    engine.num_speculative_tokens = num_speculative_tokens
    engine.stop_word_being_finished_ids = set()
    engine.stop_word_finished_request_ids = set()
    engine._spec_steps = 0
    engine._spec_tokens_proposed_per_pos = torch.zeros(num_speculative_tokens, dtype=torch.long)
    engine._spec_tokens_accepted_per_pos = torch.zeros(num_speculative_tokens, dtype=torch.long)
    engine._stage_prompt_logprob_updates = mock.Mock()
    engine._prepare_handoff_metadata_batch = mock.Mock(return_value={})
    engine._promote_recomputed_prompt_scores = mock.Mock()
    return engine


def _make_decode_request(request_id=7, output_limit=8, keep_stop=False):
    request = DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        sampling_params=SamplingParams(
            num_tokens_to_generate=output_limit,
            termination_id=-1,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            top_n_logprobs=2,
            detokenize_stop_sequence=keep_stop,
        ),
        generated_tokens=[10],
        generated_log_probs=[-0.1],
        generated_top_n_logprobs=[{"10": -0.1}],
    )
    request.stop_word_ids = None
    request.add_event_add_engine()
    request.add_event_generated_token(10)
    return request


def _top_n_rows(token_ids):
    return [(torch.tensor([-0.1, -1.1]), torch.tensor([token, token + 1])) for token in token_ids]


def test_terminal_position_uses_accepted_prefix_then_replacement():
    """Accepted EOS wins, while rejected proposal IDs are never terminal."""
    positions = TextGenerationController._get_step_termination_token_positions(
        sampled_tokens_cpu=torch.tensor([41, 42, 99, 44]),
        accepted_tokens_cpu=torch.tensor([[99, 31, -1], [30, 99, 32], [30, 31, -1], [99, -1, -1]]),
        termination_ids=torch.tensor([99, 99, 99, -1]),
    )

    # Row 0 terminates at the first accepted draft, row 1 at the second, and
    # row 2 at its replacement. Row 3 has EOS disabled even though a draft is 99.
    assert positions.device.type == "cpu"
    assert positions.tolist() == [0, 1, 2, -1]

    rejected_eos = TextGenerationController._get_step_termination_token_positions(
        sampled_tokens_cpu=torch.tensor([7]),
        accepted_tokens_cpu=torch.tensor([[-1, -1]]),
        termination_ids=torch.tensor([99]),
    )
    assert rejected_eos.tolist() == [-1]


@pytest.mark.parametrize(
    ("boundary", "accepted_count", "expected"),
    [
        pytest.param(None, 3, 3, id="no-boundary"),
        pytest.param(0, 3, 1, id="first-accepted-position"),
        pytest.param(1, 3, 2, id="second-accepted-position"),
        pytest.param(2, 2, 3, id="replacement-position"),
        pytest.param(-1, 3, 0, id="boundary-before-step"),
    ],
)
def test_usable_proposal_count_follows_logical_boundary(boundary, accepted_count, expected):
    assert (
        DynamicInferenceEngine._get_usable_speculative_proposal_count(
            num_speculative_tokens=3,
            accepted_token_count=accepted_count,
            terminal_token_position=boundary,
        )
        == expected
    )


@pytest.mark.parametrize("keep_stop", [False, True], ids=["strip-stop", "keep-stop"])
def test_stop_boundary_metrics_do_not_depend_on_output_retention(keep_stop):
    """Only proposals through the first stop endpoint enter the denominator."""
    assert DynamicInferenceEngine._find_step_stop_match([0] * 100 + [20], [20, 20], [[20, 20]]) == (
        0,
        2,
    )
    request = _make_decode_request(keep_stop=keep_stop)
    request.stop_word_ids = [[20, 20]]
    engine = _make_postprocess_engine(
        request, num_speculative_tokens=3, track_generated_token_events=True
    )

    active_ids, finished_records = engine.post_process_requests(
        request_ids=torch.tensor([request.request_id]),
        finished_request_ids=torch.empty(0, dtype=torch.long),
        evict_request_ids=None,
        step_time=0.0,
        sample=torch.tensor([20]),
        accepted_tokens=torch.tensor([[20, 20, 20]]),
        log_probs=[[-0.2, -0.3, -0.4, -0.5]],
        consumed_chunked_prefill_request_id=-1,
        termination_token_positions=torch.tensor([-1]),
        top_n_logprobs={0: _top_n_rows([20, 20, 20, 20])},
    )

    assert active_ids == [request.request_id]
    assert finished_records == []
    assert engine.stop_word_finished_request_ids == {request.request_id}
    assert engine._spec_tokens_proposed_per_pos.tolist() == [1, 1, 0]
    assert engine._spec_tokens_accepted_per_pos.tolist() == [1, 1, 0]
    assert request.generated_tokens == ([10, 20, 20] if keep_stop else [10])
    assert request.generated_log_probs == pytest.approx([-0.1, -0.2, -0.3] if keep_stop else [-0.1])
    assert len(request.generated_log_probs) == len(request.generated_tokens)
    assert len(request.generated_top_n_logprobs) == len(request.generated_tokens)
    assert all(
        str(token) in top_n
        for token, top_n in zip(request.generated_tokens, request.generated_top_n_logprobs)
    )
    generated_events = [
        event.payload["token_id"]
        for event in request.events
        if event.type is DynamicInferenceEventType.GENERATED_TOKEN
    ]
    assert generated_events == request.generated_tokens
    assert engine._get_and_clear_stop_word_finished_ids([request.request_id]) == {
        request.request_id
    }
    assert engine.stop_word_finished_request_ids == set()
    assert engine.stop_word_being_finished_ids == {request.request_id}


def test_length_boundary_trims_tokens_scores_and_proposal_metrics_together():
    request = _make_decode_request(output_limit=2)
    engine = _make_postprocess_engine(request)

    engine.post_process_requests(
        request_ids=torch.tensor([request.request_id]),
        finished_request_ids=torch.empty(0, dtype=torch.long),
        evict_request_ids=None,
        step_time=0.0,
        sample=torch.tensor([22]),
        accepted_tokens=torch.tensor([[20, 21]]),
        log_probs=[[-0.2, -0.3, -0.4]],
        consumed_chunked_prefill_request_id=-1,
        termination_token_positions=torch.tensor([-1]),
        top_n_logprobs={0: _top_n_rows([20, 21, 22])},
    )

    assert request.generated_tokens == [10, 20]
    assert request.generated_log_probs == pytest.approx([-0.1, -0.2])
    assert len(request.generated_top_n_logprobs) == 2
    assert "20" in request.generated_top_n_logprobs[-1]
    assert request.generated_top_n_logprobs[-1]["20"] == pytest.approx(-0.1)
    assert engine._spec_tokens_proposed_per_pos.tolist() == [1, 0]
    assert engine._spec_tokens_accepted_per_pos.tolist() == [1, 0]


def test_chunked_prefill_provisional_sample_is_not_output_or_metric_data():
    request = DynamicInferenceRequest(
        request_id=7,
        prompt_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        sampling_params=SamplingParams(num_tokens_to_generate=4, termination_id=99),
    )
    engine = _make_postprocess_engine(request)

    active_ids, finished_records = engine.post_process_requests(
        request_ids=torch.tensor([request.request_id]),
        finished_request_ids=torch.empty(0, dtype=torch.long),
        evict_request_ids=None,
        step_time=0.0,
        sample=torch.tensor([99]),
        accepted_tokens=None,
        log_probs=None,
        consumed_chunked_prefill_request_id=request.request_id,
        termination_token_positions=torch.tensor([0]),
    )

    assert active_ids == [request.request_id]
    assert finished_records == []
    assert request.generated_tokens == []
    assert engine._spec_steps == 0
    assert engine._spec_tokens_proposed_per_pos.tolist() == [0, 0]


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestMtpAcceptedEosLifecycle(DynamicInferenceEngineTestBase):
    """Exercise accepted-draft EOS through both production schedule paths."""

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
        "schedule_mode",
        [
            pytest.param(AsyncScheduleMode.LEGACY, id="legacy"),
            pytest.param(AsyncScheduleMode.ASYNC, id="async"),
        ],
    )
    @torch.inference_mode()
    def test_accepted_eos_compacts_lifecycle_and_keeps_aligned_output(self, schedule_mode):
        """A verified draft EOS retires one row without publishing its suffix."""
        num_speculative_tokens = 2
        first_token, eos_token, trailing_draft, replacement = 90, 91, 92, 93
        config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=8,
            max_sequence_length=12,
            num_speculative_tokens=num_speculative_tokens,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            position_embedding_type="rope",
            use_flashinfer_fused_rope=False,
            track_generated_token_events=True,
            async_sched_mode=schedule_mode,
        )
        env = self._build_test_env(config)
        engine = env.engine
        controller = engine.controller
        context = engine.context
        model = controller.inference_wrapped_model.model
        controller.tokenizer.detokenize = lambda token_ids: str(token_ids[0])
        assert model.position_embedding_type == "rope"
        assert context.use_flashinfer_fused_rope is False

        witness = {
            "base_calls": 0,
            "max_base_position": -1,
            "mtp_depths": set(),
            "accepted_eos_rows": 0,
            "accepted_eos_with_suffix_rows": 0,
        }
        real_forward = model.forward

        def deterministic_forward(*args, **kwargs):
            position_ids = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
            assert position_ids is not None
            witness["max_base_position"] = max(
                witness["max_base_position"], int(position_ids.max().item())
            )
            logits = real_forward(*args, **kwargs)
            witness["base_calls"] += 1

            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_logits.fill_(-100.0)
            flat_logits[:, first_token] = 100.0
            stride = num_speculative_tokens + 1
            assert flat_logits.size(0) >= context.num_decode_requests * stride
            for request_idx in range(context.num_decode_requests):
                offset = request_idx * stride
                flat_logits[offset, first_token] = -100.0
                flat_logits[offset, eos_token] = 100.0
                flat_logits[offset + 1, first_token] = -100.0
                flat_logits[offset + 1, trailing_draft] = 100.0
                flat_logits[offset + 2, first_token] = -100.0
                flat_logits[offset + 2, replacement] = 100.0
            return logits

        real_mtp = model.compute_mtp_single_step

        def deterministic_mtp(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            hidden_states, logits = real_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
            )
            witness["mtp_depths"].add(int(depth))
            logits.fill_(-100.0)
            logits[..., eos_token if depth == 0 else trailing_draft] = 100.0
            return hidden_states, logits

        real_terminal_positions = controller._get_step_termination_token_positions

        def witnessed_terminal_positions(samples, accepted, termination_ids):
            positions = real_terminal_positions(samples, accepted, termination_ids)
            assert positions.device.type == "cpu"
            if accepted is not None:
                accepted_counts = (accepted != -1).reshape(accepted.size(0), -1).sum(dim=1)
                witness["accepted_eos_rows"] += int(
                    ((positions >= 0) & (positions < accepted_counts)).sum().item()
                )
                witness["accepted_eos_with_suffix_rows"] += int(
                    ((positions >= 0) & (positions + 1 < accepted_counts)).sum().item()
                )
            return positions

        model.forward = deterministic_forward
        model.compute_mtp_single_step = deterministic_mtp
        controller._get_step_termination_token_positions = witnessed_terminal_positions

        for request_id in range(3):
            engine._add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=torch.zeros(4, dtype=torch.long, device="cuda"),
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=8,
                        termination_id=eos_token if request_id == 1 else -1,
                        return_log_probs=True,
                        skip_prompt_log_probs=True,
                        top_n_logprobs=3,
                        top_k=1,
                    ),
                )
            )

        records = []
        for _ in range(20):
            if not engine.has_unfinished_requests():
                break
            records.extend(engine.step_modern()["finished_request_records"])
        assert not engine.has_unfinished_requests(), "engine did not converge"

        finished = {}
        for record in records:
            request = record.merge()
            finished[request.request_id] = request
        assert set(finished) == {0, 1, 2}
        eos_request = finished[1]
        assert eos_request.status == Status.COMPLETED
        assert eos_request.generated_tokens == [first_token, eos_token]
        assert len(eos_request.generated_log_probs) == len(eos_request.generated_tokens)
        assert len(eos_request.generated_top_n_logprobs) == len(eos_request.generated_tokens)
        assert all(
            str(token) in top_n
            for token, top_n in zip(
                eos_request.generated_tokens, eos_request.generated_top_n_logprobs
            )
        )
        assert str(eos_token) in eos_request.generated_top_n_logprobs[-1]
        assert eos_request.generated_log_probs[-1] == pytest.approx(
            eos_request.generated_top_n_logprobs[-1][str(eos_token)]
        )
        for survivor_id in (0, 2):
            survivor = finished[survivor_id]
            assert len(survivor.generated_tokens) == 8
            assert trailing_draft in survivor.generated_tokens
            assert replacement in survivor.generated_tokens

        generated_events = [
            event.payload["token_id"]
            for event in eos_request.events
            if event.type is DynamicInferenceEventType.GENERATED_TOKEN
        ]
        assert generated_events == eos_request.generated_tokens
        assert eos_request.events[-1].type is DynamicInferenceEventType.FINISH

        assert witness["base_calls"] > 0
        assert witness["max_base_position"] >= context.max_sequence_length
        assert witness["max_base_position"] < context.max_sequence_length_for_model
        assert witness["mtp_depths"] == set(range(num_speculative_tokens))
        assert witness["accepted_eos_rows"] > 0
        assert witness["accepted_eos_with_suffix_rows"] > 0
        assert engine._spec_tokens_accepted_per_pos.tolist() == (
            engine._spec_tokens_proposed_per_pos.tolist()
        )
        assert (
            engine._spec_tokens_proposed_per_pos[0].item()
            > engine._spec_tokens_proposed_per_pos[1].item()
        )
        assert context.active_token_count == 0
        assert context.total_request_count == 0
        assert engine.requests == {}
        if schedule_mode is AsyncScheduleMode.ASYNC:
            assert context.async_sched_compaction_step_count > 0
