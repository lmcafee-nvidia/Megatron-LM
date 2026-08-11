# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Request-lifecycle regressions and compact real-engine parity scenarios.

The field policy is deliberately data, rather than an implicit list in the
checkpoint implementation.  It inventories both DynamicInferenceRequest and
SamplingParams state which can be present at checkpoint time:

* preserve: identity/value remains meaningful on the new segment;
* transform/recompute: prompt, remaining prompt, generation budget, and prefix
  hashes are rebuilt for the new segment;
* reset: generated output, cache epoch, and segment-local observability start
  fresh;
* aggregate: merge concatenates output/event/logprob histories;
* first/last: merge chooses the original input or terminal lifecycle state;
* derived output: generated length, text, and wire form derive from the merged
  token history.
"""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    InferenceRequest,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.test_utilities import Utils

CHECKPOINT_FIELD_POLICY = {
    # DynamicInferenceRequest dataclass and dynamic state.
    "request_id": "preserve",
    "prompt": "first",
    "prompt_tokens": "transform/recompute",
    "remaining_prompt_tokens": "transform/recompute",
    "sampling_params": "preserve",
    "stop_word_ids": "preserve",
    "policy_epoch": "preserve",
    "kv_cache_epoch": "reset",
    "block_size_tokens": "preserve",
    "enable_prefix_caching": "preserve",
    "precomputed_block_hashes": "transform/recompute",
    "generated_tokens": "aggregate",
    "generated_log_probs": "aggregate",
    "generated_top_n_logprobs": "aggregate",
    "events": "aggregate",
    "ttft": "first populated",
    "status": "last",
    "generated_length": "derived output",
    "generated_text": "derived output",
    # SamplingParams dataclass and add_attributes() extensions.
    "temperature/top_k/top_p": "preserve",
    "return_log_probs/top_n_logprobs/skip_prompt_log_probs": "preserve",
    "stop_words/detokenize_stop_sequence": "preserve",
    "return_prompt_tokens/streaming/streaming_interval": "preserve",
    "add_attributes() fields": "preserve",
    "num_tokens_to_generate": "transform/recompute",
    "num_tokens_total": "reset",
}


def _request(**overrides):
    values = {
        "request_id": 17,
        "prompt": "prompt",
        "prompt_tokens": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        "sampling_params": SamplingParams(num_tokens_to_generate=6, termination_id=-1),
        "status": Status.ACTIVE_AND_GENERATING_TOKENS,
    }
    values.update(overrides)
    return DynamicInferenceRequest(**values)


def test_checkpoint_field_policy_preserves_dynamic_state_and_merges_ttft():
    """Checkpoint all policy classes, including add_attributes() extensions."""
    request = _request(generated_tokens=[5, 6])
    request.sampling_params.add_attributes({"min_length": 3, "custom_policy": {"a": 1}})
    request.stop_word_ids = [[41, 42]]
    request.policy_epoch = [(0, 3)]
    request.kv_cache_epoch = [(0, 3)]
    request.ttft = None
    record = DynamicInferenceRequestRecord.from_request(request)

    record.checkpoint()
    checkpoint = record[-1]
    assert checkpoint.status == Status.ACTIVE_AND_GENERATING_TOKENS
    assert checkpoint.stop_word_ids == [[41, 42]]
    assert checkpoint.stop_word_ids is not request.stop_word_ids
    assert checkpoint.sampling_params.min_length == 3
    assert checkpoint.sampling_params.custom_policy == {"a": 1}
    assert checkpoint.sampling_params.custom_policy is not request.sampling_params.custom_policy
    assert checkpoint.sampling_params.num_tokens_to_generate == 4
    assert checkpoint.sampling_params.num_tokens_total is None
    assert checkpoint.policy_epoch == [(0, 3)]
    assert checkpoint.policy_epoch is not request.policy_epoch
    assert checkpoint.kv_cache_epoch is None

    checkpoint.ttft = 0.25
    checkpoint.generated_tokens = [7]
    checkpoint.status = Status.COMPLETED
    merged = record.merge()
    assert merged.generated_tokens == [5, 6, 7]
    assert merged.ttft == 0.25
    assert merged.status == Status.COMPLETED


@pytest.mark.skipif(not torch.cuda.is_available(), reason="checkpoint storage is a CUDA contract")
def test_repeated_checkpoints_archive_superseded_cuda_prompt_storage():
    """Checkpoint history keeps wire data without retaining cumulative GPU prompts."""
    request = _request(
        prompt_tokens=torch.arange(8, device="cuda", dtype=torch.int64),
        sampling_params=SamplingParams(
            num_tokens_to_generate=8, termination_id=-1, return_prompt_tokens=True
        ),
    )
    record = DynamicInferenceRequestRecord.from_request(request)
    for token in range(3):
        record[-1].generated_tokens.append(token)
        record.checkpoint()

    archived, current = record.requests[:-1], record[-1]
    assert all(not segment.prompt_tokens.is_cuda for segment in archived)
    assert all(not segment.remaining_prompt_tokens.is_cuda for segment in archived)
    assert current.prompt_tokens.is_cuda and current.remaining_prompt_tokens.is_cuda
    live_storages = {
        (
            segment.prompt_tokens.untyped_storage().data_ptr(),
            segment.prompt_tokens.untyped_storage().nbytes(),
        )
        for segment in record.requests
        if segment.prompt_tokens.is_cuda
    }
    assert len(live_storages) == 1
    assert record.merge().serialize()["prompt_tokens"] == ("tensor", list(range(8)))


def test_serialize_restores_prompt_after_serialization_failure(monkeypatch):
    """The wire-size optimization must not mutate a local request on failure."""
    prompt = torch.tensor([1, 2, 3], dtype=torch.int64)
    request = _request(prompt_tokens=prompt)

    def raise_from_base(_self):
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(InferenceRequest, "serialize", raise_from_base)
    with pytest.raises(RuntimeError, match="injected"):
        request.serialize()
    assert request.prompt_tokens is prompt


def _bare_public_engine() -> DynamicInferenceEngine:
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(block_size_tokens=4, enable_prefix_caching=False)
    engine.controller = SimpleNamespace(
        tokenizer=SimpleNamespace(),
        tokenize_prompt=lambda _tokenizer, _prompt, _add_bos=False: [1, 2, 3],
    )
    engine.requests = {}
    engine._add_request = mock.Mock()
    return engine


def test_public_add_request_defaults_params_and_rejects_duplicate_before_mutation(monkeypatch):
    """The optional API argument is usable and live IDs cannot be submitted twice."""
    engine = _bare_public_engine()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    engine.add_request(4, "hello", None)
    added = engine._add_request.call_args.args[0]
    assert isinstance(added.sampling_params, SamplingParams)

    engine.requests[4] = object()
    engine._add_request.reset_mock()
    with pytest.raises(ValueError, match="already live"):
        engine.add_request(4, "duplicate", SamplingParams())
    engine._add_request.assert_not_called()


def test_failed_queue_counts_as_unfinished_work():
    """A direct generate() loop must take one bookkeeping step for failed-only input."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(has_unfinished_requests=lambda: False)
    engine.waiting_request_ids = deque()
    engine.failed_request_ids = [9]
    assert engine.has_unfinished_requests()


class TestRequestLifecycleParity(_DynamicInferenceEngineTestBase):
    """Compare a seeded baseline against a single forced lifecycle transition."""

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
        Utils.destroy_model_parallel()

    @classmethod
    def _run_pair(cls, *, intervention, **config_overrides):
        top_n_logprobs = config_overrides.pop("top_n_logprobs", 0)
        return_prompt_tokens = config_overrides.pop("return_prompt_tokens", False)
        config = _DynamicEngineTestConfig(
            num_requests=2,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=6,
            context_max_requests=4,
            **config_overrides,
        )

        def run(with_intervention):
            env = cls._build_test_env(config)
            # The harness intentionally uses a constant test detokenizer. Use a
            # compositional one here so checkpoint segments and a final sequence
            # have the same text representation, as a real tokenizer contract
            # requires at token boundaries.
            detokenize = lambda tokens, **_kwargs: "".join(chr(33 + token) for token in tokens)
            env.engine.controller.tokenizer.detokenize = detokenize
            env.engine.controller.detokenize = lambda _tokenizer, tokens, **kwargs: detokenize(
                tokens, **kwargs
            )
            engine = env.engine
            for request in env.requests:
                request.sampling_params.top_n_logprobs = top_n_logprobs
                request.sampling_params.return_prompt_tokens = return_prompt_tokens
            futures = [engine._add_request(request) for request in env.requests]
            completed = {}
            for step in range(64):
                result = engine.step_modern()
                for record in result["finished_request_records"]:
                    merged = record.merge()
                    completed[merged.request_id] = merged
                if with_intervention and step == 2:
                    intervention(engine)
                if not engine.has_unfinished_requests():
                    break
            else:
                pytest.fail("lifecycle scenario did not drain within 64 steps")

            assert len(completed) == len(futures)
            assert not engine.requests
            assert not engine.waiting_request_ids
            assert not engine.failed_request_ids
            assert engine.context.total_request_count == 0
            assert engine.context.paused_request_count == 0
            assert all(future.done() for future in futures)
            return completed, step

        baseline, baseline_steps = run(False)
        treatment, treatment_steps = run(True)
        assert treatment_steps <= 64 and baseline_steps <= 64
        assert set(baseline) == set(treatment)
        for request_id in baseline:
            left, right = baseline[request_id], treatment[request_id]
            assert left.status == right.status == Status.COMPLETED
            assert left.generated_tokens == right.generated_tokens
            assert left.generated_text == right.generated_text
            assert left.generated_length == right.generated_length
            for logprob_field in ("prompt_log_probs", "generated_log_probs"):
                left_logprobs = getattr(left, logprob_field)
                right_logprobs = getattr(right, logprob_field)
                assert (left_logprobs is None) == (right_logprobs is None)
                if left_logprobs is not None:
                    assert left_logprobs == pytest.approx(right_logprobs, abs=1e-5)
            for top_n_field in ("prompt_top_n_logprobs", "generated_top_n_logprobs"):
                left_top_n = getattr(left, top_n_field)
                right_top_n = getattr(right, top_n_field)
                assert (left_top_n is None) == (right_top_n is None)
                if left_top_n is not None:
                    assert len(left_top_n) == len(right_top_n)
                    for left_values, right_values in zip(left_top_n, right_top_n):
                        assert left_values.keys() == right_values.keys()
                        for key in left_values:
                            assert left_values[key] == pytest.approx(right_values[key], abs=1e-5)
            # Serialization remains usable for both terminal records. Timings
            # and event timestamps intentionally differ between independent
            # runs, so compare the wire contract rather than wall-clock values.
            left_wire, right_wire = left.serialize(), right.serialize()
            assert left_wire.keys() == right_wire.keys()
            for key in (
                "request_id",
                "status",
                "generated_tokens",
                "generated_text",
                "generated_length",
            ):
                assert left_wire[key] == right_wire[key]
            baseline_events = [event["type"] for event in left_wire["events"]]
            treatment_events = [event["type"] for event in right_wire["events"]]
            assert baseline_events[0] == treatment_events[0] == "ADD_ENGINE"
            assert baseline_events[-1] == treatment_events[-1] == "FINISH"
            # RECOMPUTE creates a new context admission after checkpointing;
            # PERSIST and OFFLOAD keep their original admission. This deliberate
            # lifecycle difference must remain visible in the serialized trace.
            assert treatment_events.count("ADD_CONTEXT") >= baseline_events.count("ADD_CONTEXT")

    @pytest.mark.internal
    @pytest.mark.parametrize("mode", ["persist", "offload", "recompute"])
    @torch.inference_mode()
    def test_seeded_suspend_resume_parity(self, mode):
        """PERSIST/OFFLOAD/RECOMPUTE have identical terminal results after resume."""
        self._run_pair(
            intervention=lambda engine: (engine.suspend(), engine.resume()),
            kv_cache_management_mode=mode,
            static_kv_memory_pointers=False,
        )

    @pytest.mark.internal
    @torch.inference_mode()
    def test_recompute_preserves_prompt_and_generated_logprob_contracts(self):
        """Re-admission keeps aligned prompt/generated and top-N logprob output."""
        self._run_pair(
            intervention=lambda engine: (engine.suspend(), engine.resume()),
            kv_cache_management_mode="recompute",
            static_kv_memory_pointers=False,
            return_log_probs=True,
            materialize_only_last_token_logits=False,
            top_n_logprobs=2,
            return_prompt_tokens=True,
        )

    @pytest.mark.internal
    @torch.inference_mode()
    def test_seeded_stochastic_recompute_parity(self):
        """A seeded non-greedy request is replayed exactly through checkpoint/re-admission."""
        self._run_pair(
            intervention=lambda engine: (engine.suspend(), engine.resume()),
            kv_cache_management_mode="recompute",
            static_kv_memory_pointers=False,
            temperature=0.8,
            top_k=12,
        )

    @pytest.mark.internal
    @torch.inference_mode()
    def test_persist_static_pointer_suspend_resume_parity(self):
        """PERSIST with static KV pointers preserves the same live allocation across the pair."""

        def suspend_resume_with_pointer_check(engine):
            before = engine.context.memory_buffer.data_ptr()
            engine.suspend()
            engine.resume()
            assert engine.context.memory_buffer.data_ptr() == before

        self._run_pair(
            intervention=suspend_resume_with_pointer_check,
            kv_cache_management_mode="persist",
            static_kv_memory_pointers=True,
        )

    @pytest.mark.internal
    @torch.inference_mode()
    def test_drained_reset_preserves_coordinator_async_primitives(self):
        """A drained coordinator reset keeps its long-lived asyncio coordination objects."""
        env = self._build_test_env(_DynamicEngineTestConfig(num_requests=0))
        engine = env.engine
        engine.use_coordinator = True
        identities = (engine._loop, engine._cond, engine._state_events)
        event_identities = {state: id(event) for state, event in engine._state_events.items()}

        engine.reset()

        assert engine.use_coordinator
        assert (engine._loop, engine._cond, engine._state_events) == identities
        assert {
            state: id(event) for state, event in engine._state_events.items()
        } == event_identities
        assert engine._state_events[engine.state].is_set()

    @pytest.mark.internal
    @torch.inference_mode()
    def test_generate_drains_invalid_only_and_mixed_batches(self):
        """generate() returns one terminal FAILED record per invalid input and drains it."""
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
        assert len(invalid_only) == 1
        assert invalid_only[0].merge().status == Status.FAILED
        assert (
            not engine.requests and not engine.failed_request_ids and not engine.waiting_request_ids
        )

        mixed = engine.generate(["invalid", "valid"], params)
        assert [record.merge().status for record in mixed] == [Status.FAILED, Status.COMPLETED]
        assert (
            not engine.requests and not engine.failed_request_ids and not engine.waiting_request_ids
        )
