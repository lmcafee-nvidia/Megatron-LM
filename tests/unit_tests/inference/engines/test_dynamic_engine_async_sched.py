# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DynamicBatchControllerStepResult,
)


def _make_engine(async_sched_mode=AsyncScheduleMode.SERIAL, **overrides):
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    context = SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=async_sched_mode),
        is_hybrid_model=False,
        enable_prefix_caching=False,
    )
    model_config = SimpleNamespace(
        expert_model_parallel_size=1, num_moe_experts=None, moe_enable_routing_replay=False
    )
    engine.context = context
    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=model_config)),
        has_pending_async_forward=mock.Mock(return_value=False),
    )
    engine.enable_chunked_prefill = False
    engine.num_speculative_tokens = 0
    engine.materialize_only_last_token_logits = True

    for name, value in overrides.items():
        if name.startswith("context_"):
            setattr(context, name.removeprefix("context_"), value)
        elif name.startswith("model_config_"):
            setattr(model_config, name.removeprefix("model_config_"), value)
        else:
            setattr(engine, name, value)
    return engine


@pytest.mark.parametrize(
    "overrides, should_raise",
    [
        ({"async_sched_mode": AsyncScheduleMode.LEGACY, "num_speculative_tokens": 1}, False),
        ({}, False),
        ({"async_sched_mode": AsyncScheduleMode.OVERLAP}, False),
        ({"enable_chunked_prefill": True}, True),
        ({"num_speculative_tokens": 1}, True),
        ({"async_sched_mode": AsyncScheduleMode.OVERLAP, "num_speculative_tokens": 1}, True),
        ({"context_is_hybrid_model": True}, False),
        ({"context_enable_prefix_caching": True}, True),
        ({"materialize_only_last_token_logits": False}, True),
        ({"model_config_expert_model_parallel_size": 2}, False),
        ({"model_config_num_moe_experts": 4}, False),
        ({"model_config_expert_model_parallel_size": 2, "model_config_num_moe_experts": 4}, False),
        ({"model_config_moe_enable_routing_replay": True}, True),
    ],
)
def test_validate_async_sched_support_for_config(overrides, should_raise):
    """Ensure engine config validation accepts only supported async scheduling configs."""
    engine = _make_engine(**overrides)

    if should_raise:
        with pytest.raises(ValueError, match="Async scheduling"):
            engine._validate_async_sched_support_for_config()
    else:
        engine._validate_async_sched_support_for_config()


@pytest.mark.parametrize(
    "async_sched_mode, sampling_params, should_raise",
    [
        (AsyncScheduleMode.LEGACY, SamplingParams(top_k=0, top_p=0.5), False),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0), False),
        (AsyncScheduleMode.OVERLAP, SamplingParams(top_k=1, top_p=0.0), False),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=0, top_p=0.0), True),
        (AsyncScheduleMode.OVERLAP, SamplingParams(top_k=0, top_p=0.0), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.5), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, return_log_probs=True), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, top_n_logprobs=1), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, stop_words=["END"]), True),
    ],
)
def test_validate_async_sched_support_for_request(async_sched_mode, sampling_params, should_raise):
    """Ensure engine request validation accepts only supported async scheduling requests."""
    engine = _make_engine(async_sched_mode=async_sched_mode)
    request = SimpleNamespace(sampling_params=sampling_params)

    if should_raise:
        with pytest.raises(ValueError, match="Async scheduling"):
            engine._validate_async_sched_support_for_request(request)
    else:
        engine._validate_async_sched_support_for_request(request)


def test_add_request_runs_async_sched_request_validation():
    """Ensure request validation is called before mutating engine request state."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine._validate_async_sched_support_for_request = mock.Mock(
        side_effect=RuntimeError("validated")
    )
    request = SimpleNamespace(request_id=10)

    with pytest.raises(RuntimeError, match="validated"):
        engine._add_request(request)

    engine._validate_async_sched_support_for_request.assert_called_once_with(request)


def test_async_forward_reenters_controller_after_primer_without_rescheduling():
    """The engine immediately consumes a primer without crossing its step boundary."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.state = EngineState.RUNNING
    engine.logging_step_interval = 0
    engine.metrics_writer = None
    engine.schedule_waiting_requests = mock.Mock(return_value=False)
    engine._advance_async_sched_primer = mock.AsyncMock()
    engine.context = SimpleNamespace(
        step_count=4,
        prefix_cache_lru_clock=7,
        active_token_count=2,
        is_decode_only=mock.Mock(return_value=True),
    )
    expected_output = {"sample": "tokens"}
    engine.controller = SimpleNamespace(
        async_generate_output_tokens_dynamic_batch=mock.AsyncMock(
            side_effect=[
                DynamicBatchControllerStepResult(primer_only=True),
                DynamicBatchControllerStepResult(output=expected_output),
            ]
        )
    )

    with (
        mock.patch("megatron.core.inference.engines.dynamic_engine.nvtx_range_push") as range_push,
        mock.patch("megatron.core.inference.engines.dynamic_engine.nvtx_range_pop") as range_pop,
    ):
        output, context_state, step_time = asyncio.run(engine.async_forward())

    assert output is expected_output
    assert context_state == {"active_token_count": 2, "step_count": 4, "kv_stats": None}
    assert step_time == 0.0
    assert engine.context.step_count == 5
    assert engine.context.prefix_cache_lru_clock == 8
    engine.schedule_waiting_requests.assert_called_once_with()
    engine.controller.async_generate_output_tokens_dynamic_batch.assert_has_awaits(
        [mock.call(drain_pending_forward=False), mock.call(drain_pending_forward=False)]
    )
    engine._advance_async_sched_primer.assert_awaited_once_with()
    range_push.assert_called_once_with("Decode")
    range_pop.assert_called_once_with("Decode")


@pytest.mark.parametrize(
    "mode, has_pending_forward, expected",
    [
        (AsyncScheduleMode.LEGACY, True, False),
        (AsyncScheduleMode.SERIAL, True, True),
        (AsyncScheduleMode.OVERLAP, True, True),
        (AsyncScheduleMode.OVERLAP, False, False),
    ],
)
def test_should_defer_async_sched_admission(mode, has_pending_forward, expected):
    """Defer async admission only while a forward is pending."""
    engine = _make_engine(async_sched_mode=mode)
    engine.controller.has_pending_async_forward.return_value = has_pending_forward

    assert engine._should_defer_async_sched_admission() is expected


def test_ready_async_admission_stays_queued():
    """Leave a ready request queued until pending async logits are drained."""
    engine = _make_engine()
    request = SimpleNamespace(remaining_prompt_tokens=[1, 2, 3])
    engine.waiting_request_ids = deque([10])
    engine.get_request = mock.Mock(return_value=request)
    engine.context.check_availability = mock.Mock(return_value=(True, True, True))
    engine.context.add_request = mock.Mock()
    engine.context.enable_prefix_caching = False
    engine._cg_admission_gating_active = mock.Mock(return_value=False)
    engine._should_defer_async_sched_admission = mock.Mock(return_value=True)

    assert engine.schedule_non_chunked_prefill() is True
    assert list(engine.waiting_request_ids) == [10]
    engine.context.add_request.assert_not_called()


def test_async_forward_drains_then_admits_and_primes():
    """Drain pending logits, admit the queued request, and launch its primer."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.state = EngineState.RUNNING
    engine.logging_step_interval = 0
    engine.metrics_writer = None
    engine.schedule_waiting_requests = mock.Mock(side_effect=[True, False])
    engine.context = SimpleNamespace(
        step_count=4,
        prefix_cache_lru_clock=7,
        active_token_count=2,
        is_decode_only=mock.Mock(return_value=True),
    )
    expected_output = {"sample": "tokens"}
    engine.controller = SimpleNamespace(
        async_generate_output_tokens_dynamic_batch=mock.AsyncMock(
            side_effect=[
                DynamicBatchControllerStepResult(output=expected_output),
                DynamicBatchControllerStepResult(primer_only=True),
            ]
        )
    )

    output, _, _ = asyncio.run(engine.async_forward())

    assert output is expected_output
    assert engine.schedule_waiting_requests.call_count == 2
    engine.controller.async_generate_output_tokens_dynamic_batch.assert_has_awaits(
        [mock.call(drain_pending_forward=True), mock.call()]
    )


@pytest.mark.parametrize("consensus_due", [False, True])
def test_advance_async_sched_primer_uses_existing_ep_consensus_cadence(consensus_due):
    """Primer continuation adds no EP collective unless normal consensus is already due."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.use_coordinator = True
    engine.ep_world_size = 2
    engine.disable_ep_consensus = False
    engine.ep_consensus_interval = 20
    engine._ep_consensus_loop_counter = 20 if consensus_due else 1
    engine._last_ep_consensus = (3, False)
    engine.waiting_request_ids = deque([20])
    engine.context = SimpleNamespace(get_active_request_count=mock.Mock(return_value=2))
    engine._ep_establish_consensus = mock.AsyncMock(return_value=(3, False))

    asyncio.run(engine._advance_async_sched_primer())

    assert engine._ep_consensus_loop_counter == (21 if consensus_due else 2)
    if consensus_due:
        engine._ep_establish_consensus.assert_awaited_once_with(3, signal_consensus=False)
    else:
        engine._ep_establish_consensus.assert_not_awaited()


@pytest.mark.parametrize(
    "use_coordinator, ep_world_size, disable_ep_consensus, expected_yield",
    [(False, 2, False, False), (True, 1, False, False), (True, 2, True, True)],
)
def test_advance_async_sched_primer_skips_unneeded_ep_cadence(
    use_coordinator, ep_world_size, disable_ep_consensus, expected_yield
):
    """Primer continuation changes cadence only for coordinated EP execution."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.use_coordinator = use_coordinator
    engine.ep_world_size = ep_world_size
    engine.disable_ep_consensus = disable_ep_consensus
    engine._ep_consensus_loop_counter = 3
    engine._ep_establish_consensus = mock.AsyncMock()

    with mock.patch(
        "megatron.core.inference.engines.dynamic_engine.asyncio.sleep", new=mock.AsyncMock()
    ) as sleep:
        asyncio.run(engine._advance_async_sched_primer())

    assert engine._ep_consensus_loop_counter == 3
    engine._ep_establish_consensus.assert_not_awaited()
    if expected_yield:
        sleep.assert_awaited_once_with(0)
    else:
        sleep.assert_not_awaited()
