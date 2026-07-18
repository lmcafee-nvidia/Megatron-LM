# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.text_generation_controllers.async_schedule_step import (
    AsyncScheduleResolveResult,
    AsyncScheduleStep,
    StepResult,
    StepSpecialization,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    AsyncScheduleLogitsState,
)


def _make_step(*, prefill=False, pending=True, speculative_tokens=0, survivor_idxs=(0, 2)):
    calls = []
    sample = torch.tensor([1, 2, 3], dtype=torch.int64)
    survivors = torch.tensor(survivor_idxs, dtype=torch.int64)
    context = SimpleNamespace(
        num_prefill_requests=int(prefill),
        total_request_count=3,
        paused_request_count=0,
        active_token_count=3,
        async_sched_step_count=0,
        async_sched_compaction_step_count=0,
        prepare_requests=mock.Mock(side_effect=lambda: calls.append("prepare")),
        commit_sampled_tokens=mock.Mock(side_effect=lambda _: calls.append("commit")),
        copy_async_sched_sample_to_forward=mock.Mock(
            side_effect=lambda _: calls.append("copy_forward")
        ),
    )
    resolve_result = AsyncScheduleResolveResult(
        sampled_tokens_cpu=sample,
        active_request_ids=torch.tensor([10, 11, 12]),
        finished_request_ids=torch.tensor([11]) if len(survivor_idxs) < 3 else torch.empty(0),
        survivor_idxs=survivors,
        compaction_done_event="compaction",
    )
    logits = AsyncScheduleLogitsState(
        is_valid=pending,
        cuda_graph_request_count=7 if pending else None,
        ready_event="current" if pending else None,
    )
    controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(inference_context=context),
        num_speculative_tokens=speculative_tokens,
        _async_sched_logits=logits,
        _validate_async_sched_support_for_step=mock.Mock(
            side_effect=lambda: calls.append("validate")
        ),
        _synchronize_async_sched_event=mock.Mock(
            side_effect=lambda event: calls.append(f"wait:{event}")
        ),
        _run_async_sched_prepare=mock.Mock(
            side_effect=lambda: calls.append("prepare") or ("input_ids", "position_ids")
        ),
        _run_async_sched_sample=mock.Mock(side_effect=lambda: calls.append("sample") or sample),
        _copy_async_sched_sample_to_cpu=mock.Mock(
            side_effect=lambda _: calls.append("copy_cpu") or (sample, "sample")
        ),
        _run_async_sched_publish_bookkeeping=mock.Mock(
            side_effect=lambda: calls.append("publish") or "bookkeeping"
        ),
        _run_async_sched_forward=mock.Mock(
            side_effect=lambda *_: calls.append("forward") or "forward"
        ),
        _run_async_sched_resolve=mock.Mock(
            side_effect=lambda *_: calls.append("resolve")
            or replace(
                resolve_result, compaction_done_event="compaction" if logits.is_valid else None
            )
        ),
        _run_dummy_base_forward=mock.Mock(side_effect=lambda: calls.append("dummy_forward")),
        _reset_dummy_context=mock.Mock(side_effect=lambda: calls.append("dummy_reset")),
    )

    def run_primer():
        calls.append("forward")
        logits.set_pending(7, "forward")
        return True, "bookkeeping"

    controller._run_async_sched_forward_primer = mock.Mock(side_effect=run_primer)
    return AsyncScheduleStep(controller), context, calls, resolve_result


@pytest.mark.parametrize(
    "overlap, drain_pending_forward, expected_calls",
    [
        (
            False,
            False,
            [
                "validate",
                "wait:current",
                "prepare",
                "sample",
                "copy_forward",
                "copy_cpu",
                "wait:sample",
                "publish",
                "wait:bookkeeping",
                "forward",
                "wait:forward",
                "commit",
                "resolve",
                "wait:compaction",
            ],
        ),
        (
            True,
            False,
            [
                "validate",
                "prepare",
                "sample",
                "copy_forward",
                "copy_cpu",
                "publish",
                "forward",
                "wait:sample",
                "wait:bookkeeping",
                "commit",
                "resolve",
            ],
        ),
        (
            False,
            True,
            [
                "validate",
                "wait:current",
                "prepare",
                "sample",
                "copy_cpu",
                "wait:sample",
                "commit",
                "resolve",
            ],
        ),
        (
            True,
            True,
            ["validate", "prepare", "sample", "copy_cpu", "wait:sample", "commit", "resolve"],
        ),
    ],
)
def test_decode_full_order(overlap, drain_pending_forward, expected_calls):
    step, context, calls, _ = _make_step()

    result = asyncio.run(
        step.run(overlap=overlap, has_local_work=True, drain_pending_forward=drain_pending_forward)
    )

    assert result.specialization is StepSpecialization.FULL
    assert result.output["cuda_graph_request_count"] == 7
    assert calls == expected_calls
    assert context.async_sched_step_count == 1
    assert context.async_sched_compaction_step_count == (0 if drain_pending_forward else 1)


@pytest.mark.parametrize(
    "drain_pending_forward, survivor_idxs, expected_calls",
    [
        (
            False,
            (0, 2),
            [
                "validate",
                "sample",
                "copy_cpu",
                "wait:sample",
                "resolve",
                "prepare",
                "commit",
                "copy_forward",
                "publish",
                "wait:bookkeeping",
                "forward",
                "wait:forward",
            ],
        ),
        (
            True,
            (0, 2),
            ["validate", "sample", "copy_cpu", "wait:sample", "resolve", "prepare", "commit"],
        ),
        (False, (), ["validate", "sample", "copy_cpu", "wait:sample", "resolve"]),
    ],
)
def test_prefill_full_order(drain_pending_forward, survivor_idxs, expected_calls):
    step, context, calls, _ = _make_step(prefill=True, survivor_idxs=survivor_idxs)

    result = asyncio.run(
        step.run(overlap=False, has_local_work=True, drain_pending_forward=drain_pending_forward)
    )

    assert result.specialization is StepSpecialization.FULL
    assert calls == expected_calls
    assert context.async_sched_step_count == 1
    assert context.async_sched_compaction_step_count == 0


@pytest.mark.parametrize(
    "prefill, overlap, expected_wait", [(False, False, "forward"), (True, True, "bookkeeping")]
)
def test_primer_runs_only_forward(prefill, overlap, expected_wait):
    step, context, calls, _ = _make_step(prefill=prefill, pending=False)

    result = asyncio.run(step.run(overlap=overlap, has_local_work=True))

    assert result == StepResult(specialization=StepSpecialization.PRIMER)
    assert result.primer_only
    assert calls == ["validate", "forward", f"wait:{expected_wait}"]
    assert context.async_sched_step_count == 0


@pytest.mark.parametrize("prefill", [False, True])
def test_dummy_runs_only_base_forward_and_cleanup(prefill):
    step, context, calls, _ = _make_step(prefill=prefill)

    result = asyncio.run(step.run(overlap=True, has_local_work=False))

    assert result == StepResult(specialization=StepSpecialization.DUMMY)
    assert calls == ["validate", "dummy_forward", "dummy_reset"]
    assert context.async_sched_step_count == 0
    assert not step.controller._async_sched_logits.is_valid


def test_empty_local_step_clears_pending_logits():
    step, context, calls, _ = _make_step()
    context.total_request_count = 0
    context.active_token_count = 0

    result = asyncio.run(step.run(overlap=True, has_local_work=True))

    assert result == StepResult()
    assert calls == ["validate"]
    assert not step.controller._async_sched_logits.is_valid


def test_mtp_order_has_explicit_todo():
    step, _, calls, _ = _make_step(speculative_tokens=1)

    with pytest.raises(NotImplementedError, match="TODO: Implement async scheduling with MTP"):
        asyncio.run(step.run(overlap=True, has_local_work=True))

    assert calls == ["validate"]
