# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Orchestration for dynamic-inference async-scheduling steps."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import Tensor
from torch.cuda.nvtx import range_pop, range_push

if TYPE_CHECKING:
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )


class StepOrder(Enum):
    """Phase order for an async-scheduling step."""

    PREFILL = auto()
    DECODE = auto()
    DECODE_MTP = auto()


class StepSpecialization(Enum):
    """Work performed by an async-scheduling step."""

    FULL = auto()
    PRIMER = auto()
    DUMMY = auto()


@dataclass(frozen=True)
class StepResult:
    """Result of one dynamic-batching controller step.

    Attributes:
        output: Sampled-step output, or ``None`` when no output was produced.
        specialization: Work performed by the step.
    """

    output: Optional[Dict] = None
    specialization: StepSpecialization = StepSpecialization.FULL

    @property
    def primer_only(self) -> bool:
        """Return whether the step launched only a forward primer.

        Returns:
            bool: Whether this is a primer-only step.
        """
        return self.specialization is StepSpecialization.PRIMER


@dataclass(frozen=True)
class AsyncScheduleResolveResult:
    """State produced by async-scheduling request resolution.

    Attributes:
        sampled_tokens_cpu: Sampled CPU token IDs.
        active_request_ids: Request IDs active before resolution.
        finished_request_ids: Request IDs finished by resolution.
        survivor_idxs: Source rows for surviving requests in destination order.
        compaction_done_event: Event marking logits-compaction completion.
    """

    sampled_tokens_cpu: Tensor
    active_request_ids: Tensor
    finished_request_ids: Tensor
    survivor_idxs: Tensor
    compaction_done_event: Optional[torch.cuda.Event]


class AsyncScheduleStep:
    """Orchestrate one async-scheduling controller invocation.

    Order and specialization are independent. Supported full steps use these
    phase orders::

        PREFILL: sample -> resolve -> prepare -> forward
        DECODE:  prepare -> sample -> forward -> resolve

    ``DECODE_MTP`` reserves the planned sample/verify -> rewind -> prepare ->
    forward -> resolve order, but currently raises an explicit TODO. Primer
    and dummy specializations use the selected supported order but no-op every
    phase except forward. A controller invocation launches at most one model
    forward.

    Args:
        controller: Controller that owns phase implementations and CUDA state.
    """

    def __init__(self, controller: TextGenerationController):
        """Initialize orchestration around one controller.

        Args:
            controller: Controller that owns phase implementations and CUDA state.
        """
        self.controller = controller

    def _select_order(self) -> StepOrder:
        """Select the phase order from current request state.

        Returns:
            StepOrder: Order for the current controller invocation.
        """
        context = self.controller.inference_wrapped_model.inference_context
        if self.controller.num_speculative_tokens > 0:
            return StepOrder.DECODE_MTP
        if context.num_prefill_requests != 0:
            return StepOrder.PREFILL
        return StepOrder.DECODE

    def _select_specialization(self, has_local_work: bool) -> StepSpecialization:
        """Select the work specialization for the current rank.

        Args:
            has_local_work: Whether this rank has requests to process.

        Returns:
            StepSpecialization: Work performed by this invocation.
        """
        if not has_local_work:
            return StepSpecialization.DUMMY
        if not self.controller._async_sched_logits.is_valid:
            return StepSpecialization.PRIMER
        return StepSpecialization.FULL

    @staticmethod
    def _build_output(
        resolve_result: AsyncScheduleResolveResult, cuda_graph_request_count: Optional[int]
    ) -> Dict:
        """Build the request output returned to engine bookkeeping.

        Args:
            resolve_result: State produced by request resolution.
            cuda_graph_request_count: CUDA graph request count for consumed logits.

        Returns:
            Dict: Dynamic-batching request output.
        """
        return {
            "active_request_ids": resolve_result.active_request_ids,
            "finished_request_ids": resolve_result.finished_request_ids,
            "sample": resolve_result.sampled_tokens_cpu,
            "finished_routing_block_ids": {},
            "newly_paused_request_ids": None,
            "evict_request_ids": None,
            "accepted_tokens": None,
            "log_probs": None,
            "top_n_logprobs": None,
            "cuda_graph_request_count": cuda_graph_request_count,
        }

    def _run_primer(self, *, overlap: bool) -> StepResult:
        """Launch one forward primer and wait only at the required boundary.

        Args:
            overlap: Whether GPU work may overlap CPU work.

        Returns:
            StepResult: Primer-only result.
        """
        controller = self.controller
        primer_launched, bookkeeping_done_event = controller._run_async_sched_forward_primer()
        assert primer_launched, "Primer specialization requires missing async logits."

        if overlap:
            controller._synchronize_async_sched_event(bookkeeping_done_event)
        else:
            controller._synchronize_async_sched_event(controller._async_sched_logits.ready_event)

        return StepResult(specialization=StepSpecialization.PRIMER)

    def _run_dummy(self) -> StepResult:
        """Run the async-scheduling dummy forward and clear temporary state.

        Returns:
            StepResult: Dummy-step result.
        """
        self.controller._async_sched_logits.clear()
        self.controller._run_dummy_base_forward()
        self.controller._reset_dummy_context()
        return StepResult(specialization=StepSpecialization.DUMMY)

    async def run_prefill(
        self, specialization: StepSpecialization, *, overlap: bool, drain_pending_forward: bool
    ) -> StepResult:
        """Run the prefill phase order for one specialization.

        Args:
            specialization: Work performed by this invocation.
            overlap: Whether GPU work may overlap CPU work.
            drain_pending_forward: Whether to consume logits without a successor forward.

        Returns:
            StepResult: Result of the prefill invocation.
        """
        if specialization is StepSpecialization.PRIMER:
            return self._run_primer(overlap=overlap)
        if specialization is StepSpecialization.DUMMY:
            return self._run_dummy()

        controller = self.controller
        context = controller.inference_wrapped_model.inference_context
        with torch.inference_mode():
            cuda_graph_request_count = controller._async_sched_logits.cuda_graph_request_count

            # Sample the completed prefill or mixed forward.
            sampled_tokens_gpu = controller._run_async_sched_sample()
            sampled_tokens_cpu_view, sample_cpu_ready_event = (
                controller._copy_async_sched_sample_to_cpu(sampled_tokens_gpu)
            )
            controller._synchronize_async_sched_event(sample_cpu_ready_event)

            # Resolve before replacing prompt rows with decode rows.
            controller._async_sched_logits.clear()
            resolve_result = controller._run_async_sched_resolve(
                sampled_tokens_cpu_view, None, overlap
            )

            if resolve_result.survivor_idxs.numel() > 0:
                # Prepare one decode token for each surviving request.
                if drain_pending_forward:
                    context.prepare_requests()
                    input_ids_gpu_view = position_ids_gpu_view = None
                else:
                    input_ids_gpu_view, position_ids_gpu_view = (
                        controller._run_async_sched_prepare()
                    )

                sampled_tokens_cpu = resolve_result.sampled_tokens_cpu[resolve_result.survivor_idxs]
                context.commit_sampled_tokens(sampled_tokens_cpu)

                if not drain_pending_forward:
                    # Populate the successor input and publish its metadata.
                    survivor_idxs_gpu = resolve_result.survivor_idxs.to(sampled_tokens_gpu.device)
                    context.copy_async_sched_sample_to_forward(
                        sampled_tokens_gpu[survivor_idxs_gpu]
                    )
                    bookkeeping_done_event = controller._run_async_sched_publish_bookkeeping()
                    if not overlap:
                        controller._synchronize_async_sched_event(bookkeeping_done_event)

                    # Launch the successor decode forward.
                    forward_done_event = controller._run_async_sched_forward(
                        input_ids_gpu_view, position_ids_gpu_view
                    )
                    if not overlap:
                        controller._synchronize_async_sched_event(forward_done_event)

            context.async_sched_step_count += 1
            output = self._build_output(resolve_result, cuda_graph_request_count)

        await asyncio.sleep(0)
        return StepResult(output=output)

    async def run_decode(
        self, specialization: StepSpecialization, *, overlap: bool, drain_pending_forward: bool
    ) -> StepResult:
        """Run the decode phase order for one specialization.

        Args:
            specialization: Work performed by this invocation.
            overlap: Whether GPU work may overlap CPU work.
            drain_pending_forward: Whether to consume logits without a successor forward.

        Returns:
            StepResult: Result of the decode invocation.
        """
        if specialization is StepSpecialization.PRIMER:
            return self._run_primer(overlap=overlap)
        if specialization is StepSpecialization.DUMMY:
            return self._run_dummy()

        controller = self.controller
        context = controller.inference_wrapped_model.inference_context
        with torch.inference_mode():
            current_logits_ready_event = controller._async_sched_logits.ready_event
            cuda_graph_request_count = controller._async_sched_logits.cuda_graph_request_count

            if not overlap:
                controller._synchronize_async_sched_event(current_logits_ready_event)

            # Prepare CPU state and stable GPU views for the successor forward.
            range_push("prepare_requests")
            input_ids_gpu_view, position_ids_gpu_view = controller._run_async_sched_prepare()
            range_pop()

            # Sample pending logits and start transferring the sample to CPU.
            sampled_tokens_gpu = controller._run_async_sched_sample()
            if not drain_pending_forward:
                context.copy_async_sched_sample_to_forward(sampled_tokens_gpu)
            sampled_tokens_cpu_view, sample_cpu_ready_event = (
                controller._copy_async_sched_sample_to_cpu(sampled_tokens_gpu)
            )
            if not overlap:
                controller._synchronize_async_sched_event(sample_cpu_ready_event)

            bookkeeping_done_event = None
            forward_done_event = None
            if drain_pending_forward:
                controller._async_sched_logits.clear()
            else:
                # Publish prepared metadata before launching the successor forward.
                range_push("async_sched_transfer_bookkeeping_to_gpu")
                bookkeeping_done_event = controller._run_async_sched_publish_bookkeeping()
                range_pop()
                if not overlap:
                    controller._synchronize_async_sched_event(bookkeeping_done_event)

                range_push("async_sched_forward_pass")
                forward_done_event = controller._run_async_sched_forward(
                    input_ids_gpu_view, position_ids_gpu_view
                )
                range_pop()
                if not overlap:
                    controller._synchronize_async_sched_event(forward_done_event)

            # Resolve only after CPU samples and reusable H2D sources are available.
            if overlap:
                controller._synchronize_async_sched_event(sample_cpu_ready_event)
                if bookkeeping_done_event is not None:
                    controller._synchronize_async_sched_event(bookkeeping_done_event)
            context.commit_sampled_tokens(sampled_tokens_cpu_view)
            resolve_result = controller._run_async_sched_resolve(
                sampled_tokens_cpu_view, forward_done_event, overlap
            )
            if not overlap and resolve_result.compaction_done_event is not None:
                controller._synchronize_async_sched_event(resolve_result.compaction_done_event)

            context.async_sched_step_count += 1
            if not drain_pending_forward and resolve_result.finished_request_ids.numel() > 0:
                context.async_sched_compaction_step_count += 1
            output = self._build_output(resolve_result, cuda_graph_request_count)

        await asyncio.sleep(0)
        return StepResult(output=output)

    async def run_decode_mtp(
        self, specialization: StepSpecialization, *, overlap: bool, drain_pending_forward: bool
    ) -> StepResult:
        """Reject the not-yet-implemented async MTP phase order.

        Args:
            specialization: Work performed by this invocation.
            overlap: Whether GPU work may overlap CPU work.
            drain_pending_forward: Whether to consume logits without a successor forward.

        Raises:
            NotImplementedError: Async scheduling with MTP is not implemented.
        """
        del specialization, overlap, drain_pending_forward
        raise NotImplementedError("TODO: Implement async scheduling with MTP.")

    async def run(
        self, *, overlap: bool, has_local_work: bool, drain_pending_forward: bool = False
    ) -> StepResult:
        """Select and execute one async-scheduling step.

        Args:
            overlap: Whether GPU work may overlap CPU work.
            has_local_work: Whether this rank has requests to process.
            drain_pending_forward: Whether to consume logits without a successor forward.

        Returns:
            StepResult: Result and specialization for this invocation.
        """
        controller = self.controller
        context = controller.inference_wrapped_model.inference_context
        controller._validate_async_sched_support_for_step()

        active_request_count = context.total_request_count - context.paused_request_count
        if has_local_work and context.active_token_count == 0 and active_request_count == 0:
            controller._async_sched_logits.clear()
            return StepResult()

        assert not (
            drain_pending_forward and not controller._async_sched_logits.is_valid
        ), "Async admission drain requires pending logits."
        assert not (
            drain_pending_forward and not has_local_work
        ), "Async admission drain requires local work."

        order = self._select_order()
        specialization = self._select_specialization(has_local_work)
        if order is StepOrder.PREFILL:
            return await self.run_prefill(
                specialization, overlap=overlap, drain_pending_forward=drain_pending_forward
            )
        if order is StepOrder.DECODE:
            return await self.run_decode(
                specialization, overlap=overlap, drain_pending_forward=drain_pending_forward
            )
        if order is StepOrder.DECODE_MTP:
            return await self.run_decode_mtp(
                specialization, overlap=overlap, drain_pending_forward=drain_pending_forward
            )
        raise AssertionError(f"Unexpected async-scheduling step order: {order}")
