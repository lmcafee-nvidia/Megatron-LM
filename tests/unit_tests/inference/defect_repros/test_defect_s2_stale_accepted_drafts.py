# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected defect s2: under the default ASYNC schedule, a newly admitted request is
handed another request's stale accepted MTP drafts as its first output tokens.

Trace (read from code at ed9c2c8c1):

* ``_prepare_next_forward_pass_kernel`` writes ``accepted_tokens_per_request[pid]`` and
  ``accepted_counts[pid]`` only for ``pid < num_decode_requests``
  (``megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:291``). A prefill row
  keeps whatever the persistent ``[max_requests, S]`` buffer already held.
* The buffer is allocated once, filled with -1, in ``_init_mtp_sampling_tensors``
  (``mtp_controller_mixin.py:87-92``). The only reset is at the end of every LEGACY step
  (``text_generation_controller.py:3571-3573``).
* The ASYNC verify path ``_run_async_sched_sample_mtp`` exposes
  ``_accepted_tokens_per_request[:active_request_count]``, prefill rows included, whenever
  ``num_decode_requests > 0`` (``text_generation_controller.py:2424-2428``). It copies that slice to
  the host and hands it to the engine as the step's ``accepted_tokens`` (``:2939-2942``,
  ``:3012-3016``, ``:3111``), and to the finalize callback (``:2898-2910``). Nothing resets the
  buffer on this path.
* ``DynamicInferenceEngine._process_request_outputs`` prepends every non-(-1) entry of each row to
  that request's sampled token, with no prefill check (``dynamic_engine.py:2837-2858``).
* ASYNC is the default (``megatron/core/inference/config.py:807``).

Trigger: with MTP on, a request is admitted into a row that an earlier decode request used, and the
step that verifies its prefill still has a decode request. This is a decode-to-prefill row
transition. A full batch with a waiting queue does this every time a request finishes and a waiting
request takes its slot.

Harness:

* A real ``DynamicInferenceEngine`` + ``TextGenerationController`` + small ``GPTModel`` with MTP
  layers, built by ``DynamicInferenceEngineTestBase._build_test_env``
  (``tests/unit_tests/inference/engines/test_dynamic_engine.py``). The fixtures are those of
  ``test_speculative_mixed_prefill_decode_heterogeneous_acceptance``.
* "Counter model". As in that test, the real base forward and the real MTP step run, and their
  logits are then overwritten deterministically: the base logits at a token ``x`` select
  ``(x + 1) % 99``, and the MTP logits for ``next_token_ids = x`` also select ``(x + 1) % 99``. With
  greedy sampling (``top_k=1``), every draft is accepted. So a request whose prompt ends in ``p``
  must generate ``p+1, p+2, ...``. That depends only on its own prompt: the request's output when
  generated alone, with or without speculation. Tokens never reach the termination id 99.
* Requests: A (prompt ends 10, generates ``S + 2`` tokens, so it finishes after one decode step),
  B (ends 40, 16 tokens) and C (ends 70, 7 tokens). With ``max_requests = 2``, C waits in the queue
  until A finishes, and is then admitted into a row that held a decode request's accepted drafts.
* Two read-only spies record what happened, but every outcome assertion is on the generated tokens.
  One wraps ``_prepare_speculative_tokens_for_next_forward_pass`` to log the row layout, the decode
  count and the buffer after each verify. The other wraps ``post_process_requests`` to log the
  ``accepted_tokens`` rows the engine receives. Before any outcome check, the spy logs must show that
  the trigger was reached; otherwise the test fails with ``HARNESS s2:``.

Requirements: 1 GPU, Transformer Engine (imported by the engine-test module), flash-attn >= 2.7.3
(the dynamic-batching attention path, as for the existing speculative engine tests).
Launch: ``python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest <this file> -v``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
import torch

# Max GPUs any test here needs. Every test is single-rank.
REQUIRED_GPUS = 1

VOCAB_SIZE = 100  # DynamicEngineTestConfig.vocab_size
TERMINATION_ID = VOCAB_SIZE - 1
CYCLE = VOCAB_SIZE - 1  # generated ids stay in [0, 99), so the termination id is never produced
HIGH_LOGIT = 100.0
PROMPT_LEN = 4
MAX_TOKENS_TO_GENERATE = 16  # sizes max_sequence_length; per-request lengths are below this
MAX_STEPS = 200

REQ_A, REQ_B, REQ_C = 0, 1, 2
REQ_NAMES = {REQ_A: "A", REQ_B: "B", REQ_C: "C"}
PROMPT_LAST = {REQ_A: 10, REQ_B: 40, REQ_C: 70}


def _have_cuda() -> bool:
    return torch.cuda.is_available()


def _have_te() -> bool:
    try:
        import transformer_engine  # noqa: F401

        return True
    except Exception:
        return False


def _have_flash_attn() -> bool:
    try:
        from megatron.core.utils import is_fa_min_version

        return bool(is_fa_min_version("2.7.3"))
    except Exception:
        return False


# Every test here drives the real engine, so the requirements apply module-wide.
pytestmark = [
    pytest.mark.skipif(not _have_cuda(), reason="needs a CUDA GPU (real dynamic inference engine)"),
    pytest.mark.skipif(
        not _have_te(),
        reason="tests/unit_tests/inference/engines/test_dynamic_engine.py imports Transformer Engine",
    ),
    pytest.mark.skipif(
        not _have_flash_attn(), reason="dynamic batching attention needs flash-attn >= 2.7.3"
    ),
]


@pytest.fixture
def model_parallel():
    """Single-rank model-parallel state, as in TestDynamicInferenceEngine.setup/teardown_class."""
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    try:
        yield
    finally:
        from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
        from tests.unit_tests.inference.engines.test_dynamic_engine import reset_rounder

        delete_cuda_graphs()
        reset_rounder()
        Utils.destroy_model_parallel()


# --------------------------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------------------------- #


def _prompt(request_id: int) -> List[int]:
    last = PROMPT_LAST[request_id]
    return list(range(last - PROMPT_LEN + 1, last + 1))


def _reference_tokens(request_id: int, num_tokens: int) -> List[int]:
    """Greedy output of the counter model for this request alone: p+1, p+2, ... (mod 99)."""
    last = PROMPT_LAST[request_id]
    return [(last + k) % CYCLE for k in range(1, num_tokens + 1)]


# --------------------------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------------------------- #


@dataclass
class _VerifyRecord:
    """State right after one `_prepare_speculative_tokens_for_next_forward_pass` call."""

    num_decode: int
    row_request_ids: List[int]
    accepted_rows_gpu: Optional[torch.Tensor]  # snapshot of _accepted_tokens_per_request[:active]
    accepted_rows: Optional[List[List[int]]] = None


@dataclass
class _EngineRecord:
    """Arguments of one `post_process_requests` call."""

    request_ids: List[int]
    accepted_rows: Optional[List[List[int]]]


@dataclass
class _RunResult:
    mode_name: str
    num_spec: int
    num_tokens: Dict[int, int]
    outputs: Dict[int, List[int]]
    statuses: Dict[int, object]
    verify_records: List[_VerifyRecord]
    engine_records: List[_EngineRecord]


@dataclass
class _Admission:
    """C's prefill verify: where C sat and what its row held."""

    row: int
    num_decode: int
    row_request_ids: List[int]
    row_was_decode_before: bool
    previous_decode_owner: Optional[int]
    buffer_row: List[int]


def _install_counter_model(model) -> None:
    """Run the real forward / MTP step, then make the logits a deterministic +1 counter."""
    real_forward = model.forward
    real_mtp = model.compute_mtp_single_step

    def counter_forward(*args, **kwargs):
        logits = real_forward(*args, **kwargs)
        input_ids = args[0] if args else kwargs["input_ids"]
        assert logits.dim() == 3 and tuple(logits.shape[:2]) == tuple(input_ids.shape), (
            "HARNESS s2: expected full per-token logits [1, T, V] aligned with input_ids [1, T] "
            f"(materialize_only_last_token_logits=False); got logits {tuple(logits.shape)}, "
            f"input_ids {tuple(input_ids.shape)}"
        )
        with torch.inference_mode():
            nxt = torch.remainder(input_ids.to(torch.int64) + 1, CYCLE)
            logits.zero_()
            logits.scatter_(-1, nxt.unsqueeze(-1), HIGH_LOGIT)
        return logits

    def counter_mtp(*args, **kwargs):
        hidden, logits = real_mtp(*args, **kwargs)
        ids = kwargs["next_token_ids"] if "next_token_ids" in kwargs else args[1]
        n = logits.shape[0]
        assert logits.dim() == 3 and logits.shape[1] == 1 and ids.numel() >= n, (
            "HARNESS s2: expected MTP logits [N, 1, V] and next_token_ids with >= N entries; got "
            f"logits {tuple(logits.shape)}, next_token_ids {tuple(ids.shape)}"
        )
        with torch.inference_mode():
            nxt = torch.remainder(ids.reshape(-1)[:n].to(torch.int64) + 1, CYCLE)
            logits.zero_()
            logits.scatter_(-1, nxt.view(n, 1, 1), HIGH_LOGIT)
        return hidden, logits

    model.forward = counter_forward
    model.compute_mtp_single_step = counter_mtp


def _install_spies(engine, verify_records, engine_records) -> None:
    """Read-only wrappers: call the original, then log. They do not change behavior."""
    controller = engine.controller
    context = controller.inference_wrapped_model.inference_context

    orig_prepare = controller._prepare_speculative_tokens_for_next_forward_pass

    def prepare_spy(*args, **kwargs):
        result = orig_prepare(*args, **kwargs)
        num_decode = kwargs["num_decode_requests"] if "num_decode_requests" in kwargs else args[0]
        last_one = kwargs["last_one_indices"] if "last_one_indices" in kwargs else args[3]
        active = int(last_one.shape[0])
        start = context.paused_request_count
        rows = [int(r) for r in context.request_ids[start : start + active].tolist()]
        verify_records.append(
            _VerifyRecord(
                num_decode=int(num_decode),
                row_request_ids=rows,
                accepted_rows_gpu=controller._accepted_tokens_per_request[:active].clone(),
            )
        )
        return result

    controller._prepare_speculative_tokens_for_next_forward_pass = prepare_spy

    orig_post = engine.post_process_requests

    def post_spy(*args, **kwargs):
        request_ids = kwargs["request_ids"] if "request_ids" in kwargs else args[0]
        accepted = kwargs["accepted_tokens"] if "accepted_tokens" in kwargs else args[5]
        ids = request_ids.tolist() if torch.is_tensor(request_ids) else list(request_ids)
        engine_records.append(
            _EngineRecord(
                request_ids=[int(r) for r in ids],
                accepted_rows=None if accepted is None else accepted.tolist(),
            )
        )
        return orig_post(*args, **kwargs)

    engine.post_process_requests = post_spy


def _run_scenario(
    mode_name: str,
    num_spec: int,
    *,
    max_requests: int,
    num_tokens: Dict[int, int],
    add_c_after_ab_decode: bool,
) -> _RunResult:
    from megatron.core.inference.config import AsyncScheduleMode
    from megatron.core.inference.sampling_params import SamplingParams
    from tests.unit_tests.inference.engines.test_dynamic_engine import (
        DynamicEngineTestConfig,
        DynamicInferenceEngineTestBase,
    )

    mode = AsyncScheduleMode(mode_name)
    test_config = DynamicEngineTestConfig(
        num_requests=0,
        min_prompt_length=PROMPT_LEN,
        max_prompt_length=PROMPT_LEN,
        num_tokens_to_generate=MAX_TOKENS_TO_GENERATE,
        num_speculative_tokens=num_spec,
        materialize_only_last_token_logits=False,
        model_provider="gpt",
        context_max_requests=max_requests,
        async_sched_mode=mode,
    )
    assert test_config.vocab_size == VOCAB_SIZE
    env = DynamicInferenceEngineTestBase._build_test_env(test_config)
    engine = env.engine
    assert engine.context.config.async_sched_mode == mode
    assert engine.context.max_requests == max_requests

    _install_counter_model(engine.controller.inference_wrapped_model.model)
    verify_records: List[_VerifyRecord] = []
    engine_records: List[_EngineRecord] = []
    _install_spies(engine, verify_records, engine_records)

    finished = {}

    def add(request_id):
        engine.add_request(
            request_id=request_id,
            prompt=torch.tensor(_prompt(request_id), dtype=torch.int64, device="cuda"),
            sampling_params=SamplingParams(
                num_tokens_to_generate=num_tokens[request_id],
                termination_id=TERMINATION_ID,
                top_k=1,
            ),
        )

    def step():
        result = engine.step_modern()
        for request in result["finished_requests"]:
            finished[request.request_id] = request

    with torch.inference_mode():
        add(REQ_A)
        add(REQ_B)
        if add_c_after_ab_decode:
            # Wait until A and B have each produced their prefill token (they are decode rows now),
            # then add C so that it is admitted next to them into a never-used row.
            for _ in range(MAX_STEPS):
                if all(
                    rid in engine.requests and len(engine.get_request(rid).generated_tokens) >= 1
                    for rid in (REQ_A, REQ_B)
                ):
                    break
                step()
            else:
                pytest.fail("HARNESS s2: requests A and B never left prefill")
        add(REQ_C)

        steps = 0
        while engine.has_unfinished_requests():
            step()
            steps += 1
            if steps > MAX_STEPS:
                pytest.fail(f"HARNESS s2: engine did not finish within {MAX_STEPS} steps")

    for record in verify_records:
        record.accepted_rows = record.accepted_rows_gpu.tolist()
        record.accepted_rows_gpu = None

    return _RunResult(
        mode_name=mode_name,
        num_spec=num_spec,
        num_tokens=dict(num_tokens),
        outputs={rid: [int(t) for t in req.generated_tokens] for rid, req in finished.items()},
        statuses={rid: req.status for rid, req in finished.items()},
        verify_records=verify_records,
        engine_records=engine_records,
    )


def _backlog_num_tokens(num_spec: int) -> Dict[int, int]:
    # A: prefill token + one fully accepted decode step (S drafts + 1), so it finishes after the
    # first two-decode-row verify. B outlives C's prefill. C is short.
    return {REQ_A: num_spec + 2, REQ_B: MAX_TOKENS_TO_GENERATE, REQ_C: 7}


def _run_backlog(mode_name: str, num_spec: int) -> _RunResult:
    """A, B and C arrive together; max_requests=2, so C waits until A finishes and frees a row."""
    return _run_scenario(
        mode_name,
        num_spec,
        max_requests=2,
        num_tokens=_backlog_num_tokens(num_spec),
        add_c_after_ab_decode=False,
    )


def _locate_c_admission(result: _RunResult) -> Optional[_Admission]:
    """Find the verify of C's prefill and whether C's row had been a decode row before."""
    max_decode_before = 0
    owner_by_row: Dict[int, int] = {}
    for record in result.verify_records:
        if REQ_C in record.row_request_ids:
            row = record.row_request_ids.index(REQ_C)
            if row < record.num_decode:
                return None  # C was never seen as a prefill row
            return _Admission(
                row=row,
                num_decode=record.num_decode,
                row_request_ids=record.row_request_ids,
                row_was_decode_before=max_decode_before > row,
                previous_decode_owner=owner_by_row.get(row),
                buffer_row=record.accepted_rows[row],
            )
        max_decode_before = max(max_decode_before, record.num_decode)
        for r in range(record.num_decode):
            owner_by_row[r] = record.row_request_ids[r]
    return None


def _exposed_accepted_for_c(result: _RunResult) -> Optional[List[int]]:
    """The accepted-token row the engine received for C at C's first processed step."""
    for record in result.engine_records:
        if REQ_C in record.request_ids:
            if record.accepted_rows is None:
                return []
            row = record.accepted_rows[record.request_ids.index(REQ_C)]
            return [int(t) for t in row]
    return None


def _describe_verifies(result: _RunResult) -> str:
    lines = []
    for i, record in enumerate(result.verify_records):
        names = [REQ_NAMES.get(r, str(r)) for r in record.row_request_ids]
        lines.append(
            f"    verify {i}: rows={names} num_decode={record.num_decode} "
            f"accepted_buffer={record.accepted_rows}"
        )
    return "\n".join(lines)


def _require_trigger(result: _RunResult) -> _Admission:
    admission = _locate_c_admission(result)
    if admission is None or admission.num_decode == 0 or not admission.row_was_decode_before:
        pytest.fail(
            "HARNESS s2: the trigger was not reached. C must be verified as a prefill row, next to "
            "at least one decode row, in a row that an earlier verify used for a decode request. "
            f"Got admission={admission}.\n{_describe_verifies(result)}"
        )
    return admission


def _require_fresh_row(result: _RunResult) -> _Admission:
    admission = _locate_c_admission(result)
    if admission is None or admission.num_decode == 0 or admission.row_was_decode_before:
        pytest.fail(
            "HARNESS s2: the no-trigger control did not run as planned. C must be verified as a "
            "prefill row next to decode rows, in a row that no earlier verify used for decode. "
            f"Got admission={admission}.\n{_describe_verifies(result)}"
        )
    return admission


def _check_completed(result: _RunResult) -> None:
    for rid in (REQ_A, REQ_B, REQ_C):
        assert rid in result.outputs, (
            f"HARNESS s2: request {REQ_NAMES[rid]} did not finish "
            f"({result.mode_name}, S={result.num_spec}); finished: {sorted(result.outputs)}"
        )
        assert result.statuses[rid].name == "COMPLETED", (
            f"HARNESS s2: request {REQ_NAMES[rid]} ended with status {result.statuses[rid]} "
            f"({result.mode_name}, S={result.num_spec})"
        )


def _check_matches_reference(result: _RunResult, request_ids, label: str) -> None:
    for rid in request_ids:
        expected = _reference_tokens(rid, result.num_tokens[rid])
        got = result.outputs[rid]
        assert got == expected, (
            f"{label}: request {REQ_NAMES[rid]} ({result.mode_name}, S={result.num_spec}) "
            f"generated {got}, greedy counter-model reference {expected}.\n"
            f"{_describe_verifies(result)}"
        )


# --------------------------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_spec", [1, 2])
def test_control_legacy_backlog_admission_matches_reference(model_parallel, num_spec):
    """Same trigger as the defect test under LEGACY. It resets the buffer every step, so C is
    correct. This validates the counter-model harness, the reference and the trigger detection."""
    result = _run_backlog("legacy", num_spec)
    _check_completed(result)
    admission = _require_trigger(result)
    assert admission.buffer_row == [-1] * num_spec, (
        "HARNESS s2: LEGACY should have reset the buffer before C's prefill verify; "
        f"row {admission.row} held {admission.buffer_row}"
    )
    _check_matches_reference(result, (REQ_A, REQ_B, REQ_C), "CONTROL s2 (legacy)")


@pytest.mark.parametrize("num_spec", [1, 2])
def test_control_async_admission_into_fresh_row_matches_reference(model_parallel, num_spec):
    """ASYNC with a mixed prefill+decode MTP batch but without the trigger. C is admitted while A
    and B are both still decoding (max_requests=4), so it lands in a row no decode request used.
    The row still holds its -1 initial value, and C is correct."""
    result = _run_scenario(
        "async",
        num_spec,
        max_requests=4,
        num_tokens={REQ_A: MAX_TOKENS_TO_GENERATE, REQ_B: MAX_TOKENS_TO_GENERATE, REQ_C: 7},
        add_c_after_ab_decode=True,
    )
    _check_completed(result)
    _require_fresh_row(result)
    _check_matches_reference(result, (REQ_A, REQ_B, REQ_C), "CONTROL s2 (async, fresh row)")


# --------------------------------------------------------------------------------------------- #
# Defect
# --------------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_spec", [1, 2])
def test_defect_async_backlog_admission_prepends_stale_drafts(model_parallel, num_spec):
    """Under the default ASYNC schedule, C takes over A's freed slot and lands in a row whose
    accepted-draft entries were written when a decode request occupied it. Its output must still be
    its own greedy output."""
    result = _run_backlog("async", num_spec)
    _check_completed(result)
    admission = _require_trigger(result)
    # A and B are not involved in the transition. If they differ, something else is wrong.
    _check_matches_reference(result, (REQ_A, REQ_B), "HARNESS s2 (async, A/B)")

    expected_c = _reference_tokens(REQ_C, result.num_tokens[REQ_C])
    got_c = result.outputs[REQ_C]
    exposed = _exposed_accepted_for_c(result)
    owner = admission.previous_decode_owner
    assert got_c == expected_c, (
        f"DEFECT s2: AsyncScheduleMode.ASYNC (num_speculative_tokens={num_spec}): request C was "
        f"admitted into row {admission.row} in the step where decode rows shrank to "
        f"{admission.num_decode} (rows {[REQ_NAMES.get(r, r) for r in admission.row_request_ids]}). "
        f"It generated {got_c}, but its greedy output alone is {expected_c}. "
        f"Row {admission.row} last held accepted drafts of decode request "
        f"{REQ_NAMES.get(owner, owner)}. At C's prefill verify, "
        f"_accepted_tokens_per_request[{admission.row}] = {admission.buffer_row}: "
        "_prepare_next_forward_pass_kernel skips prefill rows (mtp_utils_triton.py:291) and the "
        "ASYNC path never resets the buffer. The step result gave the engine accepted_tokens "
        f"{exposed} for C (text_generation_controller.py:2424-2428), and "
        "_process_request_outputs prepended the non-(-1) entries to C's first token "
        "(dynamic_engine.py:2850-2858)."
    )
