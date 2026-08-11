# Request-lifecycle integration report

## Summary

Implemented bounded request/result lifecycle contracts and a 420-line unit
scenario suite. The suite documents the checkpoint field policy and compares
seeded baseline runs against bounded PERSIST, OFFLOAD, and RECOMPUTE
suspend/resume interventions. It checks terminal output, logprob/top-N
alignment, record/request serialization, futures, drainage, static pointers,
and CUDA prompt-storage retention.

## Confirmed fixes

- Checkpoint re-admission retains tokenized stop IDs and dynamic
  `SamplingParams.add_attributes()` state, deep-copies epoch history, retains
  request status, resets `num_tokens_total`, and selects the first populated
  TTFT during merge.
- Superseded checkpoint prompt views are archived off CUDA, preventing
  cumulative overlapping GPU prompt storage while preserving merge/wire data.
- Serialization restores a suppressed prompt tensor even when serialization
  raises.
- `add_request(..., sampling_params=None)` constructs default parameters, and
  a public duplicate live ID is rejected before mutation.
- Failed-only direct generation is treated as unfinished work so every failed
  input is collected and the engine drains.
- A drained coordinator reset preserves its asyncio loop, condition, state
  event identities, and coordinator mode.

## Test outcomes

- `test_dynamic_engine_request_lifecycle.py`: 13 passed.
- New lifecycle suite plus `test_inference_request.py` and
  `test_dynamic_events.py`: 35 passed.
- Focused dynamic-context pressure/all-evict/resume tests: 3 passed.
- Existing prerequisite tests: checkpoint-result reconstruction 1 passed;
  CUDA-graph OFFLOAD/RECOMPUTE lifecycle 2 passed; static UVM with managed
  allocation, hybrid Mamba state, static pointers, and graph residency 1
  passed; async pending-logit reset/suspend lifecycle 4 passed; direct
  RECOMPUTE prefix-cache re-admission 1 passed.
- Representative TP2/PP2 PERSIST graph-resume test completed successfully on
  four ranks.
- Formatting checks passed: Black, isort, pylint (10.00/10), Ruff, and both
  required diff checks. The formatter treats its unavailable optional `mypy`
  executable as non-fatal.

## Scope accounting and uncovered axes

- Net new `tests/unit_tests/` lines: 420 (limit: 2,000).
- Positive UVM coverage is included; no additional UVM defect was reproduced.
- Not credited: deterministic forced pause-budget eviction/requeue at every
  partial-prefill/before-output/after-output boundary, starvation ordering,
  actual coordinator result-parity routing, and the full TE/FP8/fused-RoPE/
  SWA/softmax-sink and chunked-prefill termination matrix. The two-rank
  hybrid Mamba/MTP/EP runner was launched, but its batch-wrapper output did
  not retain a final pytest status, so it is not claimed as coverage.

## Commits and publication

- Implementation commit: `801347f16` (`Fix dynamic inference request lifecycle contracts`, DCO signed-off).
- This report is committed separately with DCO sign-off and is pushed with the
  implementation to `fork:request-lifecycle-dfw-v0`; the final job handoff
  records the exact branch tip.
