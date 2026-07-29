# Prefix-cache unit coverage

This directory owns the generated execution-stress rows in
`tests/test_utils/prefix_cache_coverage.yaml`. A row counts as active only when
it performs the named cache operations and checks their results; accepting
`enable_prefix_caching=True` is never coverage.

The active local suites are intentionally split by the layer that can execute
their axes honestly:

- `test_hash_contract.py` runs five collected hash-chain and request-lifecycle
  rows. Each row checks seven boundaries, a 24-block ancestry chain, descendant
  mutation, serialization, checkpointing, and namespace invalidation.
- `test_kv_allocator.py` runs the allocator and paused-request matrices. Every
  allocator row performs three generations of matching, shared references,
  exhausted-pool allocation, policy-specific release or eviction, and physical
  reuse against an independent state oracle. Every paused row verifies the
  minimal right-most eviction suffix against a scalar calculation.
- `test_context_execution.py` crosses REF_ZERO and LRU with ordinary and MTP
  continuation chunks cut immediately before, on, and after a block boundary.
  Every row hides and restores the real continuation request three times, then
  checks the additional matched block IDs, references, offsets, and query work.
- `test_engine_e2e.py` runs three complete seven-request generations on fresh
  cache-disabled and cache-enabled engines. Every group contains proven hits,
  one explicit miss, and one request that exercises the selected sampling and
  output path. It compares tokens, statuses, prompt/generated log probabilities,
  and top-N results.
- `test_mamba_cache.py` stresses executable KV/Mamba match depths, relative pool
  pressure, extraction-scratch limits, pinned-slot failure, state restoration,
  and physical slot reuse.
- `test_coordinator.py` checks real handler state transitions and hundreds of
  routing decisions against an independent scalar model.
- `test_lifecycle.py`, `test_admission_rollback.py`, and
  `test_speculative_controller.py` own request checkpoint/requeue, transactional
  admission failure, and block-crossing rewind stress respectively.

## Reconciliation with earlier tests

The restructuring removed or folded tests that only checked configuration or
initial layout:

- the standalone allocator state-layout test was folded into the real
  allocate/register test;
- the disabled-mode flag/layout test was removed;
- hybrid warning-only and too-small-budget configuration tests were removed
  from prefix-cache coverage;
- the old basic hash-computation test was replaced by the five stress rows
  above;
- the disabled reset/preserve-flag check was removed.
- the one-policy speculative refcount check and the flag-off test named for
  shared prefixes were replaced by the two-policy, 200-trace rewind state
  machine; the generic controller-wrapper test remains outside prefix coverage.
- ten isolated context cases were folded into the allocator, engine,
  speculative-controller, paused-rebalance, and continuation matrices. The
  continuation matrix preserves the distinct regression where a hidden later
  chunk discovers cached blocks that were outside its first chunk.

Existing tests outside this directory remain when they own a narrower
regression that the matrices do not replace, such as collision vectors,
parent-safe LRU leaf selection, prompt-logprob bypass immutability, admission
rollback, chunk/epoch quarantine, Mamba byte immutability, and routing-state
reconstruction. Generic inference tests are not counted as prefix-cache
coverage unless the manifest names them and they prove runtime activation.

Known gaps are closed entries under `unsupported` in the manifest and never
count toward scenarios, behaviors, pairs, or collected cases. This keeps
unavailable or externally owned surfaces visible without presenting a flag-only
or unexecuted row as prefix-cache coverage.
