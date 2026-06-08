# Async Scheduling

Async scheduling for dynamic decode is transactional. A pending async forward is
launched only after the current decode step has committed its sampled tokens and
request state. CPU-only work from the committed step can then overlap the
already-launched next forward.

The dynamic decode lifecycle is:

```text
resolve forward(N)
-> sample(N)
-> hard-commit decode state(N)
-> admit/schedule next work
-> build CommittedDecodePlan(N+1)
-> launch forward(N+1)
-> overlap pure CPU post-launch work
```

Async launch is only eligible for committed decode-only state. Prefill,
chunked prefill, mixed prefill/decode, MTP rewind/verification that has not
settled, and admission that creates prefill work all skip async launch.

## Committed Plan

`CommittedDecodePlan` is the only async launch descriptor. It is immutable and
built from committed context state after hard commit and admission. It records:

- request IDs and committed row order
- active decode count and CUDA graph shape
- decode stride and active token count
- Mamba live slot IDs
- KV/resource reservations
- exact identity fingerprint
- component requirements for Mamba, MTP, logprobs, EP, and resources

The plan contains no predicted future layout and no row map. Pending async
logits can be accepted only when the pending and current committed plans match
exactly. Request reorder, finish, pause, admission, CUDA graph shape changes,
decode-stride changes, Mamba slot changes, or resource-identity changes force a
transaction rollback and synchronous fallback.

## Transaction Lifecycle

The lifecycle for one pending async forward is:

```text
prepare(plan) -> launch(plan) -> accept(plan) | rollback(plan) -> retire()
```

`AsyncDecodeTransaction` owns all launched-forward resources:

- committed plan
- H2D event
- forward-done event
- output/logit handle
- Mamba snapshot participant
- KV/resource participant
- EP participant
- diagnostics and invalidation reason

The controller orchestrates lifecycle calls. It does not manually free
transaction-owned resources. Commit, rollback, and retire paths are idempotent,
and participant-owned resources are released exactly once.

## Hard Commit And Post-Launch Work

Dynamic decode bookkeeping is split into two phases.

Hard commit happens before any async launch:

- apply sampled tokens
- update request state
- finish, pause, and resume rows
- advance offsets
- update next-step context tensors
- admit waiting requests

Post-launch CPU work happens after the next forward is queued:

- detokenization
- coordinator replies
- output formatting
- logging
- CPU-only stop and logprob work

Late CPU-discovered invalidation marks the in-flight transaction with a stable
reason. It does not mutate the committed launch identity.

## Mamba And KV

Hybrid Mamba models use one live state bank. Before an async launch, the Mamba
participant snapshots the live slots named by the committed plan. Accepting a
transaction discards the snapshot and keeps live-bank writes. Rolling back
restores the snapshot before synchronous fallback.

KV and resource reservations are transaction participants. Async KV writes are
not made visible by rollback. Unused reservations are released on rollback, and
frees that could race an in-flight forward wait on the transaction
forward-done fence.

## EP Coordination

Expert-parallel async scheduling uses one transaction consensus path:

- all ranks collectively accept or roll back the previous transaction
- all ranks collectively launch or skip the next committed async forward
- dummy ranks mirror active-rank forward cadence
- graph-shape consensus is based on committed plan state
- tagged collective diagnostics remain attached to the transaction

There is no reorder-based EP handoff protocol.

## Eligibility And Diagnostics

`classify_committed_async_launch` centralizes component gates. Decode-only state
is mandatory, admission runs before plan build, CUDA graph shape is selected
from committed state, and prefix/chunked prefill never enters async decode
launch.

Diagnostics are structured around committed launch attempts, overlap
opportunities, exact-identity accept/fallback reasons, transaction state, and
participant state. Fallback reasons are stable strings so tests and benchmark
logs can compare behavior without depending on transient object details.

## Focused Validation

Useful focused checks:

```bash
uv run python -m pytest -q tests/unit_tests/inference/test_async_scheduling_compact.py -k "committed_plan or committed_pending_forward or committed_async_launch"
uv run python -m pytest -q tests/unit_tests/inference/test_async_scheduling_compact.py -k "fast_overlap or transaction_commit_rollback"
uv run python -m pytest -q tests/unit_tests/inference/test_async_scheduling_compact.py -k "static_cleanup"
```

Inference-bench and direct-coordinator validation are separate final validation
steps and are not replaced by these unit/static checks.
