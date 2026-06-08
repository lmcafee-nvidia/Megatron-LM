# P14 C8: Restore steady-state pre-sampling async prepare

You are running inside a fresh dfw Slurm batch clone of
`git@github.com:lmcafee-nvidia/Megatron-LM.git` on branch
`context-cpu-async-schedule-weekend`.

Implement exactly one production commit and push it to the same branch.

## Goal

Fix the performance regression introduced by
`59383fdf19ae710f326987b3c4b76af8cfb4987b` without removing the correctness
checks it added.

Any performance regression relative to
`790a0f281ea878fe74cd239b131a4ce87551ef29` is unacceptable. Do not run HSG
performance benchmarks in this dfw job; the user will run HSG inference-bench
after focused dfw unit tests pass.

## Required behavior

- Preserve planned-layout snapshot and validation behavior from `59383fd`.
- Do not copy KV cache contents.
- Do not copy Mamba conv/SSM state.
- Restore pre-sampling async preparation for steady-state decode steps.
- Keep after-sampling async preparation for unsafe lifecycle-boundary cases.
- Pre-sampling deferral must not be counted as an async-disable reason.
- Keep post-sampling preparation for:
  - MTP/speculative decode,
  - pending-forward reuse paths,
  - known finish/pause/resume/evict/add/remap lifecycle boundaries,
  - any case where the dry lifecycle plan cannot prove the current sampling
    layout is preserved.

## Implementation guidance

Use the existing pure dry lifecycle plan in
`megatron/core/inference/contexts/dynamic_context.py` to decide whether a
pre-sampling prepare is safe before mutating live context.

Add a pre-sampling mode to `prepare_async_decode_next_step`, for example:

```python
def prepare_async_decode_next_step(self, *, pre_sampling: bool = False) -> bool:
    ...
```

When `pre_sampling=True`, after building the dry plan and before reserving
blocks or mutating any live context fields, return `False` unless all of these
are true:

- dry plan has the same active request count as the current active decode set,
- dry plan has no known finished requests,
- dry plan request IDs exactly equal the current active request IDs,
- dry plan source request indexes exactly equal the current active slice,
- no pause/resume/evict/add/remap effect is present in the planned active set.

The identity checks above are intentionally conservative. If a case is not
obviously steady-state, defer and let the existing after-sampling path handle it.

In
`megatron/core/inference/text_generation_controllers/text_generation_controller.py`:

- Restore `_try_prepare_async_decode_before_sampling()` so eligible steady-state
  decode calls `context.prepare_async_decode_next_step(pre_sampling=True)`.
- If pre-sampling prepare succeeds, make the EP async handoff decision before
  sampling, matching the `790a0f` fast path.
- If pre-sampling prepare fails because the step is not safe for pre-sampling,
  set `_async_prepare_deferred_until_after_sampling=True` and return `False`.
  This deferral is not an async disable reason.
- Preserve the existing after-sampling prepare path and copy sampled tokens only
  after the prepared async layout exists.
- Keep the existing planned-layout validation on pending-forward reuse.

If you need a helper, prefer a narrowly scoped internal method on
`DynamicInferenceContext`, such as `_async_decode_plan_preserves_sampling_layout`.
Avoid broad refactors.

## Focused tests to add or update

Update existing tests rather than deleting coverage.

Required test coverage:

- Context-level steady-state decode allows
  `prepare_async_decode_next_step(pre_sampling=True)` and materializes the
  planned layout.
- Context-level known max-length finish declines
  `prepare_async_decode_next_step(pre_sampling=True)` without reserving blocks
  or recording a prepared plan.
- Context-level pause/remap boundary declines
  `prepare_async_decode_next_step(pre_sampling=True)` without mutating the live
  active layout.
- Controller steady-state ordering is:
  `prepare -> handoff -> sample -> copy`.
- Controller unsafe-pre-sampling fallback ordering is:
  `sample -> prepare_after_sampling -> copy`.
- Pre-sampling deferral is not recorded in async disable reason counts.
- Existing planned-layout mismatch still discards pending-forward reuse.

Likely test files:

- `tests/unit_tests/inference/contexts/test_dynamic_context.py`
- `tests/unit_tests/inference/test_async_scheduling_compact.py`

## Required dfw test commands

Run these commands before committing:

```bash
/opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/inference/contexts/test_dynamic_context.py \
  -k "prepare_async_decode_next_step or async_lifecycle_plan"

/opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/inference/test_async_scheduling_compact.py \
  -k "prepare_async_decode_before_sampling or pending_forward_layout or reused_pending_forward"

/opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/inference/test_async_scheduling_compact.py
```

If `/opt/venv/bin/python` is unavailable in the dfw container, use the Python
interpreter already available in the repo's dfw test environment, such as
`/usr/bin/python3`, and report that substitution in the final response.

## Commit and push

If all focused tests pass:

```bash
git status --short
git add megatron/core/inference/contexts/dynamic_context.py \
        megatron/core/inference/text_generation_controllers/text_generation_controller.py \
        tests/unit_tests/inference/contexts/test_dynamic_context.py \
        tests/unit_tests/inference/test_async_scheduling_compact.py
git commit -m "P14, C8: Restore steady-state async prepare fast path"
git push origin context-cpu-async-schedule-weekend
```

Do not commit if tests fail.

Final response must include:

- final commit SHA, if committed,
- pushed branch, if pushed,
- exact test commands run,
- pass/fail counts,
- any interpreter substitution,
- a concise summary of the code change.
