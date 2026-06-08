# P14, C9: Restore Reused Pending Forward Pre-Sampling Fast Path

## Objective

Restore the batch-size-1 steady-state async scheduling fast path that was still
regressed after `e3c9b28125dcb292c63a247a0ee43123d8cbb6f5`.

The current `e3c9` fix only prepares before sampling when there is no reused
pending forward. That is wrong for the normal decode steady state: after the
first async step, the controller usually has a reusable pending async forward.
Keeping `not pending_forward_reused` in the pre-sampling prepare guard pushes
next-step preparation until after sampling on almost every steady-state step,
which loses the intended CPU/GPU overlap and produced ~5.9 ms/tok on HSG.

## Required Code Change

In `megatron/core/inference/text_generation_controllers/text_generation_controller.py`,
change the pre-sampling async prepare guard inside
`async_generate_output_tokens_dynamic_batch` back to the known-good shape:

```python
if not pending_forward_row_mapped and self.num_speculative_tokens == 0:
    async_next_prepared = self._try_prepare_async_decode_before_sampling()
```

Do not require `not pending_forward_reused`.

Keep row-mapped forwards on the conservative path. Keep MTP behavior unchanged.

## Required Tests

In `tests/unit_tests/inference/test_async_scheduling_compact.py`:

1. Update the reused-pending-forward compact test so the ordinary reused,
   non-row-mapped pending forward path proves this order:

```text
precheck -> sample -> copy
```

where `precheck` is `_try_prepare_async_decode_before_sampling()`.

2. Add a second reused-pending-forward test proving that when pre-sampling
   prepare declines, the controller falls back safely:

```text
precheck -> sample -> prepare_after -> copy
```

3. Keep existing row-mapped and unsafe-layout tests intact.

## Test Commands

Run these on dfw with `/opt/venv/bin/python` exactly:

```bash
/opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/inference/test_async_scheduling_compact.py \
  -k "reused_pending_forward_prepares_next_step_before_sampling or reused_pending_forward_falls_back_after_sampling_when_presampling_declines or prepare_async_decode_before_sampling or pending_forward_layout or reused_pending_forward"
```

```bash
/opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/inference/test_async_scheduling_compact.py
```

## Commit And Push

After tests pass, commit with:

```text
P14, C9: Restore reused pending async fast path
```

Push to `fork/context-cpu-async-schedule-weekend`.

In final output, report:

- commit SHA
- exact tests run and pass counts
- whether the worktree is clean
- any failures, if any
