## Objective

Run focused P14 async scheduling correctness tests on dfw. This is a test-only job.

## Constraints

- Do not edit files.
- Do not commit.
- Do not push.
- Use the checked-out `context-cpu-async-schedule-weekend` branch.
- Confirm the commit SHA before running tests.

## Required Commands

Run these exact focused tests:

```bash
/opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/inference/contexts/test_dynamic_context.py \
  -k "prepare_async_decode_next_step_pre_sampling_allows_steady_state or prepare_async_decode_next_step_pre_sampling_declines_known_finish or prepare_async_decode_next_step_pre_sampling_declines_remap_boundary or prepare_async_decode_matches_graph_shape_from_planned_layout"
```

```bash
/opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/inference/test_async_scheduling_compact.py \
  -k "pending_async_forward_discards_when_planned_layout_mismatches_current or reused_pending_forward_prepares_next_step_before_sampling or reused_pending_forward_falls_back_after_sampling_when_presampling_declines or prepare_async_decode_before_sampling"
```

## Final Report

Report:

- Commit SHA tested.
- Pass/fail status for each command.
- Any failures, skipped tests, or unexpected warnings relevant to these tests.
- Confirm the worktree remained clean.
