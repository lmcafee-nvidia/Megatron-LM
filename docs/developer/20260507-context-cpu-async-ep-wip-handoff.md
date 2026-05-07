# Context CPU Async EP WIP Handoff

Date: 2026-05-07

This is a WIP handoff for `context-cpu-async-schedule-weekend`. Do not treat
this commit as a finished fix or merge-ready state.

## Current Objective

Fix the EP async scheduling protocol for decode-only steps without disabling
overlap. The long-term target is a single ordered EP async handoff point that
every EP rank enters exactly once per decode step. The handoff must decide, with
a fixed payload/schema, whether the group can reuse a pending forward, launch a
new async forward, or skip/block the step.

The invariant we need:

```text
all EP ranks: enter one ordered handoff
all EP ranks: agree on reuse/launch/skip and shape
all EP ranks: continue to sampling/bookkeeping
```

The current bug violates that invariant:

```text
active rank: pending-forward reuse collective
dummy rank: async-launch collective
```

That rank divergence deadlocks.

## Current WIP Code

The WIP changes touch:

- `megatron/core/inference/contexts/dynamic_context.py`
- `megatron/core/inference/engines/dynamic_engine.py`
- `megatron/core/inference/text_generation_controllers/text_generation_controller.py`
- `tests/unit_tests/inference/engines/test_dynamic_engine.py`

Implemented so far:

- Added a separate EP ZMQ communicator for pending-forward reuse.
- Suppressed periodic EP consensus refresh while async EP handoffs are active.
- Added helper paths for dummy EP steps and idle observation.
- Added launch agreement before async graph replay and before non-MTP sampling.
- Added tests around communicator separation, launch agreement, dummy mirror,
  pending-forward reuse, and consensus suppression.

This is still not the final protocol design. The separate reuse communicator
fixed a payload-size collision but did not fix ordering divergence.

## Tests Already Run

Passed:

```bash
python -m py_compile \
  megatron/core/inference/contexts/dynamic_context.py \
  megatron/core/inference/engines/dynamic_engine.py \
  megatron/core/inference/text_generation_controllers/text_generation_controller.py \
  tests/unit_tests/inference/engines/test_dynamic_engine.py
```

Passed focused pytest subset:

```bash
/usr/bin/python -m pytest -q \
  tests/unit_tests/inference/engines/test_dynamic_engine.py::test_ep_zmq_model_collectives_use_separate_channel \
  tests/unit_tests/inference/engines/test_dynamic_engine.py::test_ep_pending_forward_reuse_agreement_matrix \
  tests/unit_tests/inference/engines/test_dynamic_engine.py::test_ep_dummy_forward_participates_in_reuse_agreement_without_pending \
  tests/unit_tests/inference/engines/test_dynamic_engine.py::test_ep_dummy_async_forward_after_mtp_mirrors_launch \
  tests/unit_tests/inference/engines/test_dynamic_engine.py::test_ep_dummy_forward_reuses_mirrored_async_forward \
  tests/unit_tests/inference/engines/test_dynamic_engine.py::test_ep_dummy_forward_discards_mirrored_async_forward_on_peer_reject
```

Passed 4-rank focused distributed subset:

```bash
/usr/bin/python -m torch.distributed.run --nproc-per-node 4 -m pytest -v \
  tests/unit_tests/inference/engines/test_dynamic_engine.py \
  -k 'ep_zmq_model_collectives_use_separate_channel or ep_consensus_refresh_suppresses_periodic or ep_dummy_forward_idle_observation_resets_cached_consensus or ep_dummy_async_forward_mirrors_non_mtp_launch_without_full_engine or ep_dummy_forward_drains_device_before_async_handoff or ep_dummy_forward_participates_in_reuse_agreement_without_pending or ep_pending_forward_reuse_agreement_matrix or ep_dummy_async_forward_after_mtp_mirrors_launch or ep_dummy_forward_reuses_mirrored_async_forward or ep_dummy_forward_discards_mirrored_async_forward_on_peer_reject or ep_async_decode_graph_launch_agrees_before_replay or ep_async_decode_graph_launch_cancels_when_peer_blocks or ep_async_launch_before_sampling_sends_skip_when_not_prepared or ep_async_launch_before_sampling_requests_greedy_forward'
```

## Latest Benchmark Failure Evidence

The last interactive benchmark used `nano-v3` through
`inference-bench/run_interactive.sh`. That was useful for exposing the protocol
bug but is not the final target model because it is not the small MTP+EP model.

Run artifact:

```text
inference-bench/experiments/20260507_060432_debug_full_fix9
```

Observed behavior:

- Server reached ready state.
- Batch size 1 started.
- GPUs stayed at 0% utilization.
- The run was killed after the batch-size-1 request exceeded the expected
  completion window.

Per-rank stacks after kill:

- `server_rank2.log`: active rank blocked in
  `_ep_pending_forward_reuse_agreement`.
- `server_rank1.log` and `server_rank3.log`: dummy ranks blocked in
  `_ep_async_launch_agreement` via `_dummy_async_forward_after_mtp`.
- `server_rank0.log`: coordinator/front-end waiting on ZMQ receive.

Interpretation: ranks are entering different collectives in the same decode
step. Splitting communicators prevents struct-unpack corruption but leaves both
collectives waiting for peers.

## Correct Fix Direction

Do not fix this by restoring a guard that disables non-MTP or MTP dummy overlap.
That would only make the benchmark stop hanging by avoiding the feature.

The correct fix is protocol-level:

- Replace optional `reuse` and `launch` collectives with one ordered handoff, or
  otherwise make the collective sequence impossible to diverge.
- Use one fixed-schema payload with fields for:
  - local pending forward exists
  - local pending forward is reusable
  - local has real work
  - local requests async launch
  - local blocks async launch
  - token/prefill/decode dimensions
  - graph/eager launch mode
- All EP ranks must call it at the same logical point before any rank can enter
  the next per-step collective.

## MTP+EP Debug Model

The small fully featured model exists on `dfw`:

```text
dfw:data/inference/megatrons/context-cpu/lawrence/checkpoints/nano-v3-mtp-300m
dfw:data/inference/megatrons/context-cpu/lawrence/models/gpt_dynamic_inference_nano-v3-mtp-300m.sh
dfw:data/inference/megatrons/context-cpu/lawrence/save_nano_v3_mtp_300m_checkpoint.py
```

The local 4-GPU job copied the checkpoint and launcher under `lawrence/`, but
`lawrence/` is a nested local repo and is not part of this pushed Megatron
commit. The checkpoint is intentionally ignored by `lawrence/.gitignore`.

The model is `EP=8, TP=1` and the checkpoint has 8 distributed shards. It cannot
be run unchanged in the current 4-GPU allocation. On `dfw`, prefer running the
8-GPU model directly. If a 4-GPU version is needed, start from
`lawrence/save_nano_v3_mtp_300m_checkpoint.py` and change the save/run args from:

```text
--nproc-per-node 8
--expert-model-parallel-size 8
```

to:

```text
--nproc-per-node 4
--expert-model-parallel-size 4
```

The matching launcher must also use `NPROC_PER_NODE=4` and
`--expert-model-parallel-size 4`.

## Suggested Next Steps On dfw

1. Check out this WIP branch/commit from the fork.
2. Use the existing `dfw` `lawrence/` directory for the 300M MTP+EP checkpoint
   and benchmark helpers.
3. Implement the fixed-schema EP handoff in `TextGenerationController`.
4. Update tests so real and dummy ranks cannot enter different collective
   phases.
5. Run focused distributed unit tests.
6. Run the `nano-v3-mtp-300m` benchmark with async scheduling on/off and inspect
   decode-only steps.

