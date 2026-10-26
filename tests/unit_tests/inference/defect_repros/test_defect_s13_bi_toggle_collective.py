# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected bug s13: a nested batch-invariant disable resets the collective.

``set_batch_invariant_mode(False)`` inside an enabled scope disables batch-invariant mode on entry
and, on exit, re-enables it with ``enable_batch_invariant_mode(backend=prev_backend)``
(``megatron/core/transformer/custom_layers/batch_invariant_kernels.py:2017-2020``). That call omits
``collective``, so ``enable_batch_invariant_mode`` falls back to its default ``"ordered"``
(``:1815``) and overwrites the module global ``_BATCH_INVARIANT_COLLECTIVE`` (``:1834``). A
process launched with ``batch_invariant_collective="multimem"``
(``megatron/training/initialize.py:140-146``) therefore silently switches its EP combine
collective after any nested disable: the inference
dispatcher picks ``ordered_reduce_scatter_v`` instead of ``multimem_reduce_scatter_v``
(``megatron/core/transformer/moe/token_dispatcher_inference.py:666-672``) and the batch-invariant
MoE reference switches its cross-rank accumulation from fp64 to fp32
(``megatron/core/transformer/moe/batch_invariant.py:78-79``).

No GPU is needed: the test only toggles the global mode. With Transformer Engine installed (the dfw
container) the real TE patch/unpatch runs; without TE (a CPU-only environment) the TE kernel patch
is replaced by a no-op, because ``_te_patch_for_batch_invariant`` imports TE unconditionally and the
patched TE kernels play no part in the collective bookkeeping under test.
"""

import importlib.util

import pytest

import megatron.core.transformer.custom_layers.batch_invariant_kernels as bik

REQUIRED_GPUS = 0

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None

# The production default backend (enable_batch_invariant_mode's default and the config default).
BACKEND = "te_native"
NON_DEFAULT_COLLECTIVE = "multimem"


@pytest.fixture
def clean_bi_state(monkeypatch):
    """Start with batch-invariant mode off; restore every global this test touches."""
    if bik.is_batch_invariant_mode_enabled():
        pytest.skip(
            "batch-invariant mode is already enabled in this process (leaked by an earlier "
            "test); s13 needs to start from the disabled state"
        )
    # monkeypatch restores these module globals at teardown (enable/disable never reset the
    # collective, so a "multimem" value would otherwise leak into later tests).
    monkeypatch.setattr(bik, "_BATCH_INVARIANT_COLLECTIVE", bik.get_batch_invariant_collective())
    monkeypatch.setattr(bik, "_BATCH_INVARIANT_BACKEND", bik.get_batch_invariant_backend())
    if not HAVE_TE:
        monkeypatch.setattr(bik, "_te_patch_for_batch_invariant", lambda *a, **k: None)
    yield
    bik.disable_batch_invariant_mode()


def _state():
    return (
        bik.is_batch_invariant_mode_enabled(),
        bik.get_batch_invariant_backend(),
        bik.get_batch_invariant_collective(),
    )


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------


def test_control_enable_records_requested_collective(clean_bi_state):
    """CONTROL: enable_batch_invariant_mode stores the collective it is given."""
    bik.enable_batch_invariant_mode(backend=BACKEND, collective=NON_DEFAULT_COLLECTIVE)
    assert _state() == (True, BACKEND, NON_DEFAULT_COLLECTIVE)


def test_control_nested_disable_restores_mode_and_backend(clean_bi_state):
    """CONTROL: a nested disable turns the mode off inside the scope and back on after it, with
    the same backend. Only the collective is under suspicion."""
    bik.enable_batch_invariant_mode(backend=BACKEND, collective=NON_DEFAULT_COLLECTIVE)
    with bik.set_batch_invariant_mode(False):
        assert not bik.is_batch_invariant_mode_enabled()
    enabled, backend, _ = _state()
    assert enabled and backend == BACKEND


def test_control_default_collective_survives_nested_disable(clean_bi_state):
    """CONTROL: with the default collective the nested toggle is indistinguishable from correct
    behaviour, which is why existing tests (all default) cannot see s13."""
    bik.enable_batch_invariant_mode(backend=BACKEND, collective="ordered")
    with bik.set_batch_invariant_mode(False):
        pass
    assert _state() == (True, BACKEND, "ordered")


def test_control_nested_enable_keeps_collective(clean_bi_state):
    """CONTROL: a nested *enable* with the same backend is a no-op, so the collective stays."""
    bik.enable_batch_invariant_mode(backend=BACKEND, collective=NON_DEFAULT_COLLECTIVE)
    with bik.set_batch_invariant_mode(True, backend=BACKEND):
        assert _state() == (True, BACKEND, NON_DEFAULT_COLLECTIVE)
    assert _state() == (True, BACKEND, NON_DEFAULT_COLLECTIVE)


# ---------------------------------------------------------------------------------------------
# Defect
# ---------------------------------------------------------------------------------------------


def test_defect_nested_disable_preserves_collective(clean_bi_state):
    """DEFECT: leaving a nested set_batch_invariant_mode(False) must restore the exact enabled
    state, including the cross-rank combine collective."""
    bik.enable_batch_invariant_mode(backend=BACKEND, collective=NON_DEFAULT_COLLECTIVE)
    before = _state()
    with bik.set_batch_invariant_mode(False):
        pass
    after = _state()
    assert after == before, (
        "DEFECT s13: leaving a nested set_batch_invariant_mode(False) re-enabled batch-invariant "
        "mode with enable_batch_invariant_mode's default collective instead of the one in force: "
        f"(enabled, backend, collective) before={before}, after={after}. The EP combine then "
        "switches from multimem_reduce_scatter_v to ordered_reduce_scatter_v mid-run "
        "(batch_invariant_kernels.py:2017-2020 omits collective=)"
    )
