# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Actual native-TE userbuffer inference under changing target batch shapes."""

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.module import base, layernorm_linear, linear

from tests.unit_tests.inference.engines import batch_invariant_test_utils as f


def test_batch_invariant_native_overlap(monkeypatch):
    case = f.Case("native-overlap", model={"tp_comm_overlap": True}, tp=2, sp=True)
    observer, active = f.ForwardWitness, []

    def witness(*args):
        result = observer(*args)
        active[:] = [result]
        return result

    monkeypatch.setattr(f, "ForwardWitness", witness)
    for owner in (linear, layernorm_linear):
        original = owner.general_gemm

        def gemm(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if active and active[0].current is not None and kwargs.get("ub") is not None:
                active[0].current.setdefault("native_overlap", []).append(kwargs["ub_type"])
            return result

        monkeypatch.setattr(owner, "general_gemm", gemm)
    with f.invariant_runtime(case) as (backend, version):
        assert backend == "te_native" and torch.distributed.get_world_size() == 2
        try:
            base.initialize_ub([512, 128], 2, dtype=torch.bfloat16, bootstrap_backend="gloo")
            reference = f.run_order(case, backend, version, "solo")
            actual = f.run_order(case, backend, version, "back")
            f.assert_same_target(reference, actual)
            assert {s["physical"] for s in reference[1].steps if s["decode"]} == {64}
            assert any(
                s["decode"] and s["physical"] == 128 and s["target_row"] > 0
                for s in actual[1].steps
            )
            for result in (reference, actual):
                for step in result[1].steps:
                    assert set(step.get("native_overlap", [])) == {
                        tex.CommOverlapType.AG,
                        tex.CommOverlapType.RS,
                    }
        finally:
            base.destroy_ub()
