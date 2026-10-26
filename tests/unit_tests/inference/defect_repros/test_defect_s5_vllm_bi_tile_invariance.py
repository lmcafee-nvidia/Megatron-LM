# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime check s5: vLLM fused MoE batch invariance across tile configurations.

In batch-invariant mode ``vllm_fused_moe`` (``megatron/core/inference/moe/vllm_fused_moe.py:
762-773``) takes the whole launch config from ``_get_default_config(M=num_tokens_hint, E, top_k)``
(``:48-102``) and pins only ``BLOCK_SIZE_K = 64``. ``BLOCK_SIZE_M`` (16/32/64/128),
``BLOCK_SIZE_N`` (64/128), ``GROUP_SIZE_M`` (1/16), ``num_warps`` (4/8) and ``num_stages`` (3/4)
still follow the token-count hint. The code comment asserts these "reorder nothing in the
accumulation", so a token's output bits depend only on the K-loop grouping. That assumption is
unverified: on Hopper, Triton may lower ``tl.dot`` for different tile shapes and warp counts to
different MMA instructions (``wgmma`` vs ``mma.sync``) whose internal accumulation need not agree
bit for bit. The only existing evidence is one pair (hint 64 vs 256,
``tests/unit_tests/inference/test_vllm_fused_moe_batch_invariant.py:298``). Upstream vLLM's
batch-invariant mode pins the whole config instead.

This file asserts the property the code relies on (it FAILS only if the assumption is false):

* ``test_defect_live_rows_bitwise_across_all_tile_configs``: for fixed weights and a fixed set of
  live rows, the live rows' outputs are bitwise identical for a hint (and equal co-batch size)
  in every distinct config ``_get_default_config`` returns for these sizes (7 configs at E=8,
  enumerated, not hand-picked), for SwiGLU and squared ReLU.
* ``test_defect_live_rows_bitwise_across_cobatch_at_fixed_config``: same, varying only the
  co-batch size at one config (isolates tile membership / row position from tile shape).

Batch-invariant mode is enabled as in ``test_vllm_fused_moe_batch_invariant.py``
(``set_batch_invariant_mode(True, backend="triton")``). Needs one GPU and Triton.
"""

import os
import tempfile

os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton_test_cache"))

import pytest
import torch

from megatron.core.inference.moe import ActivationType
from megatron.core.inference.moe.batch_invariant import HAVE_TRITON
from megatron.core.inference.moe.vllm_fused_moe import _get_default_config, vllm_fused_moe
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
)

REQUIRED_GPUS = 1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON,
    reason="s5 check needs one CUDA GPU and Triton",
)

E = 8  # num_local_experts (all local, EP=1)
TOPK = 2
HIDDEN = 512  # K of FC1: 8 BLOCK_SIZE_K=64 steps
FFN = 256
MAX_TOKENS = 2048
LIVE = 16  # rows whose outputs must not change
BI_BLOCK_K = 64  # what batch-invariant mode pins (vllm_fused_moe.py:773)
ACTIVATIONS = [ActivationType.SWIGLU, ActivationType.SQUARED_RELU]


def _bi_config(m):
    cfg = dict(_get_default_config(M=m, E=E, top_k=TOPK))
    cfg["BLOCK_SIZE_K"] = BI_BLOCK_K
    return tuple(sorted(cfg.items()))


def distinct_config_hints():
    """Smallest hint >= LIVE for every distinct batch-invariant launch config up to MAX_TOKENS."""
    hints = {}
    for m in range(LIVE, MAX_TOKENS + 1):
        hints.setdefault(_bi_config(m), m)
    return sorted(hints.values())


def make_problem(activation, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    fc1_out = 2 * FFN if activation == ActivationType.SWIGLU else FFN
    hidden = torch.randn(MAX_TOKENS, HIDDEN, generator=g, device=device).bfloat16()
    fc1 = (torch.randn(E, fc1_out, HIDDEN, generator=g, device=device) / HIDDEN**0.5).bfloat16()
    fc2 = (torch.randn(E, HIDDEN, FFN, generator=g, device=device) / FFN**0.5).bfloat16()
    # TOPK distinct experts per row; fixed for every row, live or not.
    routing = torch.argsort(torch.rand(MAX_TOKENS, E, generator=g, device=device), dim=1)
    routing = routing[:, :TOPK].contiguous().long()
    probs = torch.rand(MAX_TOKENS, TOPK, generator=g, device=device) + 0.1
    probs = probs / probs.sum(-1, keepdim=True)
    return dict(
        hidden=hidden, fc1=fc1, fc2=fc2, routing=routing, probs=probs, activation=activation
    )


def run_live_rows(p, hint, valid):
    """vllm_fused_moe output rows [:LIVE] for a co-batch of `valid` tokens and hint `hint`.
    Must be called inside batch-invariant mode."""
    out = vllm_fused_moe(
        p["hidden"],
        p["probs"],
        p["fc1"],
        p["fc2"],
        activation_type=p["activation"],
        num_local_experts=E,
        local_expert_start=0,
        valid_tokens=torch.tensor(valid, dtype=torch.int32, device=p["hidden"].device),
        routing_map=p["routing"],
        num_tokens_hint=hint,
    )
    return out[:LIVE].clone()


def _describe(hint):
    return dict(_bi_config(hint))


def _diff(a, b):
    d = (a.float() - b.float()).abs()
    return f"max abs diff {d.max().item():.3g}, {int((a != b).sum())} of {a.numel()} elements"


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------


def test_control_hints_cover_seven_distinct_configs():
    """CONTROL: the enumerated hints really select pairwise-distinct launch configs, covering
    every BLOCK_SIZE_M (16/32/64/128), BLOCK_SIZE_N (64/128), GROUP_SIZE_M (1/16),
    num_warps (4/8) and num_stages (3/4) value."""
    hints = distinct_config_hints()
    cfgs = [_describe(h) for h in hints]
    assert len({_bi_config(h) for h in hints}) == len(hints) == 7, (hints, cfgs)
    for key, values in (
        ("BLOCK_SIZE_M", {16, 32, 64, 128}),
        ("BLOCK_SIZE_N", {64, 128}),
        ("GROUP_SIZE_M", {1, 16}),
        ("num_warps", {4, 8}),
        ("num_stages", {3, 4}),
    ):
        assert {c[key] for c in cfgs} == values, (key, cfgs)
    assert all(c["BLOCK_SIZE_K"] == BI_BLOCK_K for c in cfgs)


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=lambda a: a.name)
def test_control_identical_calls_repeat_bitwise(activation):
    """CONTROL: at every config, an identical call repeats bitwise (the indirection-table
    atomics may reorder rows inside a tile; that must not change bits)."""
    p = make_problem(activation)
    with set_batch_invariant_mode(True, backend="triton"):
        for hint in distinct_config_hints():
            first = run_live_rows(p, hint, hint)
            for _ in range(2):
                again = run_live_rows(p, hint, hint)
                assert torch.equal(first, again), f"hint {hint}: {_diff(first, again)}"
            assert torch.isfinite(first).all()


# ---------------------------------------------------------------------------------------------
# Runtime checks of the batch-invariance assumption
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=lambda a: a.name)
def test_defect_live_rows_bitwise_across_all_tile_configs(activation):
    """DEFECT/runtime check: the live rows are bitwise identical in every distinct launch
    config batch-invariant mode can pick (hint = co-batch size)."""
    p = make_problem(activation)
    hints = distinct_config_hints()
    with set_batch_invariant_mode(True, backend="triton"):
        outs = {h: run_live_rows(p, h, h) for h in hints}
    ref_hint = hints[0]
    ref = outs[ref_hint]
    bad = [
        f"hint {h} {_describe(h)}: {_diff(outs[h], ref)}"
        for h in hints[1:]
        if not torch.equal(outs[h], ref)
    ]
    assert not bad, (
        f"DEFECT s5: vllm_fused_moe in batch-invariant mode ({activation.name}) changes the bits "
        f"of the same {LIVE} live rows with the launch config; only BLOCK_SIZE_K is pinned "
        f"(vllm_fused_moe.py:762-773). Reference hint {ref_hint} {_describe(ref_hint)}. "
        "Mismatches: " + "; ".join(bad)
    )


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=lambda a: a.name)
def test_defect_live_rows_bitwise_across_cobatch_at_fixed_config(activation):
    """DEFECT/runtime check: at one fixed config (hint 256: BLOCK_SIZE_M 64, BLOCK_SIZE_N 128,
    8 warps), the live rows are bitwise identical for co-batch sizes 16..256."""
    p = make_problem(activation)
    hint = 256
    valids = [LIVE, 40, 130, 200, 256]
    with set_batch_invariant_mode(True, backend="triton"):
        outs = {v: run_live_rows(p, hint, v) for v in valids}
    ref = outs[valids[0]]
    bad = [
        f"co-batch {v}: {_diff(outs[v], ref)}"
        for v in valids[1:]
        if not torch.equal(outs[v], ref)
    ]
    assert not bad, (
        f"DEFECT s5: vllm_fused_moe in batch-invariant mode ({activation.name}) changes the bits "
        f"of the same {LIVE} live rows with the co-batch size at a fixed launch config "
        f"{_describe(hint)} (tile membership / row position, not tile shape). Mismatches: "
        + "; ".join(bad)
    )
