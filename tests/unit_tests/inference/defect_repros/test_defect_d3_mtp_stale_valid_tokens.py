# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for defect d3: MTP draft forwards reuse a stale MoE valid-token count.

Defect (read from code at ed9c2c8c1):

* ``InferenceAllGatherDispatcherBase._valid_tokens_tensor`` (and ``_host_valid_tokens_estimate``)
  is one class-level value shared by every inference MoE layer
  (``megatron/core/transformer/moe/token_dispatcher_inference.py:67-205``). At EP=1 both
  dispatchers refresh it in ``token_dispatch`` only when ``_runs_metadata_sync`` is set
  (``:194-202``, ``:582-593``); at NCCL EP>1 the refresh is ``update_metadata`` under the same flag.
* ``set_moe_metadata_sync`` (``megatron/core/inference/utils.py:219-235``) sets the flag only on
  the first MoE layer of the module walk. ``GPTModel`` / ``HybridModel`` register ``decoder``
  before ``mtp``, so whenever the decoder has a MoE layer the MTP layer's MoE never refreshes.
* MTP draft forwards (``mtp_controller_mixin.py:601-626``) and the KV-seed pass (``:388``) run only
  MTP layers, with request-shaped rows: R rows < the main forward's V tokens whenever
  ``num_speculative_tokens >= 1``. The MTP layer's experts therefore read ``valid_tokens = V``
  while ``routing_map`` has R rows.
* The count kernels (``megatron/core/inference/moe/permute.py:54-115``) mask only by
  ``valid_tokens * topk``, so they count pairs past the end of ``routing_map``; the permute kernel
  clamps to ``routing_map.numel()`` (``:286``), so counts and placements disagree.

Layered evidence in this file:

(a) Mechanism. A real ``GPTModel`` (one MoE decoder layer + one MTP layer whose inner layer is
    MoE, ``transformer_impl='inference_optimized'``), the real ``set_moe_metadata_sync``, and the
    real ``MoELayer.forward`` (router, inference dispatcher, combine) for "main forward with V
    tokens" then "MTP forward with R rows". Only ``experts.forward`` is replaced by a recorder
    that reads the scalar the real experts would pass to the fused-MoE kernels
    (``experts.py:1723``, ``:1742``). A second, model-free test drives two real dispatcher objects
    whose flags are set by the real ``set_moe_metadata_sync``.
(b) Consequence at kernel level, made deterministic with canaries: ``routing_map`` is a view of
    the first R rows of a larger buffer whose tail rows route to a canary expert. With
    ``valid_tokens = V > R`` the per-expert counts must equal a plain-torch count over the R real
    rows. Checked for both count kernels, through ``permute_tokens`` (torch backend), through the
    vLLM-backend align step, and, for the default vLLM backend, that ``vllm_fused_moe`` does not
    write output rows past ``hidden_states.shape[0]``.
(c) Controls: the same harnesses with ``valid_tokens == R`` (and the decoder MoE in (a)) pass.

All canary reads stay inside the larger base buffers, so the defect tests do not read or write
outside any allocation (they are safe to run normally and report nothing under memcheck).

Requirements: 1 GPU. (a) also needs Transformer Engine. (b)/(c) kernel tests run on CPU under
``TRITON_INTERPRET=1`` when no GPU is present.
Launch: ``python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest <this file> -v``.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.moe import permute as permute_mod
from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.permute import compute_local_tokens_per_expert, permute_tokens
from megatron.core.inference.moe.vllm_fused_moe import (
    _moe_align_block_size_cuda_graphable,
    vllm_fused_moe,
)

# Max GPUs any test here needs. Every test is single-rank (EP=1).
REQUIRED_GPUS = 1


def _kernel_device():
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    return None


DEVICE = _kernel_device()

needs_kernel_device = pytest.mark.skipif(
    DEVICE is None, reason="needs a CUDA GPU, or TRITON_INTERPRET=1 to interpret the kernels on CPU"
)
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU (NCCL process groups, model build)"
)

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

needs_te = pytest.mark.skipif(
    not HAVE_TE, reason="transformer_impl='inference_optimized' layers need Transformer Engine"
)


@pytest.fixture
def cpu_sm_count(monkeypatch):
    """The persistent count kernel sizes its grid from the SM count; give the CPU run one."""
    if DEVICE == "cpu":
        monkeypatch.setattr(permute_mod, "_NUM_SMS", 4)


# --------------------------------------------------------------------------------------------- #
# (b)/(c) Kernel-level canary harness
# --------------------------------------------------------------------------------------------- #

TOPK = 2
CANARY_EXPERT = 7  # real rows never route here; canary rows route one slot here, one to -1
R_ROWS = 6  # rows that really belong to this forward (the MTP rows)
STALE_VALID = 10  # the main forward's count left in the shared scalar
BUF_ROWS = 64  # backing buffer; rows [R_ROWS, BUF_ROWS) are canaries
HIDDEN = 32
HIDDEN_SENTINEL = 1.0e4
PROB_SENTINEL = -7.0

# (local_expert_start, num_local_experts): EP=1 (all 8 experts local) and the rank-1 slice of
# EP=2 (NCCL EP>1 also hands the MoE only the live rows). The canary expert is local in both.
SLICES = [
    pytest.param(0, 8, id="ep1-all-local"),
    pytest.param(4, 4, id="ep2-rank1-slice"),
]


def _routing_buffers(device):
    """Return (base, routing_map) where routing_map = base[:R_ROWS] (a contiguous view).

    Real rows pick TOPK distinct experts from [0, CANARY_EXPERT). Canary rows are
    [CANARY_EXPERT, -1]: one countable pair each, so a stale count inflates the canary expert
    by exactly STALE_VALID - R_ROWS and the inflated total stays inside permute_tokens' output
    buffer (no out-of-bounds write in the test itself).
    """
    g = torch.Generator().manual_seed(1234)
    real = torch.stack([torch.randperm(CANARY_EXPERT, generator=g)[:TOPK] for _ in range(R_ROWS)])
    base = torch.empty(BUF_ROWS, TOPK, dtype=torch.int64)
    base[:, 0] = CANARY_EXPERT
    base[:, 1] = -1
    base[:R_ROWS] = real
    base = base.to(device)
    routing_map = base[:R_ROWS]
    assert routing_map.is_contiguous() and routing_map.data_ptr() == base.data_ptr()
    return base, routing_map


def _hidden_and_probs(device):
    """hidden/probs as views of the first R_ROWS of sentinel-filled buffers."""
    g = torch.Generator().manual_seed(4321)
    hidden_base = torch.full((BUF_ROWS, HIDDEN), HIDDEN_SENTINEL, dtype=torch.bfloat16)
    hidden_base[:R_ROWS] = torch.randn(R_ROWS, HIDDEN, generator=g).to(torch.bfloat16)
    probs_base = torch.full((BUF_ROWS, TOPK), PROB_SENTINEL, dtype=torch.float32)
    probs_base[:R_ROWS] = torch.rand(R_ROWS, TOPK, generator=g)
    hidden_base, probs_base = hidden_base.to(device), probs_base.to(device)
    return hidden_base[:R_ROWS], probs_base[:R_ROWS]


def _ref_counts(routing_map_real, local_start, num_local):
    """Plain-torch per-local-expert pair count over the real rows only."""
    lid = routing_map_real.reshape(-1).cpu().long() - local_start
    lid = lid[(lid >= 0) & (lid < num_local)]
    return torch.bincount(lid, minlength=num_local).to(torch.int32)


def _aligned(counts, alignment):
    return torch.where(counts > 0, (counts + alignment - 1) // alignment * alignment, counts)


def _valid(n):
    return torch.tensor([n], dtype=torch.int32, device=DEVICE)


def _check_counts(counts, routing_map, local_start, num_local, prefix, what):
    ref = _ref_counts(routing_map, local_start, num_local)
    got = counts.cpu()
    canary_lid = CANARY_EXPERT - local_start
    assert torch.equal(got, ref), (
        f"{prefix}{what}: per-expert counts {got.tolist()} != plain-torch count over the "
        f"{routing_map.shape[0]} routing_map rows {ref.tolist()}. Local canary expert "
        f"{canary_lid} (global {CANARY_EXPERT}) holds {int(got[canary_lid])}, so the kernel read "
        f"{int(got[canary_lid])} pair(s) past the end of routing_map."
    )


def _check_permutation(result, hidden, probs, routing_map, local_start, num_local, alignment, prefix):
    permuted_hidden, permuted_probs, permutation_map, inclusive = result
    ref = _ref_counts(routing_map, local_start, num_local)
    ref_inc = torch.cumsum(_aligned(ref, alignment), 0).to(torch.int32)
    ref_exc = ref_inc - _aligned(ref, alignment)
    got_inc = inclusive.cpu()
    assert torch.equal(got_inc, ref_inc), (
        f"{prefix}permute_tokens inclusive expert offsets {got_inc.tolist()} != reference "
        f"{ref_inc.tolist()} from the {routing_map.shape[0]} real rows (alignment {alignment}); "
        f"n_used={int(got_inc[-1])} vs {int(ref_inc[-1])}."
    )
    pm = permutation_map.cpu()
    rm = routing_map.cpu()
    for e in range(num_local):
        lo, hi = int(ref_exc[e]), int(ref_inc[e])
        block = pm[lo:hi]
        claimed = block[block >= 0]
        expected = sorted(
            t for t in range(rm.shape[0]) for k in range(TOPK) if int(rm[t, k]) - local_start == e
        )
        assert sorted(claimed.tolist()) == expected, (
            f"{prefix}local expert {e}: permuted rows [{lo}, {hi}) hold source tokens "
            f"{sorted(claimed.tolist())}, expected {expected}."
        )
        assert int((block < 0).sum()) == hi - lo - len(expected), (
            f"{prefix}local expert {e}: {int((block < 0).sum())} unclaimed rows in its block, "
            f"expected {hi - lo - len(expected)} alignment-padding rows; counts and placement "
            f"disagree."
        )
        rows = lo + (block >= 0).nonzero().flatten()
        assert torch.equal(permuted_hidden.cpu()[rows], hidden.cpu()[claimed.long()]), (
            f"{prefix}local expert {e}: permuted hidden rows do not match their source tokens."
        )
        probs_cpu = probs.cpu()
        expected_probs = sorted(
            float(probs_cpu[t, k])
            for t in range(rm.shape[0])
            for k in range(TOPK)
            if int(rm[t, k]) - local_start == e
        )
        assert sorted(permuted_probs.cpu()[rows].tolist()) == expected_probs, (
            f"{prefix}local expert {e}: permuted probs do not match the routed pairs."
        )


@needs_kernel_device
@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
@pytest.mark.parametrize("local_start,num_local", SLICES)
def test_control_count_kernels_valid_equals_rows(persistent, local_start, num_local, cpu_sm_count):
    """valid_tokens == routing_map rows: both count kernels match the plain-torch count."""
    _, routing_map = _routing_buffers(DEVICE)
    counts = compute_local_tokens_per_expert(
        routing_map, local_start, num_local, _valid(R_ROWS), persistent=persistent
    )
    _check_counts(counts, routing_map, local_start, num_local, "", "control")


@needs_kernel_device
@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
@pytest.mark.parametrize("local_start,num_local", SLICES)
def test_defect_count_kernels_ignore_rows_past_routing_map(
    persistent, local_start, num_local, cpu_sm_count
):
    """valid_tokens = V > R (the stale main-forward count): counts must cover only the R rows."""
    _, routing_map = _routing_buffers(DEVICE)
    kernel = "_count_local_tokens_kernel" + ("_persistent" if persistent else "")
    counts = compute_local_tokens_per_expert(
        routing_map, local_start, num_local, _valid(STALE_VALID), persistent=persistent
    )
    _check_counts(
        counts,
        routing_map,
        local_start,
        num_local,
        f"DEFECT d3: valid_tokens={STALE_VALID} > routing_map rows={R_ROWS}; ",
        kernel,
    )


@needs_kernel_device
@pytest.mark.parametrize("alignment", [1, 16])
@pytest.mark.parametrize("local_start,num_local", SLICES)
def test_control_permute_tokens_valid_equals_rows(alignment, local_start, num_local):
    """valid_tokens == rows: offsets, placement and copied data match the reference."""
    _, routing_map = _routing_buffers(DEVICE)
    hidden, probs = _hidden_and_probs(DEVICE)
    result = permute_tokens(
        hidden, probs, routing_map, local_start, num_local, _valid(R_ROWS), alignment=alignment
    )
    _check_permutation(result, hidden, probs, routing_map, local_start, num_local, alignment, "")


@needs_kernel_device
@pytest.mark.parametrize("alignment", [1, 16])
@pytest.mark.parametrize("local_start,num_local", SLICES)
def test_defect_permute_tokens_stale_valid_tokens(alignment, local_start, num_local):
    """Through the public permute_tokens entry (torch backend path) with the stale count.

    Shapes are chosen so the inflated n_used stays <= output_size: the test shows the
    count/offset corruption without itself writing out of bounds.
    """
    _, routing_map = _routing_buffers(DEVICE)
    hidden, probs = _hidden_and_probs(DEVICE)
    result = permute_tokens(
        hidden, probs, routing_map, local_start, num_local, _valid(STALE_VALID), alignment=alignment
    )
    output_size = result[2].shape[0]
    assert int(result[3][-1]) <= output_size  # harness: the test itself stayed in bounds
    _check_permutation(
        result,
        hidden,
        probs,
        routing_map,
        local_start,
        num_local,
        alignment,
        f"DEFECT d3: valid_tokens={STALE_VALID} > routing_map rows={R_ROWS}; ",
    )


# vLLM backend (the default inference_grouped_gemm_backend): the persistent count kernel's only
# caller builds the indirection table.
VLLM_BLOCK_M = 16


def _check_align(sorted_ids, num_post_padded, routing_map, local_start, num_local, prefix):
    ref = _ref_counts(routing_map, local_start, num_local)
    ref_post = int(_aligned(ref, VLLM_BLOCK_M).sum())
    got_post = int(num_post_padded.item())
    assert got_post == ref_post, (
        f"{prefix}_moe_align_block_size num_tokens_post_padded={got_post} != {ref_post} from the "
        f"{routing_map.shape[0]} real rows."
    )
    sentinel = routing_map.numel()
    ids = sorted_ids.cpu()
    placed = sorted(ids[ids != sentinel].tolist())
    rm = routing_map.cpu().reshape(-1)
    expected = sorted(
        p for p in range(rm.numel()) if 0 <= int(rm[p]) - local_start < num_local
    )
    assert placed == expected, (
        f"{prefix}sorted_token_ids holds pair ids {placed}, expected the local real pairs "
        f"{expected} (ids >= {sentinel} are pairs past the end of routing_map)."
    )


@needs_kernel_device
@pytest.mark.parametrize("local_start,num_local", SLICES)
def test_control_vllm_align_valid_equals_rows(local_start, num_local, cpu_sm_count):
    _, routing_map = _routing_buffers(DEVICE)
    sorted_ids, _, post = _moe_align_block_size_cuda_graphable(
        routing_map, VLLM_BLOCK_M, num_local, local_start, _valid(R_ROWS)
    )
    _check_align(sorted_ids, post, routing_map, local_start, num_local, "")


@needs_kernel_device
@pytest.mark.parametrize("local_start,num_local", SLICES)
def test_defect_vllm_align_stale_valid_tokens(local_start, num_local, cpu_sm_count):
    _, routing_map = _routing_buffers(DEVICE)
    sorted_ids, _, post = _moe_align_block_size_cuda_graphable(
        routing_map, VLLM_BLOCK_M, num_local, local_start, _valid(STALE_VALID)
    )
    _check_align(
        sorted_ids,
        post,
        routing_map,
        local_start,
        num_local,
        f"DEFECT d3: valid_tokens={STALE_VALID} > routing_map rows={R_ROWS}; ",
    )


# vllm_fused_moe output rows. _moe_sum_kernel loops token_id over [0, valid_tokens) with no bound
# against max_tokens (vllm_fused_moe.py:612) and stores every row it visits (:641). The output
# here is a view of a sentinel-filled buffer, so any store past max_tokens is visible.
VLLM_EXPERTS = 4
VLLM_HIDDEN = 64  # (hidden, ffn, topk, experts) as in test_vllm_fused_moe.py::test_matches_reference
VLLM_FFN = 64
OUT_SENTINEL = 12345.0

# Under TRITON_INTERPRET the loop index of `tl.range(pid, valid_tokens, ...)` is a Python int, so
# _moe_sum_kernel's `token_id.to(tl.int64)` raises; these two tests are GPU-only.
needs_cuda_moe_sum = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA GPU (_moe_sum_kernel does not run under TRITON_INTERPRET)",
)


def _vllm_inputs():
    g = torch.Generator().manual_seed(77)
    hidden_base = torch.full((BUF_ROWS, VLLM_HIDDEN), HIDDEN_SENTINEL, dtype=torch.bfloat16)
    hidden_base[:R_ROWS] = torch.randn(R_ROWS, VLLM_HIDDEN, generator=g).to(torch.bfloat16)
    probs_base = torch.full((BUF_ROWS, TOPK), PROB_SENTINEL, dtype=torch.float32)
    probs_base[:R_ROWS] = torch.rand(R_ROWS, TOPK, generator=g)
    # Rows past R route nowhere (-1, masked padding), so no kernel reads intermediates for them;
    # the only effect of a stale count is then the _moe_sum store.
    rm_base = torch.full((BUF_ROWS, TOPK), -1, dtype=torch.int64)
    rm_base[:R_ROWS] = torch.stack(
        [torch.randperm(VLLM_EXPERTS, generator=g)[:TOPK] for _ in range(R_ROWS)]
    )
    fc1 = (torch.randn(VLLM_EXPERTS, VLLM_FFN, VLLM_HIDDEN, generator=g) / VLLM_HIDDEN**0.5).to(
        torch.bfloat16
    )
    fc2 = (torch.randn(VLLM_EXPERTS, VLLM_HIDDEN, VLLM_FFN, generator=g) / VLLM_FFN**0.5).to(
        torch.bfloat16
    )
    out_base = torch.full((BUF_ROWS, VLLM_HIDDEN), OUT_SENTINEL, dtype=torch.float32)
    t = [x.to(DEVICE) for x in (hidden_base, probs_base, rm_base, fc1, fc2, out_base)]
    hidden_base, probs_base, rm_base, fc1, fc2, out_base = t
    return (
        hidden_base[:R_ROWS],
        probs_base[:R_ROWS],
        rm_base[:R_ROWS],
        fc1,
        fc2,
        out_base,
    )


def _ref_squared_relu_moe(hidden, probs, routing_map, fc1, fc2):
    """Per-token squared-ReLU MoE in fp32 with bf16 rounding of the two intermediates."""
    h, p, rm = hidden.cpu().float(), probs.cpu(), routing_map.cpu()
    w1, w2 = fc1.cpu().float(), fc2.cpu().float()
    out = torch.zeros(h.shape[0], VLLM_HIDDEN)
    for t in range(h.shape[0]):
        for k in range(TOPK):
            e = int(rm[t, k])
            if 0 <= e < VLLM_EXPERTS:
                a = torch.relu(w1[e] @ h[t]).square().to(torch.bfloat16).float()
                y = (w2[e] @ a).to(torch.bfloat16).float()
                out[t] += float(p[t, k]) * y
    return out


def _run_vllm(valid):
    hidden, probs, routing_map, fc1, fc2, out_base = _vllm_inputs()
    out = out_base[:R_ROWS]
    result = vllm_fused_moe(
        hidden,
        probs,
        fc1,
        fc2,
        ActivationType.SQUARED_RELU,
        num_local_experts=VLLM_EXPERTS,
        local_expert_start=0,
        valid_tokens=_valid(valid),
        routing_map=routing_map,
        out=out,
    )
    assert result.data_ptr() == out_base.data_ptr()  # harness: wrote into the provided buffer
    ref = _ref_squared_relu_moe(hidden, probs, routing_map, fc1, fc2)
    return out_base.cpu(), ref


@needs_cuda_moe_sum
def test_control_vllm_fused_moe_valid_equals_rows(cpu_sm_count):
    """valid_tokens == rows: live rows match the reference, rows past the output are untouched."""
    out_base, ref = _run_vllm(R_ROWS)
    # Same tolerance as test_vllm_fused_moe.py, but with O(1) outputs so it is not vacuous.
    assert ref.abs().max() > 0.5
    torch.testing.assert_close(out_base[:R_ROWS], ref, atol=5e-2, rtol=5e-2)
    assert torch.all(out_base[R_ROWS:] == OUT_SENTINEL)


@needs_cuda_moe_sum
def test_defect_vllm_fused_moe_stale_valid_writes_past_output(cpu_sm_count):
    """Default backend with the stale count: no store may land past out[max_tokens - 1]."""
    out_base, ref = _run_vllm(STALE_VALID)
    clobbered = (out_base[R_ROWS:] != OUT_SENTINEL).any(dim=1).nonzero().flatten() + R_ROWS
    assert clobbered.numel() == 0, (
        f"DEFECT d3: valid_tokens={STALE_VALID} > hidden_states rows={R_ROWS}; vllm_fused_moe "
        f"(_moe_sum_kernel) stored output rows {clobbered.tolist()} past the end of the "
        f"{R_ROWS}-row output tensor."
    )
    torch.testing.assert_close(out_base[:R_ROWS], ref, atol=5e-2, rtol=5e-2)


# --------------------------------------------------------------------------------------------- #
# (a) Mechanism: who refreshes the shared valid-token count
# --------------------------------------------------------------------------------------------- #

# A decode step with 4 requests and num_speculative_tokens=3: the main (verification) forward
# carries 4 * (1 + 3) tokens; each MTP draft forward carries one row per request.
MAIN_TOKENS = 16
MTP_ROWS = 4
MODEL_HIDDEN = 32
MODEL_EXPERTS = 4


@pytest.fixture
def ep1_model_parallel():
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


@pytest.fixture
def fresh_moe_metadata_state(monkeypatch):
    """Fresh process-global MoE metadata state, restored afterwards."""
    from megatron.core.inference import utils as inference_utils
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher,
    )

    monkeypatch.setattr(inference_utils, "moe_layer_cache", None)
    monkeypatch.setattr(inference_utils, "_moe_metadata_sync_initialized", False)
    monkeypatch.setattr(InferenceAllGatherDispatcherBase, "_valid_tokens_tensor", None)
    monkeypatch.setattr(InferenceAllGatherDispatcherBase, "_host_valid_tokens_estimate", None)
    # Standalone use: no context has wired the NVLS routing-mask count (see token_dispatch).
    monkeypatch.setattr(NVLSAllGatherVDispatcher, "_real_token_count_tensor", None)
    # What DynamicInferenceContext does at init for every EP size (dynamic_context.py:920).
    InferenceAllGatherDispatcherBase.allocate_valid_tokens_tensor()
    return inference_utils


def _moe_config(dispatcher_type, **extra):
    from megatron.core.activations import squared_relu
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=MODEL_HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        activation_func=squared_relu,
        num_moe_experts=MODEL_EXPERTS,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        inference_moe_token_dispatcher_type=dispatcher_type,
        **extra,
    )


def _dispatcher(dispatcher_type):
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        NCCLAllGatherDispatcher,
        NVLSAllGatherVDispatcher,
    )

    cls = NVLSAllGatherVDispatcher if dispatcher_type == "nvls" else NCCLAllGatherDispatcher
    return cls(
        num_local_experts=MODEL_EXPERTS,
        local_expert_indices=list(range(MODEL_EXPERTS)),
        config=_moe_config(dispatcher_type),
        pg_collection=get_default_pg_collection(),
    )


def _dispatch(dispatcher, rows):
    """The inference dispatch steps MoELayer.forward runs: dispatch_preprocess, token_dispatch."""
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
    )

    hidden = torch.randn(rows, 1, MODEL_HIDDEN, device="cuda", dtype=torch.bfloat16)
    routing_map = torch.randint(0, MODEL_EXPERTS, (rows, 2), device="cuda")
    probs = torch.rand(rows, 2, device="cuda", dtype=torch.float32)
    hidden, probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)
    hidden, probs = dispatcher.token_dispatch(hidden, probs)
    return dict(
        rows=int(dispatcher.routing_map.shape[0]),
        valid=int(InferenceAllGatherDispatcherBase._valid_tokens().item()),
        host_estimate=InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate(),
    )


def _dispatchers_flagged_by_set_moe_metadata_sync(inference_utils, dispatcher_type):
    """Two real dispatchers whose flags the real set_moe_metadata_sync assigns.

    The cache is seeded in GPTModel's module-walk order (decoder registered before mtp,
    gpt_model.py:267/280); the model-level tests below derive that order from a real model.
    """
    decoder_dispatcher = _dispatcher(dispatcher_type)
    mtp_dispatcher = _dispatcher(dispatcher_type)
    inference_utils.moe_layer_cache = [
        SimpleNamespace(_inference_token_dispatcher=decoder_dispatcher),
        SimpleNamespace(_inference_token_dispatcher=mtp_dispatcher),
    ]
    inference_utils.set_moe_metadata_sync(None)
    return decoder_dispatcher, mtp_dispatcher


@needs_cuda
@pytest.mark.parametrize("dispatcher_type", ["nccl", "nvls"])
def test_control_dispatcher_first_layer_refreshes_valid_tokens(
    dispatcher_type, ep1_model_parallel, fresh_moe_metadata_state
):
    decoder_dispatcher, _ = _dispatchers_flagged_by_set_moe_metadata_sync(
        fresh_moe_metadata_state, dispatcher_type
    )
    seen = _dispatch(decoder_dispatcher, MAIN_TOKENS)
    assert seen["rows"] == MAIN_TOKENS and seen["valid"] == MAIN_TOKENS, seen


@needs_cuda
@pytest.mark.parametrize("dispatcher_type", ["nccl", "nvls"])
def test_defect_dispatcher_mtp_layer_sees_stale_valid_tokens(
    dispatcher_type, ep1_model_parallel, fresh_moe_metadata_state
):
    decoder_dispatcher, mtp_dispatcher = _dispatchers_flagged_by_set_moe_metadata_sync(
        fresh_moe_metadata_state, dispatcher_type
    )
    _dispatch(decoder_dispatcher, MAIN_TOKENS)  # main (verification) forward
    seen = _dispatch(mtp_dispatcher, MTP_ROWS)  # MTP draft forward
    assert seen["valid"] == seen["rows"] == MTP_ROWS, (
        f"DEFECT d3: {dispatcher_type} dispatcher of the second MoE layer (the MTP layer) "
        f"dispatched {seen['rows']} rows but the shared valid-token count read by its experts is "
        f"{seen['valid']} (host estimate {seen['host_estimate']}), left over from the "
        f"{MAIN_TOKENS}-token main forward: set_moe_metadata_sync gave it "
        f"_runs_metadata_sync={mtp_dispatcher._runs_metadata_sync}."
    )


def _build_gpt_with_moe_mtp(dispatcher_type):
    """Smallest GPTModel whose decoder and MTP layer both contain an inference MoE layer.

    Construction follows tests/unit_tests/inference/engines/test_dynamic_engine.py
    (_build_test_env, model_provider='gpt') and test_text_generation_controller.py (setup_model).
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_inference_spec,
        get_gpt_mtp_block_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import clear_nvte_env_vars

    # TE attention with the auto backend conflicts with the NVTE_*_ATTN=0 pinned by the unit-test
    # conftest; the engine and controller tests clear them the same way (restored per test).
    clear_nvte_env_vars()
    model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
    config = _moe_config(dispatcher_type, mtp_num_layers=1)
    layer_spec = get_gpt_layer_with_inference_spec(num_experts=MODEL_EXPERTS)
    mtp_block_spec = get_gpt_mtp_block_spec(
        config=config, spec=layer_spec, use_transformer_engine=False
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=128,
        max_sequence_length=64,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        mtp_block_spec=mtp_block_spec,
        position_embedding_type="none",
    ).cuda()
    model.eval()
    return model


def _run_main_then_mtp_moe(dispatcher_type, inference_utils, monkeypatch):
    """Build the model, apply set_moe_metadata_sync, run decoder MoE (V) then MTP MoE (R).

    Returns per-layer records of what the experts saw. Only experts.forward is replaced: the
    recorder reads the exact scalar InferenceGroupedMLP passes to mcore_fused_moe /
    vllm_fused_moe as valid_tokens, then returns zeros (so the defective kernels never run here).
    The router's torch.compile wrapper is swapped for the function it wraps to keep the test
    fast; routing values are irrelevant to the count under test.
    """
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
    from megatron.core.transformer.moe.router import InferenceTopKRouter
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
    )

    monkeypatch.setattr(
        InferenceTopKRouter,
        "_compiled_topk_routing",
        staticmethod(topk_routing_with_score_function),
    )
    model = _build_gpt_with_moe_mtp(dispatcher_type)
    decoder_moe = [m for m in model.decoder.modules() if isinstance(m, MoELayer)]
    mtp_moe = [m for m in model.mtp.modules() if isinstance(m, MoELayer)]
    assert len(decoder_moe) == 1 and len(mtp_moe) == 1, (len(decoder_moe), len(mtp_moe))
    decoder_moe, mtp_moe = decoder_moe[0], mtp_moe[0]

    inference_utils.set_moe_metadata_sync(model)  # production: text_generation_controller.py:852

    seen = []

    def recorder(tag):
        def fake_experts_forward(
            dispatched_input, tokens_per_expert, permuted_probs, routing_map=None
        ):
            seen.append(
                dict(
                    layer=tag,
                    rows=int(routing_map.shape[0]),
                    valid=int(InferenceAllGatherDispatcherBase._valid_tokens().item()),
                    host_estimate=InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate(),
                )
            )
            out = torch.zeros(
                dispatched_input.shape, dtype=torch.float32, device=dispatched_input.device
            )
            return out, None

        return fake_experts_forward

    monkeypatch.setattr(decoder_moe.experts, "forward", recorder("decoder"))
    monkeypatch.setattr(mtp_moe.experts, "forward", recorder("mtp"))

    with InferenceMode.active(), torch.inference_mode():
        decoder_moe(torch.randn(MAIN_TOKENS, 1, MODEL_HIDDEN, device="cuda", dtype=torch.bfloat16))
        mtp_moe(torch.randn(MTP_ROWS, 1, MODEL_HIDDEN, device="cuda", dtype=torch.bfloat16))

    walk = [
        "decoder" if m is decoder_moe else "mtp" if m is mtp_moe else "?"
        for m in (inference_utils.moe_layer_cache or [])
    ]
    flags = dict(
        decoder=decoder_moe._inference_token_dispatcher._runs_metadata_sync,
        mtp=mtp_moe._inference_token_dispatcher._runs_metadata_sync,
    )
    assert [s["layer"] for s in seen] == ["decoder", "mtp"], seen
    return seen, walk, flags


@needs_cuda
@needs_te
@pytest.mark.parametrize("dispatcher_type", ["nccl", "nvls"])
def test_control_model_decoder_moe_reads_its_own_valid_tokens(
    dispatcher_type, ep1_model_parallel, fresh_moe_metadata_state, monkeypatch
):
    seen, walk, flags = _run_main_then_mtp_moe(
        dispatcher_type, fresh_moe_metadata_state, monkeypatch
    )
    decoder, mtp = seen
    assert decoder["rows"] == decoder["valid"] == MAIN_TOKENS, (decoder, walk, flags)
    assert mtp["rows"] == MTP_ROWS, (mtp, walk, flags)  # harness: the MTP forward had R rows


@needs_cuda
@needs_te
@pytest.mark.parametrize("dispatcher_type", ["nccl", "nvls"])
def test_defect_model_mtp_moe_reads_stale_valid_tokens(
    dispatcher_type, ep1_model_parallel, fresh_moe_metadata_state, monkeypatch
):
    seen, walk, flags = _run_main_then_mtp_moe(
        dispatcher_type, fresh_moe_metadata_state, monkeypatch
    )
    _, mtp = seen
    assert mtp["valid"] == mtp["rows"] == MTP_ROWS, (
        f"DEFECT d3: GPTModel with MoE in decoder and MTP ({dispatcher_type} dispatcher, EP=1): "
        f"the MTP layer's MoE ran on {mtp['rows']} routing rows but its experts read "
        f"valid_tokens={mtp['valid']} (host estimate {mtp['host_estimate']}), the count left by "
        f"the {MAIN_TOKENS}-token decoder forward. set_moe_metadata_sync walk order {walk}, "
        f"_runs_metadata_sync {flags}."
    )
