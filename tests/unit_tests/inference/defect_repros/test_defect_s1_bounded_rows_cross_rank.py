# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected bug s1: bounded decode rows are chosen per EP rank, with no EP agreement.

The controller enables the bounded FlashInfer row prefix from the rank's *local* phase
(``text_generation_controller.py:932-943``: ``InferenceMode.set_bounded_mxfp8_rows(
context.is_decode_only() and capacity and max_requests * (spec + 1) * EP <= capacity)``). With the
NVLS dispatcher the EP ranks never agree on a phase (``match_ep_token_counts`` is false for NVLS,
``dynamic_context.py:2620-2627``). A bounded rank processes only ``[:capacity]`` of the gathered
AGV rows and copies that prefix into the shared RSV ``out`` buffer
(``flashinfer_mxfp8.py:330-376``; NVFP4 ``flashinfer_nvfp4.py:355-424``). RSV rows ``>= capacity``
keep whatever they held, and ``token_combine`` sums every rank's RSV rows into each peer's tokens
(``token_dispatcher_inference.py:639-679``; no copy because the expert output *is* the RSV tensor).
A peer whose prefill tokens sit at global AGV rows ``>= capacity`` then gets the bounded rank's
stale rows instead of that rank's expert contributions.

The tests drive the real pieces below the engine: ``NVLSAllGatherVDispatcher`` (symmetric-memory
AGV / RSV, real multimem collectives), the production expert glue
``InferenceGroupedMLP._flashinfer_forward`` (``experts.py:1666-1693``, called with a stand-in
``self`` as ``test_flashinfer_mxfp8.py`` does in its NVLS copy-fallback test)
and the real FlashInfer kernels behind ``flashinfer_routed_mxfp8_moe`` /
``flashinfer_routed_nvfp4_moe``. The bounded flag is computed with the controller's predicate and
set through ``InferenceMode.set_bounded_mxfp8_rows``, exactly as the controller does.

Reference: a plain fp64 per-token MoE, ``out_t = sum_k p_tk * W2_e relu(W1_e x_t)^2``, using the
BF16 weights. Inputs and weights are E2M1-grid values times a power of two, with one ``+-6 * 2^-k``
in every 16-element block, so every MXFP8 (32-block) and NVFP4 (16-block) block has the same amax
and the weight / input quantization is exact (the fixture checks this for MXFP8). Routing weights
are multiples of 1/16, exact in BF16. The only quantization error left is the kernel's
intermediate requantization (CPU emulation: per-token relative L2 error <= 0.041 MXFP8, <= 0.105
NVFP4), against tolerances 0.15 / 0.35. Every token routes slot 0 to an expert of rank 0 with
weight >= 0.625, so a missing rank-0 contribution is a relative error >= 0.72, and a stale
sentinel row (1000.0) is >= 450.

Tests (EP = world size, power of two; rank 0 is the decode-only rank):
* CONTROL all ranks unbounded (every rank in a non-decode step): matches the reference.
* CONTROL all ranks bounded, every token inside the capacity, RSV pre-filled with the sentinel:
  matches the reference.
* DEFECT rank 0 bounded (decode-only, 4 tokens), ranks >= 1 unbounded with 160 prefill tokens each,
  so peer tokens sit at global rows >= capacity; RSV pre-filled with a sentinel (a stale previous
  step) or with zeros (missing contributions). Every rank's outputs must match the reference.

Requirements: >= 2 Blackwell (SM100-family) GPUs in one NVLink domain with multicast (GB200),
power-of-two world size, torch symmetric memory + Triton, TE (for the MXFP8 ``TransformerConfig``),
FlashInfer with the routed-MXFP8 API (Megatron's own import gate, ``flashinfer_mxfp8.py:12-25``;
present in 0.6.8 and 0.7.0.post1). The NVFP4 cases need FlashInfer >= 0.7.0
(``flashinfer_nvfp4.py:28-31``) and skip otherwise. Launch:
``python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest <file> -v``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.activations import squared_relu
from megatron.core.inference.moe.flashinfer_mxfp8 import (
    _FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR,
    HAVE_FLASHINFER_ROUTED_MXFP8,
)
from megatron.core.inference.moe.flashinfer_nvfp4 import (
    _FLASHINFER_NVFP4_IMPORT_ERROR,
    HAVE_FLASHINFER_ROUTED_NVFP4,
)
from megatron.core.inference.utils import InferenceMode
from tests.unit_tests.test_utilities import Utils

# Max GPUs any test here needs (it uses all ranks of the launch as one EP group; 2 is the minimum).
REQUIRED_GPUS = 2

HIDDEN = 256
FFN = 256
TOPK = 2
CAPACITY = 128  # inference_flashinfer_mxfp8_token_capacity
PER_RANK_MAX_TOKENS = 256  # NVLS per-rank worst case; global_max = 256 * EP
DECODE_TOKENS = 4
PREFILL_TOKENS = 160
NUM_SPECULATIVE_TOKENS = 0
SENTINEL = 1000.0
TOLERANCE = {"mxfp8": 0.15, "nvfp4": 0.35}
RELU2 = 6  # flashinfer ActivationType.Relu2
# Power-of-two exponents: x in [-1.5, 1.5], |W1| <= 6/32, |W2| <= 6/64 -> outputs O(1).
EXP_X, EXP_FC1, EXP_FC2 = 2, 5, 6

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _num_experts(ep):
    return max(8, 2 * ep)


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs CUDA GPUs"
    ws = Utils.world_size
    if ws < REQUIRED_GPUS:
        return (
            f"needs >= {REQUIRED_GPUS} GPUs launched with torch.distributed.run "
            f"(WORLD_SIZE={ws}); the defect is a cross-EP-rank interaction"
        )
    if ws & (ws - 1):
        return f"the NVLS dispatcher's Triton barrier needs a power-of-two EP size (got {ws})"
    if torch.cuda.get_device_capability()[0] != 10:
        return (
            "routed MXFP8 (TRT-LLM gen, get_trtllm_moe_sm100_module) and per-token NVFP4 "
            "(CuTe DSL) "
            f"MoE kernels are SM100-family only; device is SM{torch.cuda.get_device_capability()}"
        )
    if not HAVE_FLASHINFER_ROUTED_MXFP8:
        return (
            "Megatron's routed-MXFP8 import gate failed (needs flashinfer.mxfp8_quantize, "
            "shuffle_matrix_a, fused_moe.trtllm_fp8_block_scale_routed_moe / Fp8QuantizationType / "
            f"WeightLayout, utils.get_shuffle_matrix_sf_a_row_indices): "
            f"{_FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR!r}"
        )
    return None


def _grid(shape, exponent, gen):
    """E2M1-grid values times 2**-exponent with one +-6 * 2**-exponent in every 16-block (last dim).

    Every MXFP8 32-block and NVFP4 16-block then has amax 6 * 2**-exponent, so the MXFP8 scale is
    2**(-6 - exponent) (ceil or nearest e8m0 rule alike) and the codes E2M1 * 64 are exact in E4M3;
    the NVFP4 block scale is E4M3(448) and the codes are exact in E2M1.
    """
    numel = 1
    for s in shape:
        numel *= s
    vals = _E2M1[torch.randint(0, 8, (numel,), generator=gen)]
    sign = torch.randint(0, 2, (numel,), generator=gen).float() * 2 - 1
    v = (vals * sign).reshape(-1, 16)
    pos = torch.randint(0, 16, (v.shape[0],), generator=gen)
    pin = 6.0 * (torch.randint(0, 2, (v.shape[0],), generator=gen).float() * 2 - 1)
    v[torch.arange(v.shape[0]), pos] = pin
    return (v.reshape(shape) * 2.0**-exponent).to(torch.bfloat16)


def _controller_bounded_rows(is_decode_only, ep_size, token_capacity):
    """The controller's predicate, text_generation_controller.py:932-943 (max_requests chosen so
    that the static EP-wide bound fits the capacity, i.e. the feature is enabled)."""
    max_requests = CAPACITY // ep_size
    decode_token_upper_bound = max_requests * (NUM_SPECULATIVE_TOKENS + 1) * ep_size
    return (
        is_decode_only
        and token_capacity is not None
        and decode_token_upper_bound <= token_capacity
    )


def _deployment_config(backend, ep, num_experts):
    """The deployment the defect needs; constructing it shows config validation accepts it."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=FFN,
        moe_router_topk=TOPK,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_token_dispatcher_type="nvls",
        inference_flashinfer_mxfp8_token_capacity=CAPACITY,
        activation_func=squared_relu,
    )
    if backend == "mxfp8":
        kwargs.update(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True)
    else:
        kwargs.update(inference_moe_nvfp4_recipe="per_token")
    return TransformerConfig(**kwargs)


def _prepare_mxfp8(weight_stack):
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    quantized = [MXFP8Tensor.from_bf16(w, backend="triton") for w in weight_stack]
    stacked = MXFP8Tensor(
        data=torch.stack([q.data for q in quantized]).contiguous(),
        scale=torch.stack([q.scale for q in quantized]).contiguous(),
        backend="triton",
        dtype=torch.bfloat16,
    )
    return quantized, prepare_routed_mxfp8_weights(stacked)


def _assert_mxfp8_exact(quantized, weight_stack, exponent):
    """Harness check: MXFP8 storage of the grid weights is lossless (data = w * 2**(6+k), every
    scale byte = 127 - 6 - k; K = 256 and rows % 128 == 0 leave no padding bytes)."""
    for q, w in zip(quantized, weight_stack):
        exact = w.float() * 2.0 ** (6 + exponent)
        assert torch.equal(q.data.float(), exact), "MXFP8 weight not exact"
        assert torch.all(q.scale.view(torch.uint8) == 127 - 6 - exponent), "MXFP8 scale != 2^(-6-k)"


def _warm_up_kernels(env):
    """Build / load FlashInfer JIT modules and cubins outside the multi-rank sections."""
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        flashinfer_routed_mxfp8_moe,
        quantize_routed_mxfp8_input,
    )

    gen = torch.Generator().manual_seed(7)
    tokens = 16
    x = _grid((tokens, HIDDEN), EXP_X, gen).cuda()
    first = env.rank * env.local + torch.arange(tokens) % env.local
    second = env.rank * env.local + (torch.arange(tokens) + 1) % env.local  # distinct (local >= 2)
    routing = torch.stack([first, second], dim=1).cuda()
    probs = torch.full((tokens, TOPK), 0.5, device="cuda")
    # FlashInfer's MXFP8 input quantizer is exact on the grid (harness check for the reference).
    data, scale = quantize_routed_mxfp8_input(x, env.mx_fc1.padded_cols)
    assert torch.equal(data.float(), x.float() * 2.0 ** (6 + EXP_X)), "MXFP8 input not exact"
    assert torch.all(scale == 127 - 6 - EXP_X), "MXFP8 input scale not 2^(-6-k)"
    for bounds in ({}, dict(token_capacity=8, use_bounded_rows=True)):
        flashinfer_routed_mxfp8_moe(
            x,
            routing,
            probs,
            env.mx_fc1,
            env.mx_fc2,
            num_experts=env.num_experts,
            local_expert_offset=env.rank * env.local,
            activation_type=RELU2,
            **bounds,
        )
    if env.nvfp4 is not None:
        from megatron.core.inference.moe.flashinfer_nvfp4 import flashinfer_routed_nvfp4_moe

        flashinfer_routed_nvfp4_moe(
            x,
            routing,
            probs,
            env.nvfp4,
            num_experts=env.num_experts,
            local_expert_offset=env.rank * env.local,
            activation_type=RELU2,
        )
    torch.cuda.synchronize()


@pytest.fixture(scope="module")
def env():
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.parallel_state import get_expert_model_parallel_group
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
    ep_group = get_expert_model_parallel_group()
    rank, ep = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    try:
        # Create the EP communicator while all ranks are in lockstep (store timeout is 1 min).
        dist.barrier(group=ep_group)
        NVLSAllGatherVDispatcher._delete_buffers()
        try:
            NVLSAllGatherVDispatcher.allocate_buffers(
                per_rank_worst_case_token_count=PER_RANK_MAX_TOKENS,
                topk=TOPK,
                hidden_size=HIDDEN,
                ep_group=ep_group,
            )
        except RuntimeError as exc:
            pytest.skip(f"NVLS symmetric memory unavailable: {exc}")
        # Multicast must work on every rank (pattern: d1 test, _require_nvls).
        handle = NVLSAllGatherVDispatcher._symm_rsv["handle"]
        multicast = handle is not None and bool(getattr(handle, "multicast_ptr", 0))
        ok = torch.tensor([1.0 if multicast else 0.0], device="cuda")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item() != 1.0:
            pytest.skip("NVLS multicast pointer unavailable on at least one rank")
        # Rows past the live count are quantized but routed -1; keep them finite.
        NVLSAllGatherVDispatcher._symm_agv_hidden["tensor"].zero_()
        NVLSAllGatherVDispatcher._symm_agv_probs["tensor"].zero_()

        num_experts = _num_experts(ep)
        local = num_experts // ep
        gen = torch.Generator().manual_seed(1234)  # identical global weights on every rank
        fc1 = _grid((num_experts, FFN, HIDDEN), EXP_FC1, gen).cuda()
        fc2 = _grid((num_experts, HIDDEN, FFN), EXP_FC2, gen).cuda()
        own = slice(rank * local, (rank + 1) * local)
        q1, mx_fc1 = _prepare_mxfp8(fc1[own])
        q2, mx_fc2 = _prepare_mxfp8(fc2[own])
        _assert_mxfp8_exact(q1, fc1[own], EXP_FC1)
        _assert_mxfp8_exact(q2, fc2[own], EXP_FC2)
        nvfp4 = None
        if HAVE_FLASHINFER_ROUTED_NVFP4:
            from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights

            nvfp4 = prepare_routed_nvfp4_weights(
                fc1[own].contiguous(), fc2[own].contiguous()
            )
        env = SimpleNamespace(
            rank=rank,
            ep=ep,
            ep_group=ep_group,
            num_experts=num_experts,
            local=local,
            fc1=fc1,
            fc2=fc2,
            mx_fc1=mx_fc1,
            mx_fc2=mx_fc2,
            nvfp4=nvfp4,
        )
        # A cold FlashInfer build takes minutes and is serialized across processes; take turns.
        for turn in range(ep):
            if turn == rank:
                _warm_up_kernels(env)
            dist.barrier()
        yield env
    finally:
        torch.cuda.synchronize()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _reset_inference_mode():
    InferenceMode.unset_active()
    yield
    InferenceMode.unset_active()


def _skip_unless_backend(backend):
    if backend == "nvfp4" and not HAVE_FLASHINFER_ROUTED_NVFP4:
        import flashinfer

        pytest.skip(
            "Megatron's per-token NVFP4 MoE needs FlashInfer >= 0.7.0 (flashinfer_nvfp4.py:28-31); "
            f"installed {flashinfer.__version__}: {_FLASHINFER_NVFP4_IMPORT_ERROR!r}"
        )


def _experts_view(env, backend):
    """Stand-in ``self`` for InferenceGroupedMLP._flashinfer_forward (production attributes)."""
    from flashinfer.fused_moe.core import ActivationType

    return SimpleNamespace(
        _activation_clamp_scale=None,
        _uses_mxfp8_weights=backend == "mxfp8",
        _nvfp4_weights=env.nvfp4 if backend == "nvfp4" else None,
        _fc1_weight=env.mx_fc1 if backend == "mxfp8" else None,
        _fc2_weight=env.mx_fc2 if backend == "mxfp8" else None,
        num_local_experts=env.local,
        ep_group=env.ep_group,
        _flashinfer_activation_type=ActivationType.Relu2,
        _nvls_dispatcher=True,
        _flashinfer_mxfp8_token_capacity=CAPACITY,
    )


def _global_tokens(env, total, seed):
    """Same global batch on every rank. Slot 0 -> an expert of rank 0 (weight 10/16..14/16);
    slot 1 -> an expert of another rank."""
    gen = torch.Generator().manual_seed(seed)
    hidden = _grid((total, HIDDEN), EXP_X, gen)
    slot0 = torch.randint(0, env.local, (total,), generator=gen)
    slot1 = torch.randint(env.local, env.num_experts, (total,), generator=gen)
    routing = torch.stack([slot0, slot1], dim=1).to(torch.int64)
    p0 = torch.randint(10, 15, (total,), generator=gen).float() / 16
    probs = torch.stack([p0, 1 - p0], dim=1).to(torch.float32)
    return hidden, routing, probs


def _reference_moe(x, routing, probs, fc1, fc2):
    out = torch.zeros(x.shape[0], fc2.shape[1], dtype=torch.float64, device=x.device)
    xd = x.double()
    for slot in range(routing.shape[1]):
        for expert in routing[:, slot].unique().tolist():
            rows = routing[:, slot] == expert
            h = torch.relu(xd[rows] @ fc1[expert].double().t()).square()
            out[rows] += probs[rows, slot : slot + 1].double() * (h @ fc2[expert].double().t())
    return out


def _sync():
    torch.cuda.synchronize()
    dist.barrier()


def _run_step(env, backend, tokens_per_rank, decode_only, rsv_fill, seed):
    """One MoE step on every rank; returns every rank's per-token stats (identical on all ranks)."""
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    hidden, routing, probs = _global_tokens(env, sum(tokens_per_rank), seed)
    start = sum(tokens_per_rank[: env.rank])
    n = tokens_per_rank[env.rank]
    x = hidden[start : start + n].cuda()
    r = routing[start : start + n].cuda()
    p = probs[start : start + n].cuda()
    bounded = _controller_bounded_rows(decode_only[env.rank], env.ep, CAPACITY)

    config = _deployment_config(backend, env.ep, env.num_experts)
    dispatcher = NVLSAllGatherVDispatcher(
        num_local_experts=env.local,
        local_expert_indices=[env.rank * env.local + i for i in range(env.local)],
        config=config,
        pg_collection=get_default_pg_collection(),
        runs_metadata_sync=True,
    )
    experts = _experts_view(env, backend)
    rsv = NVLSAllGatherVDispatcher._get_rsv_tensor()

    _sync()  # no peer may still be reading this rank's RSV from the previous step
    rsv.fill_(rsv_fill)  # what a previous step left behind
    _sync()
    with torch.no_grad(), InferenceMode.active():
        InferenceMode.set_bounded_mxfp8_rows(bounded)
        h, pr = dispatcher.dispatch_preprocess(x, r, p)
        h, pr = dispatcher.token_dispatch(h, pr)
        expert_out, _ = InferenceGroupedMLP._flashinfer_forward(
            experts, h, dispatcher.routing_map, pr
        )
        assert expert_out is rsv, "expert output must be written into the RSV buffer"
        out = dispatcher.combine_postprocess(dispatcher.token_combine(expert_out))
    _sync()

    ref = _reference_moe(x, r, p, env.fc1, env.fc2)
    diff = out.double() - ref
    rel = diff.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
    stats = dict(
        rank=env.rank,
        start=start,
        n=n,
        bounded=bounded,
        rel=rel.cpu().tolist(),
        offset=diff.mean(dim=-1).cpu().tolist(),
    )
    gathered = [None] * env.ep
    dist.all_gather_object(gathered, stats)
    return sorted(gathered, key=lambda s: s["rank"])


def _tokens(gathered):
    """(global AGV row, rank, rel error, mean offset) for every token of every rank."""
    return [
        (g["start"] + i, g["rank"], rel, off)
        for g in gathered
        for i, (rel, off) in enumerate(zip(g["rel"], g["offset"]))
    ]


def _ranges(rows):
    rows = sorted(rows)
    out, begin, prev = [], None, None
    for row in rows:
        if begin is None:
            begin = prev = row
        elif row == prev + 1:
            prev = row
        else:
            out.append(f"[{begin}, {prev + 1})")
            begin = prev = row
    if begin is not None:
        out.append(f"[{begin}, {prev + 1})")
    return ", ".join(out)


def _assert_control(gathered, backend, label):
    tol = TOLERANCE[backend]
    bad = [t for t in _tokens(gathered) if not t[2] <= tol]
    worst = max(t[2] for t in _tokens(gathered))
    assert not bad, (
        f"harness/reference invalid ({backend}, {label}): {len(bad)} tokens exceed relative L2 "
        f"error {tol} (worst {worst:.4g}) on ranks {sorted({t[1] for t in bad})}, rows "
        f"{_ranges(t[0] for t in bad)}"
    )


@pytest.mark.parametrize("backend", ["mxfp8", "nvfp4"])
def test_control_all_ranks_unbounded_match_reference(env, backend):
    """Every rank runs a non-decode step, so no rank bounds its rows; same token layout as the
    DEFECT test (rank 0 has 4 tokens, peers 160 each, peer rows run past the capacity)."""
    _skip_unless_backend(backend)
    tokens_per_rank = [DECODE_TOKENS] + [PREFILL_TOKENS] * (env.ep - 1)
    gathered = _run_step(
        env, backend, tokens_per_rank, [False] * env.ep, rsv_fill=SENTINEL, seed=11
    )
    assert [g["bounded"] for g in gathered] == [False] * env.ep
    assert sum(tokens_per_rank) > CAPACITY
    _assert_control(gathered, backend, "all ranks unbounded")


@pytest.mark.parametrize("backend", ["mxfp8", "nvfp4"])
def test_control_all_ranks_bounded_within_capacity_match_reference(env, backend):
    """Every rank is decode-only and bounded, every token sits inside the capacity, and the RSV
    rows past it hold the sentinel: the bounded prefix is correct when all ranks agree."""
    _skip_unless_backend(backend)
    tokens_per_rank = [DECODE_TOKENS] * env.ep
    gathered = _run_step(env, backend, tokens_per_rank, [True] * env.ep, rsv_fill=SENTINEL, seed=12)
    assert [g["bounded"] for g in gathered] == [True] * env.ep
    assert sum(tokens_per_rank) <= CAPACITY
    _assert_control(gathered, backend, "all ranks bounded, tokens within capacity")


@pytest.mark.parametrize("stale", ["sentinel", "zeros"])
@pytest.mark.parametrize("backend", ["mxfp8", "nvfp4"])
def test_defect_bounded_rank_corrupts_peer_rows_past_capacity(env, backend, stale):
    """Rank 0 decode-only (bounded), ranks >= 1 prefill (unbounded); peer tokens at global rows
    >= capacity must still equal the reference."""
    _skip_unless_backend(backend)
    tokens_per_rank = [DECODE_TOKENS] + [PREFILL_TOKENS] * (env.ep - 1)
    decode_only = [True] + [False] * (env.ep - 1)
    fill = SENTINEL if stale == "sentinel" else 0.0
    gathered = _run_step(env, backend, tokens_per_rank, decode_only, rsv_fill=fill, seed=13)

    # Harness preconditions (not the defect): mixed per-rank decisions, peer rows past capacity.
    assert [g["bounded"] for g in gathered] == decode_only, "controller predicate not as intended"
    tokens = _tokens(gathered)
    past = [t for t in tokens if t[0] >= CAPACITY]
    assert past and all(t[1] != 0 for t in past), "no peer token placed past the capacity"

    tol = TOLERANCE[backend]
    bad = [t for t in tokens if not t[2] <= tol]
    inside = [t[2] for t in tokens if t[0] < CAPACITY]
    what = ""
    if bad:
        mean_offset = sum(t[3] for t in bad) / len(bad)
        what = (
            f"mean output offset {mean_offset:.4g} = the {SENTINEL:g} sentinel left in rank 0's RSV"
            if stale == "sentinel"
            else "rank 0's expert contributions are missing (its RSV rows past the capacity are 0)"
        )
    assert not bad, (
        f"DEFECT s1: [{backend}, RSV pre-fill={stale}] EP rank 0 ran a bounded decode step "
        f"(capacity {CAPACITY}; its {DECODE_TOKENS} tokens fit) while ranks "
        f"{list(range(1, env.ep))} ran unbounded prefill steps. {len(bad)} of {len(past)} peer "
        f"tokens at global AGV rows >= "
        f"{CAPACITY} differ from the per-token MoE reference (relative L2 error up to "
        f"{max(t[2] for t in bad):.4g}, tolerance {tol}) on ranks {sorted({t[1] for t in bad})}, "
        f"rows {_ranges(t[0] for t in bad)}; {what}. Tokens at rows < {CAPACITY}: max error "
        f"{max(inside):.4g}. The bounded rank writes only RSV[:capacity] "
        f"(flashinfer_{backend}.py) and token_combine sums its stale rows into the peers' tokens; "
        f"the bounded-row choice needs EP-wide agreement under NVLS."
    )
