# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected bug s6: MTP MoE layers reuse stale NVLS EP step metadata.

This is defect d3's root cause under the NVLS dispatcher at EP > 1 (read at ed9c2c8c1):

* ``NVLSAllGatherVDispatcher._step_metadata`` holds ``[valid_tokens, rank_token_offset,
  ep_max_tokens]`` for the whole step. Only the dispatcher with ``_runs_metadata_sync`` refreshes
  it (``token_dispatcher_inference.py:595-596`` -> ``update_metadata`` -> the fused metadata
  kernel, ``megatron/core/inference/moe/metadata.py:47``). ``set_moe_metadata_sync``
  (``megatron/core/inference/utils.py:219-235``) gives that flag to the first MoE layer of the
  module walk, which is a decoder layer, never the MTP layer.
* MTP forwards (draft depths, KV-seed/commit pass, dummy passes) run only MTP layers, with R
  request-shaped rows per rank, after a main forward with V >= R tokens per rank. The MTP layer's
  all-gather-v (``:603-620``) and reduce-scatter-v (``:675-682``) then use the main forward's
  ``rank_token_offset`` / ``ep_max_tokens``, and its experts the main forward's ``valid_tokens``.

What the kernels take from where (``megatron/core/inference/communication/torch_symm_triton/
variable_collectives.py``):

* AGV local row count = ``input_tensor_0.shape[0]`` (``:712``), i.e. R. Placement offset and the
  CTA early-exit bound come from the metadata tensors (``:416-421``).
* RSV local row count = ``output_tensor.shape[0]`` (``:322``); the output is allocated with
  ``self._local_tokens = hidden_states.shape[0]`` (``token_dispatcher_inference.py:548``, ``:659-664``).

So by reading, nothing is read past row R. Both collectives use the *same* stale offset, so each
rank's R rows land at the main forward's offset and are read back from there, and the experts'
stale ``valid_tokens = sum(V)`` window covers them. The round trip is therefore expected to be
correct on ``main``; the defect is that the MTP layer's metadata, gathered layout and expert window
describe the main forward: the experts process ``sum(V - R)`` stale rows left by the previous MoE
layer, and a fix that refreshes only ``valid_tokens`` (as a d3-only fix might) would leave rank 1's
rows outside the expert window.

Tests (2 ranks, EP=2, the real dispatcher API in the order ``MoELayer.forward`` calls it, with the
experts replaced by an identity over the rows ``[0, valid_tokens)``; rows past it are NaN):

* ``test_control_fresh_metadata_round_trip`` -- both forwards go through a metadata-refreshing
  dispatcher: the gathered prefix is exactly the real rows, the metadata is fresh, and every rank
  gets back ``EP x`` its own rows. Must PASS.
* ``test_defect_mtp_dispatch_window_holds_only_real_rows`` -- the MTP dispatcher is flagged by the
  real ``set_moe_metadata_sync``: asserts fresh metadata and a compact gathered window. FAILS on
  ``main`` with ``DEFECT s6: ...``.
* ``test_defect_mtp_round_trip_reads_only_real_rows`` -- runtime check of the reading above on the
  stale path: the local inputs are views into canary-filled buffers, and the test asserts no canary
  reaches the gathered buffers and the combined outputs are exactly ``EP x`` the R real rows.
  Expected to PASS on ``main``; a FAIL means the stale metadata corrupts data (out-of-bounds reads
  or wrong outputs), which is worse than the reading concluded.

Requirements: 2 GPUs (``REQUIRED_GPUS = 2``), SM >= 9 with NVLS multicast (H100 SXM/NVSwitch),
torch symmetric memory and Triton. Skips on every rank when NVLS is unavailable.
Launch: ``python -m torch.distributed.run --standalone --nproc-per-node=2 -m pytest <file> -v``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils

REQUIRED_GPUS = 2

EP = 2
HIDDEN = 64  # bf16 rows of 128 bytes: the 128-bit AGV/RSV paths
TOPK = 2
NUM_EXPERTS = 4
PER_RANK_MAX = 32  # per_rank_worst_case_token_count; AGV/RSV grid = 32 CTAs

# Main (verification) forward: V tokens per EP rank. MTP forward: R <= V rows per rank, as in the
# engine (draft depths: padded request count; commit pass: <= the main forward's tokens; dummies:
# 1 row vs a dummy main forward of num_speculative_tokens + 1 tokens).
MAIN_TOKENS = (8, 6)
MTP_ROWS = (3, 2)

# Canaries for rows [R, PER_RANK_MAX) of the local MTP input buffers. Real rows are randn
# (|x| < 6), expert ids in [0, NUM_EXPERTS) and probs in [0, 1).
HIDDEN_CANARY = -4096.0
ROUTING_CANARY = 77
PROB_CANARY = -7.0

MAIN_SEED = 11
MTP_SEED = 22


def _offsets(counts):
    return [sum(counts[:r]) for r in range(len(counts))]


def _fresh_metadata(counts, rank):
    """[valid_tokens, rank_token_offset, ep_max_tokens] the metadata kernel computes for counts."""
    return [sum(counts), _offsets(counts)[rank], max(counts)]


# --------------------------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------------------------- #


def _all_min(flag: bool) -> bool:
    t = torch.tensor([1.0 if flag else 0.0], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item() == 1.0)


def _require_nvls(ep_group):
    """Allocate the NVLS dispatcher buffers (collective) and skip on every rank unless usable.

    Detection follows test_defect_d1_fused_tp_residual.py::_require_nvls: SM >= 9, a
    symmetric-memory handle and a non-zero multicast pointer on every rank.
    """
    from megatron.core.inference.communication.torch_symm_triton.utils import (
        is_device_nvls_capable,
    )
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    capable = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
    if not _all_min(capable):
        pytest.skip("NVLS dispatcher needs SM >= 9 on every rank")

    error = None
    try:
        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=PER_RANK_MAX,
            topk=TOPK,
            hidden_size=HIDDEN,
            ep_group=ep_group,
        )
    except RuntimeError as e:  # raised when any symmetric buffer failed to initialize
        error = f"{type(e).__name__}: {e}"

    bufs = [
        getattr(NVLSAllGatherVDispatcher, name, None)
        for name in ("_symm_agv_hidden", "_symm_agv_routing", "_symm_agv_probs", "_symm_rsv",
                     "_symm_metadata")
    ]
    multicast = error is None and all(
        b is not None
        and b["handle"] is not None
        and bool(getattr(b["handle"], "multicast_ptr", 0))
        for b in bufs
    )
    if not _all_min(multicast):
        pytest.skip(
            "NVLS unavailable (needs symmetric memory with a multicast pointer on every rank; "
            f"error={error})"
        )

    # Start from known buffer contents so nothing seen later is left over from another test.
    for b in bufs[:4]:
        b["tensor"].zero_()
    torch.cuda.synchronize()
    dist.barrier()


@pytest.fixture
def nvls_ep2(monkeypatch):
    """EP=2 model parallel state, fresh MoE metadata globals and allocated NVLS buffers."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if Utils.world_size < EP or Utils.world_size % EP != 0:
        pytest.skip(f"requires a world size divisible by EP={EP} (world size {Utils.world_size})")

    from megatron.core import parallel_state
    from megatron.core.inference import utils as inference_utils
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher,
    )

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=EP)

    # Process-global state touched below; monkeypatch restores it afterwards.
    monkeypatch.setattr(inference_utils, "moe_layer_cache", None)
    monkeypatch.setattr(inference_utils, "_moe_metadata_sync_initialized", False)
    monkeypatch.setattr(InferenceAllGatherDispatcherBase, "_valid_tokens_tensor", None)
    monkeypatch.setattr(InferenceAllGatherDispatcherBase, "_host_valid_tokens_estimate", None)
    monkeypatch.setattr(
        NVLSAllGatherVDispatcher,
        "_per_rank_worst_case_token_count",
        NVLSAllGatherVDispatcher._per_rank_worst_case_token_count,
    )
    # Standalone use: no context has wired the routing-mask count (see token_dispatch).
    monkeypatch.setattr(NVLSAllGatherVDispatcher, "_real_token_count_tensor", None)

    SymmetricMemoryManager.destroy()
    NVLSAllGatherVDispatcher._delete_buffers()
    try:
        # What DynamicInferenceContext does at init (dynamic_context.py:920-931).
        InferenceAllGatherDispatcherBase.allocate_valid_tokens_tensor()
        ep_group = parallel_state.get_expert_model_parallel_group()
        _require_nvls(ep_group)
        yield SimpleNamespace(
            inference_utils=inference_utils,
            ep_rank=dist.get_rank(ep_group),
            ep_group=ep_group,
        )
    finally:
        torch.cuda.synchronize()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


def _moe_config():
    """Inference MoE config (as in test_defect_d3_mtp_stale_valid_tokens.py::_moe_config)."""
    from megatron.core.activations import squared_relu
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        activation_func=squared_relu,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_dtype="fp32",
        expert_model_parallel_size=EP,
        inference_moe_token_dispatcher_type="nvls",
    )


def _dispatcher(ep_rank, runs_metadata_sync=True):
    """NVLS dispatcher as built in test_moe_dispatching_and_routing.py::TestNVLSAllGatherVDispatcher."""
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    num_local = NUM_EXPERTS // EP
    return NVLSAllGatherVDispatcher(
        num_local_experts=num_local,
        local_expert_indices=[ep_rank * num_local + i for i in range(num_local)],
        config=_moe_config(),
        pg_collection=get_default_pg_collection(),
        runs_metadata_sync=runs_metadata_sync,
    )


def _dispatchers_flagged_by_set_moe_metadata_sync(env):
    """Decoder and MTP dispatchers whose flags the real set_moe_metadata_sync assigns.

    The cache is seeded in the GPTModel/HybridModel module-walk order (decoder before mtp,
    gpt_model.py:267/280, hybrid_model.py:392/420), as in test_defect_d3.
    """
    decoder = _dispatcher(env.ep_rank)
    mtp = _dispatcher(env.ep_rank)
    env.inference_utils.moe_layer_cache = [
        SimpleNamespace(_inference_token_dispatcher=decoder),
        SimpleNamespace(_inference_token_dispatcher=mtp),
    ]
    env.inference_utils.set_moe_metadata_sync(None)
    assert decoder._runs_metadata_sync and not mtp._runs_metadata_sync, (
        "harness: set_moe_metadata_sync did not flag only the first (decoder) dispatcher"
    )
    return decoder, mtp


def _global_rows(seed, counts):
    """Rank-major rows of one forward, identical on every rank (CPU generator)."""
    g = torch.Generator().manual_seed(seed)
    total = sum(counts)
    hidden = torch.randn(total, HIDDEN, generator=g).to(torch.bfloat16).cuda()
    routing = torch.randint(0, NUM_EXPERTS, (total, TOPK), generator=g).cuda()
    probs = torch.rand(total, TOPK, generator=g).cuda()
    return hidden, routing, probs


def _local_rows(rows, counts, rank):
    start = _offsets(counts)[rank]
    return tuple(t[start : start + counts[rank]] for t in rows)


def _canary_views(rows, counts, rank):
    """This rank's rows as views of the first R rows of canary-filled PER_RANK_MAX-row buffers."""
    h, r, p = _local_rows(rows, counts, rank)
    n = counts[rank]
    base_h = torch.full((PER_RANK_MAX, 1, HIDDEN), HIDDEN_CANARY, dtype=torch.bfloat16, device="cuda")
    base_r = torch.full((PER_RANK_MAX, TOPK), ROUTING_CANARY, dtype=torch.int64, device="cuda")
    base_p = torch.full((PER_RANK_MAX, TOPK), PROB_CANARY, dtype=torch.float32, device="cuda")
    base_h[:n, 0] = h
    base_r[:n] = r
    base_p[:n] = p
    return (base_h[:n], base_r[:n], base_p[:n]), (base_h, base_r, base_p)


def _step_metadata():
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    return [int(v) for v in NVLSAllGatherVDispatcher._step_metadata.tolist()]


def _valid_tokens():
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
    )

    return int(InferenceAllGatherDispatcherBase._valid_tokens().item())


def _moe_layer_round_trip(dispatcher, hidden, routing, probs):
    """One inference MoE layer: dispatch, identity experts over [0, valid_tokens), combine.

    The dispatcher methods are called in the order MoELayer.forward calls them. The experts read
    the shared valid-token count exactly as InferenceGroupedMLP does (experts.py:1723) and produce
    output rows only inside that window; rows past it are NaN, so a combine that reads a row the
    experts did not produce returns NaN.
    """
    h, p = dispatcher.dispatch_preprocess(hidden, routing, probs)
    h, p = dispatcher.token_dispatch(h, p)
    h, _, p = dispatcher.dispatch_postprocess(h, p)
    torch.cuda.synchronize()
    seen = SimpleNamespace(
        metadata=_step_metadata(),
        valid=_valid_tokens(),
        gathered_h=h.clone(),
        gathered_r=dispatcher.routing_map.clone(),
        gathered_p=p.clone(),
    )
    expert_out = torch.full(h.shape, float("nan"), dtype=torch.float32, device=h.device)
    expert_out[: seen.valid] = h[: seen.valid].float()
    out = dispatcher.combine_preprocess(expert_out)
    out = dispatcher.token_combine(out)
    seen.combined = dispatcher.combine_postprocess(out)
    torch.cuda.synchronize()
    return seen


def _run_main_then_mtp(env, main_dispatcher, mtp_dispatcher):
    rank = env.ep_rank
    main_rows = _global_rows(MAIN_SEED, MAIN_TOKENS)
    mtp_rows = _global_rows(MTP_SEED, MTP_ROWS)

    # Main forward: plain (canary-free) local inputs.
    mh, mr, mp = _local_rows(main_rows, MAIN_TOKENS, rank)
    main = _moe_layer_round_trip(main_dispatcher, mh.unsqueeze(1).clone(), mr.clone(), mp.clone())

    # MTP forward: local inputs are views of canary-filled buffers.
    (h, r, p), bases = _canary_views(mtp_rows, MTP_ROWS, rank)
    mtp = _moe_layer_round_trip(mtp_dispatcher, h, r, p)
    dist.barrier()
    return SimpleNamespace(
        rank=rank,
        main=main,
        mtp=mtp,
        main_rows=main_rows,
        main_local=(mh, mr, mp),
        mtp_rows=mtp_rows,
        mtp_local=_local_rows(mtp_rows, MTP_ROWS, rank),
        bases=bases,
    )


def _row_positions(gathered_h, rows):
    """Index of each of `rows` in the gathered hidden buffer (-1 if absent)."""
    hits = (gathered_h.unsqueeze(0) == rows.unsqueeze(1)).all(dim=-1)  # [len(rows), global_max]
    return [int(row.nonzero()[0]) if bool(row.any()) else -1 for row in hits]


def _gathered_prefix_matches(seen, rows):
    n = rows[0].shape[0]
    return (
        torch.equal(seen.gathered_h[:n], rows[0])
        and torch.equal(seen.gathered_r[:n], rows[1])
        and torch.equal(seen.gathered_p[:n], rows[2])
    )


def _canaries_in_gathered(seen):
    return dict(
        hidden_rows=int((seen.gathered_h == HIDDEN_CANARY).any(dim=-1).sum()),
        routing_rows=int((seen.gathered_r == ROUTING_CANARY).any(dim=-1).sum()),
        prob_rows=int((seen.gathered_p == PROB_CANARY).any(dim=-1).sum()),
    )


def _canary_tails_intact(bases, n):
    base_h, base_r, base_p = bases
    return (
        bool((base_h[n:] == HIDDEN_CANARY).all())
        and bool((base_r[n:] == ROUTING_CANARY).all())
        and bool((base_p[n:] == PROB_CANARY).all())
    )


def _expected_combined(local_hidden):
    # Identity experts on both ranks, summed by the RSV: exactly EP x each row (fp32 sum of equal
    # bf16 values, cast back to bf16).
    return (local_hidden.float() * EP).to(torch.bfloat16).view(-1, 1, HIDDEN)


# --------------------------------------------------------------------------------------------- #
# CONTROL (must pass on main)
# --------------------------------------------------------------------------------------------- #


def test_control_fresh_metadata_round_trip(nvls_ep2):
    """Each forward refreshes the metadata: gathered prefix, metadata and round trip are exact."""
    first_layer = _dispatcher(nvls_ep2.ep_rank, runs_metadata_sync=True)
    obs = _run_main_then_mtp(nvls_ep2, first_layer, first_layer)
    rank = obs.rank

    # Main forward (V tokens per rank).
    assert obs.main.metadata == _fresh_metadata(MAIN_TOKENS, rank), (
        f"harness: main-forward metadata {obs.main.metadata} != "
        f"{_fresh_metadata(MAIN_TOKENS, rank)}"
    )
    assert _gathered_prefix_matches(obs.main, obs.main_rows), (
        "harness: main-forward AGV prefix != the rank-major rows of all EP ranks"
    )
    assert torch.equal(obs.main.combined, _expected_combined(obs.main_local[0])), (
        "harness: main-forward combine != EP x the local rows"
    )

    # MTP-shaped forward (R rows per rank) with fresh metadata: what a fix must produce.
    assert obs.mtp.metadata == _fresh_metadata(MTP_ROWS, rank), (
        f"harness: refreshed MTP metadata {obs.mtp.metadata} != {_fresh_metadata(MTP_ROWS, rank)}"
    )
    assert obs.mtp.valid == sum(MTP_ROWS)
    assert _gathered_prefix_matches(obs.mtp, obs.mtp_rows), (
        f"harness: refreshed MTP AGV prefix != real rows; positions "
        f"{_row_positions(obs.mtp.gathered_h, obs.mtp_rows[0])}"
    )
    assert _canaries_in_gathered(obs.mtp) == dict(hidden_rows=0, routing_rows=0, prob_rows=0), (
        f"harness: canaries in the gathered buffers: {_canaries_in_gathered(obs.mtp)}"
    )
    assert _canary_tails_intact(obs.bases, MTP_ROWS[rank]), "harness: canary tail modified"
    assert torch.equal(obs.mtp.combined, _expected_combined(obs.mtp_local[0])), (
        "harness: refreshed MTP combine != EP x the local rows"
    )


# --------------------------------------------------------------------------------------------- #
# DEFECT (must fail on main)
# --------------------------------------------------------------------------------------------- #


def test_defect_mtp_dispatch_window_holds_only_real_rows(nvls_ep2):
    """The MTP layer's metadata and expert window must describe the MTP forward's R rows."""
    decoder, mtp = _dispatchers_flagged_by_set_moe_metadata_sync(nvls_ep2)
    obs = _run_main_then_mtp(nvls_ep2, decoder, mtp)
    rank = obs.rank

    fresh = _fresh_metadata(MTP_ROWS, rank)
    n_real = sum(MTP_ROWS)
    positions = _row_positions(obs.mtp.gathered_h, obs.mtp_rows[0])
    in_window = sum(1 for pos in positions if 0 <= pos < obs.mtp.valid)
    stale_rows_in_window = obs.mtp.valid - in_window
    starts = _offsets(MTP_ROWS)
    by_rank = {r: positions[starts[r] : starts[r] + MTP_ROWS[r]] for r in range(EP)}
    expected_by_rank = {r: list(range(starts[r], starts[r] + MTP_ROWS[r])) for r in range(EP)}
    ok = (
        obs.mtp.metadata == fresh
        and obs.mtp.valid == n_real
        and _gathered_prefix_matches(obs.mtp, obs.mtp_rows)
    )
    assert ok, (
        f"DEFECT s6: NVLS EP={EP} MTP-layer dispatch of R={list(MTP_ROWS)} rows per rank after a "
        f"main forward of V={list(MAIN_TOKENS)} tokens per rank reused the main forward's step "
        f"metadata: [valid_tokens, rank_token_offset, ep_max_tokens] = {obs.mtp.metadata} on rank "
        f"{rank}, expected {fresh}. The MTP layer's dispatcher has _runs_metadata_sync="
        f"{mtp._runs_metadata_sync} (set_moe_metadata_sync), so the AGV placed each rank's rows at "
        f"the main forward's offsets (gathered row indices per rank {by_rank}, expected "
        f"{expected_by_rank}) and the experts' window [0, {obs.mtp.valid}) holds {stale_rows_in_window} stale rows "
        f"from the previous MoE layer besides the {in_window} real ones."
    )


def test_defect_mtp_round_trip_reads_only_real_rows(nvls_ep2):
    """Runtime check of the reading: on the stale path nothing past row R is read and the
    combined outputs are exactly the R real rows' outputs. Expected to PASS on main."""
    decoder, mtp = _dispatchers_flagged_by_set_moe_metadata_sync(nvls_ep2)
    obs = _run_main_then_mtp(nvls_ep2, decoder, mtp)
    rank = obs.rank
    n = MTP_ROWS[rank]

    canaries = _canaries_in_gathered(obs.mtp)
    assert canaries == dict(hidden_rows=0, routing_rows=0, prob_rows=0), (
        f"DEFECT s6: the MTP-layer all-gather-v (stale metadata {obs.mtp.metadata}) read local rows "
        f"past R={n}: canary rows found in the gathered buffers {canaries}."
    )
    assert _canary_tails_intact(obs.bases, n), (
        f"DEFECT s6: the MTP-layer dispatch wrote local input rows past R={n} (canary tail changed)."
    )
    expected = _expected_combined(obs.mtp_local[0])
    got = obs.mtp.combined
    same_shape = got.shape == expected.shape
    max_diff = (
        (got.float() - expected.float()).abs().nan_to_num(float("inf")).max().item()
        if same_shape
        else "n/a"
    )
    nonfinite_rows = int((~torch.isfinite(got.float())).reshape(got.shape[0], -1).any(-1).sum())
    assert same_shape and torch.equal(got, expected), (
        f"DEFECT s6: MTP-layer combine under stale metadata {obs.mtp.metadata} (valid_tokens "
        f"{obs.mtp.valid}) returned wrong rows on rank {rank}: shape {tuple(got.shape)} vs "
        f"{tuple(expected.shape)}, non-finite rows {nonfinite_rows} (rows the experts never "
        f"produced), max |diff| {max_diff}."
    )
