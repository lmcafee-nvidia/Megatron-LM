# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected bug s12: processed log-probs with top-p < 1 are not run-to-run deterministic.

``FlashInferSampling.log_probs_kernel`` (``megatron/core/inference/sampling/flashinfer_sampling.py:
144-194``) renormalizes the temperature-scaled softmax with
``flashinfer.sampling.top_k_renorm_probs`` and then ``flashinfer.sampling.top_p_renorm_probs``
(``:187-188``). The top-p call does not pass ``is_deterministic``, while every sampling call in the
same class passes ``deterministic=True`` (``:113-136``). FlashInfer's default is
``is_deterministic=False`` (``flashinfer/sampling.py:1742-1746`` at tag ``v0.6.16.post3``, the tested
container's version, identical at ``v0.7.0.post1``). For vocabularies of at least 2048 that selects
the AIR radix top-p kernel with float ``atomicAdd`` histograms
(``include/flashinfer/air_top_p.cuh:184-191, 271-275``, ``csrc/renorm.cu:43-55``). The histogram
sums decide the top-p threshold bucket (``air_top_p.cuh:367-385``), so a row whose cumulative mass
lies within rounding noise of ``top_p`` can keep a different token set on different runs, which
changes every kept log-prob of that row.

Found while building this repro (by reading FlashInfer ``v0.6.16.post3``): ``top_k_renorm_probs``
has no determinism switch at all. When the vocabulary does not fit one CTA's shared memory
(``> ~57.5K`` fp32 entries on H100: 227 KiB opt-in smem, ``topk.cuh:1855-1874``), each row is split
over several CTAs whose partial sums are combined with a float ``atomicAdd(&state->sum_topk, ...)``
(``topk.cuh:1701, 1796``). With three or more CTAs (vocab > ~115K, e.g. 128K/152K vocabularies) the
normalizer depends on arrival order. Megatron runs this stage for every filtered row, including
top-p-only rows (``top_k=vocab``), so passing ``is_deterministic=True`` to the top-p call alone does
not make large-vocabulary log-probs deterministic.

Harness: the real ``FlashInferSampling`` with a stand-in context carrying the GPU sampling view and
the real ``DynamicInferenceContext.active_sampling_filter_flags`` (pattern from
``tests/unit_tests/inference/contexts/test_dynamic_context.py::
test_flashinfer_no_op_filter_rows_get_exact_log_softmax``).

Needs one GPU and FlashInfer.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from megatron.core.inference.sampling_params import MIN_SAMPLING_TEMPERATURE

try:
    import flashinfer
    import flashinfer.sampling  # noqa: F401

    HAVE_FLASHINFER = True
except ImportError:
    flashinfer = None
    HAVE_FLASHINFER = False

REQUIRED_GPUS = 1

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="s12 repro needs one CUDA GPU"),
    pytest.mark.skipif(not HAVE_FLASHINFER, reason="s12 repro needs FlashInfer"),
]

# 32000 >= 2048 selects the AIR radix top-p kernel, and fits one top-k CTA on H100 (single-CTA
# top_k_renorm_probs is deterministic), so the top-p stage is the only suspect at this size.
SMALL_VOCAB = 32000
# 152064 (Qwen2) needs three top-k CTAs per row on H100: the multi-CTA float-atomic normalizer.
LARGE_VOCAB = 152064
REPEATS = 64


def _determinism_kwarg():
    """Name of top_p_renorm_probs's determinism switch in the installed FlashInfer, or None."""
    params = inspect.signature(flashinfer.sampling.top_p_renorm_probs).parameters
    for name in ("is_deterministic", "deterministic"):
        if name in params:
            return name
    return None


def _context(temperature, top_k, top_p):
    """Stand-in dynamic context: GPU sampling view + real active_sampling_filter_flags."""
    temperature = torch.as_tensor(temperature, dtype=torch.float32)
    top_k = torch.as_tensor(top_k, dtype=torch.int32)
    top_p = torch.as_tensor(top_p, dtype=torch.float32)
    n = temperature.numel()
    ctx = SimpleNamespace(
        gpu_view=SimpleNamespace(
            temperature=temperature.cuda(), top_k=top_k.cuda(), top_p=top_p.cuda()
        ),
        active_request_metadata={"temperature": temperature, "top_k": top_k, "top_p": top_p},
        total_request_count=n,
        paused_request_count=0,
    )
    ctx.active_sampling_filter_flags = lambda count=None: (
        DynamicInferenceContext.active_sampling_filter_flags(ctx, count)
    )
    return ctx


def _logits(rows, vocab, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # LM-like spread: a few thousand plausible tokens and a long tail.
    return torch.randn(rows, vocab, generator=g, device="cuda", dtype=torch.float32) * 3.0


def _scaled_probs(logits, temperature):
    t = torch.as_tensor(temperature, dtype=torch.float32, device="cuda")
    scaled = logits / t.clamp(min=MIN_SAMPLING_TEMPERATURE).unsqueeze(1)
    return scaled, torch.softmax(scaled, dim=-1)


def _repeat_diff(fn, repeats, rows=None):
    """Run fn() `repeats` times; return (#runs differing bitwise from the first, differing rows)."""
    first = fn().clone()
    bad_runs, bad_rows = 0, set()
    for _ in range(repeats - 1):
        out = fn()
        a, b = (first, out) if rows is None else (first[rows], out[rows])
        # Bitwise comparison that also treats -inf == -inf and NaN payloads exactly.
        diff = (a.view(torch.int32) != b.view(torch.int32)).any(dim=-1)
        if bool(diff.any()):
            bad_runs += 1
            idx = torch.nonzero(diff).flatten().tolist()
            bad_rows.update(idx if rows is None else [rows[i] for i in idx])
    return bad_runs, sorted(bad_rows)


def _stage_report(probs, top_k_safe, top_p_safe, repeats):
    """Run-to-run check of each FlashInfer stage alone, for attribution in failure messages."""
    fis = flashinfer.sampling
    k_runs, _ = _repeat_diff(lambda: fis.top_k_renorm_probs(probs, top_k_safe), repeats)
    renormed = fis.top_k_renorm_probs(probs, top_k_safe)
    p_runs, _ = _repeat_diff(lambda: fis.top_p_renorm_probs(renormed, top_p_safe), repeats)
    parts = [
        f"top_k_renorm_probs alone differed in {k_runs}/{repeats - 1} reruns",
        f"top_p_renorm_probs (default flags, as Megatron calls it) in {p_runs}/{repeats - 1}",
    ]
    kw = _determinism_kwarg()
    if kw is not None:
        d_runs, _ = _repeat_diff(
            lambda: fis.top_p_renorm_probs(renormed, top_p_safe, **{kw: True}), repeats
        )
        parts.append(f"top_p_renorm_probs({kw}=True) in {d_runs}/{repeats - 1}")
    else:
        parts.append("installed top_p_renorm_probs has no determinism parameter")
    return "; ".join(parts)


def _boundary_top_p(probs_after_topk, targets):
    """Per-row top_p placed exactly on a cumulative-mass boundary of the row's sorted probs.

    At such a p the AIR kernel's threshold decision `prev + hist >= total * p` compares two
    quantities that are equal in exact arithmetic, so float-atomic accumulation order decides it.
    """
    sorted_p, _ = torch.sort(probs_after_topk.double(), dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    total = cum[:, -1:]
    t = torch.as_tensor(targets, dtype=torch.float64, device=cum.device).unsqueeze(1)
    j = torch.searchsorted(cum / total, t).clamp(max=cum.shape[1] - 2)
    return (cum.gather(1, j) / total).squeeze(1).float().cpu()


def _small_vocab_case():
    """64 rows at SMALL_VOCAB: 16 boundary-tied top-p rows, 32 random top-p rows, 16 unfiltered."""
    rows = 64
    logits = _logits(rows, SMALL_VOCAB, seed=1234)
    temperature = torch.full((rows,), 1.0)
    top_k = torch.zeros(rows, dtype=torch.int32)  # 0 = top-k disabled (sentinel)
    top_p = torch.ones(rows)
    g = torch.Generator().manual_seed(5)
    tie_rows, rand_rows, free_rows = list(range(16)), list(range(16, 48)), list(range(48, 64))
    top_p[rand_rows] = 0.5 + 0.49 * torch.rand(len(rand_rows), generator=g)
    _, probs = _scaled_probs(logits, temperature)
    # Megatron feeds top-p the top-k renorm output; with k = vocab and one CTA that is a
    # deterministic renormalization, so the boundaries are computed on exactly that input.
    renormed_k = flashinfer.sampling.top_k_renorm_probs(
        probs, torch.full((rows,), SMALL_VOCAB, dtype=torch.int32, device="cuda")
    )
    targets = 0.5 + 0.45 * torch.rand(len(tie_rows), generator=g)
    top_p[tie_rows] = _boundary_top_p(renormed_k[tie_rows], targets)
    return logits, temperature, top_k, top_p, tie_rows + rand_rows, free_rows


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------


def test_control_record_flashinfer_top_p_renorm_signature(record_property):
    """CONTROL/probe: record the installed FlashInfer version and the top_p_renorm_probs
    parameters (the fix needs the determinism switch's name)."""
    params = list(inspect.signature(flashinfer.sampling.top_p_renorm_probs).parameters)
    version = getattr(flashinfer, "__version__", "unknown")
    msg = (
        f"s12 probe: flashinfer {version}; top_p_renorm_probs{tuple(params)}; "
        f"determinism kwarg = {_determinism_kwarg()!r}; top_k_renorm_probs"
        f"{tuple(inspect.signature(flashinfer.sampling.top_k_renorm_probs).parameters)}"
    )
    record_property("s12_probe", msg)
    print(msg)  # pyproject addopts include -s
    assert "probs" in params and "top_p" in params


def test_control_unfiltered_rows_exact_and_repeatable():
    """CONTROL: in a mixed batch, rows with no top-k/top-p bypass the renorm kernels and are
    bitwise the log_softmax of the scaled logits on every run. Validates the harness."""
    logits, temperature, top_k, top_p, _, free_rows = _small_vocab_case()
    sampler = FlashInferSampling(SMALL_VOCAB, torch.Generator(device="cuda"))
    ctx = _context(temperature, top_k, top_p)
    assert ctx.active_sampling_filter_flags() == (True, False)  # top-p-only batch

    scaled, _ = _scaled_probs(logits, temperature)
    expected = torch.log_softmax(scaled, dim=-1)
    out = sampler.log_probs_kernel(logits, ctx)
    assert torch.equal(out[free_rows], expected[free_rows])
    bad_runs, _ = _repeat_diff(lambda: sampler.log_probs_kernel(logits, ctx), REPEATS, free_rows)
    assert bad_runs == 0


def test_control_top_k_noop_single_cta_is_repeatable():
    """CONTROL: at SMALL_VOCAB the top-k stage (k = vocab, one CTA per row on H100) is bitwise
    repeatable, so any nondeterminism in the small-vocab defect test comes from top-p."""
    logits, temperature, _, _, _, _ = _small_vocab_case()
    _, probs = _scaled_probs(logits, temperature)
    k = torch.full((probs.shape[0],), SMALL_VOCAB, dtype=torch.int32, device="cuda")
    bad_runs, _ = _repeat_diff(
        lambda: flashinfer.sampling.top_k_renorm_probs(probs, k), REPEATS
    )
    assert bad_runs == 0, f"single-CTA top_k_renorm_probs differed in {bad_runs} reruns"


def test_control_top_p_renorm_deterministic_flag_is_repeatable():
    """CONTROL: FlashInfer's own deterministic top-p mode, on the same inputs, is bitwise
    repeatable. This is the switch a fix would pass, and proves the harness can tell."""
    kw = _determinism_kwarg()
    if kw is None:
        pytest.skip("installed flashinfer.sampling.top_p_renorm_probs has no determinism switch")
    logits, temperature, _, top_p, filtered, _ = _small_vocab_case()
    _, probs = _scaled_probs(logits, temperature)
    renormed = flashinfer.sampling.top_k_renorm_probs(
        probs, torch.full((probs.shape[0],), SMALL_VOCAB, dtype=torch.int32, device="cuda")
    )
    p = top_p.cuda()
    bad_runs, _ = _repeat_diff(
        lambda: flashinfer.sampling.top_p_renorm_probs(renormed, p, **{kw: True}),
        REPEATS,
        filtered,
    )
    assert bad_runs == 0, f"top_p_renorm_probs({kw}=True) differed in {bad_runs} reruns"


# ---------------------------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------------------------


def test_defect_top_p_logprobs_repeat_bitwise():
    """DEFECT: Megatron's processed log-probs for top-p rows must be bitwise identical across
    identical calls (small vocabulary: only the top-p stage can differ)."""
    logits, temperature, top_k, top_p, filtered, _ = _small_vocab_case()
    sampler = FlashInferSampling(SMALL_VOCAB, torch.Generator(device="cuda"))
    ctx = _context(temperature, top_k, top_p)
    bad_runs, bad_rows = _repeat_diff(
        lambda: sampler.log_probs_kernel(logits, ctx), REPEATS, filtered
    )
    stages = ""
    if bad_runs:
        _, probs = _scaled_probs(logits, temperature)
        k_safe = torch.full((probs.shape[0],), SMALL_VOCAB, dtype=torch.int32, device="cuda")
        stages = _stage_report(probs, k_safe, top_p.cuda(), REPEATS)
    assert bad_runs == 0, (
        "DEFECT s12: FlashInferSampling.log_probs_kernel is not run-to-run deterministic for "
        f"top-p rows (vocab {SMALL_VOCAB}): {bad_runs}/{REPEATS - 1} identical reruns differed "
        f"bitwise from the first, rows {bad_rows} (0-15 boundary-tied top_p, 16-47 random). "
        "flashinfer_sampling.py:188 calls top_p_renorm_probs without its determinism switch "
        f"(installed name: {_determinism_kwarg()!r}; FlashInfer default False). Stage check: "
        f"{stages}"
    )


def test_defect_large_vocab_logprobs_repeat_bitwise():
    """DEFECT: at a 152K vocabulary Megatron's processed log-probs for top-k and top-p rows must
    be bitwise identical across identical calls. The stage report attributes any difference to
    the multi-CTA top_k_renorm_probs normalizer and/or the AIR top-p kernel."""
    rows = 32
    logits = _logits(rows, LARGE_VOCAB, seed=4321)
    temperature = torch.full((rows,), 0.8)
    top_k = torch.zeros(rows, dtype=torch.int32)
    top_p = torch.ones(rows)
    top_k[:16] = 50  # top-k-only rows
    g = torch.Generator().manual_seed(6)
    top_p[16:] = 0.5 + 0.49 * torch.rand(16, generator=g)  # top-p-only rows
    sampler = FlashInferSampling(LARGE_VOCAB, torch.Generator(device="cuda"))
    ctx = _context(temperature, top_k, top_p)
    assert ctx.active_sampling_filter_flags() == (False, False)  # mixed batch: both stages run

    repeats = REPEATS // 2
    bad_runs, bad_rows = _repeat_diff(lambda: sampler.log_probs_kernel(logits, ctx), repeats)
    stages = ""
    if bad_runs:
        _, probs = _scaled_probs(logits, temperature)
        k_safe = top_k.masked_fill(top_k <= 0, LARGE_VOCAB).cuda()
        stages = _stage_report(probs, k_safe, top_p.cuda(), repeats)
    assert bad_runs == 0, (
        "DEFECT s12: FlashInferSampling.log_probs_kernel is not run-to-run deterministic at "
        f"vocab {LARGE_VOCAB}: {bad_runs}/{repeats - 1} identical reruns differed bitwise from "
        f"the first, rows {bad_rows} (0-15 top_k=50, 16-31 top-p). Neither renorm call is "
        "deterministic here: top_p_renorm_probs is called without its determinism switch and "
        "top_k_renorm_probs (no switch) sums >= 3 per-CTA partials with float atomicAdd. "
        f"Stage check: {stages}"
    )
