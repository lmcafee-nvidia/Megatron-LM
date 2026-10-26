# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for suspected bug s3: selectively-BF16 FlashInfer expert layers break batch invariance.

Config validation explicitly allows BF16 FlashInfer expert layers inside a batch-invariant MXFP8
model (``transformer_config.py:2280-2290``). Their forward, the BF16 branch of
``InferenceGroupedMLP._flashinfer_forward`` (``experts.py:1694-1708``), calls
``flashinfer.fused_moe.cutlass_fused_moe`` without ``use_fused_finalize=False``. FlashInfer
defaults it to True and documents the fused finalize as non-associative atomics that are not
run-to-run deterministic (``flashinfer/fused_moe/core.py:1443-1446`` at v0.7.0.post1). The
per-token NVFP4 path makes the opposite choice under batch-invariant mode
(``flashinfer_nvfp4.py:312-316``).

How FlashInfer actually selects the fused finalize (read at v0.7.0.post1 and v0.6.8; identical):
* ``use_fused_finalize`` only decides whether FINALIZE-fused GEMM2 tactics are *offered*
  (``moe_kernels.h`` ``mayHaveFinalizeFused``; ``moe_gemm_template_dispatch.h`` duplicates every TMA
  warp-specialized config with ``EpilogueFusionType::FINALIZE`` and appends the copies). Whether
  the fused finalize runs depends on the GEMM2 tactic (``cutlass_fused_moe_kernels.cuh``:
  ``using_fused_finalize = use_fused_finalize_ && gemm2 tactic is FINALIZE``), and then the GEMM2
  epilogue reduces the top-k routes with BF16 ``red.global.add`` (``epilogue_moe_finalize.hpp``,
  ``sm90_visitor_scatter.hpp``; plain stores when top_k == 1).
* Without autotuning, ``AutoTuner.choose_one`` returns tactic -1 and the runner uses the *first*
  GEMM2 tactic, which is a non-fused config (the FINALIZE copies come after it). Megatron never
  enters ``flashinfer.autotune`` (grep), so an untuned Megatron run uses the deterministic
  finalize kernel by default. The fused finalize is reached once the autotuner holds a tuned
  tactic for the op (``with flashinfer.autotune():`` warmup, a loaded autotune cache, or
  ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1`` with a matching bundled config).

So the tests sweep *every* GEMM2 tactic the runner offers, pinned through a patched
``AutoTuner.choose_one`` that stands in for a tuned autotuner cache (GEMM1 keeps its default; the
Megatron call itself is unmodified). Batch-invariant mode must hold whichever tactic is tuned.
The untuned default (-1) is measured too and reported separately.

Measurement for one tactic: a 1024-token batch (top_k = 8 of 16 experts, H = 1024, I = 512,
squared-ReLU) replayed ``MCORE_REPRO_S3_REPLAYS`` times (default 8) and compared bitwise; a probe
token alone (replayed too), at row 5 of a 37-token batch and at row 73 of the 1024-token batch,
compared bitwise (batch invariance).

Tests:
* CONTROL Megatron BF16 path, batch-invariant mode, top_k = 1, every GEMM2 tactic (FINALIZE ones
  included: with one route the fused epilogue stores instead of reducing): deterministic and
  batch-invariant. Validates the harness (pinning reaches FINALIZE tactics; no false positives)
  on every FlashInfer version.
* CONTROL ``cutlass_fused_moe(..., use_fused_finalize=False)`` called directly, top_k = 8, every
  GEMM2 tactic: deterministic and batch-invariant. Needs ``use_fused_finalize`` (FlashInfer >=
  0.6.15; absent in 0.6.8, where this test skips).
* DEFECT Megatron BF16 path (``_flashinfer_forward``), batch-invariant mode, top_k = 8, every GEMM2
  tactic: must be deterministic and batch-invariant. Expected to FAIL on main for the FINALIZE
  tactics (atomic arrival order varies), with high probability rather than certainty.

Hardware: 1 GPU, SM90 (H100) or SM100/SM103 (B200/GB200/GB300). FlashInfer offers FINALIZE GEMM2
tactics only where BF16 has TMA warp-specialized grouped GEMMs, i.e. SM >= 90 except SM120/121
(``moe_tma_warp_specialized_traits.h``: SM120 TMA WS is FP4-only); SM89 builds only Ampere-style
configs, and SM < 89 is rejected by ``get_cutlass_fused_moe_module`` ("Invalid backend").
``python -m pytest <file> -v`` (no distributed launch needed).
"""

import inspect
import os
from types import SimpleNamespace

import pytest
import torch

# Max GPUs any test here needs.
REQUIRED_GPUS = 1

NUM_EXPERTS = 16
TOPK = 8
HIDDEN = 1024
FFN = 512
N_BATCH, PROBE_ROW = 1024, 73
N_SMALL, PROBE_ROW_SMALL = 37, 5
REPLAYS = max(2, int(os.environ.get("MCORE_REPRO_S3_REPLAYS", "8")))

# Archs whose get_cutlass_fused_moe_module builds BF16 TMA warp-specialized MoE GEMMs (and with
# them FINALIZE-fused GEMM2 tactics): "90", and "100"/"103"/"107"/"110".
_FUSED_FINALIZE_ARCHS = (90, 100, 103, 107, 110)
_GEMM1_OP = "trtllm::fused_moe::gemm1"
_GEMM2_OP = "trtllm::fused_moe::gemm2"


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs a CUDA GPU"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm not in _FUSED_FINALIZE_ARCHS:
        return (
            f"SM{sm}: FlashInfer's BF16 cutlass_fused_moe has FINALIZE-fused GEMM2 tactics only on "
            "TMA warp-specialized archs SM90/SM100/SM103 (SM120/121 TMA WS is FP4-only, SM89 is "
            "Ampere-style only, SM<89 is rejected as an invalid backend)"
        )
    try:
        import flashinfer.fused_moe  # noqa: F401
        from flashinfer.autotuner import AutoTuner  # noqa: F401
    except ImportError as exc:
        return f"FlashInfer fused_moe / autotuner not importable: {exc!r}"
    from megatron.core.transformer.moe import experts

    if not experts.HAVE_FLASHINFER:
        return "megatron.core.transformer.moe.experts could not import flashinfer.fused_moe"
    return None


_SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))


def _flashinfer_version():
    import flashinfer

    return getattr(flashinfer, "__version__", "unknown")


def _has_use_fused_finalize():
    """cutlass_fused_moe gained use_fused_finalize in 0.6.15 (absent in 0.6.8 and 0.6.14)."""
    from flashinfer import fused_moe

    return "use_fused_finalize" in inspect.signature(fused_moe.cutlass_fused_moe).parameters


@pytest.fixture
def pin(monkeypatch):
    """Stand-in for a tuned FlashInfer autotuner cache.

    ``state.tactic`` is an index into the runner's GEMM2 tactic list (None = FlashInfer's untuned
    default, -1). GEMM1 always keeps its default. The GEMM2 id encoding follows the binding:
    absolute (GEMM1 count + index) from 0.6.15 on; relative-if-in-range before (0.6.8).
    """
    from flashinfer.autotuner import AutoTuner

    original = AutoTuner.choose_one
    absolute_ids = _has_use_fused_finalize()
    state = SimpleNamespace(tactic=None, counts=None, fused_finalize=None)

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op not in (_GEMM1_OP, _GEMM2_OP):
            return original(self, custom_op, runners, tuning_config, inputs, **kwargs)
        runner = runners[0]
        gemm1 = int(runner.fused_moe_runner.get_gemm1_tactic_count())
        gemm2 = int(runner.fused_moe_runner.get_gemm2_tactic_count())
        state.counts = (gemm1, gemm2)
        # 0.6.8's runner has no such attribute: its C++ default use_fused_finalize_ is true.
        state.fused_finalize = getattr(runner, "use_fused_finalize", True)
        if custom_op == _GEMM1_OP or state.tactic is None:
            return runner, -1
        return runner, (gemm1 + state.tactic) if absolute_ids else state.tactic

    monkeypatch.setattr(AutoTuner, "choose_one", choose_one)
    return state


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator().manual_seed(2026)
    fc1 = (torch.randn(NUM_EXPERTS, FFN, HIDDEN, generator=gen) * HIDDEN**-0.5).bfloat16()
    fc2 = (torch.randn(NUM_EXPERTS, HIDDEN, FFN, generator=gen) * FFN**-0.5).bfloat16()
    return fc1.cuda(), fc2.cuda()


def _batches(topk):
    """(alone, small, big) batches sharing one probe token, as CUDA tensors."""
    gen = torch.Generator().manual_seed(100 + topk)
    probe_hidden = torch.randn(HIDDEN, generator=gen).bfloat16()
    probe_routing = torch.randperm(NUM_EXPERTS, generator=gen)[:topk]
    probe_probs = torch.softmax(torch.randn(topk, generator=gen) * 2, dim=-1)

    def batch(n, row):
        hidden = torch.randn(n, HIDDEN, generator=gen).bfloat16()
        routing = torch.stack([torch.randperm(NUM_EXPERTS, generator=gen)[:topk] for _ in range(n)])
        probs = torch.softmax(torch.randn(n, topk, generator=gen) * 2, dim=-1)
        hidden[row], routing[row], probs[row] = probe_hidden, probe_routing, probe_probs
        return (
            hidden.cuda(),
            routing.to(torch.int64).cuda(),
            probs.to(torch.float32).contiguous().cuda(),
            row,
        )

    return batch(1, 0), batch(N_SMALL, PROBE_ROW_SMALL), batch(N_BATCH, PROBE_ROW)


def _megatron_call(fc1, fc2):
    """The production BF16 branch of InferenceGroupedMLP._flashinfer_forward (stand-in self)."""
    from flashinfer.fused_moe.core import ActivationType

    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    layer = SimpleNamespace(
        _activation_clamp_scale=None,
        _uses_mxfp8_weights=False,
        _nvfp4_weights=None,
        _fc1_weight=fc1,
        _fc2_weight=fc2,
        # squared ReLU: the only expert activation an MXFP8 FlashInfer model accepts.
        _flashinfer_activation_type=ActivationType.Relu2,
        _nvls_dispatcher=False,
        ep_group=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
    )

    def call(hidden, routing, probs):
        output, _ = InferenceGroupedMLP._flashinfer_forward(layer, hidden, routing, probs)
        return output

    return call


def _unfused_call(fc1, fc2):
    """FlashInfer called directly with the deterministic (non-fused) finalize requested."""
    from flashinfer import fused_moe
    from flashinfer.fused_moe.core import ActivationType

    def call(hidden, routing, probs):
        output = fused_moe.cutlass_fused_moe(
            hidden,
            routing.int(),
            probs,
            fc1,
            fc2,
            hidden.dtype,
            quant_scales=None,
            activation_type=ActivationType.Relu2,
            use_fused_finalize=False,
        )
        return output[0] if isinstance(output, (list, tuple)) else output

    return call


def _bits(t):
    return t.contiguous().view(torch.int16)


def _measure(call, batches):
    """Bitwise replay and batch-invariance mismatch counts for the currently pinned tactic."""
    alone, small, big = batches
    big_outs = [call(*big[:3]).clone() for _ in range(REPLAYS)]
    alone_outs = [call(*alone[:3]).clone() for _ in range(max(2, REPLAYS // 3))]
    small_out = call(*small[:3]).clone()
    torch.cuda.synchronize()
    replay = max(
        int((_bits(o) != _bits(outs[0])).sum())
        for outs in (big_outs, alone_outs)
        for o in outs[1:]
    )
    probe = _bits(alone_outs[0][0])
    batch = max(
        int((_bits(small_out[small[3]]) != probe).sum()),
        int((_bits(big_outs[0][big[3]]) != probe).sum()),
    )
    return dict(replay=replay, batch=batch)


def _sweep(pin, call, batches):
    """Measure FlashInfer's untuned default and then every GEMM2 tactic of the call's runner."""
    pin.tactic = None
    results = {"default": _measure(call, batches)}
    counts, fused_finalize = pin.counts, pin.fused_finalize
    assert counts is not None, "cutlass_fused_moe did not consult the autotuner"
    for index in range(counts[1]):
        pin.tactic = index
        try:
            results[index] = _measure(call, batches)
        except Exception as exc:  # tactic not runnable for this shape / device
            results[index] = f"unrunnable ({type(exc).__name__})"
    pin.tactic = None
    torch.cuda.synchronize()
    return results, counts, fused_finalize


def _violations(results):
    return {
        key: r
        for key, r in results.items()
        if isinstance(r, dict) and (r["replay"] or r["batch"])
    }


def _runnable(results):
    return [key for key, r in results.items() if isinstance(r, dict)]


def _describe(violations, limit=8):
    parts = []
    for key, r in list(violations.items())[:limit]:
        name = "untuned default (-1)" if key == "default" else f"GEMM2 tactic {key}"
        parts.append(
            f"{name}: {r['replay']} output elements differ between replays, probe token differs "
            f"across co-batch sizes in {r['batch']} of {HIDDEN} elements"
        )
    more = len(violations) - limit
    return "; ".join(parts) + (f"; and {more} more tactics" if more > 0 else "")


def test_control_megatron_bf16_path_single_route_is_deterministic(pin, weights):
    """top_k = 1: no top-k reduction, so every tactic (FINALIZE ones only store) is stable."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    call = _megatron_call(*weights)
    batches = _batches(topk=1)
    with torch.no_grad(), set_batch_invariant_mode(True):
        results, counts, _ = _sweep(pin, call, batches)
    assert _runnable(results), "no runnable tactic"
    violations = _violations(results)
    assert not violations, (
        f"harness invalid: with top_k=1 (no atomic reduction) the BF16 path still varies "
        f"(GEMM1/GEMM2 tactic counts {counts}): {_describe(violations)}"
    )


def test_control_cutlass_unfused_finalize_is_deterministic(pin, weights):
    """cutlass_fused_moe(use_fused_finalize=False): every tactic replays and is batch-invariant."""
    if not _has_use_fused_finalize():
        pytest.skip(
            f"FlashInfer {_flashinfer_version()}: cutlass_fused_moe has no use_fused_finalize "
            "argument (added in 0.6.15), so the non-fused finalize cannot be requested"
        )
    call = _unfused_call(*weights)
    batches = _batches(topk=TOPK)
    with torch.no_grad():
        results, counts, fused_finalize = _sweep(pin, call, batches)
    assert fused_finalize is False, "runner was not built with use_fused_finalize=False"
    assert _runnable(results), "no runnable tactic"
    violations = _violations(results)
    assert not violations, (
        f"control failed: non-fused finalize is not deterministic / batch-invariant "
        f"(GEMM1/GEMM2 tactic counts {counts}): {_describe(violations)}"
    )


def test_defect_megatron_bf16_path_is_deterministic_in_batch_invariant_mode(pin, weights):
    """The selectively-BF16 FlashInfer expert path must replay bitwise and be batch-invariant in
    batch-invariant mode, whichever GEMM2 tactic FlashInfer's autotuner holds."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    call = _megatron_call(*weights)
    batches = _batches(topk=TOPK)
    with torch.no_grad(), set_batch_invariant_mode(True):
        results, counts, fused_finalize = _sweep(pin, call, batches)
    assert _runnable(results), "no runnable tactic"

    unfused_gemm2 = None
    if _has_use_fused_finalize():  # diagnostic: GEMM2 tactics offered without the fused finalize
        with torch.no_grad():
            _unfused_call(*weights)(*batches[0][:3])
        unfused_gemm2 = pin.counts[1]
    offered = (
        f"{counts[1]} GEMM2 tactics offered vs {unfused_gemm2} with use_fused_finalize=False"
        if unfused_gemm2 is not None
        else f"{counts[1]} GEMM2 tactics offered (FlashInfer {_flashinfer_version()} always "
        "offers FINALIZE tactics)"
    )
    violations = _violations(results)
    default = "violated" if "default" in violations else "deterministic"
    assert not violations, (
        f"DEFECT s3: in batch-invariant mode the selectively-BF16 FlashInfer expert path "
        f"(experts.py:1694-1708) is not deterministic / batch-invariant for {len(violations)} of "
        f"{len(_runnable(results))} runnable GEMM2 tactics FlashInfer's autotuner can select: "
        f"{_describe(violations)}. The call does not pass use_fused_finalize=False "
        f"(runner use_fused_finalize={fused_finalize}; {offered}), so FINALIZE-fused GEMM2 "
        f"tactics, which reduce the top-{TOPK} routes with BF16 atomics, stay selectable. "
        f"FlashInfer's untuned default tactic: {default}."
    )
