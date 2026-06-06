# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-100%-busy + correctness verification for the async launch-before-commit overlap.

This is a *measurement / benchmark* harness for the ``enable_async_scheduling``
launch-before-commit overlap flip (commit ``c89cb70e``): in steady-state plain GPT
decode under a CUDA graph the next forward is launched speculatively BEFORE the
current step's ``update_requests``, so the commit (sampling + bookkeeping) runs in
the launched forward's GPU shadow. The intent is that the GPU stays ~100% busy in
steady-state decode (no inter-forward CPU bubble), while remaining token-exact vs
the serial path.

It builds the real dynamic-inference engine in-process (so per-forward CUDA-event
instrumentation is possible) using the same construction path as
``examples/inference/advanced/gpt_dynamic_inference.py`` (parse args ->
initialize_megatron -> get_model_for_inference -> context/wrapper/controller/engine).
No production inference code is modified: the per-forward timer is a script-level
monkeypatch of the controller's single forward entry point
(``_dynamic_step_forward_logits``), which is the one call site shared by the serial
path and both overlap modes (PRIME head + launch-before-commit).

Modes (``--gbd-mode``):

* ``correctness`` -- run one (async on/off) config to completion and dump per-request
  generated token ids + status + the overlap diagnostic counter. Run twice (off / on)
  and feed both dumps to ``compareA`` for the token-exact verdict.
* ``gpubusy`` -- prefill + warm up to steady-state decode, then measure N steady
  decode steps with ``torch.cuda.Event`` pairs bracketing each forward. Reports the
  inter-forward GPU gap (mean/median/p90, us) and the GPU-active fraction. Optionally
  brackets the steady window in ``cudaProfilerStart/Stop`` so an external ``nsys
  profile --capture-range=cudaProfilerApi`` captures only steady-state decode.
* ``compareA`` -- token-exact diff of two ``correctness`` dumps (serial vs async).
* ``nsysparse`` -- compute true GPU busy/idle fraction + inter-kernel gap stats from a
  ``nsys export --type sqlite`` database (the authoritative Part-B artifact).

The model is selected by ``--gbd-model`` (only ``357m`` is wired today; add a preset
in ``MODEL_PRESETS`` to re-run for nano-v3 later -- the measurement logic is
model-agnostic).
"""

import argparse
import json
import sys
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Model presets: the Megatron arg list (architecture + checkpoint + tokenizer).
# Mirrors examples/inference/gpt/gpt_dynamic_inference_357m.sh.
# ---------------------------------------------------------------------------
def _preset_357m() -> Tuple[List[str], Dict[str, str]]:
    base = "/lustre/fsw/portfolios/adlr/users/lmcafee/checkpoints"
    ckpt = f"{base}/357m/core-local-tp1-pp1"
    vocab = f"{base}/357m/vocab/gpt2-vocab.json"
    merge = f"{base}/357m/vocab/gpt2-merges.txt"
    argv = [
        "--transformer-impl", "local",
        "--model-provider", "gpt",
        "--load", ckpt,
        "--exit-on-missing-checkpoint",
        "--inference-ckpt-non-strict",
        "--tokenizer-type", "GPT2BPETokenizer",
        "--vocab-file", vocab,
        "--merge-file", merge,
        "--max-position-embeddings", "2048",
        "--seq-length", "2048",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--num-layers", "24",
        "--num-attention-heads", "16",
        "--hidden-size", "1024",
        "--bf16",
        "--micro-batch-size", "1",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--seed", "42",
        "--use-flash-attn",
        "--inference-rng-tracker",
        "--inference-dynamic-batching",
    ]
    return argv, {"vocab_size_hint": "50304"}


def _preset_12b() -> Tuple[List[str], Dict[str, str]]:
    """12B GPT (40L / hidden 5120 / GQA 8 / SwiGLU / RoPE / RMSNorm), bf16, TP1/PP1.

    Architecture matches the checkpoint at
    ``/lustre/.../checkpoints/12b/core-local-tp1-pp1`` (read off its saved args). We use the
    NullTokenizer with the checkpoint's padded vocab size (131072) rather than the original
    TikTokenizer: the gpubusy / correctness harness only needs ``tokenizer.vocab_size`` (prompts
    are synthetic in-vocab token ids; detokenize is not exercised in gpubusy), so this avoids a
    tokenizer-file dependency while keeping the embedding/output sizes identical to the checkpoint.
    """
    base = "/lustre/fsw/portfolios/adlr/users/lmcafee/checkpoints"
    ckpt = f"{base}/12b/core-local-tp1-pp1"
    argv = [
        "--transformer-impl", "local",
        "--model-provider", "gpt",
        "--load", ckpt,
        "--exit-on-missing-checkpoint",
        "--inference-ckpt-non-strict",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", "131072",
        "--make-vocab-size-divisible-by", "128",
        "--max-position-embeddings", "8192",
        "--seq-length", "8192",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--num-layers", "40",
        "--hidden-size", "5120",
        "--num-attention-heads", "32",
        "--group-query-attention",
        "--num-query-groups", "8",
        "--ffn-hidden-size", "14336",
        "--kv-channels", "128",
        "--swiglu",
        "--position-embedding-type", "rope",
        "--rotary-base", "1000000",
        "--rotary-percent", "1.0",
        "--normalization", "RMSNorm",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--bf16",
        "--micro-batch-size", "1",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--seed", "42",
        "--use-flash-attn",
        "--inference-rng-tracker",
        "--inference-dynamic-batching",
    ]
    return argv, {"vocab_size_hint": "131072"}


MODEL_PRESETS = {"357m": _preset_357m, "12b": _preset_12b}


def build_megatron_argv(knobs) -> List[str]:
    """Build the full Megatron argv from the model preset + measurement knobs.

    The ``enable_async_scheduling`` flag is intentionally NOT placed here -- it only
    affects the InferenceConfig (not the model / the captured CUDA graphs), so it is
    toggled per-engine via dataclasses.replace. The block-scope CUDA graph machinery
    is always built (both serial and async run with identical graphs; only the
    launch-before-commit ordering differs)."""
    preset = MODEL_PRESETS[knobs.gbd_model]
    argv, _ = preset()
    argv += [
        "--inference-dynamic-batching-buffer-size-gb", str(knobs.gbd_buffer_gb),
        "--cuda-graph-impl", "local",
        "--inference-cuda-graph-scope", "block",
        "--inference-dynamic-batching-num-cuda-graphs", str(knobs.gbd_num_cuda_graphs),
        "--inference-dynamic-batching-max-requests", str(knobs.gbd_max_requests),
        "--num-tokens-to-generate", str(knobs.gbd_gen_tokens),
    ]
    return argv


# ---------------------------------------------------------------------------
# Small numeric helpers (avoid a hard numpy dependency in the hot path).
# ---------------------------------------------------------------------------
def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q / 100.0 * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summ(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "median": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "min": s[0],
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# Engine construction (mirrors gpt_dynamic_inference.py main()).
# ---------------------------------------------------------------------------
def build_engine(args, model, tokenizer, *, enable_async: bool, max_sequence_length: int):
    """Build a fresh context/wrapper/controller/engine on top of an already-loaded model.

    enable_async is injected into the InferenceConfig (the only place it matters); the
    CUDA graphs were captured from the model config and are shared across variants."""
    import dataclasses

    import torch  # noqa: F401

    from megatron.core import parallel_state
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.engines import DynamicInferenceEngine
    from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
        GPTInferenceWrapper,
    )
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )
    from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
    from megatron.inference.utils import get_inference_config_from_model_and_args

    delete_cuda_graphs()

    inference_config = get_inference_config_from_model_and_args(model, args)
    inference_config = dataclasses.replace(
        inference_config,
        enable_async_scheduling=enable_async,
        max_sequence_length=max_sequence_length,
    )
    context = DynamicInferenceContext(model.config, inference_config)
    wrapped = GPTInferenceWrapper(model, context)
    wrapped.model_is_pipeline_parallel = not (
        parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
    )
    controller = TextGenerationController(inference_wrapped_model=wrapped, tokenizer=tokenizer)
    engine = DynamicInferenceEngine(controller, context)
    return engine, controller, context


def make_requests(knobs, *, vocab_size: int, sampling: str, batch_size: int,
                  prompt_len: int, gen_tokens, stagger_finishes: bool):
    """Build deterministic DynamicInferenceRequests with explicit prompt tokens.

    Prompt token ids are fully determined by (request_id, prompt_len) so the prompt
    set is byte-identical across processes -- required for the cross-process
    token-exact comparison in correctness mode."""
    import torch

    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    requests = []
    for rid in range(batch_size):
        # Deterministic, in-vocab prompt tokens (avoid id 0 / very large ids).
        start = 100 + 7 * rid
        ids = [(start + 3 * j) % (vocab_size - 2) + 1 for j in range(prompt_len)]
        prompt = torch.tensor(ids, dtype=torch.int64, device="cuda")

        if isinstance(gen_tokens, int):
            n_gen = gen_tokens
        else:
            n_gen = gen_tokens
        if stagger_finishes:
            n_gen = n_gen + (rid % 8) * 4  # staggered finishes across steps

        if sampling == "greedy":
            sp = SamplingParams(num_tokens_to_generate=n_gen, top_k=1, top_p=0.0,
                                temperature=1.0, termination_id=-1)
        elif sampling == "torch":
            # Pure multinomial draw (no top-k/top-p filtering) through the C3 per-request
            # keyed RNG -> deterministic given the fixed --seed.
            sp = SamplingParams(num_tokens_to_generate=n_gen, top_k=0, top_p=0.0,
                                temperature=1.0, termination_id=-1)
        else:
            raise ValueError(sampling)
        requests.append(
            DynamicInferenceRequest(request_id=rid, prompt_tokens=prompt, sampling_params=sp)
        )
    return requests


# ---------------------------------------------------------------------------
# Mode: correctness (Part A).
# ---------------------------------------------------------------------------
def run_correctness(knobs, args, model, tokenizer) -> Dict:
    import torch  # noqa: F401

    from megatron.core.inference.inference_request import Status

    vocab_size = tokenizer.vocab_size
    max_seq_len = knobs.gbd_prompt_len + knobs.gbd_gen_tokens + 8 + 32

    engine, controller, context = build_engine(
        args, model, tokenizer,
        enable_async=bool(knobs.gbd_async), max_sequence_length=max_seq_len,
    )
    requests = make_requests(
        knobs, vocab_size=vocab_size, sampling=knobs.gbd_sampling,
        batch_size=knobs.gbd_batch_size, prompt_len=knobs.gbd_prompt_len,
        gen_tokens=knobs.gbd_gen_tokens, stagger_finishes=True,
    )
    engine.reset()
    by_id = {}
    for req in requests:
        engine._add_request(req)
        by_id[req.request_id] = req

    steps = 0
    max_steps = knobs.gbd_prompt_len + knobs.gbd_gen_tokens + knobs.gbd_batch_size + 64
    while engine.has_unfinished_requests() and steps < max_steps:
        result = engine.step_modern()
        steps += 1
        for rec in result["finished_request_records"]:
            fin = rec.merge()
            req = by_id[fin.request_id]
            req._gbd_tokens = list(fin.generated_tokens)
            req._gbd_status = str(fin.status)

    tokens = {}
    statuses = {}
    for rid, req in by_id.items():
        toks = getattr(req, "_gbd_tokens", None)
        if toks is None:
            toks = list(getattr(req, "generated_tokens", []) or [])
        tokens[rid] = [int(t) for t in toks]
        statuses[rid] = getattr(req, "_gbd_status", str(getattr(req, "status", None)))

    out = {
        "mode": "correctness",
        "async": bool(knobs.gbd_async),
        "sampling": knobs.gbd_sampling,
        "batch_size": knobs.gbd_batch_size,
        "prompt_len": knobs.gbd_prompt_len,
        "gen_tokens": knobs.gbd_gen_tokens,
        "steps": steps,
        "launch_before_commit_count": int(controller._async_launch_before_commit_count),
        "committed_with_inflight_forward": bool(controller._async_committed_with_inflight_forward),
        "block_size_tokens": int(context.block_size_tokens),
        "tokens": tokens,
        "statuses": statuses,
        "num_requests": len(by_id),
        "gen_lengths": {rid: len(t) for rid, t in tokens.items()},
    }
    return out


# ---------------------------------------------------------------------------
# Mode: gpubusy (Part B, programmatic).
# ---------------------------------------------------------------------------
def run_gpubusy(knobs, args, model, tokenizer, enable_async=None) -> Dict:
    import torch

    if enable_async is None:
        enable_async = bool(knobs.gbd_async)
    enable_async = bool(enable_async)

    vocab_size = tokenizer.vocab_size
    warmup = knobs.gbd_warmup_steps
    steady = knobs.gbd_steady_steps
    # gen long enough that no request finishes during [0, warmup+steady]; prompt short
    # enough that no 256-token block boundary is crossed during the window either.
    gen_tokens = warmup + steady + 16
    max_seq_len = knobs.gbd_prompt_len + gen_tokens + 32

    engine, controller, context = build_engine(
        args, model, tokenizer,
        enable_async=enable_async, max_sequence_length=max_seq_len,
    )

    # ---- per-forward CUDA-event timer (script-level monkeypatch; no prod change) ----
    state = {"recording": False, "starts": [], "ends": []}
    orig_forward = controller._dynamic_step_forward_logits

    def timed_forward(input_ids, position_ids):
        if state["recording"]:
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            orig_forward(input_ids, position_ids)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            state["starts"].append(s)
            state["ends"].append(e)
        else:
            orig_forward(input_ids, position_ids)

    controller._dynamic_step_forward_logits = timed_forward

    requests = make_requests(
        knobs, vocab_size=vocab_size, sampling="greedy",
        batch_size=knobs.gbd_batch_size, prompt_len=knobs.gbd_prompt_len,
        gen_tokens=gen_tokens, stagger_finishes=False,
    )
    engine.reset()
    for req in requests:
        engine._add_request(req)

    # ---- driver: how decode steps are pumped ----
    #
    # step_modern: one engine.step_modern() (== _run_coroutine_sync(async_step())) per step, so
    #   every step is its own run_until_complete -- the host pipeline of step K and the forward of
    #   step K strictly alternate at the run_until_complete boundary (the historical driver).
    # continuous: warmup + the entire steady window run inside ONE run_until_complete, awaiting
    #   engine.async_step() in a loop. This mirrors the production engine loop (run_engine's body):
    #   there is no per-step event-loop boundary, so the launch-before-commit host work of one step
    #   (commit + prestage + bookkeeping) overlaps the next forward across step boundaries -- the
    #   pipelining the per-step run_until_complete driver cannot exercise.
    driver = knobs.gbd_driver
    nsys = bool(knobs.gbd_nsys)
    box = {"active": 0, "launch_before": 0, "non_decode_steps": 0}

    def _profiler(start):
        if nsys:
            (torch.cuda.cudart().cudaProfilerStart if start else
             torch.cuda.cudart().cudaProfilerStop)()

    def _enter_steady_window():
        assert engine.is_decode_only, "did not reach steady-state decode during warmup"
        box["active"] = context.total_request_count - context.paused_request_count
        box["launch_before"] = int(controller._async_launch_before_commit_count)
        _profiler(True)
        state["recording"] = True

    def _exit_steady_window():
        state["recording"] = False
        _profiler(False)

    if driver == "step_modern":
        # Prefill + warm up to steady-state decode.
        warm = 0
        decode_seen = 0
        while warm < warmup + 64 and decode_seen < warmup:
            engine.step_modern()
            warm += 1
            if engine.is_decode_only:
                decode_seen += 1
        _enter_steady_window()
        for _ in range(steady):
            engine.step_modern()
            if not engine.is_decode_only:
                box["non_decode_steps"] += 1
        _exit_steady_window()
    elif driver == "continuous":
        async def _drive():
            warm = 0
            decode_seen = 0
            while warm < warmup + 64 and decode_seen < warmup:
                await engine.async_step()
                warm += 1
                if engine.is_decode_only:
                    decode_seen += 1
            _enter_steady_window()
            for _ in range(steady):
                await engine.async_step()
                if not engine.is_decode_only:
                    box["non_decode_steps"] += 1
            _exit_steady_window()

        engine._loop.run_until_complete(_drive())
    else:
        raise ValueError(driver)

    torch.cuda.synchronize()
    active = box["active"]
    launch_before = box["launch_before"]
    non_decode_steps = box["non_decode_steps"]
    launch_during = int(controller._async_launch_before_commit_count) - launch_before

    # ---- compute metrics from CUDA events ----
    starts, ends = state["starts"], state["ends"]
    n = len(starts)
    fwd_ms = [starts[i].elapsed_time(ends[i]) for i in range(n)]
    gap_ms = [ends[i].elapsed_time(starts[i + 1]) for i in range(n - 1)]
    span_ms = starts[0].elapsed_time(ends[-1]) if n >= 2 else float("nan")
    total_fwd_ms = sum(fwd_ms)
    active_fraction = (total_fwd_ms / span_ms) if (n >= 2 and span_ms > 0) else float("nan")

    gap_us = [g * 1000.0 for g in gap_ms]
    fwd_us = [f * 1000.0 for f in fwd_ms]

    out = {
        "mode": "gpubusy",
        "method": "programmatic_cuda_events",
        "driver": driver,
        "model": knobs.gbd_model,
        "async": enable_async,
        "batch_size": knobs.gbd_batch_size,
        "active_decode_requests": int(active),
        "prompt_len": knobs.gbd_prompt_len,
        "warmup_steps": warmup,
        "steady_steps": steady,
        "num_forwards_recorded": n,
        "non_decode_steps_in_window": non_decode_steps,
        "launch_before_commit_in_window": launch_during,
        "prestage_in_shadow_count": int(controller._async_prestage_in_shadow_count),
        "block_size_tokens": int(context.block_size_tokens),
        "forward_us": _summ(fwd_us),
        "inter_forward_gap_us": _summ(gap_us),
        "gpu_active_fraction": active_fraction,
        "window_span_ms": span_ms,
        "total_forward_ms": total_fwd_ms,
    }
    return out


# ---------------------------------------------------------------------------
# Mode: compareA (token-exact diff of two correctness dumps).
# ---------------------------------------------------------------------------
def run_compareA(knobs) -> Dict:
    with open(knobs.gbd_serial_json) as f:
        serial = json.load(f)
    with open(knobs.gbd_async_json) as f:
        asyncd = json.load(f)

    s_tokens = {int(k): v for k, v in serial["tokens"].items()}
    a_tokens = {int(k): v for k, v in asyncd["tokens"].items()}
    s_status = {int(k): v for k, v in serial["statuses"].items()}
    a_status = {int(k): v for k, v in asyncd["statuses"].items()}

    keys_match = set(s_tokens) == set(a_tokens)
    mismatches = []
    first_div = None
    for rid in sorted(set(s_tokens) & set(a_tokens)):
        st, at = s_tokens[rid], a_tokens[rid]
        ss, as_ = s_status.get(rid), a_status.get(rid)
        if st != at or ss != as_:
            # locate first divergent token position
            pos = None
            for i in range(min(len(st), len(at))):
                if st[i] != at[i]:
                    pos = i
                    break
            if pos is None and len(st) != len(at):
                pos = min(len(st), len(at))
            mismatches.append({
                "request_id": rid,
                "serial_status": ss, "async_status": as_,
                "serial_len": len(st), "async_len": len(at),
                "first_divergent_pos": pos,
                "serial_tok_at_pos": (st[pos] if pos is not None and pos < len(st) else None),
                "async_tok_at_pos": (at[pos] if pos is not None and pos < len(at) else None),
            })
            if first_div is None:
                first_div = mismatches[-1]

    token_exact = keys_match and not mismatches
    total_tokens = sum(len(v) for v in s_tokens.values())
    out = {
        "mode": "compareA",
        "sampling": serial.get("sampling"),
        "batch_size": serial.get("batch_size"),
        "request_keys_match": keys_match,
        "token_exact": token_exact,
        "num_requests": len(s_tokens),
        "total_serial_tokens": total_tokens,
        "serial_launch_before_commit_count": serial.get("launch_before_commit_count"),
        "async_launch_before_commit_count": asyncd.get("launch_before_commit_count"),
        "num_mismatches": len(mismatches),
        "first_divergence": first_div,
        "mismatches": mismatches[:16],
    }
    return out


# ---------------------------------------------------------------------------
# Mode: nsysparse (authoritative GPU busy/idle from an nsys sqlite export).
# ---------------------------------------------------------------------------
def run_nsysparse(knobs) -> Dict:
    import sqlite3

    con = sqlite3.connect(knobs.gbd_sqlite)
    cur = con.cursor()

    def table_exists(name):
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cur.fetchone() is not None

    kernel_table = None
    for cand in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        if table_exists(cand):
            kernel_table = cand
            break
    if kernel_table is None:
        con.close()
        return {"mode": "nsysparse", "error": "no kernel table in sqlite",
                "sqlite": knobs.gbd_sqlite}

    cur.execute(f"SELECT start, end FROM {kernel_table} ORDER BY start ASC")
    rows = cur.fetchall()
    con.close()

    if not rows:
        return {"mode": "nsysparse", "error": "no kernels in capture window",
                "sqlite": knobs.gbd_sqlite}

    # Merge intervals to get true GPU-busy time (union of kernel exec spans).
    span_start = rows[0][0]
    span_end = max(r[1] for r in rows)
    span_ns = span_end - span_start

    busy_ns = 0
    cur_s, cur_e = rows[0]
    # inter-kernel gaps measured on the merged-as-you-go boundary (true GPU idle gaps).
    gaps_ns = []
    for s, e in rows[1:]:
        if s > cur_e:
            busy_ns += cur_e - cur_s
            gaps_ns.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy_ns += cur_e - cur_s

    idle_ns = span_ns - busy_ns
    gaps_us = [g / 1000.0 for g in gaps_ns]
    out = {
        "mode": "nsysparse",
        "sqlite": knobs.gbd_sqlite,
        "kernel_table": kernel_table,
        "num_kernels": len(rows),
        "window_span_ms": span_ns / 1e6,
        "gpu_busy_ms": busy_ns / 1e6,
        "gpu_idle_ms": idle_ns / 1e6,
        "gpu_busy_fraction": busy_ns / span_ns if span_ns > 0 else float("nan"),
        "gpu_idle_fraction": idle_ns / span_ns if span_ns > 0 else float("nan"),
        "num_idle_gaps": len(gaps_us),
        "inter_kernel_idle_gap_us": _summ(gaps_us),
    }
    return out


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_knobs(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gbd-mode", required=True,
                   choices=["correctness", "gpubusy", "compareA", "nsysparse"])
    p.add_argument("--gbd-model", default="357m", choices=list(MODEL_PRESETS))
    p.add_argument("--gbd-async", type=int, default=0)
    p.add_argument("--gbd-driver", default="step_modern",
                   choices=["step_modern", "continuous"])
    p.add_argument("--gbd-batch-size", type=int, default=1)
    p.add_argument("--gbd-sampling", default="greedy", choices=["greedy", "torch"])
    p.add_argument("--gbd-prompt-len", type=int, default=8)
    p.add_argument("--gbd-gen-tokens", type=int, default=100)
    p.add_argument("--gbd-warmup-steps", type=int, default=24)
    p.add_argument("--gbd-steady-steps", type=int, default=200)
    p.add_argument("--gbd-num-cuda-graphs", type=int, default=16)
    p.add_argument("--gbd-max-requests", type=int, default=320)
    p.add_argument("--gbd-buffer-gb", type=float, default=20.0)
    p.add_argument("--gbd-nsys", type=int, default=0)
    p.add_argument("--gbd-seed", type=int, default=42)
    p.add_argument("--gbd-out", default=None)
    p.add_argument("--gbd-serial-json", default=None)
    p.add_argument("--gbd-async-json", default=None)
    p.add_argument("--gbd-sqlite", default=None)
    knobs, _ = p.parse_known_args(argv)
    return knobs


def main():
    knobs = parse_knobs(sys.argv[1:])

    # Pure-python modes: no Megatron / GPU needed.
    if knobs.gbd_mode == "compareA":
        out = run_compareA(knobs)
    elif knobs.gbd_mode == "nsysparse":
        out = run_nsysparse(knobs)
    else:
        # Megatron modes: replace argv with the model preset, then parse + init.
        sys.argv = ["gpu_busy_decode.py"] + build_megatron_argv(knobs)

        import torch  # noqa: F401

        from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
        from megatron.inference.utils import add_inference_args, get_model_for_inference
        from megatron.training import initialize_megatron
        from megatron.training.arguments import parse_and_validate_args

        args = parse_and_validate_args(
            extra_args_provider=add_inference_args,
            args_defaults={"no_load_rng": True, "no_load_optim": True},
        )
        initialize_megatron()
        tokenizer = build_tokenizer(args)
        model = get_model_for_inference()

        if knobs.gbd_mode == "correctness":
            out = run_correctness(knobs, args, model, tokenizer)
        elif knobs.gbd_mode == "gpubusy":
            out = run_gpubusy(knobs, args, model, tokenizer)
        else:
            raise ValueError(knobs.gbd_mode)

    text = json.dumps(out, indent=2, sort_keys=True)
    print("GBD_RESULT_BEGIN")
    print(text)
    print("GBD_RESULT_END")
    if knobs.gbd_out:
        with open(knobs.gbd_out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
