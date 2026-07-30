# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

READINESS_MARKER = "Running on http"
READINESS_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 60
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
LOGPROB_P95_ATOL = 0.048790164169432  # A 5% probability ratio in log space.
LOGPROB_MAX_ATOL = 0.182321556793955  # A 20% probability ratio in log space.


def build_server_cmd(
    checkpoint_dir: str,
    tokenizer_model: str,
    server_log_dir: str = None,
    prefix_cache: bool = False,
    stress_mode: bool = False,
) -> list[str]:
    log_args = ["--log-dir", server_log_dir, "--tee", "3"] if server_log_dir else []
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *log_args,
        "--nproc-per-node=8",
        "-m",
        "examples.inference.launch_inference_server",
        "--tiktoken-pattern",
        "v2",
        "--use-mcore-models",
        "--tokenizer-type",
        "TikTokenizer",
        "--tokenizer-model",
        tokenizer_model,
        "--auto-detect-ckpt-format",
        "--max-tokens-to-oom",
        "1024" if stress_mode else "3600000",
        "--inference-max-seq-length",
        "4096",
        "--attention-backend",
        "flash",
        "--use-checkpoint-args",
        "--micro-batch-size",
        "1",
        "--no-load-optim",
        "--no-use-tokenizer-model-from-checkpoint-args",
        "--load",
        checkpoint_dir,
        "--distributed-backend",
        "nccl",
        "--transformer-impl",
        "inference_optimized",
        "--sequence-parallel",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--deterministic-mode",
        "--ckpt-format",
        "torch_dist",
        "--bf16",
        "--num-layers",
        "24",
        "--hidden-size",
        "1152",
        "--num-attention-heads",
        "16",
        "--max-position-embeddings",
        "1024",
        "--seq-length",
        "1024",
        "--inference-logging-step-interval=1",
        "--inference-dynamic-batching-buffer-size-gb",
        "0.25" if stress_mode else "20",
        "--dist-ckpt-strictness",
        "log_unexpected",
        "--inference-ckpt-non-strict",
        "--port",
        str(SERVER_PORT),
        "--host",
        SERVER_HOST,
        *(
            [
                "--inference-dynamic-batching-prefix-caching",
                "--inference-dynamic-batching-prefix-caching-eviction-policy",
                "lru",
                "--inference-dynamic-batching-prefix-caching-coordinator-policy",
                "load_balanced",
            ]
            if prefix_cache
            else []
        ),
    ]


def cleaned_env() -> dict:
    env = os.environ.copy()
    for v in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ):
        env.pop(v, None)
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NCCL_ALGO"] = "Ring"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", MEGATRON_LOGGING_LEVEL="20")
    return env


def post_completion() -> dict:
    body = json.dumps(
        {"model": "EMPTY", "prompt": "Hello, world!", "max_tokens": 10, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{SERVER_PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        if resp.status != 200:
            raise AssertionError(f"server returned status {resp.status}")
        return json.loads(resp.read())


def post_chat(prompt: str) -> dict:
    body = json.dumps(
        {
            "model": "EMPTY",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 1,
            "return_tokenized_data": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{SERVER_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        assert resp.status == 200
        return json.loads(resp.read())


def collect_numeric_pairs(reference, actual, label, pairs):
    if isinstance(reference, dict):
        assert tuple(reference) == tuple(actual)
        for key in reference:
            collect_numeric_pairs(reference[key], actual[key], f"{label}.{key}", pairs)
    elif isinstance(reference, list):
        assert len(reference) == len(actual)
        for idx, (left, right) in enumerate(zip(reference, actual)):
            collect_numeric_pairs(left, right, f"{label}.{idx}", pairs)
    elif isinstance(reference, float):
        assert isinstance(actual, float)
        pairs.append((label, reference, actual))
    else:
        assert reference == actual


def assert_numeric_pairs(pairs):
    assert all(math.isfinite(value) for _, left, right in pairs for value in (left, right))
    differences = [abs(left - right) for _, left, right in pairs]
    p95 = sorted(differences)[math.ceil(0.95 * len(differences)) - 1]
    worst = max(range(len(pairs)), key=differences.__getitem__)
    stats = (
        f"count={len(pairs)}, mean={sum(differences) / len(pairs):.6g}, p95={p95:.6g}, "
        f"max={differences[worst]:.6g}, "
        f"over_5pct={sum(value > LOGPROB_P95_ATOL for value in differences)}, "
        f"worst={pairs[worst]!r}"
    )
    assert p95 <= LOGPROB_P95_ATOL and differences[worst] <= LOGPROB_MAX_ATOL, stats


def run_server(args, prefix_cache=False):
    cmd = build_server_cmd(
        args.checkpoint_dir,
        args.tokenizer_model,
        args.server_log_dir,
        prefix_cache,
        args.prefix_cache_compare,
    )
    print(f"[smoke] spawning server: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=cleaned_env(),
    )

    ready = threading.Event()

    def watch():
        for line in proc.stdout:
            print(f"[server] {line}", end="", flush=True)
            if READINESS_MARKER in line:
                ready.set()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    try:
        if not ready.wait(READINESS_TIMEOUT_S):
            raise AssertionError(f"readiness banner not seen in {READINESS_TIMEOUT_S}s")

        time.sleep(2)

        if not args.prefix_cache_compare:
            body = post_completion()
            assert (body.get("choices") or [{}])[0].get("text")
        else:
            outputs, cached_by_wave, memory = [], [], []
            for cycle in range(args.prefix_cache_stress_cycles):
                shared = (f"deterministic shared prefix {cycle} for cache stress. " * 80).strip()
                for wave in (
                    [shared],
                    [shared] * 8,
                    [shared] * 8,
                    [(f"pressure {cycle:02d} {idx:03d}. " * 120).strip() for idx in range(32)],
                    [shared] * 8,
                    [shared] * 8,
                ):
                    cached = []
                    with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                        responses = executor.map(post_chat, wave)
                    for response in responses:
                        choice = response["choices"][0]
                        message = choice["message"]
                        token_ids = message["generation_token_ids"]
                        outputs.append((token_ids, message["content"], choice["logprobs"]))
                        cached.append(response["usage"]["prompt_tokens_details"]["cached_tokens"])
                    cached_by_wave.append((min(cached), max(cached)))
                output = subprocess.check_output(
                    "nvidia-smi --query-compute-apps=used_memory "
                    "--format=csv,noheader,nounits".split(),
                    text=True,
                )
                memory.append(sum(map(int, output.split())))
            return outputs, cached_by_wave, memory
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print("[smoke] server didn't exit on SIGTERM; SIGKILL", flush=True)
                proc.kill()
                proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--server-log-dir", default=None)
    parser.add_argument("--prefix-cache-compare", action="store_true")
    parser.add_argument("--prefix-cache-stress-cycles", type=int, default=3)
    args = parser.parse_args()
    if not args.prefix_cache_compare:
        run_server(args)
        print("[smoke] PASS", flush=True)
        return 0

    assert args.prefix_cache_stress_cycles >= 3
    reference, _, reference_memory = run_server(args, prefix_cache=False)
    cached, activation, memory = run_server(args, prefix_cache=True)
    pairs = []
    for idx, (ref_output, cached_output) in enumerate(zip(reference, cached, strict=True)):
        assert ref_output[:2] == cached_output[:2], f"HTTP output mismatch at request {idx}"
        assert len(ref_output[2]["content"]) == len(ref_output[0])
        collect_numeric_pairs(ref_output[2], cached_output[2], f"request {idx}.logprobs", pairs)
    assert_numeric_pairs(pairs)
    assert all(low == 0 < high for low, high in activation[1::6])
    assert all(value == (0, 0) for value in activation[4::6])
    assert all(low > 0 for wave in (activation[2::6], activation[5::6]) for low, _ in wave)
    assert max(memory) <= max(reference_memory) + 64 * 8
    assert memory[-1] <= memory[-2] + 64 * 8  # Allow 64 MiB/GPU for lazy workspaces.
    return 0


if __name__ == "__main__":
    sys.exit(main())
