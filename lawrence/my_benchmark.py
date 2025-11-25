# lawrence mcafee

# python -m lawrence.my_benchmark --incoming-requests-per-sec 128 -n 2000 -u 0 -b 25
# (cd megatron-inference-benchmarking/ && bash dynamic_benchmarks/mcore_benchmark_llama3.1-8b.sh --checkpoint /lustre/fsw/portfolios/llmservice/users/ksanthanam/llama3.1-8b-mcore/ --hf-home /lustre/fsw/portfolios/llmservice/users/ksanthanam/hf_home/ --output-dir mcore_outputs/ --tp 1)

# pip install transformers
# pip install simpy
# pip install flashinfer-python

# ~~~~~~~~ import ~~~~~~~~
import argparse
import os
import subprocess

CHECKPOINT = "/lustre/fsw/portfolios/llmservice/users/ksanthanam/llama3.1-8b-mcore/"
HF_HOME = "/lustre/fsw/portfolios/llmservice/users/ksanthanam/hf_home/"
# OUTPUT_DIR = "mcore_outputs/"
TP = 1

# os.environ["PIP_CONSTRAINT"] = '"" pip install flashinfer-python'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming-requests-per-sec", type=int, required=True)
    parser.add_argument("-n", "--max-num-requests", type=int, required=True)
    parser.add_argument("-u", "--unified-memory-level", type=int, required=True,
                        choices=[0, 1])
    parser.add_argument("-b", "--buffer-size-gb", type=int, required=True)
    args = parser.parse_args()

    os.environ["MAX_NUM_REQUESTS"] = str(args.max_num_requests)

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "16"
    # os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "16"
    # os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    # os.environ["HF_HOME"] = HF_HOME

    SERVER_PORT = 12346
    # UVM_LEVEL = 0
    # BUFFER_SIZE_GB = 25

    MODEL_ARGS = [
        "--load", CHECKPOINT,
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", "meta-llama/Llama-3.1-8B",
        "--max-position-embeddings", "8192",
        "--seq-length", "8192",
        "--te-rng-tracker",
        "--no-use-tokenizer-model-from-checkpoint-args",
        "--inference-max-seq-length", "8192",
        "--use-checkpoint-args",
        "--transformer-impl", "transformer_engine",
        "--attention-backend", "flash",
        "--inference-ckpt-non-strict",
    ]

    INFERENCE_ARGS = [
        "--bf16",
        "--micro-batch-size", "1",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--tensor-model-parallel-size", str(TP),
        "--pipeline-model-parallel-size", "1",
        "--seed", "12345",
        "--inference-rng-tracker",
        "--inference-dynamic-batching",
        "--inference-ckpt-non-strict",
        # "--inference-dynamic-batching-buffer-size-gb", str(BUFFER_SIZE_GB),
        # "--inference-dynamic-batching-unified-memory-level", str(UVM_LEVEL),
        "--cuda-graph-impl", "local",
        "--cuda-graph-scope", "full",
        "--inference-dynamic-batching-num-cuda-graphs", "32",
        "--top_p", "-1", "--top_k", "1", "--use-flashinfer-fused-rope",
        "--disable-chunked-prefill",
        "--inference-coordinator-port", str(SERVER_PORT),

        # >>>
        "--prompt-file", "/lustre/fsw/portfolios/llmservice/users/ksanthanam/sharegpt_filtered_benchmark.json",
        "--incoming-requests-per-sec", str(args.incoming_requests_per_sec), # 32, 64, 128

        "--inference-dynamic-batching-buffer-size-gb", str(args.buffer_size_gb),
        "--inference-dynamic-batching-unified-memory-level", str(args.unified_memory_level),

        # <<<
    ]

    subprocess.run([

        # "torchrun",
        # "--nproc_per_node",
        # str(TP) --no_python \
        # python benchmark/run_megatron_server.py \

        "python",
        "-m",
        "examples.inference.gpt.gpt_dynamic_inference_with_coordinator",

        *MODEL_ARGS,
        *INFERENCE_ARGS,

    ])

# eof
