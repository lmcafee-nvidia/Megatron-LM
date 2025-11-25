# lawrence mcafee

# ~~~~~~~~ command ~~~~~~~~
# # For debugging.
# RUN apt-get update && apt-get install -y gdb strace && apt-get install -y vim
# torchrun --nproc_per_node=1 -m ...
# python -m scripts_ignore_me.run_inference -e mcore-dynamic -m 357m -s 8 -u 0 -g 4

# ~~~~~~~~ import ~~~~~~~~
import argparse
# import datetime
import os
import subprocess

# USER_DIR = "/lustre/fs11/portfolios/adlr/users/lmcafee"
USER_DIR = "/lustre/fsw/portfolios/adlr/users/lmcafee"

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # ~~~~~~~~ args ~~~~~~~~
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="357m",
                        choices=["357m", "12b"])
    parser.add_argument("-r", "--repo")
    # parser.add_argument("-n", "--nsight", action="store_true")
    # parser.add_argument("-b", "--launch-block", action="store_true")
    parser.add_argument("-g", "--num-graphs", type=int, default=0)
    parser.add_argument("-s", "--suspend-resume-interval", type=int, required=True) # , default=999999)
    parser.add_argument("-u", "--unified-memory-level", type=int, required=True) # default=0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("-c", "--use_coordinator", action="store_true")
    parser.add_argument("-d", "--duration", type=float, default=1.)
    args = parser.parse_args()

    if args.repo is None:
        REPO_DIR = os.getcwd()
        args.repo = os.path.basename(REPO_DIR)
    else:
        REPO_DIR = f"{USER_DIR}/inference/megatrons/{args.repo}"

    # pax({"REPO_DIR": REPO_DIR, "repo": args.repo})

    # ~~~~~~~~ model ~~~~~~~~
    BASE_CKPT_DIR = f"{USER_DIR}/checkpoints"
    if args.model == "357m":
        os.environ["CHECKPOINT_DIR"] = f"{BASE_CKPT_DIR}/357m/core-local-tp1-pp1"
        os.environ["VOCAB_FILE"] = f"{BASE_CKPT_DIR}/357m/vocab/gpt2-vocab.json"
        os.environ["MERGE_FILE"] = f"{BASE_CKPT_DIR}/357m/vocab/gpt2-merges.txt"

    elif args.model == "12b":
        os.environ["CHECKPOINT_DIR"] = f"{BASE_CKPT_DIR}/12b/core-local-tp1-pp1"
        os.environ["TOKENIZER_MODEL"] = f"{BASE_CKPT_DIR}/12b/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"

    else:
        raise Exception(f"specialize for model '{args.model}'.")

    # ~~~~~~~~ env ~~~~~~~~
    if 0:
        os.environ["NUM_TOKENS_TO_PROMPT"] = "4 7" # "4 32"
        os.environ["NUM_TOKENS_TO_GENERATE"] = "16"
        os.environ["INCOMING_REQUESTS_DURATION"] = str(args.duration)
        os.environ["INCOMING_REQUESTS_PER_SEC"] = "100." # 100
    else:
        os.environ["PROMPTS"] = " ".join(f'"{p}"' for p in (
            "Lawrence would like to",
            "NVIDIA is best at",
            "The inventor of the GPU is",
            "Michigan is best known for",
            "All I want for Christmas is",
        ))
        os.environ["NUM_TOKENS_TO_GENERATE"] = "64" # *16

    active_buffer_size_gb = 40. # 50
    if args.unified_memory_level == 0:
        active_buffer_size_gb /= 2
    # pax("active_buffer_size_gb")
    os.environ["ACTIVE_BUFFER_SIZE_GB"] = str(active_buffer_size_gb)

    os.environ["EXTRA_ARGS"] = ""
    # if args.launch_block:
    #     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ~~~~~~~~ nsight ~~~~~~~~
    # if args.nsight:
    #     date_str = datetime.date.today().strftime("%Y%m%d")
    #     os.environ["NSIGHT_PREFIX"] = f"{REPO_DIR}/scripts/nsight/{date_str}/{args.engine}-{args.model}"
    #     os.makedirs(os.path.dirname(os.environ["NSIGHT_PREFIX"]), exist_ok=True)

    # ~~~~~~~~ run ~~~~~~~~
    del os.environ["NCCL_DEBUG"]
    os.environ["EXTRA_ARGS"] += " --inference-ckpt-non-strict"
    os.environ["EXTRA_ARGS"] += f" --inference-dynamic-batching-unified-memory-level {args.unified_memory_level}"
    os.environ["EXTRA_ARGS"] += " --return-log-probs"
    os.environ["EXTRA_ARGS"] += " --inference-repeat-n 2"
    os.environ["NUM_CUDA_GRAPHS"] = str(args.num_graphs)
    if args.max_requests is not None:
        os.environ["EXTRA_ARGS"] += f" --inference-dynamic-batching-max-requests-override {args.max_requests}"
    if args.suspend_resume_interval is not None:
        os.environ["EXTRA_ARGS"] += f" --suspend-resume-interval {args.suspend_resume_interval}"
    os.environ["USE_COORDINATOR"] = str(int(args.use_coordinator))

    subprocess.run([
        "bash",
        f"examples/inference/gpt/gpt_dynamic_inference_{args.model}.sh",
    ], cwd=REPO_DIR)

# eof
