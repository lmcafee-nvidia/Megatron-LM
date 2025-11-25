MEGATRON_DIR="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/users/lmcafee/inference/megatrons/optimize-context-uvm"
ACCOUNT="llmservice_nlp_fm"

IMAGE="/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/torch2305-symmetric-fa3.sqsh"
# IMAGE="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/users/lmcafee/images/pytorch-25.05.sqsh"

echo "~~~"
echo "image ... '${IMAGE}'."
echo "~~~"

srun -p interactive --account=$ACCOUNT -t 4:00:00 --nodes=1 --exclusive --gpus-per-node=8 \
--container-image=${IMAGE} \
--container-mounts="/home:/home,/lustre:/lustre" \
--container-workdir=${MEGATRON_DIR}/megatron-inference-benchmarking  \
--pty bash
