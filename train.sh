#!/bin/sh
#SBATCH --job-name=aigc_finetune_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300G
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64

echo "Number of nodes: $SLURM_NNODES"
export HYDRA_FULL_ERROR=1

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

export DISABLE_VERSION_CHECK=1

# Getting the node names
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_addr" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

printenv

# srun torchrun \
#     --nnodes $SLURM_NNODES \
#     --nproc-per-node $SLURM_GPUS_PER_NODE \
#     --rdzv-backend c10d \
#     --rdzv-endpoint $head_node_ip:$port \
#     --rdzv_id $SLURM_JOB_ID \
#     ./src/train.py examples/train_full/qwen2_5vl_3b_full_sft.yaml

srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc-per-node $SLURM_GPUS_PER_NODE \
    --rdzv-backend c10d \
    --rdzv-endpoint $head_node_ip:$port \
    --rdzv_id $SLURM_JOB_ID \
    ./src/train.py examples/train_full/gemma-3-4b-it_full_sft.yaml