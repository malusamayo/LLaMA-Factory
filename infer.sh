#!/bin/sh
#SBATCH --job-name=aigc_inference
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


cd ~/LLaMA-Factory/
# python3 scripts/vllm_infer.py --model_name_or_path ./saves/qwen2_5vl-3b/full/sft/checkpoint-2531/ \
#     --template qwen2_vl --dataset chameleon --pipeline_parallel_size 2 \
#     --save_name generated_predictions_qwen2_vl.jsonl

# python3 scripts/vllm_infer.py --model_name_or_path ./saves/qwen2_5vl-3b/full/sft_fakeclue/checkpoint-816/ \
#     --template qwen2_vl --dataset fakeclue_eval --pipeline_parallel_size 2 \
#     --save_name generated_predictions_fakeclue_qwen2_5vl-3b.jsonl

# python3 scripts/vllm_infer.py --model_name_or_path ./saves/gemma-3-4b-it/full/sft/checkpoint-2532/ \
#     --template gemma3 --dataset chameleon --pipeline_parallel_size 2 \
#     --save_name generated_predictions_gemma-3-4b.jsonl

# python3 scripts/hg_infer.py --model_name_or_path ./saves/gemma-3-4b-it/full/sft/checkpoint-2532/ \
#     --file_name oss_chameleon_test.json \
#     --save_name generated_predictions_gemma-3-4b.jsonl

python3 scripts/hg_infer.py --model_name_or_path ./saves/gemma-3-4b-it/full/sft_fakeclue/checkpoint-816/ \
    --file_name oss_fakeclue_image_val.json \
    --save_name generated_predictions_fakeclue_gemma-3-4b.jsonl

# llamafactory-cli eval --model_name_or_path ./saves/qwen2_5vl-3b/full/sft/checkpoint-2531/ \
#     --template qwen2_vl --dataset chameleon
