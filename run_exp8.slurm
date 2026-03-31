#!/bin/bash
#SBATCH --job-name=bird-exp8
#SBATCH --partition=256C8G1H_MI325X_Ubuntu22
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation
#SBATCH --nodelist=gpu-21
#SBATCH --output=slurm_exp8_%j.out
#SBATCH --error=slurm_exp8_%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "=========================================="

export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token_here"}
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
echo ""

python3 run_all.py 8

echo ""
echo "=========================================="
echo "EXP 8 COMPLETE — $(date)"
echo "=========================================="
