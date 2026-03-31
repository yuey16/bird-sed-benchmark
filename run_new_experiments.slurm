#!/bin/bash
#SBATCH --job-name=bird-sed-adv
#SBATCH --partition=256C8G1H_MI325X_Ubuntu22
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation
#SBATCH --nodelist=gpu-21
#SBATCH --output=slurm_adv_%j.out
#SBATCH --error=slurm_adv_%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $SLURM_GPUS_ON_NODE"
echo "Date:   $(date)"
echo "=========================================="

export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token_here"}
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

echo ""
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
echo ""

# Run GPU experiments first (6, 7), then CPU-only ones (4, 5)
# BirdNET (exp5) imports TensorFlow which can steal GPU memory from PyTorch
python3 run_all.py 6 7 4 5

echo ""
echo "=========================================="
echo "ALL ADVANCED EXPERIMENTS COMPLETE"
echo "Date: $(date)"
echo "=========================================="
