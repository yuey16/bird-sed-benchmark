#!/bin/bash
#SBATCH --job-name=bird-sed-exps
#SBATCH --partition=256C8G1H_MI325X_Ubuntu22
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation
#SBATCH --nodelist=gpu-21
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

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
echo "Python: $(which python3)"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA avail: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
echo ""

echo "######################################################################"
echo "# Experiment 1: Energy-VAD"
echo "######################################################################"
python3 exp1_energy_vad.py
echo ""
echo "[Exp 1 done at $(date)]"
echo ""

echo "######################################################################"
echo "# Experiment 2: Off-the-shelf Speech VAD (Silero)"
echo "######################################################################"
python3 exp2_silero_vad.py
echo ""
echo "[Exp 2 done at $(date)]"
echo ""

echo "######################################################################"
echo "# Experiment 3: Speech VAD + BEATs Linear Classifier"
echo "######################################################################"
python3 exp3_vad_beats_classifier.py
echo ""
echo "[Exp 3 done at $(date)]"
echo ""

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
