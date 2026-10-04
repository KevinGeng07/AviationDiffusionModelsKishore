#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: CFM (flow matching) + SWI_DiT training run, ADS-B trajectory
# ---------------------------------------------------------------------------
#SBATCH --job-name=cfm_swi_dit_trajectory_full
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/cfm_swi_dit_%j.out
#SBATCH --error=logs/cfm_swi_dit_%j.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
mkdir -p logs

module load anaconda3
source activate adsb

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
REPO=/global/scratch/users/kevingeng/aviation-bayen/AviationDiffusionModelsKishore
OUTPUT_DIR=$REPO/training/checkpoints/swi_dit

mkdir -p $OUTPUT_DIR

cd $REPO/training

python train_cfm_swi_dit_savio.py \
    --nc_path $NC_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs 100 \
    --batch_size 64

echo "Job finished at $(date)"
