#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: test-set ADE/FDE for finished checkpoints (eval/evaluate.py), 1 GPU
#
# Usage (on Savio, from the repo root):
#   sbatch savio_batchfiles/EvalSavio.sh                                  # every folder in FOLDERS in eval/evaluate.py
#   sbatch savio_batchfiles/EvalSavio.sh finished_ckpt/6_5_SWiDiT training/checkpoints/distill_4L_384_a07
#   EVAL_ARGS="--n_samples 5 --max_batches 50" sbatch savio_batchfiles/EvalSavio.sh   # extra evaluate.py options
#
# Results (per model + a side-by-side table at the end) go to logs/eval_<jobid>.out
# ---------------------------------------------------------------------------
#SBATCH --job-name=eval
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out      # %x = job name, %j = job id
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load anaconda3
source activate adsb
PYTHON=$HOME/.conda/envs/adsb/bin/python

NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
REPO=/global/scratch/users/kevingeng/aviation-bayen/AviationDiffusionModelsKishore
cd $REPO

echo "Folders: ${*:-FOLDERS list in eval/evaluate.py} | extra args: ${EVAL_ARGS:-none}"
# -u: unbuffered so results show up in the log as each model finishes
$PYTHON -u eval/evaluate.py "$@" --nc_path $NC_PATH --num_workers ${SLURM_CPUS_PER_TASK:-2} $EVAL_ARGS
