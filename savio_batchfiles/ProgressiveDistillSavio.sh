#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: CFM + SWI_DiT progressive distillation (fewer sampling steps), 1 GPU
#
# Run settings can be passed at submit time (defaults below):
#   sbatch --job-name=<name> --export=ALL,RUN_NAME=<name>,BASE=6_4_256,LR=5e-5 \
#          savio_batchfiles/ProgressiveDistillSavio.sh
#
# BASE is any model spec accepted by train_cfm_swi_dit_progressive_savio.py --base:
# a shorthand like 6_4_256 (→ finished_ckpt/6_4_256_SWiDiT), 'teacher', a run folder or a .pt.
# Resubmitting with the same RUN_NAME skips finished rounds and resumes the current one.
# ---------------------------------------------------------------------------
#SBATCH --job-name=progressive_distill
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out      # %x = job name, %j = job id
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
mkdir -p logs

module load anaconda3
source activate adsb

PYTHON=$HOME/.conda/envs/adsb/bin/python
echo "Python: $PYTHON"

$PYTHON -c "import torch, numpy, netCDF4; print('torch', torch.__version__, '| netCDF4', netCDF4.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" \
    || { echo "Environment check failed (see .err log)"; exit 1; }

# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
REPO=/global/scratch/users/kevingeng/aviation-bayen/AviationDiffusionModelsKishore

# Per-run settings (override with sbatch --export=ALL,VAR=value,...)
BASE=${BASE:-6_4_256}
RUN_NAME=${RUN_NAME:-progressive_$BASE}
LR=${LR:-5e-5}
MIN_LR_FRAC=${MIN_LR_FRAC:-0.2}
EPOCHS_PER_ROUND=${EPOCHS_PER_ROUND:-10}
START_STEPS=${START_STEPS:-32}
MIN_STEPS=${MIN_STEPS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}
OUTPUT_DIR=$REPO/training/checkpoints/$RUN_NAME

echo "Run: $RUN_NAME | base: $BASE | lr=$LR (floor ×$MIN_LR_FRAC) | epochs/round=$EPOCHS_PER_ROUND | steps $START_STEPS → $MIN_STEPS | batch=$BATCH_SIZE | seed=$SEED"
echo "Output: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR

# ---------------------------------------------------------------------------
# Output-folder guard: never let two jobs write the same round folders
# ---------------------------------------------------------------------------
LOCK=$OUTPUT_DIR/.active_job
if [ -f "$LOCK" ]; then
    OTHER=$(cat "$LOCK")
    if [ "$OTHER" != "$SLURM_JOB_ID" ] && squeue -h -j "$OTHER" 2>/dev/null | grep -q .; then
        echo "ERROR: $OUTPUT_DIR is in use by job $OTHER (still queued/running)."
        echo "       Submit this run with a different RUN_NAME."
        exit 1
    fi
fi
echo "$SLURM_JOB_ID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd $REPO

# -u: unbuffered stdout so per-epoch lines appear in logs/ as they happen.
# Per-round metrics go to $OUTPUT_DIR/steps_NN/metrics.csv.
$PYTHON -u training/train_cfm_swi_dit_progressive_savio.py \
    --nc_path $NC_PATH \
    --base $BASE \
    --output_dir $OUTPUT_DIR \
    --lr $LR \
    --min_lr_frac $MIN_LR_FRAC \
    --epochs_per_round $EPOCHS_PER_ROUND \
    --start_steps $START_STEPS \
    --min_steps $MIN_STEPS \
    --batch_size $BATCH_SIZE \
    --seed $SEED
STATUS=$?

echo "Job finished at $(date) with exit code $STATUS"
exit $STATUS    # non-zero → Slurm marks the job FAILED instead of COMPLETED
