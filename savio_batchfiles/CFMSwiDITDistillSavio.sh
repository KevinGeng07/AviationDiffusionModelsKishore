#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: CFM + SWI_DiT knowledge distillation (1 GPU per job), ADS-B trajectory
#
# Run settings can be passed at submit time (defaults below), so several runs can be
# submitted as separate 1-GPU jobs — see savio_batchfiles/submit_distill_runs.sh:
#   sbatch --job-name=<name> --export=ALL,RUN_NAME=<name>,STUDENT_CONFIG=<json>,ALPHA=<a> \
#          savio_batchfiles/CFMSwiDITDistillSavio.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=cfm_swi_dit_distill
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

# Call the env's interpreter by path so the job can't fall back to another Python
# (e.g. base anaconda, which has torch but not netCDF4) via the inherited environment.
PYTHON=$HOME/.conda/envs/adsb/bin/python
echo "Python: $PYTHON"

# Fail fast with a clear message if the env or GPU isn't usable.
$PYTHON -c "import torch, numpy, netCDF4; print('torch', torch.__version__, '| netCDF4', netCDF4.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" \
    || { echo "Environment check failed (see .err log)"; exit 1; }

# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
REPO=/global/scratch/users/kevingeng/aviation-bayen/AviationDiffusionModelsKishore
TEACHER_CKPT=$REPO/training/checkpoints/swi_dit/best_swi_dit.pt

# Per-run settings (override with sbatch --export=ALL,VAR=value,...)
RUN_NAME=${RUN_NAME:-swi_dit_distill_4L}
STUDENT_CONFIG=${STUDENT_CONFIG:-$REPO/configs/swi_dit_student.json}
ALPHA=${ALPHA:-0.7}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}
# one output folder per run: runs must never share last.pt / best.pt / metrics.csv
OUTPUT_DIR=$REPO/training/checkpoints/$RUN_NAME

echo "Run: $RUN_NAME | student: $STUDENT_CONFIG | alpha=$ALPHA | epochs=$EPOCHS | batch=$BATCH_SIZE | seed=$SEED"
echo "Output: $OUTPUT_DIR"
[ -f "$STUDENT_CONFIG" ] || { echo "Student config not found: $STUDENT_CONFIG"; exit 1; }

mkdir -p $OUTPUT_DIR

cd $REPO/training

# -u: unbuffered stdout so per-epoch lines appear in logs/ as they happen.
# Per-epoch student metrics are also appended to $OUTPUT_DIR/metrics.csv.
$PYTHON -u train_cfm_swi_dit_distill_savio.py \
    --nc_path $NC_PATH \
    --teacher_config $REPO/configs/swi_dit_teacher.json \
    --student_config $STUDENT_CONFIG \
    --teacher_ckpt $TEACHER_CKPT \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --alpha $ALPHA \
    --seed $SEED
STATUS=$?

echo "Job finished at $(date) with exit code $STATUS"
exit $STATUS    # non-zero → Slurm marks the job FAILED instead of COMPLETED
