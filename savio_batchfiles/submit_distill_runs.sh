#!/bin/bash
# ---------------------------------------------------------------------------
# Submit several distillation runs at once — one Slurm job (= 1 GPU) per run.
#
# Usage, on Savio from anywhere:
#   bash savio_batchfiles/submit_distill_runs.sh
#
# Edit RUNS below: one line per run →  <run_name>  <student_config>  <alpha>
#   - run_name   : job name, log prefix (logs/<run_name>_<jobid>.out) and output folder
#                  (training/checkpoints/<run_name>/). Must be unique per run.
#   - Re-submitting a run with the same name resumes it from its own last.pt.
#   - Runs beyond your GPU QoS limit wait as PD and start when a GPU frees up.
# ---------------------------------------------------------------------------
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p logs        # Slurm writes logs/ relative to the submit directory

RUNS=(
  "distill_2L_256_a07   configs/students/swi_dit_2L_256.json   0.7"
  "distill_4L_128_a07   configs/students/swi_dit_4L_128.json   0.7"
  "distill_4L_256_a07   configs/students/swi_dit_4L_256.json   0.7"
)

for run in "${RUNS[@]}"; do
  read -r name cfg alpha <<< "$run"
  [ -f "$cfg" ] || { echo "Missing student config: $cfg"; exit 1; }
  sbatch --job-name="$name" \
         --export=ALL,RUN_NAME="$name",STUDENT_CONFIG="$REPO/$cfg",ALPHA="$alpha" \
         savio_batchfiles/CFMSwiDITDistillSavio.sh
done

squeue -u "$USER" -o "%.10i %.22j %.3t %.10M %b %R"
