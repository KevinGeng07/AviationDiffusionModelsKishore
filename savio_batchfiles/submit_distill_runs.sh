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
#   - Runs whose folder already exists are skipped (never overwritten). To continue one:
#       bash savio_batchfiles/submit_distill_run.sh --name <name> --student <cfg> --resume
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

# each run goes through submit_distill_run.sh, which skips (does not overwrite) any run
# whose folder already exists or whose job is already queued/running
for run in "${RUNS[@]}"; do
  read -r name cfg alpha <<< "$run"
  bash savio_batchfiles/submit_distill_run.sh --name "$name" --student "$cfg" --alpha "$alpha" \
      || echo "  → skipped $name"
done

squeue -u "$USER" -o "%.10i %.22j %.3t %.10M %b %R"
