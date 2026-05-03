#!/bin/bash
# SLURM: run GigaTIME training pipeline on a Torch *compute* node (not the login node).
# Login nodes OOM-kill heavy SVS preprocessing (exit 137). Submit this with: sbatch torch_train_pipeline.sh
#
# Before first use:
#   1) sinfo                    — pick a partition your PI/account can use
#   2) sacctmgr show assoc user:$USER  — note Account / Partition if required
#   3) Edit #SBATCH lines below (--partition, --account, --mem, --time).
#
# Docs: https://hpc.nyu.edu/docs/hpc/running_jobs/batch_slurm/

#SBATCH --job-name=gigatime-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
## SBATCH --partition=FILL_ME          # e.g. cpu, faculty-..., student-... per `sinfo`
## SBATCH --account=FILL_ME          # if your lab requires -A / --account=
#
# GPU (optional — uncomment if your partition gives GPUs and you want CUDA for inference):
## SBATCH --gres=gpu:1

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/$USER}"
export SCRATCH
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TMPDIR="$SCRATCH/tmp"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="${HF_HOME:-$SCRATCH/.hf}"
export MPLCONFIGDIR="$SCRATCH/.organoid_analyzer_cache/matplotlib"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR"

VENV="${VENV:-$SCRATCH/venv/organoid}"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

REPO="${REPO:-$SCRATCH/Organoid-Analyzer}"
cd "$REPO"

SVS_DIR="${SVS_DIR:-$SCRATCH/gdc_svs}"
ANNOTATIONS="${ANNOTATIONS:-data/gigatime/annotations.csv}"
COX_OUT="${COX_OUT:-$SCRATCH/cox_model.pkl}"

echo "=== $(date) host=$(hostname) cwd=$(pwd)"
echo "=== SVS_DIR=$SVS_DIR COX_OUT=$COX_OUT"

exec python -m gigatime_analyzer.scripts.run_pipeline --mode train \
  --svs_dir "$SVS_DIR" \
  --annotations_csv "$ANNOTATIONS" \
  --cox_model_path "$COX_OUT"
