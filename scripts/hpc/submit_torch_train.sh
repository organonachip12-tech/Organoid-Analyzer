#!/bin/bash
# Wrapper for NYU Torch: discovers your torch_pr_* Slurm account + partition (required since 2024+).
# Run from repo root on a login node: bash scripts/hpc/submit_torch_train.sh
#
# If discovery fails: https://projects.hpc.nyu.edu — your PI must register an HPC project and add you.

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/$USER}"
REPO="${REPO:-$SCRATCH/Organoid-Analyzer}"
cd "$REPO"

LINE=""
if command -v sacctmgr >/dev/null 2>&1; then
  LINE=$(sacctmgr -n -P show associations where user="$USER" format=account,partition 2>/dev/null | grep -E 'torch_pr_' | head -1) || true
fi

if [ -z "${LINE:-}" ]; then
  echo "Could not auto-find a torch_pr_* account for user=$USER."
  echo "Ask your PI to add you to an HPC project: https://projects.hpc.nyu.edu"
  echo "Docs: https://services.rt.nyu.edu/docs/hpc/getting_started/Slurm_Accounts/hpc_project_management_portal/"
  if command -v sacctmgr >/dev/null 2>&1; then
    echo "--- sacctmgr associations (if any) ---"
    sacctmgr show associations where user="$USER" format=account,partition 2>/dev/null || true
  fi
  exit 1
fi

ACCOUNT=$(echo "$LINE" | cut -d'|' -f1)
PARTITION=$(echo "$LINE" | cut -d'|' -f2 | cut -d',' -f1 | tr -d ' ' | tr -d '*')

if [ -z "$ACCOUNT" ] || [ -z "$PARTITION" ]; then
  echo "Parsed empty account/partition from: $LINE"
  exit 1
fi

echo "Submitting with --account=$ACCOUNT --partition=$PARTITION"

exec sbatch --account="$ACCOUNT" --partition="$PARTITION" --cpus-per-task=8 --mem=128G --time=48:00:00 \
  --job-name=gigatime-train --output=slurm-%x-%j.out --error=slurm-%x-%j.err \
  scripts/hpc/torch_train_pipeline.sh
