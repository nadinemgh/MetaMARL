#!/bin/bash
#SBATCH --account=def-gdumas85
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --output=slurm-%x-%j.out
#SBATCH --job-name=water-tag-50

set -euo pipefail

cd "$HOME/MetaMARL"

module load python/3.12
source .venv/bin/activate

# Prevent BLAS/PyTorch from oversubscribing CPU cores inside Ray workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

EXPERIMENT="tagging_fixed_50"
JOB_ROOT="$SCRATCH/metamarl/experiments/$EXPERIMENT/$SLURM_JOB_ID"
mkdir -p "$JOB_ROOT/wandb" "$JOB_ROOT/wandb_cache"

# The reporter creates its own W&B run; offline files remain on SCRATCH.
export WANDB_MODE=offline
export WANDB_PROJECT=MetaMARL
export WANDB_RUN_GROUP="schelling_2026_10"
export WANDB_NAME="$EXPERIMENT-job-$SLURM_JOB_ID"
export WANDB_DIR="$JOB_ROOT/wandb"
export WANDB_CACHE_DIR="$JOB_ROOT/wandb_cache"

# Node-local scratch is isolated for each job and avoids Raven cache collisions.
NODE_WORK="${SLURM_TMPDIR:-$JOB_ROOT/node_tmp}"
mkdir -p "$NODE_WORK"
export TMPDIR="$NODE_WORK/tmp"
export RAY_TMPDIR="$NODE_WORK/ray"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

export RAVEN_CWD="$NODE_WORK/raven"
mkdir -p "$RAVEN_CWD"
rsync -a --exclude='.cache' \
    "$HOME/MetaMARL/examples/fresh_water/raven/" \
    "$RAVEN_CWD/"
export RAVEN_CMD="$RAVEN_CWD/2_Raven/Raven.exe"

# Check that the executable can run on Narval's Linux nodes.
test -d "$RAVEN_CWD"
test -x "$RAVEN_CMD"
if file "$RAVEN_CMD" | grep -q 'Mach-O'; then
    echo "ERROR: Raven.exe is macOS Mach-O. Build a Linux executable for Narval." >&2
    exit 1
fi

printf 'Experiment: %s\nJob ID: %s\nHost: %s\nRAVEN_CWD: %s\nRAVEN_CMD: %s\nWANDB_DIR: %s\n' \
    "$EXPERIMENT" "$SLURM_JOB_ID" "$(hostname)" "$RAVEN_CWD" "$RAVEN_CMD" "$WANDB_DIR"

python -m examples.fresh_water.run_tagging_fixed_50

printf 'Completed: %s\nW&B offline files: %s\n' "$EXPERIMENT" "$WANDB_DIR"
