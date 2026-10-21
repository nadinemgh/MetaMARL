#!/bin/bash
set -euo pipefail
cd "$HOME/MetaMARL"
for condition in baseline tagging_fixed_0 tagging_fixed_20 tagging_fixed_40 tagging_fixed_50 tagging_fixed_80; do
    sbatch "slurm/run_${condition}.sh"
done
squeue -u "$USER"
