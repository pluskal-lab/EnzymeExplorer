#!/bin/bash
# Per-batch cleanup of AF-DB downloads. Spawned as a SLURM array task
# by tps_screening_manager.sh AFTER the corresponding prediction task
# has finished (via --dependency=aftercorr). Removes the
# batch_<idx>/ directory of downloaded PDBs so the cluster scratch
# fills don't accumulate.
#
# Only scheduled when the manager triggered the AF-DB download array;
# pre-existing externally-managed structures directories are never
# touched.
#
# Usage (positional, set by the manager):
#   tps_screening_cleanup_batch.sh <structures_root>

#SBATCH --job-name=tps_screen_cleanup
#SBATCH --partition=standard
#SBATCH --account=project_465000660
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:30:00

set -euo pipefail

structures_root="$1"
batch_idx="${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array task}"
batch_name=$(printf "batch_%06d" "$batch_idx")
batch_dir="$structures_root/$batch_name"

if [[ -d "$batch_dir" ]]; then
    echo "[cleanup] removing $batch_dir"
    rm -rf "$batch_dir"
else
    echo "[cleanup] nothing to remove at $batch_dir"
fi
