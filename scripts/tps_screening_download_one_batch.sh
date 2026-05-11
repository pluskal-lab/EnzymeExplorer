#!/bin/bash
# Per-batch AF-DB structure downloader. Spawned as a SLURM array task
# by tps_screening_manager.sh; $SLURM_ARRAY_TASK_ID picks which batch
# this task downloads structures for.
#
# Usage (positional, set by the manager):
#   tps_screening_download_one_batch.sh \
#       <fasta> <batch_size> <structures_root> <missing_csvs_root>
#
# Slice covered by this task: [batch * batch_size, (batch+1) * batch_size).
# Output:
#   <structures_root>/batch_<idx>/<uid>.pdb           per downloaded PDB
#   <missing_csvs_root>/batch_<idx>.csv               UniProt IDs missing from AF-DB
#
# CPU-only. No GPU requested — AF-DB ingress is the bottleneck.

#SBATCH --job-name=tps_screen_dl
#SBATCH --partition=standard
#SBATCH --account=project_465000660
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=12:00:00

# See manager note: cluster bashrc files may reference unbound vars;
# keep -u off while sourcing, re-enable for our own code.
set -eo pipefail
source ~/.bashrc
conda activate enzyme_explorer
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fasta="$1"
batch_size="$2"
structures_root="$3"
missing_csvs_root="$4"

batch_idx="${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array task}"
start_i=$(( batch_idx * batch_size ))
end_i=$(( start_i + batch_size ))
batch_name=$(printf "batch_%06d" "$batch_idx")

batch_struct_dir="$structures_root/$batch_name"
missing_csv="$missing_csvs_root/$batch_name.csv"

echo "[dl] batch $batch_idx  range [$start_i, $end_i)"
echo "[dl] structures_dir = $batch_struct_dir"
echo "[dl] missing_csv    = $missing_csv"

python -m enzymeexplorer.src.screening.tps_download_af_structures \
    --fasta-path "$fasta" \
    --output-dir "$batch_struct_dir" \
    --missing-csv "$missing_csv" \
    --start-i "$start_i" \
    --end-i   "$end_i" \
    --n-workers 16
