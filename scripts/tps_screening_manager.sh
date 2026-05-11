#!/bin/bash
# Top-level orchestrator for FASTA TPS screening on a SLURM cluster.
#
# This is a tiny CPU job that:
#   1. Counts records in the input FASTA.
#   2. Splits them into ``n_batches = ceil(N / batch_size)`` slices.
#   3. Submits four dependent SLURM array jobs:
#        a. (optional) AF-DB structure download   — CPU,  one task per batch
#        b. TPS prediction                        — 1 GPU, one task per batch
#        c. (optional) per-batch structure cleanup — CPU,  one task per batch
#        d. final results unification             — CPU,  one task total
#   4. Exits immediately. All actual screening happens in the spawned
#      array jobs.
#
# Each batch runs on its own node with one GPU. There is no Python-side
# multi-GPU forking — SLURM handles concurrency.
#
# Usage (sbatch):
#   sbatch scripts/tps_screening_manager.sh \
#       <fasta>          \
#       <output_root>    \
#       <classifier>     \      # plm | plm_domains | both
#       <batch_size>     \      # e.g. 40000
#       [<structures_dir>]      # optional; when set, skips the download
#                               # step and points every prediction task at
#                               # the same shared directory of <uid>.pdb files
#
# Output layout under <output_root>:
#   shards/<classifier>/batch_<idx>.csv      per-batch raw predictions
#   structures/batch_<idx>/<uid>.pdb         only when structures were downloaded
#   missing_csvs/batch_<idx>.csv             only when structures were downloaded
#   plm.csv  plm_domains.csv  plm_domains_fallback.csv  no_structure.csv
#                                            final unified outputs (gather step)

#SBATCH --job-name=tps_screening_mgr
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=01:00:00

set -euo pipefail

source ~/.bashrc
conda activate enzyme_explorer

# ---- args ----------------------------------------------------------------

if (( $# < 4 )); then
    sed -n '2,40p' "$0" >&2
    exit 1
fi
fasta="$1"
output_root="$2"
classifier="$3"
batch_size="$4"
structures_dir="${5:-}"

case "$classifier" in plm|plm_domains|both) ;;
  *) echo "[mgr] ERROR: classifier must be plm|plm_domains|both, got '$classifier'" >&2; exit 1 ;;
esac
[[ -f "$fasta" ]] || { echo "[mgr] ERROR: fasta not found: $fasta" >&2; exit 1; }
[[ "$batch_size" =~ ^[0-9]+$ && "$batch_size" -ge 1 ]] || {
    echo "[mgr] ERROR: batch_size must be a positive integer (got '$batch_size')" >&2; exit 1; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scripts_dir="$repo_root/scripts"
cd "$repo_root"

mkdir -p "$output_root/shards"

# ---- count records → number of batches -----------------------------------

n_seqs=$(python -c "from Bio import SeqIO; print(sum(1 for _ in SeqIO.parse('$fasta','fasta')))")
n_batches=$(( (n_seqs + batch_size - 1) / batch_size ))
last_idx=$(( n_batches - 1 ))
echo "[mgr] $n_seqs sequences  /  batch_size=$batch_size  →  n_batches=$n_batches"

if (( n_batches < 1 )); then
    echo "[mgr] ERROR: empty FASTA" >&2; exit 1
fi

# ---- helper: format batch dir suffix --------------------------------------
# Kept consistent with the .sh scripts (batch_$(printf "%06d" idx)).

# ---- decide whether we need an AF-DB download step -----------------------

needs_structures=0
case "$classifier" in plm_domains|both) needs_structures=1 ;; esac

dl_array_id=""
predict_extra=()
if (( needs_structures )) && [[ -z "$structures_dir" ]]; then
    mkdir -p "$output_root/structures" "$output_root/missing_csvs"
    dl_array_id=$(sbatch --parsable \
        --array=0-${last_idx} \
        "$scripts_dir/tps_screening_download_one_batch.sh" \
            "$fasta" "$batch_size" \
            "$output_root/structures" \
            "$output_root/missing_csvs")
    echo "[mgr] submitted download array job: $dl_array_id"
    # Each predict task looks under <structures_root>/batch_<idx>/.
    predict_extra+=(--structures-root "$output_root/structures")
elif (( needs_structures )); then
    # Shared, pre-populated structures dir.
    predict_extra+=(--structures-dir "$structures_dir")
    echo "[mgr] structures supplied externally: $structures_dir (no download)"
fi

# ---- predict array (1 GPU per task) --------------------------------------

predict_deps=()
if [[ -n "$dl_array_id" ]]; then
    # aftercorr: predict array task i waits for download array task i,
    # and is skipped if that download failed. This keeps a single bad
    # batch from blocking the rest.
    predict_deps+=(--dependency=aftercorr:"$dl_array_id")
fi

pred_array_id=$(sbatch --parsable \
    --array=0-${last_idx} \
    "${predict_deps[@]}" \
    "$scripts_dir/tps_screening_predict_one_batch.sh" \
        "$fasta" "$batch_size" \
        "$output_root/shards" \
        "$classifier" \
        "${predict_extra[@]}")
echo "[mgr] submitted predict array job: $pred_array_id"

# ---- per-batch cleanup array (only when we downloaded structures) --------

if [[ -n "$dl_array_id" ]]; then
    cleanup_array_id=$(sbatch --parsable \
        --array=0-${last_idx} \
        --dependency=aftercorr:"$pred_array_id" \
        "$scripts_dir/tps_screening_cleanup_batch.sh" \
            "$output_root/structures")
    echo "[mgr] submitted cleanup array job: $cleanup_array_id"
fi

# ---- final gather (runs once after all predict tasks have completed) -----
# `afterany` instead of `afterok` so partial results are still aggregated
# if a small fraction of batches fails — the per-classifier sub-dirs each
# get written if at least one shard exists.

gather_id=$(sbatch --parsable \
    --dependency=afterany:"$pred_array_id" \
    "$scripts_dir/tps_screening_gather.sh" "$output_root")
echo "[mgr] submitted gather job: $gather_id"

echo "[mgr] all jobs submitted. Final outputs will land at:"
echo "[mgr]   $output_root/{plm,plm_domains,plm_domains_fallback,no_structure}.csv"
