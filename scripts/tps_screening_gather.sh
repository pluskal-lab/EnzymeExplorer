#!/bin/bash
# Final unification job: concatenate every per-batch shard CSV into
# one combined CSV per classifier, then clean up caches if and only
# if every expected batch is present.
#
# Spawned by tps_screening_manager.sh with
# --dependency=afterany:<predict_array_id>, so partial results still
# aggregate when a small fraction of batches fails. ``afterany`` means
# we run regardless of how the prediction array ended.
#
# Completeness check
# ------------------
# Before merging, the script counts shard CSVs per expected
# sub-directory:
#   * --classifier plm         → expects <n_batches> files under shards/plm/
#   * --classifier plm_domains → expects <n_batches> under each of
#                                shards/{plm_domains,plm_domains_fallback,no_structure}/
#   * --classifier both        → expects <n_batches> under all four
# If every expected sub-dir has at least <n_batches> shards, the run
# is considered FULLY COMPLETE and the gather job removes the retry
# caches (``structures/``, ``embeddings_cache/``, ``missing_csvs/``)
# AFTER a successful merge. The final CSVs and the per-batch logs
# are kept.
#
# If any expected sub-dir is short of shards, the merge still runs
# (so the user gets partial results) but the caches are PRESERVED so
# a follow-up ``sbatch`` of the manager pointed at the same
# --output-root can pick up the missing batches and reuse every
# already-downloaded PDB + already-computed embedding.
#
# Usage (positional, set by the manager):
#   tps_screening_gather.sh <output_root> <n_batches> <classifier>
#
# Reads:   <output_root>/shards/<classifier>/batch_*.csv
# Writes:  <output_root>/{plm,plm_domains,plm_domains_fallback,no_structure}.csv
# Deletes: per-shard CSVs after successful aggregation
#          (via gather_detections_to_csv --delete-shards)
# When the run is fully complete, ALSO deletes:
#   <output_root>/structures/, embeddings_cache/, missing_csvs/

#SBATCH --job-name=tps_screen_gather
#SBATCH --partition=standard
#SBATCH --account=project_465000660
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# See manager note: cluster bashrc files may reference unbound vars;
# keep -u off while sourcing, re-enable for our own code.
set -eo pipefail
source ~/.bashrc
conda activate enzyme_explorer_prod
set -u

# No ``cd`` needed: $output_root is absolute (manager resolved it) and
# ``python -m`` finds the package via sys.path.

output_root="$1"
n_batches="$2"
classifier="$3"

echo "[gather] aggregating shards under $output_root (n_batches=$n_batches, classifier=$classifier)"

# ---- completeness check (runs BEFORE --delete-shards) ---------------------
# Decide whether the run is fully complete by counting shard CSVs per
# expected classifier subdir. Has to happen pre-merge because
# ``gather_detections_to_csv --delete-shards`` removes the per-shard
# files (and the empty subdirs) on its way out.

case "$classifier" in
    plm)          expected_subdirs=(plm) ;;
    plm_domains)  expected_subdirs=(plm_domains plm_domains_fallback no_structure) ;;
    both)         expected_subdirs=(plm plm_domains plm_domains_fallback no_structure) ;;
    *) echo "[gather] WARN: unknown classifier '$classifier'; skipping completeness check"; expected_subdirs=() ;;
esac

all_complete=1
for sd in "${expected_subdirs[@]}"; do
    sd_path="$output_root/shards/$sd"
    if [[ -d "$sd_path" ]]; then
        n_have=$(find "$sd_path" -maxdepth 1 -name '*.csv' | wc -l)
    else
        n_have=0
    fi
    if (( n_have < n_batches )); then
        echo "[gather] incomplete: $sd has $n_have/$n_batches shards"
        all_complete=0
    else
        echo "[gather] complete:   $sd has $n_have/$n_batches shards"
    fi
done
if (( ${#expected_subdirs[@]} == 0 )); then
    all_complete=0
fi

# ---- merge per-classifier shards into final CSVs --------------------------
# ``--delete-shards`` is gated on completeness. If we deleted shards on a
# partial run, the rerun's worker-side skip-done check (which looks for
# ``shards/<classifier>/batch_<i>.csv``) would see every shard missing
# and redo *every* batch — including the ones that already succeeded.
# Keeping the shards on partial runs lets the rerun skip the
# already-finished batches; the merged CSV is still produced as a
# snapshot of progress-so-far, just kept in sync with the next merge.

gather_args=(
    --shards-root "$output_root/shards"
    --output-root "$output_root"
)
if (( all_complete )); then
    gather_args+=(--delete-shards)
fi

python -m enzymeexplorer.src.screening.gather_detections_to_csv \
    "${gather_args[@]}"

# ---- cache cleanup (only when fully complete) -----------------------------

if (( all_complete )); then
    echo "[gather] run is fully complete — clearing retry caches"
    for cache_subdir in structures embeddings_cache missing_csvs; do
        if [[ -d "$output_root/$cache_subdir" ]]; then
            echo "[gather]   removing $output_root/$cache_subdir/"
            rm -rf "$output_root/$cache_subdir"
        fi
    done
    echo "[gather] final outputs at $output_root/{plm,plm_domains,plm_domains_fallback,no_structure}.csv"
else
    echo "[gather] run is INCOMPLETE — keeping shards + caches for retry"
    # Each merged CSV at <output_root>/<classifier>.csv is a snapshot of
    # progress so far; the next rerun's gather will overwrite it once
    # the missing batches finish. The per-batch shard CSVs under
    # shards/<classifier>/ are intentionally kept so the rerun's
    # skip-done logic recognises them and re-runs only what failed.
    if [[ -d "$output_root/shards" ]]; then
        n_shards=$(find "$output_root/shards" -mindepth 2 -name '*.csv' | wc -l)
        echo "[gather]   $n_shards per-batch shard CSV(s) preserved under $output_root/shards/"
    fi
    if [[ -d "$output_root/structures" ]]; then
        n_left=$(find "$output_root/structures" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "[gather]   $n_left batch struct dir(s) preserved under $output_root/structures/"
    fi
    if [[ -d "$output_root/embeddings_cache" ]]; then
        n_emb=$(find "$output_root/embeddings_cache" -maxdepth 1 -name '*.npy' | wc -l)
        echo "[gather]   $n_emb embedding cache file(s) preserved under $output_root/embeddings_cache/"
    fi
    echo "[gather]   Re-run the manager with the same --output-root to retry only the missing batches."
fi
