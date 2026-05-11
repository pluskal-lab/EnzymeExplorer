#!/bin/bash
# Final unification job: concatenate every per-batch shard CSV into
# one combined CSV per classifier, then sweep any leftover per-batch
# structure directories.
#
# Spawned by tps_screening_manager.sh with
# --dependency=afterany:<predict_array_id>, so partial results still
# aggregate when a small fraction of batches fails. ``afterany`` means
# we run regardless of how the prediction array ended — what we're
# guaranteeing is "no more shard CSVs will be written after this
# point" + "the prediction array is no longer in flight, so it's safe
# to delete the per-batch structure dirs as a final cleanup pass".
#
# Usage (positional, set by the manager):
#   tps_screening_gather.sh <output_root>
#
# Reads:   <output_root>/shards/<classifier>/batch_*.csv
# Writes:  <output_root>/{plm,plm_domains,plm_domains_fallback,no_structure}.csv
# Deletes: per-shard CSVs after successful aggregation
#          (via gather_detections_to_csv --delete-shards)
# Final sweep: <output_root>/structures/ if it exists, in case any
#              predict task was SIGKILLed and its EXIT trap never
#              fired. Only kicks in when the structures dir actually
#              exists (i.e. the manager triggered the AF-DB download
#              array); externally-supplied --structures-dir paths
#              live outside <output_root> and are never touched.

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

echo "[gather] aggregating shards under $output_root"

python -m enzymeexplorer.src.screening.gather_detections_to_csv \
    --shards-root "$output_root/shards" \
    --output-root "$output_root" \
    --delete-shards

# Belt-and-braces: nuke any per-batch structure dir that survived a
# SIGKILLed predict task. The healthy path already removed each batch's
# dir via the predict-batch EXIT trap, so this sweep usually finds
# nothing.
if [[ -d "$output_root/structures" ]]; then
    n_left=$(find "$output_root/structures" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if (( n_left > 0 )); then
        echo "[gather] WARN: $n_left batch struct dirs survived (probably SIGKILLed predict tasks); removing"
    fi
    rm -rf "$output_root/structures"
fi
