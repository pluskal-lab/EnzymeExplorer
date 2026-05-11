#!/bin/bash
# Final unification job: concatenate every per-batch shard CSV into
# one combined CSV per classifier. Spawned by tps_screening_manager.sh
# with --dependency=afterany:<predict_array_id>, so partial results
# still aggregate when a small fraction of batches fails.
#
# Usage (positional, set by the manager):
#   tps_screening_gather.sh <output_root>
#
# Reads:   <output_root>/shards/<classifier>/batch_*.csv
# Writes:  <output_root>/{plm,plm_domains,plm_domains_fallback,no_structure}.csv
# Deletes: the per-shard CSVs after successful aggregation
# (via gather_detections_to_csv --delete-shards).

#SBATCH --job-name=tps_screen_gather
#SBATCH --partition=standard
#SBATCH --account=project_465000660
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

set -euo pipefail

source ~/.bashrc
conda activate enzyme_explorer

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_root="$1"

echo "[gather] aggregating shards under $output_root"

python -m enzymeexplorer.src.screening.gather_detections_to_csv \
    --shards-root "$output_root/shards" \
    --output-root "$output_root" \
    --delete-shards
