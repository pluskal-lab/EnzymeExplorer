#!/bin/bash

#SBATCH --job-name=uniprot_screening
#SBATCH --time=44:00:0
#SBATCH --mem 50GB
#SBATCH --cpus-per-task 50
#SBATCH --partition standard-g
#SBATCH --account=project_465000660
#SBATCH --gpus 8

set -euo pipefail

source ~/.bashrc
conda activate enzyme_explorer
cd /scratch/project_465000659/samusevi/EnzymeExplorer

input_fasta_path="$1"
output_root_path="$2"
echo "Performing TPS screening on: $input_fasta_path → $output_root_path (session $SLURM_ARRAY_TASK_ID)"

python -m enzymeexplorer.src.screening.tps_screening_cluster_launcher \
    --session-i "$SLURM_ARRAY_TASK_ID" \
    --fasta-path "$input_fasta_path" \
    --output-root "$output_root_path"

# Concatenate per-shard CSVs only on the last array task. The check assumes
# SLURM_ARRAY_TASK_MAX is set; if running outside SLURM, run gather_detections_to_csv
# manually after all sessions have finished.
if [[ "${SLURM_ARRAY_TASK_ID:-1}" == "${SLURM_ARRAY_TASK_MAX:-1}" ]]; then
    python -m enzymeexplorer.src.screening.gather_detections_to_csv \
        --shards-dir "$output_root_path/shards" \
        --output-path "$output_root_path/detections.csv" \
        --delete-shards
fi
