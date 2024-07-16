#!/bin/bash

#SBATCH --job-name=uniprot_screening
#SBATCH --time=44:00:0
#SBATCH --mem 50GB
#SBATCH --cpus-per-task 50
#SBATCH --partition standard-g
#SBATCH --account=project_465000660
#SBATCH --gpus 8

source ~/.bashrc
conda activate tps_ml_discovery
cd /scratch/project_465000659/samusevi/TPS_ML_Discovery

input_fasta_path="$1"
output_root_path="$2"
echo "Performing TPS screening with input fasta: $1, storing individual detections into: $2"


python -m src.screening.tps_screening_cluster_launcher --session-i $SLURM_ARRAY_TASK_ID --fasta-path "$input_fasta_path" --output-root "$output_root_path"
