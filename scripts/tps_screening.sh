#!/bin/bash

#SBATCH --job-name=uniprot_screening
#SBATCH --time=44:00:0
#SBATCH --mem 50GB
#SBATCH --cpus-per-task 50
#SBATCH --partition standard-g
#SBATCH --account=project_465000660
#SBATCH --gpus 8

source ~/.bashrc
conda activate terpene_miner
cd /scratch/project_465000659/samusevi/TerpeneMiner

input_fasta_path="$1"
output_root_path="$2"
detection_threshold="$3"
echo "Performing TPS screening with input fasta: $1 (detection threshold is $3), storing individual detections into: $2"

python -m terpeneminer.src.screening.tps_screening_cluster_launcher --session-i $SLURM_ARRAY_TASK_ID --fasta-path "$input_fasta_path" --output-root "$output_root_path" --detection-threshold "$detection_threshold"

