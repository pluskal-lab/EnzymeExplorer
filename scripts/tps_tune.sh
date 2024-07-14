#!/bin/bash

#SBATCH --job-name=ANM_ATLAS_REFOLDED
#SBATCH --time=20:00:0
#SBATCH --mem 30GB
#SBATCH --cpus-per-task 30
#SBATCH --partition small-g
#SBATCH --account=project_465000660
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate tps_ml_discovery
cd /scratch/project_465000659/samusevi/TPS_ML_Discovery

python -m src.modeling_main tune --hyperparameter-combination-i $SLURM_ARRAY_TASK_ID
#python -m src.modeling_main tune --classes "precursor substr" isTPS --hyperparameter-combination-i $SLURM_ARRAY_TASK_ID
