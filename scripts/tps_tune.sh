#!/bin/bash

#SBATCH --job-name=tps_hyperparams
#SBATCH --time=40:00:0
#SBATCH --mem 50GB
#SBATCH --cpus-per-task 50
#SBATCH --partition small-g
#SBATCH --account=project_465000660
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate enzyme_explorer
cd /scratch/project_465000659/samusevi/EnzymeExplorer

enzyme_explorer_main tune --hyperparameter-combination-i $SLURM_ARRAY_TASK_ID
