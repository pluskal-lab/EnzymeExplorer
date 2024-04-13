#!/bin/bash

# Create env
conda create -n tps_ml_discovery python==3.10 scikit-learn pandas numpy scipy jupyter seaborn -y
conda activate tps_ml_discovery
pip install epam.indigo
pip install openpyxl
pip install rdkit-pypi

