#!/bin/bash

# Create env and install required packages
conda create -n tps_ml_discovery python==3.10.0 scikit-learn pandas numpy scipy jupyter seaborn pytorch pytorch-cuda pymol pymol-psico tmalign -c pytorch -c nvidia -c schrodinger -c speleo3 -c conda-forge -y
conda activate tps_ml_discovery
pip install epam.indigo
pip install openpyxl
pip install rdkit-pypi
pip install fair-esm
pip install tqdm
conda install -c bioconda mafft -y
conda install -c bioconda iqtree -y
conda install -c conda-forge biopython -y
conda install -c schrodinger pymol-psico -y
pip install py3Dmol
pip install hdbscan
pip install scikit-learn-extra
pip install plotly
