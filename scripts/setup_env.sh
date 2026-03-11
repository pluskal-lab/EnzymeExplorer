#!/bin/bash

# Create env and install required packages
conda create -n enzyme_explorer python==3.10.0 pandas==2.2.2 numpy==1.26.4 jupyter matplotlib seaborn foldseek==9.427df8a pymol-bundle pymol-psico==3.4.19 tmalign==20170708 bioconda::mmseqs2 mafft==7.525 iqtree==2.3.0 biopython==1.83 fastapi -c schrodinger -c speleo3 -c conda-forge -c bioconda -y
conda activate enzyme_explorer

pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install torch --index-url https://download.pytorch.org/whl/rocm6.0  # for amd gpu's
pip install rdkit==2022.9.5 
pip install scikit-learn==1.5.1
pip install scipy==1.13.0 
pip install goatools
pip install ConfigArgParse
pip install epam.indigo
pip install openpyxl
pip install py3Dmol
pip install hdbscan==0.8.33
pip install scikit-learn-extra
pip install plotly
pip install fair-esm==2.0.0
pip install ankh==1.10.0
pip install tables
pip install tqdm
pip install py-mcc-f1
pip install inquirer
pip install dataclasses-json
pip install scikit-optimize
pip install xgboost
pip install GPUtil
pip install wget
pip install git+https://github.com/SamusRam/ProFun.git # one needs to install prerequisites of individual models separately, see https://github.com/SamusRam/ProFun
pip install gdown

# installing CLEAN
cwd=$(pwd)
cd ..
if [ -d CLEAN ]; then
    rm -rf CLEAN
fi
git clone https://github.com/tttianhao/CLEAN.git
cd CLEAN/app/src
echo "export PATH=\$PATH:$(pwd)" >> ~/.bashrc
source ~/.bashrc
cd $cwd
