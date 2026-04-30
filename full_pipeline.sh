#!/bin/bash
set -e
# get_mmseqs_based_folds --config ../enz_exp_playground/data_prep/configs/enzyme_explorer_dataprep.yaml 
detect_domains --config enzymeexplorer/configs/martsDB_domain_detection_config.yaml
detect_domains --config enzymeexplorer/configs/martsDB_initial_domain_detection_config.yaml
detect_domains --config enzymeexplorer/configs/enzyme_explorer_domain_detection_config.yaml
python -m enzymeexplorer.src.structure_processing.get_structural_features --config enzymeexplorer/configs/martsDB_structural_features_config.yaml
python -m enzymeexplorer.src.structure_processing.get_structural_features --config enzymeexplorer/configs/martsDB_initial_structural_features_config.yaml
python -m enzymeexplorer.src.structure_processing.get_structural_features --config enzymeexplorer/configs/enzyme_explorer_structural_features_config.yaml
# . scripts/extract_all_embeddings.sh
enzyme_explorer_main run