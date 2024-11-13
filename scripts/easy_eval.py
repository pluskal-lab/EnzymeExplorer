"""This script is used to predict the presence of terpene synthases in a given FASTA file, using TPS language model only, no domains."""

import os
import argparse
from pathlib import Path
from shutil import copytree, rmtree
import gdown  # type: ignore
import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--leave-downloaded-configs", action="store_true")
    parser.add_argument("--leave-downloaded-workflow-outputs", action="store_true")
    return parser.parse_args()


def main():
    # moving the current "outputs" folder to "outputs_old"
    if Path("outputs").exists():
        if Path("outputs_old").exists():
            rmtree("outputs_old")
        os.rename("outputs", "outputs_old")

    logger.info("Downloading the precomputed outputs..")
    url = "https://drive.google.com/uc?id=11lLiYRGR5kA2ceqFHb8vhCE4uiF6P6Eu"
    output_path = Path("outputs.zip")
    if output_path.exists():
        os.remove(output_path)
    gdown.download(url, str(output_path), quiet=False)
    os.system(f"unzip {output_path}")

    # moving the current "configs" folder to "configs_old"
    if Path("terpeneminer/configs").exists():
        if Path("terpeneminer/configs_old").exists():
            rmtree("terpeneminer/configs_old")
        os.rename("terpeneminer/configs", "terpeneminer/configs_old")
        
    url = "https://drive.google.com/uc?id=1TEMxUlnIHyc3RiXko10boLUgrofe47Xf"
    output_path = Path("terpeneminer/configs.zip")
    if output_path.exists():
        os.remove(output_path)
    gdown.download(url, str(output_path), quiet=False)
    os.system(f"unzip {output_path} -d terpeneminer")
    # Evaluate all models
    os.system("terpene_miner_main evaluate")

    # Evaluate TPS detection
    os.system("terpene_miner_main evaluate --classes \"isTPS\" --output-filename tps_detection")
    os.system("terpene_miner_main evaluate --classes \"isTPS\" --id-2-category-path data/id_2_kingdom_dataset.pkl --output-filename tps_detection_per_kingdom")

    # Evaluate per protein signatures
    os.system("terpene_miner_main evaluate --id-2-category-path data/id_2_domains_presence.pkl --output-filename per_interpro_signatures")

    # Visualize main results
    os.system("terpene_miner_main visualize")

    # Visualize ablation study
    os.system(
        "terpene_miner_main visualize --models "
        "DomainsRandomForest__with_minor_reactions_global_tuning "
        "PlmRandomForest__esm-1v_with_minor_reactions_global_tuning "
        "PlmRandomForest__tps_esm-1v-subseq_with_minor_reactions_global_tuning "
        "PlmDomainsRandomForest__tps_esm-1v-subseq_foldseek_with_minor_reactions_global_tuning_domains_subset_plm_subset "
        "--model-names \"Domain comparisons only\" \"PLM only\" \"Finetuned PLM only\" \"Finetuned PLM + Domain comparisons\" "
        "--subset-name \"ablation_study\""
    )

    # Visualize different models comparison
    os.system(
        "terpene_miner_main visualize --models "
        "PlmDomainsMLP__tps_esm-1v-subseq_with_minor_reactions_global_tuning "
        "PlmDomainsLogisticRegression__tps_esm-1v-subseq_with_minor_reactions_global_tuning "
        "PlmDomainsRandomForest__tps_esm-1v-subseq_with_minor_reactions_global_tuning "
        "--model-names \"Feed-Forward Neural Net\" \"Logistic Regression\" \"Random Forest\" "
        "--subset-name \"different_models_best_feats\""
    )
    
    os.system("terpene_miner_main visualize --plot-boxplots-per-type --models  "
            "CLEAN__with_minor_reactions HMM__with_minor_reactions Foldseek__with_minor_reactions Blastp__with_minor_reactions "
            "PlmDomainsRandomForest__tps_esm-1v-subseq_foldseek_with_minor_reactions_global_tuning_domains_subset_plm_subset "
            "--model-names \"CLEAN*\" \"pHMM\" \"Foldseek\" \"BLASTp\" \"Ours\" ")

    
    os.system("terpene_miner_main visualize --eval-output-filename tps_detection --plot-tps-detection --models  "
            "CLEAN__with_minor_reactions Foldseek__with_minor_reactions HMM__with_minor_reactions Blastp__with_minor_reactions PfamSUPFAM__supfam PfamSUPFAM__pfam "
            "PlmDomainsRandomForest__tps_esm-1v-subseq_foldseek_with_minor_reactions_global_tuning_domains_subset_plm_subset "
            "--model-names \"CLEAN*\" \"Foldseek\" \"BLASTp\" \"pHMM\" \"SUPFAM\" \"Pfam\" \"Ours\" ")
    
    os.system("terpene_miner_main visualize --eval-output-filename tps_detection_per_kingdom --plot-per-category --category-name \"Taxon\" --categories-order \"Plants\" \"Fungi\" \"Bacteria\"  \"Animals\" \"Protists\" --models  "
            "CLEAN__with_minor_reactions Foldseek__with_minor_reactions HMM__with_minor_reactions Blastp__with_minor_reactions PfamSUPFAM__supfam PfamSUPFAM__pfam "
            "PlmDomainsRandomForest__tps_esm-1v-subseq_foldseek_with_minor_reactions_global_tuning_domains_subset_plm_subset "
            "--model-names \"CLEAN*\" \"Foldseek\" \"BLASTp\" \"pHMM\" \"SUPFAM\" \"Pfam\" \"Ours\" ")
    
    # Clean up downloaded folders and restore old ones
    args = parse_args()
    if not args.leave_downloaded_workflow_outputs:
        if Path("outputs").exists():
            # Create outputs_old directory if it doesn't exist
            Path("outputs_old").mkdir(exist_ok=True)
            # Copy evaluation results to outputs_old
            copytree("outputs/evaluation_results", "outputs_old/evaluation_results", dirs_exist_ok=True)
            rmtree("outputs")
        if Path("outputs_old").exists():
            Path("outputs_old").rename("outputs")
    if not args.leave_downloaded_configs:
        if Path("terpeneminer/configs").exists():
            rmtree("terpeneminer/configs")    
        if Path("terpeneminer/configs_old").exists():
            Path("terpeneminer/configs_old").rename("terpeneminer/configs")


if __name__ == "__main__":
    main()
