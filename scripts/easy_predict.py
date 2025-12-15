from functools import partial
from uuid import uuid4
import argparse
import gdown
import pandas as pd
import torch

from pathlib import Path
import os
import pickle
from shutil import copyfile, rmtree
import logging
import subprocess
from dataclasses import dataclass
import re
import numpy as np
from enzymeexplorer.src.embeddings_extraction.esm_transformer_utils import (
    compute_embeddings,
    get_model_and_tokenizer,
)
from Bio import SeqIO
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)

logger.info(f'Cuda available is {torch.cuda.is_available()}')

def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory-with-structures", required=True, help="Path to the directory with input PDB files", type=str
    )
    parser.add_argument("--needed-proteins-csv-path", required=True, help="Path to the CSV file containing proteins to be screened", type=str)
    parser.add_argument("--csv-id-column", required=True, help="Name of the column with IDs in the CSV file", type=str)
    parser.add_argument("--output-csv-path", required=True, help="Path to the output CSV file with the results", type=str)

    parser.add_argument("--is-bfactor-confidence", action="store_true")
    parser.add_argument("--detect-precursor-synthases", help="Flag to detect precursor synthases as well. Set to False with `--no-detect-precursor-synthases`.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--detection-threshold", help="Threshold for detection", type=float, default=0.0)
    parser.add_argument("--n-jobs", help="Number of jobs to run in parallel", type=int, default=16)
    parser.add_argument("--plm-batch-size", help="Batch size for embeddings computation", type=int, default=4)
    parser.add_argument("--plm-max-seq-len", help="Max sequence length for embeddings computation", type=int, default=1022)
    parser.add_argument("--clf-batch-size", help="Batch size for classifier", type=int, default=4096)
    return parser.parse_args()


def main():

    args = parse_args()

    logger.info("checking TPS language model checkpoint presence")
    plm_chkpt_path = Path("data/plm_checkpoints")
    if not plm_chkpt_path.exists():
        plm_chkpt_path.mkdir(parents=True)
    plm_path = plm_chkpt_path / "checkpoint-tps-esm1v-t33-subseq.ckpt"
    if not plm_path.exists():
        logger.info("Downloading TPS language model checkpoint..")
        url = "https://drive.google.com/uc?id=1jU76oUl0-CmiB9m3XhaKmI2HorFhyxC7"
        gdown.download(url, str(plm_path), quiet=False)
    clf_chkpt_path = Path("data/classifier_plm_checkpoints.pkl")
    if not clf_chkpt_path.exists():
        logger.info("Downloading PLM-based classifier checkpoints..")
        url = "https://drive.google.com/uc?id=15_OFrrVUy9r9Urj-R2CjTRj_DHcazdAl"
        gdown.download(url, str(clf_chkpt_path), quiet=False)
        
    clf_esm1v_chkpt_path = Path("data/classifier_plm_checkpoints_esm1v.pkl")
    if not clf_esm1v_chkpt_path.exists():
        logger.info("Downloading ESM-1v-based classifier checkpoints..")
        url = "https://drive.google.com/uc?id=1917A4wyLqI5pSUSJQsUc9Ffd1loljEFJ"
        gdown.download(url, str(clf_esm1v_chkpt_path), quiet=False)
        
    clf_main_chkpt_path = Path("data/classifier_domain_and_plm_checkpoints.pkl")
    if not clf_main_chkpt_path.exists():
        logger.info("Downloading main classifier checkpoints..")
        url = "https://drive.google.com/uc?id=1ulaZUev6HJC237-t4S_BK41yMNXw9SNG"
        gdown.download(url, str(clf_main_chkpt_path), quiet=False)
        
    model, batch_converter, alphabet = get_model_and_tokenizer(
        "esm-1v-finetuned-subseq", return_alphabet=True
    )

    compute_embeddings_partial = partial(
        compute_embeddings,
        bert_model=model,
        converter=batch_converter,
        padding_idx=alphabet.padding_idx,
        model_repr_layer=33,
        max_len=1022,
    )
    
    with open('data/classifier_plm_checkpoints.pkl', 'rb') as file:
        fold_plm_classifiers = pickle.load(file)

    model_fallback, batch_converter_fallback, alphabet_fallback = get_model_and_tokenizer(
            "esm-1v", return_alphabet=True
        )
    compute_embeddings_partial_fallback = partial(
        compute_embeddings,
        bert_model=model_fallback,
        converter=batch_converter_fallback,
        padding_idx=alphabet_fallback.padding_idx,
        model_repr_layer=33,
        max_len=1022,
    )
    with open('data/classifier_plm_checkpoints_esm1v.pkl', 'rb') as file:
        fold_plm_classifiers_fallback = pickle.load(file)

    with open('data/classifier_domain_and_plm_checkpoints.pkl', 'rb') as file:
        fold_classifiers = pickle.load(file)
        
    url = "https://drive.google.com/uc?id=1oEdqSf9iXhfjGtCADPqebFwTKeV_Jhd1"
    output_path = Path("data/domain_templates.zip")
    if output_path.exists():
        logger.info("domain_templates.zip exists and will be unzipped")
    else:
        logger.info("Downloading domain_templates.zip")
        gdown.download(url, str(output_path), quiet=False)
    os.system(f"unzip -o {output_path} -d {args.input_directory_with_structures}")
    
    url = "https://drive.google.com/uc?id=1x_7DT4NIZSimwJo2HLhOmoGoFL55jwFC"
    output_path = Path("data/tps_detected_domains.zip")
    tps_detected_domains_path = Path("data/tps_detected_domains")
    if tps_detected_domains_path.exists():
        logger.info("tps_detected_domains already exists")
    else:
        logger.info("tps_detected_domains does not exist")
        if output_path.exists():
            logger.info("tps_detected_domains.zip exists and will be unzipped")
        else:
            logger.info("Downloading tps_detected_domains.zip")
            gdown.download(url, str(output_path), quiet=False)
        tps_detected_domains_path.mkdir(parents=True)
        os.system(f"unzip {output_path} -d {tps_detected_domains_path}")
    domains_subset_path = Path("data/domains_subset.pkl")
    if not domains_subset_path.exists():
        logger.info("Downloading domains subset..")
        url = "https://drive.google.com/uc?id=1KwD8enIwwwvXrZUEh6-gYyodZHM1VjSp"
        gdown.download(url, str(domains_subset_path), quiet=False)

    
    ############################################################
    # detecting domains
    ############################################################
    logger.info("Detecting domains..")
    working_dir_temp = Path("_temp")
    if not working_dir_temp.exists():
        working_dir_temp.mkdir()
    domain_detections_path = working_dir_temp / f"filename_2_detected_domains_completed_confident_{uuid4()}.pkl"
    detected_domain_structures_root = working_dir_temp / f"detected_domains_{uuid4()}"
    logger.info(f"domain_detections_path: {domain_detections_path}")
    if not detected_domain_structures_root.exists():
        detected_domain_structures_root.mkdir()
    domain_detection_out = os.system(
        "python -m enzymeexplorer.src.structure_processing.domain_detections "
        f'--needed-proteins-csv-path "{args.needed_proteins_csv_path}" '
        f'--csv-id-column {args.csv_id_column} '
        f'--n-jobs {args.n_jobs} '
        f'--input-directory-with-structures {args.input_directory_with_structures} '
        f"{'--is-bfactor-confidence ' if args.is_bfactor_confidence else ''}"
        f'--detections-output-path "{domain_detections_path}" '
        f'--detected-regions-root-path "{detected_domain_structures_root}" '
        f'--domains-output-path "{detected_domain_structures_root}" '
        "--store-domains "
        "--recompute-existing-secondary-structure-residues "
        "--do-not-store-intermediate-files"
    )
    logger.info(f"domain_detections output: {domain_detection_out}")
    
    ############################################################
    # comparing detected domains to the known ones
    ############################################################
    with open(domain_detections_path, "rb") as file:
        detected_domains = pickle.load(file)

    logger.info("Detected %d domains. Starting comparison to the known domains..", len(detected_domains))
    if detected_domains:
        current_computation_id = uuid4()
        comparison_results_path = working_dir_temp / f"filename_2_regions_vs_known_reg_dists_{current_computation_id}.pkl"
        os.system("python -m enzymeexplorer.src.structure_processing.comparing_to_known_domains_foldseek "
                  f'--known-domain-structures-root data/tps_detected_domains/all '
                  f'--detected-domain-structures-root "{detected_domain_structures_root}" '
                  '--path-to-known-domains-subset data/domains_subset.pkl '
                  f'--output-path "{comparison_results_path}" ')

        logger.info("Compared detected domains to the known ones!")

        with open(comparison_results_path, "rb") as file:
            comparison_results = pickle.load(file)
        os.remove(comparison_results_path)
    else:
        comparison_results = {}

    ############################################################
    # detecting TPS and predicting TPS substrates
    ############################################################
    logger.info("Detecting TPS and predicting TPS substrates..")
    # reading the needed proteins
    proteins_df = pd.read_csv(args.needed_proteins_csv_path)
    relevant_protein_ids = set(proteins_df[args.csv_id_column].values)

    input_directory = Path(args.input_directory_with_structures)

    # getting the files
    blacklist_files = {"1ps1.pdb", "5eat.pdb", "3p5r.pdb"}
    pdb_files_to_process = list(input_directory.glob("*.pdb")) + list(input_directory.glob("*.cif"))
    pdb_files_to_process = [
        filepath
        for filepath in pdb_files_to_process
        if str(filepath.name) not in blacklist_files
        and (
            filepath.stem in relevant_protein_ids
            or "".join(filepath.stem.replace("(", "").replace(")", "").replace("-", ""))
            in relevant_protein_ids
        )
    ]
    
    results_output_root = Path(args.output_csv_path).parent / f"detections_plm_{uuid4()}"
    if not results_output_root.exists():
        results_output_root.mkdir(parents=True)
    
    # First, get proteins that have domain features and proteins that do not
    with_domain_comparisons = []
    no_domain_comparisons = []
    for pdb_file_path in tqdm(
        pdb_files_to_process,
        total=len(pdb_files_to_process),
        desc=f"Getting proteins with meaningful domain comparisons..."
    ):
        pdb_id = pdb_file_path.stem
        if pdb_id in comparison_results:
            with_domain_comparisons.append(pdb_file_path)
        else:
            no_domain_comparisons.append(pdb_file_path)

    logger.info(f"Found {len(with_domain_comparisons)} proteins with domain comparisons and {len(no_domain_comparisons)} with none.")

    #First take those with domain comparisons
    next_batch = []
    next_batch_embeddings = []
    batch_domain_features = {}
    next_batch_ids = []
    batch_counter = 0
    final_cycle = len(with_domain_comparisons)
    counter = 0
    for pdb_file_path in tqdm(
        with_domain_comparisons,
        total=len(with_domain_comparisons),
        desc=f"Generating PLM embeddings and TPS-activity predictions for proteins with domain comparisons..."
    ):
        pdb_id = pdb_file_path.stem
        file_type = pdb_file_path.suffix[1:]
        counter += 1

        predictions = []
        # Extract sequence from pdb file
        input_seq = str(SeqIO.read(pdb_file_path, file_type + "-atom").seq)
        if len(input_seq) > args.plm_max_seq_len:
            input_seq = input_seq[: (args.plm_max_seq_len - 2)]
        # Get domain features from every classifier
        meaningful_comparison = True
        for classifier_i, classifier in enumerate(fold_classifiers):
            logger.info("Comparing domain detections to the selected known examples")
            dom_features_count = sum(map(len, classifier.domain_type_2_order_of_domain_modules.values()))
            dom_feat = np.zeros(dom_features_count)
            current_comparison_results = comparison_results[pdb_id]
            was_alpha_observed = False
            for domain_detection in detected_domains[pdb_id]:
                domain_type = domain_detection.domain
                detection_id = domain_detection.module_id
                known_domain_id_2_tmscore = dict(current_comparison_results[detection_id])
                if domain_type == 'alpha':
                    if not was_alpha_observed:
                        alpha_idx = 1
                        was_alpha_observed = True
                    else:
                        alpha_idx = 2
                    domain_type = f"alpha{alpha_idx}"
                for known_module_id, dom_feat_idx in classifier.domain_type_2_order_of_domain_modules[domain_type]:
                    # assert known_module_id in known_domain_id_2_tmscore, f"Known module {known_module_id} not found in comparison results"
                    dom_feat[dom_feat_idx] = known_domain_id_2_tmscore.get(known_module_id, 0)
            if np.max(dom_feat) < 0.4:
                logger.warning(f"No meaningful domain comparisons in a model for {pdb_id}")
                # This protein will be predicted in no domain comparisons
                no_domain_comparisons.append(pdb_file_path)
                meaningful_comparison = False
                break
            else:
                dom_feat = 1 - dom_feat.reshape(1, -1)
                batch_domain_features[(pdb_id, classifier_i)] = dom_feat
        if meaningful_comparison:
            next_batch.append(input_seq)
            next_batch_ids.append(pdb_id)
        #Do embedding by batch
        if len(next_batch) == args.plm_batch_size or (counter == final_cycle and next_batch):
            logger.info(f"Creating embedding for batch")
            (enzyme_encodings_np_batch,_,) = compute_embeddings_partial(input_seqs=next_batch)
            next_batch_embeddings.append(enzyme_encodings_np_batch)
            next_batch = []
        #Classifier by batch
        if len(next_batch_embeddings) == args.clf_batch_size or (counter == final_cycle and next_batch_embeddings):
            logger.info(f"Predicting for batch {batch_counter}")
            #Concatenate the embeddings
            next_batch_embeddings = np.concatenate(next_batch_embeddings, axis=0)
            batch_counter += 1         
            n_samples = len(next_batch_embeddings)
            for classifier_i, classifier in enumerate(fold_classifiers):
                logger.info(f"Predicting with classifier {classifier_i + 1}/{len(fold_classifiers)}..")
                if classifier.plm_feat_indices_subset is not None:
                    emb_plm = np.apply_along_axis(lambda i: i[classifier.plm_feat_indices_subset], 1, next_batch_embeddings)
                else:
                    emb_plm = next_batch_embeddings                
                domain_features = list()
                #Get previously selected domain features for this classifier and concatonate to every emebdding in this batch
                for one_id in next_batch_ids:
                    domain_features.append(batch_domain_features[(one_id, classifier_i)])
                embs = [np.concatenate((embedding, other[0])) for embedding, other in zip(emb_plm, domain_features)]
                embs = np.stack(embs, axis=0)
                y_pred_proba = classifier.predict_proba(embs)
                #stroe to predictions list
                for sample_i in range(n_samples):
                    predictions_raw = {}
                    for class_i, class_name in enumerate(classifier.classes_):
                        if class_name != "Unknown":
                            predictions_raw[class_name] = y_pred_proba[class_i][sample_i, 1]
                    if len(predictions) < sample_i + 1:
                        predictions.append(
                            {
                                class_name: [value]
                                for class_name, value in predictions_raw.items()
                            }
                        )
                    else:
                        for class_name, value in predictions_raw.items():
                            predictions[sample_i][class_name].append(value)
            logger.info("Averaging predictions over all models..")
            predictions_avg = []
            for prediction in predictions:
                predictions_avg.append(
                    {
                        class_name: np.mean(values)
                        for class_name, values in prediction.items()
                    }
                )
            for protein_id, avg_pred in zip(next_batch_ids, predictions_avg):
                protein_id_short = protein_id.replace("/", "")
                if avg_pred["isTPS"] >= args.detection_threshold or (
                    args.detect_precursor_synthases
                    and avg_pred["precursor substr"] >= args.detection_threshold
                ):
                    output_file = results_output_root / protein_id_short
                    with open(output_file, "w", encoding="utf-8") as outputs_file:
                        json.dump(avg_pred, outputs_file)
            #After predicting the batch, start a new one:
            next_batch_embeddings = []
            next_batch_ids = []
            batch_domain_features = {}
    

    #Now bathces without any domain comparisons
    next_batch = []
    next_batch_embeddings = []
    batch_domain_features = {}
    next_batch_ids = []
    batch_counter = 0
    final_cycle = len(no_domain_comparisons)
    counter = 0 
    for pdb_file_path in tqdm(
        no_domain_comparisons,
        total=len(no_domain_comparisons),
        desc=f"Generating PLM embeddings and TPS-activity predictions for proteins with no domain comparisons..."
    ):
        pdb_id = pdb_file_path.stem
        file_type = pdb_file_path.suffix[1:]
        counter += 1
        predictions = []
        # Extract sequence from pdb file
        input_seq = str(SeqIO.read(pdb_file_path, file_type + "-atom").seq)
        if len(input_seq) > args.plm_max_seq_len:
            input_seq = input_seq[: (args.plm_max_seq_len - 2)]

        next_batch.append(input_seq)
        next_batch_ids.append(pdb_id) 

        #Do embedding by batch
        if len(next_batch) == args.plm_batch_size or (counter == final_cycle and next_batch):
            logger.info(f"Creating embedding for batch")
            (enzyme_encodings_np_batch,_,) = compute_embeddings_partial_fallback(input_seqs=next_batch)
            next_batch_embeddings.append(enzyme_encodings_np_batch)
            next_batch = [] 

        #Predict batch              
        if len(next_batch_embeddings) == args.clf_batch_size or (counter == final_cycle and next_batch_embeddings):
            logger.info(f"Predicting for batch {batch_counter}")
            batch_counter += 1
            #Concatenate the embeddings
            next_batch_embeddings = np.concatenate(next_batch_embeddings, axis=0)
            #Batch compute embeddings
            n_samples = len(next_batch_embeddings)
            for classifier_i, classifier in enumerate(fold_plm_classifiers_fallback):
                logger.info(f"Predicting with classifier {classifier_i + 1}/{len(fold_classifiers)}...")
                y_pred_proba = classifier.predict_proba(next_batch_embeddings)          
                #Store to predictions list
                for sample_i in range(n_samples):
                    predictions_raw = {}
                    for class_i, class_name in enumerate(classifier.classes_):
                        if class_name != "Unknown":
                            predictions_raw[class_name] = y_pred_proba[class_i][sample_i, 1]
                    if len(predictions) < sample_i + 1:
                        predictions.append(
                            {
                                class_name: [value]
                                for class_name, value in predictions_raw.items()
                            }
                        )
                    else:
                        for class_name, value in predictions_raw.items():
                            predictions[sample_i][class_name].append(value)

            logger.info("Averaging predictions over all models...")
            predictions_avg = []
            for prediction in predictions:
                predictions_avg.append(
                    {
                        class_name: np.mean(values)
                        for class_name, values in prediction.items()
                    }
                )
            for protein_id, avg_pred in zip(next_batch_ids, predictions_avg):
                protein_id_short = protein_id.replace("/", "")
                if avg_pred["isTPS"] >= args.detection_threshold or (
                    args.detect_precursor_synthases
                    and avg_pred["precursor substr"] >= args.detection_threshold
                ):
                    output_file = results_output_root / protein_id_short
                    with open(output_file, "w", encoding="utf-8") as outputs_file:
                        json.dump(avg_pred, outputs_file)
            #After predicting the batch, start a new one:
            next_batch_ids = []
            next_batch_embeddings = []
            batch_domain_features = {}
    
    os.system(
        f"python -m enzymeexplorer.src.screening.gather_detections_to_csv --screening-results-root {results_output_root} --output-path {args.output_csv_path} --delete-individual-files"
    )
    os.remove(domain_detections_path)
    rmtree(detected_domain_structures_root)
        
if __name__ == "__main__":
    main()
