"""This script extracts TPS language model embeddings for protein sequences and detects TPS"""

import os
import argparse
from dataclasses import dataclass
import json
import pickle
from functools import partial
from pathlib import Path
import esm  # type: ignore
import numpy as np  # type: ignore
from tqdm.auto import tqdm  # type: ignore

# import torch  # type: ignore
# torch.hub.set_dir("/scratch/project_465000659/samusevi")
# os.environ["TRANSFORMERS_CACHE"] = "/scratch/project_465000659/samusevi/cache"

from terpeneminer.src.embeddings_extraction.esm_transformer_utils import (
    compute_embeddings,
    get_model_and_tokenizer,
)
from terpeneminer.src.embeddings_extraction.ankh_transformer_utils import (
    compute_embeddings as ankh_compute_embeddings,
    get_model_and_tokenizer as ankh_get_model_and_tokenizer
)


def _extract_id_from_entry(entry: tuple) -> str:
    return entry[0]


def _extract_seq_from_entry(entry: tuple) -> str:
    return entry[1]


def _is_sequence_good(sequence: str, max_seq_len: int) -> bool:
    return len(sequence) <= max_seq_len


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clf-batch-size", type=int, default=4096)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--model", type=str, default="esm-1v-finetuned-subseq")
    parser.add_argument("--fasta-path", type=str, default="data/uniref90.fasta")
    parser.add_argument("--starting-i", type=int, default=0)
    parser.add_argument("--end-i", type=int, default=700_000)
    parser.add_argument("--output-id", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-root", type=str, default="detected_tps")
    parser.add_argument(
        "--ckpt-root-path", type=str, default="data/classifier_checkpoints.pkl"
    )
    parser.add_argument("--detection-threshold", type=float, default=0.2)
    parser.add_argument("--detect-precursor-synthases", type=bool, default=False)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


@dataclass
class PredictionResults:
    """
    A data class to store batches of model predictions
    """

    uniprot_id: list[str]
    confidence: list[float]


def main(arguments: argparse.Namespace):
    """
    Main function for processing protein sequences and predicting class probabilities using pre-trained PLM embeddings
    and downstream classifiers.

    This function performs the following steps:
    1. Loads the pre-trained model and necessary utilities for embedding computation.
    2. Reads the input FASTA file containing protein sequences.
    3. Processes the sequences in batches to generate embeddings using a pre-trained PLM model.
    4. Applies trained classifiers on the generated embeddings to predict class probabilities.
    5. Outputs prediction results for each sequence, saving them to the specified directory.

    Args:
        arguments (argparse.Namespace): Parsed command-line arguments that include:
            - model: The pre-trained model to be used for generating embeddings.
            - max_len: Maximum sequence length for embeddings.
            - fasta_path: Path to the input FASTA file containing sequences.
            - batch_size: Number of sequences to process per batch.
            - detection_threshold: Threshold for considering a prediction as positive.
            - clf_batch_size: Number of samples processed in each classification batch.
            - output_root: Directory to store prediction outputs.
            - ckpt_root_path: Path to the checkpoint file containing pre-trained classifiers.
            - detect_precursor_synthases: Boolean flag to detect precursor synthases.
            - starting_i, end_i: Range of indices to process sequences.
            - gpu: GPU identifier for processing sequences.

    Returns:
        None. The function writes the prediction results as JSON files for each processed sequence
        in the specified output directory.
    """

    if "esm" in args.model:
        model, batch_converter, alphabet = get_model_and_tokenizer(
            arguments.model, return_alphabet=True
        )

        compute_embeddings_partial = partial(
            compute_embeddings,
            bert_model=model,
            converter=batch_converter,
            padding_idx=alphabet.padding_idx,
            model_repr_layer=33,
            max_len=arguments.max_len,
        )
    elif "ankh" in args.model:
        model, tokenizer = ankh_get_model_and_tokenizer(args.model)
        compute_embeddings_partial = partial(
            ankh_compute_embeddings, bert_model=model, tokenizer=tokenizer
        )
    else:
        raise NotImplementedError(
            f"Model {args.model} is not supported. Currently only esm, ankh model families are supported"
        )

    uniprot_generator = esm.data.read_fasta(arguments.fasta_path)

    detection_threshold = arguments.detection_threshold
    clf_batch_size = arguments.clf_batch_size

    output_root = Path(arguments.output_root)
    with open(arguments.ckpt_root_path, "rb") as file:
        all_classifiers = pickle.load(file)

    def process_embeddings(
        enzyme_encodings_np_batch: np.ndarray, classifiers: list
    ) -> list[dict[str, float]]:
        """
        This function processes embeddings. It predicts class probabilities for each sample in the batch
        :param enzyme_encodings_np_batch: plm embeddings
        :param classifiers: downstream classifiers
        :return:
        """
        predictions = []
        n_samples = len(enzyme_encodings_np_batch)
        for classifier_i, classifier in enumerate(classifiers):
            if hasattr(classifier, "plm_feat_indices_subset") and classifier.plm_feat_indices_subset is not None:
                emb_plm = np.apply_along_axis(lambda i: i[classifier.plm_feat_indices_subset], 1,
                                              enzyme_encodings_np_batch)
            else:
                emb_plm = enzyme_encodings_np_batch
            y_pred_proba = classifier.predict_proba(emb_plm)
            for sample_i in range(n_samples):
                predictions_raw = {}
                for class_i, class_name in enumerate(classifier.classes_):
                    if class_name != "Unknown":
                        predictions_raw[class_name] = y_pred_proba[class_i][sample_i, 1]
                if sample_i == 0:
                    print('predictions_raw: ', predictions_raw)
                if classifier_i == 0:
                    predictions.append(
                        {
                            class_name: [value]
                            for class_name, value in predictions_raw.items()
                        }
                    )
                else:
                    for class_name, value in predictions_raw.items():
                        predictions[sample_i][class_name].append(value)
        # average the predictions
        predictions_avg = []
        for prediction in predictions:
            predictions_avg.append(
                {
                    class_name: np.mean(values)
                    for class_name, values in prediction.items()
                }
            )
            print({
                    class_name: len(values)
                    for class_name, values in prediction.items()
                })
        print('predictions_avg: ', predictions_avg)
        return predictions_avg

    next_batch = []
    next_batch_ids = []
    results_output_root = output_root / "detections_plm"

    if not results_output_root.exists():
        results_output_root.mkdir(parents=True)

    def _batch_predict(
        batch_to_process: list[str],
        batch_ids: list[str],
        enzyme_encodings_list_to_process: list[np.ndarray],
        enzyme_ids_list_to_process: list[str],
        last_call: bool = False,
    ):
        """
        A helper function to predict from a batch of sequences
        """
        if len(batch_to_process):
            (
                enzyme_encodings_np_batch,
                _,
            ) = compute_embeddings_partial(input_seqs=batch_to_process)
            enzyme_encodings_list_to_process.extend(enzyme_encodings_np_batch)
            enzyme_ids_list_to_process.extend(batch_ids)

        if len(enzyme_encodings_list_to_process) >= clf_batch_size or last_call:
            predictions = process_embeddings(
                np.stack(enzyme_encodings_list_to_process), all_classifiers
            )
            for protein_id, class_2_prob in zip(
                enzyme_ids_list_to_process, predictions
            ):
                protein_id_short = protein_id.split()[0].replace("/", "")
                if class_2_prob["isTPS"] >= detection_threshold or (
                    arguments.detect_precursor_synthases
                    and class_2_prob["precursor substr"] >= detection_threshold
                ):
                    output_file = results_output_root / protein_id_short
                    with open(output_file, "w", encoding="utf-8") as outputs_file:
                        json.dump(class_2_prob, outputs_file)
            enzyme_encodings_list_to_process = []
            enzyme_ids_list_to_process = []
        return enzyme_encodings_list_to_process, enzyme_ids_list_to_process

    enzyme_encodings_list: list[np.ndarray] = []
    enzyme_ids_list: list[str] = []
    for i, uniprot_entry in tqdm(
        enumerate(uniprot_generator),
        total=arguments.end_i,
        desc=f"Processing sequences on GPU {arguments.gpu}",
    ):
        if i < arguments.starting_i:
            continue
        if i == arguments.end_i:
            break
        uniprot_id = _extract_id_from_entry(uniprot_entry)
        seq = _extract_seq_from_entry(uniprot_entry)
        if not _is_sequence_good(uniprot_entry[1], max_seq_len=arguments.max_len):
            seq = seq[: (arguments.max_len - 2)]
        next_batch.append(seq)
        next_batch_ids.append(uniprot_id)

        if len(next_batch) == arguments.batch_size:
            enzyme_encodings_list, enzyme_ids_list = _batch_predict(
                next_batch, next_batch_ids, enzyme_encodings_list, enzyme_ids_list
            )
            next_batch = []
            next_batch_ids = []

    if enzyme_encodings_list or next_batch:
        _batch_predict(
            next_batch,
            next_batch_ids,
            enzyme_encodings_list,
            enzyme_ids_list,
            last_call=True,
        )


if __name__ == "__main__":
    args = parse_args()
    # for AMD GPUs
    os.environ["HIP_VISIBLE_DEVICES"] = args.gpu
    os.environ["AMD_SERIALIZE_KERNEL"] = "3"
    main(args)
