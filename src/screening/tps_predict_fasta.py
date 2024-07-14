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
import torch  # type: ignore

from src.embeddings_extraction.esm_transformer_utils import (
    compute_embeddings,
    get_model_and_tokenizer,
)

torch.hub.set_dir("~")
os.environ["TRANSFORMERS_CACHE"] = "~/cache/"


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
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clf-batch-size", type=int, default=4096)
    parser.add_argument("--max-len", type=int, default=1600)
    parser.add_argument("--model", type=str, default="ankh_tps")
    parser.add_argument("--fasta-path", type=str, default="data/uniref90.fasta")
    parser.add_argument("--starting-i", type=int, default=0)
    parser.add_argument("--end-i", type=int, default=700_000)
    parser.add_argument("--output-id", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-root", type=str, default="detected_tps")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument(
        "--ckpt-root-path", type=str, default="data/ankh_random_forests.pkl"
    )
    parser.add_argument("--detection-threshold", type=float, default=0.2)
    return parser.parse_args()


@dataclass
class PredictionResults:
    """
    A data class to store batches of model predictions
    """

    uniprot_id: list[str]
    confidence: list[float]


if __name__ == "__main__":
    args = parse_args()
    model, batch_converter, alphabet = get_model_and_tokenizer(
        args.model, return_alphabet=True
    )
    MAX_LEN = args.max_len

    compute_embeddings_partial = partial(
        compute_embeddings,
        bert_model=model,
        converter=batch_converter,
        padding_idx=alphabet.padding_idx,
        model_repr_layer=33,
        max_len=MAX_LEN,
    )

    uniprot_generator = esm.data.read_fasta(args.fasta_path)

    BATCH_SIZE = args.batch_size
    detection_threshold = args.detection_threshold
    clf_batch_size = args.clf_batch_size

    output_root = Path(args.output_root)
    with open(args.ckpt_root_path, "rb") as file:
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
            y_pred_proba = classifier.predict_proba(enzyme_encodings_np_batch)
            for sample_i in range(n_samples):
                predictions_raw = {}
                for class_i, class_name in enumerate(classifier.classes_):
                    if class_name != "Unknown":
                        predictions_raw[class_name] = y_pred_proba[class_i][sample_i, 1]
                if classifier_i == 0:
                    predictions.append(
                        {
                            class_name.replace("-", "_"): value / len(classifiers)
                            for class_name, value in predictions_raw.items()
                        }
                    )
                else:
                    for class_name, value in predictions_raw.items():
                        predictions[sample_i][
                            class_name.replace("-", "_")
                        ] += value / len(classifiers)
        return predictions

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
            ) = compute_embeddings_partial(batch_to_process)
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
                if class_2_prob["isTPS"] >= detection_threshold:
                    output_file = results_output_root / protein_id_short
                    with open(output_file, "w", encoding="utf-8") as outputs_file:
                        json.dump(class_2_prob, outputs_file)
            enzyme_encodings_list_to_process = []
            enzyme_ids_list_to_process = []
        return enzyme_encodings_list_to_process, enzyme_ids_list_to_process

    enzyme_encodings_list: list[np.ndarray] = []
    enzyme_ids_list: list[str] = []
    starting_i = (args.ciirc_session - 1) * args.delta
    end_i = starting_i + args.delta
    for i, uniprot_entry in tqdm(enumerate(uniprot_generator), total=end_i):
        if i < starting_i:
            continue
        if i == end_i:
            break
        uniprot_id = _extract_id_from_entry(uniprot_entry)
        seq = _extract_seq_from_entry(uniprot_entry)
        if not _is_sequence_good(uniprot_entry[1], max_seq_len=MAX_LEN):
            seq = seq[: (MAX_LEN - 2)]
        next_batch.append(seq)
        next_batch_ids.append(uniprot_id)

        if len(next_batch) == BATCH_SIZE:
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
