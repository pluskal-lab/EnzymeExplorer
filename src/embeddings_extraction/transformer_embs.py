"""This script extracts PLM embeddings for UniProt proteins"""
# pylint: disable=R0801
import argparse  # type: ignore
import logging  # type: ignore
import os  # type: ignore
import pickle  # type: ignore
from functools import partial
from pathlib import Path
import pandas as pd  # type: ignore
import torch  # type: ignore
from tqdm.auto import trange  # type: ignore
from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore

from src.embeddings_extraction.ankh_transformer_utils import (
    compute_embeddings as ankh_compute_embeddings,
)
from src.embeddings_extraction.ankh_transformer_utils import (
    get_model_and_tokenizer as ankh_get_model_and_tokenizer,
)
from src.embeddings_extraction.esm_transformer_utils import (
    compute_embeddings as esm_compute_embeddings,
)
from src.embeddings_extraction.esm_transformer_utils import (
    get_model_and_tokenizer as esm_get_model_and_tokenizer,
)

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="esm-1v-1")
    parser.add_argument("--model-repr-layer", type=int, default=33)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=11_000)
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    )
    parser.add_argument("--id-column", type=str, default="Uniprot ID")
    parser.add_argument("--seq-column", type=str, default="Amino acid sequence")
    parser.add_argument("--output-root-path", type=str, default="outputs/ankh_embs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.gpu
    root_path = Path(cli_args.output_root_path)
    if "esm" in cli_args.model:
        logger.info("Loading an ESM-family model %s", cli_args.model)
        model, batch_converter, alphabet = esm_get_model_and_tokenizer(
            cli_args.model, return_alphabet=True
        )
        MAX_LEN = 424242_000 if "esm-2" in cli_args.model else 1022

        compute_embeddings_partial = partial(
            esm_compute_embeddings,
            bert_model=model,
            converter=batch_converter,
            padding_idx=alphabet.padding_idx,
            model_repr_layer=cli_args.model_repr_layer,
            max_len=MAX_LEN,
        )
    elif "ankh" in cli_args.model:
        logger.info("Loading an Ankh-family model %s", cli_args.model)
        model, tokenizer = ankh_get_model_and_tokenizer(cli_args.model)
        compute_embeddings_partial = partial(
            ankh_compute_embeddings, bert_model=model, tokenizer=tokenizer
        )
    else:
        raise NotImplementedError(
            f"Model {cli_args.model} is not supported. Currently only esm, ankh model families are supported"
        )
    logger.info("Model was loaded! Reading data...")
    df = pd.read_csv(cli_args.csv_path)
    df = df.drop_duplicates(cli_args.id_column)
    df = df.sort_values(by=cli_args.id_column)
    ids_list = df[cli_args.id_column].values[
        cli_args.start_index : cli_args.end_index + 1
    ]
    seqs_list = df[cli_args.seq_column].values[
        cli_args.start_index : cli_args.end_index + 1
    ]

    logger.info("Data are ready!")
    batch_size = cli_args.batch_size
    if not (root_path / f"uniprot_embs_{cli_args.model}").exists():
        (root_path / f"uniprot_embs_{cli_args.model}").mkdir(parents=True)

    processed_seqs = []
    with logging_redirect_tqdm([logger]):
        for batch_i in trange(len(seqs_list) // batch_size + 1):
            input_seq_list_batch = seqs_list[
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]
            if len(input_seq_list_batch):
                try:
                    (
                        enzyme_encodings_np_batch,
                        enzyme_encoding_seqs_batch,
                    ) = compute_embeddings_partial(input_seqs=input_seq_list_batch)
                    processed_seqs.extend(input_seq_list_batch)
                    with open(
                        root_path
                        / f"uniprot_embs_{cli_args.model}/batch_{cli_args.gpu}_{batch_i}_embs_avg.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(enzyme_encodings_np_batch, f)

                    with open(
                        root_path
                        / f"uniprot_embs_{cli_args.model}/batch_{cli_args.gpu}_{batch_i}_embs_seqs.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(enzyme_encoding_seqs_batch, f)

                    with open(
                        root_path
                        / f"uniprot_embs_{cli_args.model}/batch_{cli_args.gpu}_{batch_i}_ids.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(
                            ids_list[batch_i * batch_size : (batch_i + 1) * batch_size],
                            f,
                        )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "Batch %d from index range %d - %d resulted in cuda OutOfMemoryError and was skipped",
                        batch_i,
                        cli_args.start_index,
                        cli_args.end_index,
                    )
                    continue

    count_of_unprocessed_entries = len(set(seqs_list).difference(set(processed_seqs)))
    if count_of_unprocessed_entries:
        logger.warning("Unprocessed seqs count: %d", count_of_unprocessed_entries)
