"""This script performs sampling of non-TPS proteins from the Swiss-Prot"""

import argparse
import logging
import pickle
from random import sample
from typing import Generator, Optional, Union

import esm  # type: ignore
import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uniprot-fasta-path", type=str, default="data/uniprot_sprot.fasta"
    )
    parser.add_argument("--uniprot-size-to-check", type=int, default=566_509)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--output-path", type=str, default="data/sampled_id_2_seq.pkl")
    parser.add_argument("--blacklist-path", type=str, default=None)
    args = parser.parse_args()
    return args


def get_random_sample(  # pylint: disable=R0913
    generator: Generator,
    subset_size: int,
    selected_indices_set: set[int],
    uniprot_size_to_check: int = 230_000_000,
    blacklisted_ids: Optional[set] = None,
) -> Union[list[str], tuple[list[str], dict[str, str]]]:
    """
    This function returns a requested random sample from UniProt.
    :param generator: iterator over UniProt entries
    :param selected_indices_set: randomly sampled indices of UniProt entries
    :param uniprot_size_to_check: number of UniProt entries to go through
    :param blacklisted_ids: Blacklisting ids (e.g. of known TPS)
    :return: list of sampled UniProt ID's and UniProt ID -> amino acid sequence mapping when
    return_dict_to_seq is set to True
    """
    uniprot_ids_sampled: list = []
    uniprot_id_2_seq = {}
    append_next_suitable = False

    def _extract_id_from_entry(entry: tuple) -> str:
        return entry[0].split("|")[1]

    def _extract_seq_from_entry(entry: tuple) -> str:
        return entry[1]

    for i, uniprot_entry in tqdm(
        enumerate(generator),
        total=uniprot_size_to_check,
        desc="Sampling uniprot fasta",
    ):
        uniprot_id = _extract_id_from_entry(uniprot_entry)
        if len(uniprot_ids_sampled) == subset_size or i >= uniprot_size_to_check:
            break  # breaking instead of return to enable mypy check typing
        if blacklisted_ids is not None and uniprot_id in blacklisted_ids:
            append_next_suitable = (
                selected_indices_set is not None and i in selected_indices_set
            )
            continue
        if append_next_suitable:
            uniprot_ids_sampled.append(uniprot_id)
            uniprot_id_2_seq[uniprot_id] = _extract_seq_from_entry(uniprot_entry)
            append_next_suitable = False
        elif selected_indices_set is not None and i in selected_indices_set:
            uniprot_ids_sampled.append(uniprot_id)
            uniprot_id_2_seq[uniprot_id] = _extract_seq_from_entry(uniprot_entry)
    return uniprot_ids_sampled, uniprot_id_2_seq


def main() -> None:
    """
    The main-entry-point function to sample non-TPS proteins from the Swiss-Prot
    @return:
    """
    cli_args = parse_args()
    tps_df = pd.read_csv("data/TPS-Nov19_2023_verified_all_reactions.csv")

    sample_indices = set(
        sample(
            list(range(cli_args.uniprot_size_to_check)),
            cli_args.sample_size,
        )
    )
    logger.info("Starting Swiss-Prot negatives sampling.")
    uniprot_generator = esm.data.read_fasta(cli_args.uniprot_fasta_path)
    current_blacklisted_ids = set(tps_df["Uniprot ID"].values)
    if cli_args.blacklist_path is not None:
        with open(cli_args.blacklist_path, "rb") as file:
            blacklisted_id_2_seq = pickle.load(file)
        current_blacklisted_ids = set(blacklisted_id_2_seq.keys()).union(
            current_blacklisted_ids
        )
    _, id_2_seq = get_random_sample(
        uniprot_generator,
        cli_args.sample_size,
        selected_indices_set=sample_indices,
        uniprot_size_to_check=cli_args.uniprot_size_to_check,
        blacklisted_ids=current_blacklisted_ids,
    )
    with open(cli_args.output_path, "wb") as file_write:
        pickle.dump(id_2_seq, file_write)
    logger.info("Swiss-Prot negatives sampled and stored to data/sampled_id_2_seq.pkl.")


if __name__ == "__main__":
    main()
