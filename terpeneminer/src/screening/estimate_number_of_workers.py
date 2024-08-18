"""This module estimates the number of workers for the screening pipeline"""

import argparse
import logging
import esm  # type: ignore
from tqdm.auto import tqdm  # type: ignore

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=int, default=40000)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--fasta-path", type=str, default="data/uniprot_trembl.fasta")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uniprot_generator = esm.data.read_fasta(args.fasta_path)
    for i, _ in tqdm(enumerate(uniprot_generator), total=300_000_000):
        pass
    try:
        TOTAL_NUMBER_OF_PROTEINS = i + 1  # pylint: disable=W0631
    except NameError:
        TOTAL_NUMBER_OF_PROTEINS = 0
    logger.info(
        "Total number of proteins in the fasta file: %s", TOTAL_NUMBER_OF_PROTEINS
    )
    logger.info(
        "Estimated number of workers: %s",
        TOTAL_NUMBER_OF_PROTEINS // (args.delta * args.n_gpus) + 1,
    )
