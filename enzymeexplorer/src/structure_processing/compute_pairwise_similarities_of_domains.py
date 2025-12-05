"""A script for pairwise comparisons of detected TPS domains"""

import argparse
import logging
import os
import pickle
from functools import partial
from multiprocessing import Pool

from enzymeexplorer.src.structure_processing.structural_algorithms import (
    compute_region_distances,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="alpha")
    parser.add_argument("--start-i", type=int, default=0)
    parser.add_argument("--end-i", type=int, default=1000000)
    parser.add_argument("--precomputed-scores-path", type=str, default="")
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--name-tag", type=str, default="")
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    if args.precomputed_scores_path != "":
        with open(args.precomputed_scores_path, "rb") as file:
            PRECOMPUTED_SCORES = pickle.load(file)
    else:
        PRECOMPUTED_SCORES = None
    os.chdir("data/alphafold_structs")
    logger.info("Loading data...")
    with open("file_2_all_residues.pkl", "rb") as f:
        file_2_all_residues = pickle.load(f)

    regions_path = {
        "alpha": "regions_completed_very_confident_alpha_ALL.pkl",
        "beta": "regions_completed_very_confident_beta_ALL.pkl",
        "gamma": "regions_completed_very_confident_gamma_ALL.pkl",
        "delta": "regions_completed_very_confident_delta_ALL.pkl",
        "epsilon": "regions_completed_very_confident_epsilon_ALL.pkl",
        "all": "regions_completed_very_confident_all_ALL.pkl",
    }[args.name]

    with open(regions_path, "rb") as file:
        regions_all = pickle.load(file)

    logger.info("Data were loaded.")

    partial_dist_compute = partial(
        compute_region_distances,
        regions=regions_all,
        file_2_all_residues=file_2_all_residues,
        output_name=args.name,
        name_tag=args.name_tag,
        precomputed_scores=PRECOMPUTED_SCORES,
    )
    region_indices = list(range(len(regions_all)))[args.start_i : args.end_i]
    logger.info(
        "Started parallel pairwise comparison with %d workers.", args.n_jobs
    )
    with Pool(args.n_jobs) as p:
        list_of_distances_list = p.map(partial_dist_compute, region_indices)


if __name__ == "__main__":
    args = parse_args()
    main(args)
