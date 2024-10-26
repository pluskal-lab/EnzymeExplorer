"""This script detects TPS domains in protein structures"""

import os
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict
import pickle
import time
import logging
import subprocess
from datetime import datetime
from shutil import copyfile
from uuid import uuid4

from pymol import cmd  # type: ignore
import pandas as pd  # type: ignore
from Bio import PDB  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from terpeneminer.src.structure_processing.structural_algorithms import (
    SUPPORTED_DOMAINS,
    DOMAIN_2_THRESHOLD,
    MappedRegion,
    get_alignments,
    get_remaining_residues,
    get_mapped_regions_with_surroundings_parallel,
    compress_selection_list,
get_pairwise_tmscore
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to compare detected TPS domains to the known ones"
    )
    parser.add_argument(
        "--input-directory-with-structures",
        help="A directory containing PDB structures",
        type=str,
        default="data/alphafold_structs/",
    )
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument(
        "--domain-detections-path",
        help="A path to a dictionary with the detected domains",
        type=str,
        default="_temp/filename_2_detected_domains_completed_confident.pkl",
    )
    parser.add_argument(
        "--domain-detections-residues-path",
        help="A path to a dictionary with the secondary-structure residues per file",
        type=str,
        default="_temp/file_2_all_residues.pkl",
    )
    parser.add_argument("--path-to-all-known-domains", type=str, default="data/alphafold_structs/regions_completed_very_confident_all_ALL.pkl")
    parser.add_argument("--path-to-known-domains-subset", type=str, default="data/domains_subset.pkl")
    parser.add_argument("--number-of-workers", type=int, default=16)
    parser.add_argument("--output-path", type=str, default="_temp/filename_2_regions_vs_known_reg_dists.pkl")
    parser.add_argument("--pdb-filepath", type=str, default="")
    return parser.parse_args()


def compute_distances_to_known_regions(
    segment_i: int,
    current_region_segments: list[list[tuple[str, MappedRegion]]],
    region_i: tuple[str, MappedRegion],
    filename_2_all_residues: dict,
    computation_id: str
):
    """
        Computes pairwise distances (TM-scores) between a given region and all other regions within the same segment
        and saves the results to a pickle file.

        This function calculates the TM-score (a structural similarity score) between a specific region
        (`region_i`) and all regions in the same segment (`segment_i`) as defined by the `region_segments`.
        If a results file already exists for the given region and segment, it skips the computation.
        Otherwise, the function computes the distances, stores them in a list, and writes the results
        to a temporary pickle file named based on the segment index and computation ID.

        Parameters
        ----------
        segment_i : int
            The index of the segment for which the distances should be computed.
        current_region_segments : list[list[tuple[str, MappedRegion]]]
            A list of segments, where each segment is a list of tuples. Each tuple consists of a string identifier
            and a `MappedRegion` object representing a region within the segment.
        region_i : tuple[str, MappedRegion]
            A tuple containing a string identifier and a `MappedRegion` object for the region of interest
            (the region whose distances to all other regions in the same segment are to be computed).
        filename_2_all_residues : dict
            A dictionary mapping filenames to residue data, which is used during the TM-score calculation.
        computation_id : str
            A unique string identifier for the computation, used to construct the output filename.

        Returns
        -------
        list of tuple
            A list of tuples, where each tuple contains the module ID of the compared region and the computed
            TM-score distance to `region_i`. If the result already exists, the function will return `None`.

        Notes
        -----
        The function checks if the results file already exists and skips computation if so.
        The results are stored in a pickle file with the format:
        `_temp_tm_region_<module_id>_segment_<segment_i>_<computations_id>.pkl`.

        The TM-score computation relies on the external function `get_pairwise_tmscore`,
        which takes in a PyMOL command object (`cmd`), the regions being compared, and residue data.
        """
    computation_results_path = f"_temp_tm_region_{region_i[1].module_id}_segment_{segment_i}_{computation_id}.pkl"
    if os.path.exists(computation_results_path):
        return
    results = []
    for region_j in current_region_segments[segment_i]:
        tmscore = get_pairwise_tmscore(cmd, region_i, region_j, filename_2_all_residues)
        results.append((region_j[1].module_id, tmscore))
    with open(computation_results_path, "wb") as write_file:
        pickle.dump(results, write_file)
    return results


if __name__ == "__main__":
    args = parse_args()

    # loading secondary structure residues
    input_directory = Path(args.input_directory_with_structures)
    with open(args.domain_detections_residues_path, "rb") as f:
        file_2_all_residues = pickle.load(f)
    all_secondary_structure_residues_path = input_directory / "file_2_all_residues.pkl"
    with open(all_secondary_structure_residues_path, "rb") as f:
        file_2_all_residues_all = pickle.load(f)
    file_2_all_residues.update(file_2_all_residues_all)

    # loading all known domains
    with open(args.path_to_all_known_domains, "rb") as file:
        regions_completed_very_confident_all_ALL = pickle.load(file)
    with open(args.path_to_known_domains_subset, "rb") as file:
        dom_subset, feat_indices_subset = pickle.load(file)
    regions_all = [reg for reg in regions_completed_very_confident_all_ALL if reg[1].module_id in dom_subset]

    # preparing the data for parallel processing
    n_workers = args.number_of_workers
    temp_struct_name = input_directory / Path(args.pdb_filepath).name
    if args.pdb_filepath != str(temp_struct_name):
        copyfile(args.pdb_filepath, temp_struct_name)

    # loading detected domains
    with open(args.domain_detections_path, "rb") as file:
        filename_2_detected_regions_completed_confident = pickle.load(file)
    filename_2_regions_vs_known_reg_dists = {}
    type_2_regions = dict()

    cwd = os.getcwd()
    os.chdir(input_directory)

    for filename, regions in filename_2_detected_regions_completed_confident.items():
        region_2_known_reg_dists = defaultdict(list)
        for region in regions:
            domain_type = region.domain
            if domain_type not in type_2_regions:
                type_2_regions[domain_type] = [el for el in regions_all if el[1].domain == domain_type]
            regions_all_current_type = type_2_regions[domain_type]

            regions_segment_len = len(regions_all_current_type) // n_workers + 1
            region_segments = []
            start_i = 0
            print('len(regions_all): ', len(regions_all))
            while start_i < len(regions_all):
                region_segments.append(regions_all_current_type[start_i:start_i + regions_segment_len])
                print()
                start_i += regions_segment_len


            print('region_segments cout: ', sum([len(x) for x in region_segments]))

            computations_id = str(uuid4())
            partial_dist_compute = partial(
                compute_distances_to_known_regions,
                region_i=(filename, region),
                current_region_segments=region_segments,
                filename_2_all_residues=file_2_all_residues,
                computation_id=computations_id
            )
            print('list(range(len(region_segments))): ', list(range(len(region_segments))))
            with Pool(n_workers - 2) as p:
                list_of_distances_list = p.map(partial_dist_compute, list(range(len(region_segments))))
            for results_path in Path('.').glob(f'*{computations_id}.pkl'):
                with open(results_path, "rb") as f:
                    results_partial = pickle.load(f)
                region_2_known_reg_dists[region.module_id].extend(results_partial)
        filename_2_regions_vs_known_reg_dists[filename] = region_2_known_reg_dists

    os.chdir(cwd)

    with open(args.output_path, "wb") as file:
        pickle.dump(filename_2_regions_vs_known_reg_dists, file)

    os.remove(temp_struct_name)
