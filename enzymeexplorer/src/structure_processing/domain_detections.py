"""This script detects TPS domains in protein structures"""

import os
import tempfile
import yaml
import configargparse
from pathlib import Path
from collections import defaultdict
import pickle
import logging
from pymol import cmd  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from Bio import PDB  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import re
from enzymeexplorer.src.structure_processing.structural_algorithms import (
    MappedRegion,
    get_alignments,
    get_remaining_residues,
    find_continuous_segments_longer_than,
    get_mapped_regions_with_surroundings_parallel,
)
from enzymeexplorer.src.structure_processing.utils import (
    save_file_to_all_residues,
    get_pdb_files,
    prefilter_pdb_files_by_foldseek_alignments,
    store_domains,
    store_domain_separately,
    detect_domains_roughly,
    plot_aligned_domains,
    get_confident_residue_mappings
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def parse_args() -> configargparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = configargparse.ArgumentParser(
        description="A script to detect TPS domains in protein structures"
    )

    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
        default="configs/domain_detection_default_config.yaml",
    )

    parser.add_argument(
        "--needed-proteins-csv-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--csv-id-column",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input-directory-with-structures",
        help="A directory containing PDB structures",
        type=str,
        default="data/structs/",
    )
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument(
        "--detections-output-path",
        help="A path to save a dictionary with the detected domains to",
        type=str,
    )
    parser.add_argument(
        "--store-domains",
        help="A flag to store detected domains",
        action="store_true",
    )
    parser.add_argument("--detected-regions-root-path", type=str)
    parser.add_argument(
        "--domains-output-path",
        help="A root path for saving the detected domains to",
        type=str,
    )
    parser.add_argument("--is-bfactor-confidence", action="store_true")
    parser.add_argument("--do-not-store-intermediate-files", action="store_true")
    parser.add_argument(
        "--secondary-structure-residues-path",
        type=str,
        default="data/secondary_structure_residues.pkl",
    )
    parser.add_argument(
        "--recompute-existing-secondary-structure-residues",
        action="store_true",
    )
    parser.add_argument(
        "--prefilter-pdbs-by-foldseek-alignment",
        action="store_true",
    )
    parser.add_argument(
        "--domain-templates",
        nargs="+",
        default=[
            {
                "name": "alpha",
                "path": "data/domain_templates/1ps1.pdb",
                "residues": "chain A & ss H+S",
                "thresholds": {"tmscore": 0.2, "min_align_len": 70},
            },
            {
                "name": "beta",
                "path": "data/domain_templates/5eat.pdb",
                "residues": "resi 37-57+64-97+104-117+123-129+138-156+162-195+203-213+223-239 & chain A & ss H+S",
                "thresholds": {"tmscore": 0.2, "min_align_len": 50},
            },
            {
                "name": "gamma",
                "path": "data/domain_templates/3p5r.pdb",
                "residues": "resi 138-151+157-171+185-222+233-248+258-275+281-304+313-339 & chain A & ss H+S",
                "thresholds": {"tmscore": 0.2, "min_align_len": 50},
            },
        ],
    )
    return parser.parse_args()



def main():
    args = parse_args()
    input_directory = Path(args.input_directory_with_structures).absolute()

    domain_templates = args.domain_templates
    domain_templates = [yaml.safe_load(template) for template in domain_templates]

    for domain_template in domain_templates:
        domain_template["path"] = Path(domain_template["path"]).absolute()
    supported_domains = [template["name"] for template in domain_templates]
    domain_2_threshold = {
        template["name"]: template["thresholds"] for template in domain_templates
    }

    # reading the needed proteins
    pdb_files = get_pdb_files(
        args.needed_proteins_csv_path,
        args.csv_id_column,
        input_directory,
    )
    if args.prefilter_pdbs_by_foldseek_alignment:
        logger.info(f"Filtering PDB files by foldseek alignment to domain templates")
        pdb_files = prefilter_pdb_files_by_foldseek_alignments(
            pdb_files, domain_templates
        )

    secondary_structure_residues_path = Path(args.secondary_structure_residues_path)
    if (
        not secondary_structure_residues_path.exists()
        or args.recompute_existing_secondary_structure_residues
    ):
        save_file_to_all_residues(
            secondary_structure_residues_path=secondary_structure_residues_path,
            pdb_files=pdb_files,
            domain_templates=domain_templates,
        )

    with open(secondary_structure_residues_path, "rb") as file:
        file_2_all_residues = pickle.load(file)

    # getting the files
    cwd = os.getcwd()
    os.chdir(input_directory)

    filename_2_known_regions: dict[str, list[MappedRegion]] = defaultdict(list)
    filename_2_remaining_residues: dict[str, set[str]] = file_2_all_residues.copy()

    for detection_iter in range(args.n_iters):
        logger.info(f"Starting detection iteration {detection_iter + 1}")

        # Detecting TPS domains in protein structures
        filename_2_potential_regions = detect_domains_roughly(
            [
                pdb_file
                for pdb_file in pdb_files
                if len(filename_2_remaining_residues.get(pdb_file.stem, [])) >= 10
            ],  # only considering files for which there are remaining residues
            filename_2_remaining_residues,
            domain_templates=domain_templates,
            args=args,
            iteration=detection_iter + 1,
        )

        filename_2_detected_region = {
            filename: (
                [
                    sorted(
                        filename_2_potential_regions[filename],
                        key=lambda r: r.tmscore,
                        reverse=True,
                    )[0]
                ]
                if len(filename_2_potential_regions[filename]) > 0
                else []
            )
            for filename in filename_2_potential_regions
        }

        filename_2_detected_region_with_potential_expansion = (
            get_mapped_regions_with_surroundings_parallel(
                list(filename_2_detected_region.keys()),
                file_2_all_residues,
                filename_2_detected_region,
                n_jobs=args.n_jobs,
            )
        )

        # Get unsegmented parts
        filename_2_remaining_residues = get_remaining_residues(
            filename_2_detected_region_with_potential_expansion,
            filename_2_remaining_residues,
        )
        for filename in filename_2_detected_region:
            if len(filename_2_detected_region[filename]) == 0:
                continue
            region = filename_2_detected_region[filename][0]
            num_regions_of_same_domain_type = len(
                [
                    reg
                    for reg in filename_2_known_regions[filename]
                    if reg.domain == region.domain
                ]
            )
            region.module_id = (
                f"{filename}_{region.domain}_{num_regions_of_same_domain_type}"
            )
            filename_2_known_regions[filename].append(region)

    filename_2_known_regions_completed = get_mapped_regions_with_surroundings_parallel(
        list(filename_2_known_regions.keys()),
        file_2_all_residues,
        filename_2_known_regions,
        n_jobs=args.n_jobs,
    )

    # Getting confident residues
    if args.is_bfactor_confidence:
        filename_2_known_regions_completed_confident = get_confident_residue_mappings(
            filename_2_known_regions_completed,
            file_2_all_residues,
            domain_2_threshold
        )
    else:
        filename_2_known_regions_completed_confident = filename_2_known_regions_completed

    (Path(cwd) / args.detected_regions_root_path).mkdir(parents=True, exist_ok=True)
    if not args.do_not_store_intermediate_files:
        plot_aligned_domains(
            filename_2_known_regions_completed_confident,
            supported_domains,
            Path(cwd) / args.detected_regions_root_path,
        )

    (Path(cwd) / args.domains_output_path).mkdir(parents=True, exist_ok=True)
    store_domain_separately(
        filename_2_known_regions_completed_confident,
        supported_domains,
        Path(cwd) / args.detected_regions_root_path,
    )
    if args.store_domains:
        store_domains(
            filename_2_known_regions_completed_confident,
            supported_domains,
            Path(cwd) / args.domains_output_path,
            cmd,
        )

    os.chdir(cwd)
    # save the confident regions
    with open(args.detections_output_path, "wb") as f:
        pickle.dump(filename_2_known_regions_completed_confident, f)


if __name__ == "__main__":
    main()
