"""This script detects TPS domains in protein structures"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import pickle
import time
import logging
import subprocess
from datetime import datetime
from pymol import cmd  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from Bio import PDB  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from enzymeexplorer.src.structure_processing.structural_algorithms import (
    SUPPORTED_DOMAINS,
    DOMAIN_2_THRESHOLD,
    MappedRegion,
    get_alignments,
    plot_aligned_domains,
    get_remaining_residues,
    find_continuous_segments_longer_than,
    get_mapped_regions_with_surroundings_parallel,
    compress_selection_list,
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


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to detect TPS domains in protein structures"
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
        default="data/alphafold_structs/",
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
        "--recompute-existing-secondary-structure-residues", action="store_true"
    )
    return parser.parse_args()


def detect_domains_roughly(
    specified_pdb_files: list[Path],
    file_2_all_residues_mapping: dict[str, set[str]],
    domain_2_threshold: dict[str, tuple[float, int]],
    output_root: Path,
    supported_domains: set[str],
    n_jobs: int = 16,
    iteration: int = 0,
) -> dict[str, list[MappedRegion]]:
    """
    Detects protein domains in multiple structures based on alignment scores and domain-specific thresholds.

    :param file_2_all_residues_mapping: A dictionary mapping file identifiers to sets of residue sequences present in those files
    :param domain_2_threshold: A dictionary mapping domain names to a tuple containing the TM-score threshold (float)
                               and the minimum mapping size (int) required to consider a match valid
    :param output_root: The root directory where output images and serialized results will be saved
    :param supported_domains: A set of domain names to consider for detection. Defaults to SUPPORTED_DOMAINS
    :param n_jobs: The number of parallel jobs to use for alignment calculations. Defaults to 16

    :return: A dictionary mapping each filename to a list of known MappedRegion objects representing the detected
             reliable domains, while ensuring that no overlaying domains are included.
    """
    file_2_possible_regions: dict = defaultdict(list)
    for domain_this in supported_domains:
        logger.info("Started detection of domain %s", domain_this)
        start_t = time.time()
        file_2_tmscore_residues_domain = get_alignments(
            specified_pdb_files,
            domain_name=domain_this,
            file_2_current_residues=file_2_all_residues_mapping,
            n_jobs=n_jobs,
        )
        logger.info(
            "Detection of %s domain. Execution took %d seconds",
            domain_this,
            time.time() - start_t,
        )

        execution_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if (
            len(file_2_tmscore_residues_domain)
            and not args.do_not_store_intermediate_files
        ):
            plot_aligned_domains(
                file_2_tmscore_residues_domain,
                title=f"{domain_this} domain detections",
                save_path=output_root
                / f"{domain_this}_detections_{execution_timestamp}.png",
            )

        for sequence_id, current_detections in file_2_tmscore_residues_domain.items():
            logger.info(sequence_id)
            for i, (tm_score, res_mapping) in enumerate(current_detections):
                logger.info(f"tm_score: {tm_score:.2f}")
                logger.info(f"len of res_mapping: {len(res_mapping)}")
                if len(res_mapping) >= domain_2_threshold[domain_this][1] and tm_score >= domain_2_threshold[domain_this][0]:
                    file_2_possible_regions[sequence_id].append(
                        MappedRegion(
                            module_id=f"{sequence_id}_{domain_this}_{i}",
                            domain=domain_this,
                            tmscore=tm_score,
                            residues_mapping=res_mapping,
                        ),
                    )

        logger.info(
            "Detected %d potential %s domains", len([dom for doms in file_2_possible_regions.values() for dom in doms]), domain_this
        )

        if not args.do_not_store_intermediate_files:
            with open(
                output_root
                / f"potential_regions_{domain_this}_{execution_timestamp}_iter_{iteration}.pkl",
                "wb",
            ) as result_file:
                pickle.dump(file_2_possible_regions, result_file)

    return file_2_possible_regions


def is_similar_to_known_region(
    region_known: MappedRegion,
    region_new: MappedRegion,
    threshold_recall_threshold: float = 0.5,
) -> bool:
    """
    Checks whether two regions overlap sufficiently based on a recall threshold.

    :param region_known: The known region to compare against
    :param region_new: The new region to be compared
    :param threshold_recall_threshold: The minimum recall threshold for the regions to be considered similar, defaults to 0.5

    :return: True if the overlap between the two regions meets or exceeds the threshold, otherwise False
    """
    mapped_residues_known = set(region_known.residues_mapping.keys())
    mapped_residues_new = set(region_new.residues_mapping.keys())
    if len(mapped_residues_new) == 0 or len(mapped_residues_known) == 0:
        return False
    return (
        len(mapped_residues_new.intersection(mapped_residues_known))
        / len(mapped_residues_new)
        >= threshold_recall_threshold
    )


def is_similar_to_anything_known(
    file_name: str,
    struct_region: MappedRegion,
    file_2_known_regions: dict[str, list[MappedRegion]],
    threshold_recall_threshold: float = 0.5,
) -> bool:
    """
    Checks if `region_new` overlaps with any of the known regions in the given file.

    :param file_name: A filename to be compared
    :param file_2_known_regions: A dictionary mapping filenames to lists of known MappedRegion objects
    :param threshold_recall_threshold: The minimum recall threshold for the regions to be considered similar, defaults to 0.5

    :return: True if the new region overlaps with any known region according to the threshold, otherwise False
    """
    for region_known in file_2_known_regions[file_name]:
        if is_similar_to_known_region(
            region_known, struct_region, threshold_recall_threshold
        ):
            return True
    return False


def can_there_be_unassigned_domain(
    file_name: str,
    filename_2_remaining_residues_mapping: dict[str, set[str]],
    filename_2_known_regions_mapping: dict[str, list[MappedRegion]],
    min_continuous_len: int = 15,
    max_allowed_gap: int = 3,
) -> bool:
    """
    Determines whether there could be an unassigned domain in the given file based on the remaining residues.

    :param file_name: The name of the file to check for unassigned domains
    :param filename_2_remaining_residues_mapping: A dictionary mapping filenames to sets of remaining residues not yet assigned to any domain
    :param filename_2_known_regions_mapping: A dictionary mapping filenames to lists of known MappedRegion objects
    :param min_continuous_len: The minimum length of residues required to consider the presence of an unassigned domain, defaults to 15
    :param max_allowed_gap: The maximum gap allowed between residues in a continuous segment, defaults to 3

    :return: True if there could be an unassigned domain in the file, otherwise False
    """
    if file_name not in filename_2_known_regions_mapping:
        return False
    region_types = {reg.domain for reg in filename_2_known_regions_mapping[file_name]}
    if "alpha" not in region_types:
        return (
            len(filename_2_remaining_residues_mapping[file_name]) > min_continuous_len
        )
    return (
        len(
            find_continuous_segments_longer_than(
                filename_2_remaining_residues_mapping[file_name],
                min_secondary_struct_len=min_continuous_len,
                max_allowed_gap=max_allowed_gap,
            )
        )
        > 0
    )


def get_confident_af_residues(
    sequence_id: str, confidence_threshold: int = 70
) -> set[int]:
    """
    Retrieves a set of residues from an AlphaFold PDB file that have a confidence score (B-factor) above the specified threshold.

    :param sequence_id: The ID of the protein for which the PDB file is to be parsed
    :param confidence_threshold: The minimum B-factor required for a residue to be considered confident, defaults to 70

    :return: A set of residue numbers that have a confidence score above the specified threshold
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure(sequence_id, f"{sequence_id}.pdb")

    confident_residues = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_bfactor() >= confidence_threshold:
                        confident_residues.add(residue.get_id()[1])
                    break
    return confident_residues


def get_all_confidence_values(sequence_id: str) -> list[int]:
    """
    Retrieves a set of residues from an AlphaFold PDB file that have a confidence score (B-factor) above the specified threshold.

    :param sequence_id: The ID of the protein for which the PDB file is to be parsed
    :param confidence_threshold: The minimum B-factor required for a residue to be considered confident, defaults to 70

    :return: A set of residue numbers that have a confidence score above the specified threshold
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure(sequence_id, f"{sequence_id}.pdb")

    values = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    values.append(atom.get_bfactor())
                    break
    return values


def main(args: argparse.Namespace):
    # reading the needed proteins
    if args.needed_proteins_csv_path is not None:
        proteins_df = pd.read_csv(args.needed_proteins_csv_path)
        relevant_protein_ids = set(proteins_df[args.csv_id_column].values)

    input_directory = Path(args.input_directory_with_structures)
    all_secondary_structure_residues_path = input_directory / "file_2_all_residues.pkl"
    if (
        not all_secondary_structure_residues_path.exists()
        or args.recompute_existing_secondary_structure_residues
    ):
        subprocess.check_output(
            f"python -m enzymeexplorer.src.structure_processing.compute_secondary_structure_residues --input-directory {input_directory} --output-path {all_secondary_structure_residues_path}".split(),
        )
    with open(all_secondary_structure_residues_path, "rb") as file:
        file_2_all_residues = pickle.load(file)

    # getting the files
    cwd = os.getcwd()
    os.chdir(input_directory)
    blacklist_files = (
        {"1ps1.pdb", "5eat.pdb", "3p5r.pdb", "1w6j.pdb", "1ubw.pdb"}
        .union({f"{domain}.pdb" for domain in SUPPORTED_DOMAINS})
        .union({f"{domain}_object.pdb" for domain in SUPPORTED_DOMAINS})
    )
    all_pdb_files = [filepath for filepath in Path(".").glob("*.pdb")]
    pdb_files = []
    found_relevant_protein_ids = set()
    for filepath in all_pdb_files:
        if (str(filepath) in blacklist_files) or (
            filepath.stem not in file_2_all_residues
        ):
            continue
        if args.needed_proteins_csv_path is not None:
            protein_id = filepath.stem
            simple_protein_id = "".join(
                filepath.stem.replace("(", "").replace(")", "").replace("-", "")
            )
            if protein_id in relevant_protein_ids:
                found_relevant_protein_ids.add(protein_id)
            elif simple_protein_id in relevant_protein_ids:
                found_relevant_protein_ids.add(simple_protein_id)
            else:
                continue
        pdb_files.append(filepath)

    # Warn user if PDB files are missing for any proteins specified by needed_proteins_csv_path
    if args.needed_proteins_csv_path is not None:
        missing_protein_ids = relevant_protein_ids - found_relevant_protein_ids
        if missing_protein_ids:
            logger.warning(
                f"Missing PDB files for the following {len(missing_protein_ids)} protein(s) specified in {args.needed_proteins_csv_path}: {', '.join(missing_protein_ids)}\nProteins with missing PDB files will be skipped."
            )
            
    filename_2_known_regions: dict[str, list[MappedRegion]] = defaultdict(list)
    filename_2_remaining_residues: dict[str, set[str]] = file_2_all_residues.copy()
            
    for detection_iter in range(args.n_iters):
        logger.info(f"Starting detection iteration {detection_iter + 1}")

    # Detecting TPS domains in protein structures
        filename_2_potential_regions = detect_domains_roughly(
            pdb_files,
            filename_2_remaining_residues,
            DOMAIN_2_THRESHOLD,
            supported_domains=SUPPORTED_DOMAINS,
            output_root=Path("."),
            n_jobs=args.n_jobs,
            iteration=detection_iter + 1,
        )
        
        filename_2_detected_region = {
            filename: ([sorted(
                filename_2_potential_regions[filename],
                key=lambda r: r.tmscore,
                reverse=True,
            )[0]]
            if len(filename_2_potential_regions[filename]) > 0
            else [])
            for filename in filename_2_potential_regions
        }
        
        filename_2_detected_region_with_potential_expansion = get_mapped_regions_with_surroundings_parallel(
            list(filename_2_detected_region.keys()),
            file_2_all_residues,
            filename_2_detected_region,
            n_jobs=args.n_jobs,
        )

        # Get unsegmented parts
        filename_2_remaining_residues = get_remaining_residues(
            filename_2_detected_region_with_potential_expansion, filename_2_remaining_residues
        )
        for filename in filename_2_detected_region:
            if len(filename_2_detected_region[filename]) == 0:
                continue
            region = filename_2_detected_region[filename][0]
            num_regions_of_same_domain_type = len([reg for reg in filename_2_known_regions[filename] if reg.domain == region.domain])
            region.module_id = f"{filename}_{region.domain}_{num_regions_of_same_domain_type}"
            filename_2_known_regions[filename].append(region)
            
    filename_2_known_regions_completed = get_mapped_regions_with_surroundings_parallel(
        list(filename_2_known_regions.keys()),
        file_2_all_residues,
        filename_2_known_regions,
        n_jobs=args.n_jobs,
    )

    # Getting confident residues
    filename_2_known_regions_completed_confident = {}
    for filename, regions in tqdm(
        filename_2_known_regions_completed.items(), desc="Filtering confident residues"
    ):
        if args.is_bfactor_confidence:
            conf_residues = get_confident_af_residues(filename)
            if len(conf_residues) < 0.6 * len(file_2_all_residues[filename]):
                logger.warning(
                    f"Too few confident residues for {filename}, leaving top-80% most confident residues"
                )
                all_confidence_values = get_all_confidence_values(filename)
                conf_residues = get_confident_af_residues(
                    filename, np.percentile(all_confidence_values, 20)
                )
            new_regions = []
            for mapped_region_init in regions:
                new_residues_mapping = {
                    res: res_dom
                    for res, res_dom in mapped_region_init.residues_mapping.items()
                    if res in conf_residues
                }
                if (
                    len(new_residues_mapping)
                    >= DOMAIN_2_THRESHOLD[mapped_region_init.domain][1]
                ):
                    new_regions.append(
                        MappedRegion(  # pylint: disable=R0801
                            module_id=mapped_region_init.module_id,
                            domain=mapped_region_init.domain,
                            tmscore=mapped_region_init.tmscore,
                            residues_mapping=new_residues_mapping,
                        )
                    )
            filename_2_known_regions_completed_confident[filename] = new_regions
        else:
            filename_2_known_regions_completed_confident[filename] = regions

    # for further convenience, storing also regions separately per domain
    domain_2_regions_completed_confident = defaultdict(list)

    for (
        filename,
        protein_regions,
    ) in filename_2_known_regions_completed_confident.items():
        for region in protein_regions:
            domain_2_regions_completed_confident["all"].append((filename, region))
            domain_2_regions_completed_confident[region.domain].append(
                (filename, region)
            )
    (Path(cwd) / args.detected_regions_root_path).mkdir(parents=True, exist_ok=True)
    with open(
        Path(cwd)
        / args.detected_regions_root_path
        / "regions_completed_very_confident_all_ALL.pkl",
        "wb",
    ) as f:
        pickle.dump(domain_2_regions_completed_confident["all"], f)
    for domain_name in SUPPORTED_DOMAINS:
        with open(
            Path(cwd)
            / args.detected_regions_root_path
            / f"regions_completed_very_confident_{domain_name}_ALL.pkl",
            "wb",
        ) as f:
            pickle.dump(domain_2_regions_completed_confident[domain_name], f)

    (Path(cwd) / args.domains_output_path).mkdir(parents=True, exist_ok=True)
    if args.store_domains:
        domains_output_path = Path(cwd) / args.domains_output_path
        if not domains_output_path.exists():
            domains_output_path.mkdir(parents=True)
        for domain_name in SUPPORTED_DOMAINS:
            PATH = domains_output_path / f"tps_domain_detections_{domain_name}"
            if not PATH.exists():
                PATH.mkdir(parents=True)

        for filename, protein_regions in tqdm(
            filename_2_known_regions_completed_confident.items(),
            desc="Storing detected domains",
        ):
            for region in protein_regions:
                PATH = Path(
                    domains_output_path / f"tps_domain_detections_{region.domain}"
                )
                mapped_residues = list(set(region.residues_mapping.keys()))
                cmd.delete(filename)
                cmd.load(f"{filename}.pdb")
                print(
                    f"{region.module_id}",
                    f"{filename} & resi {compress_selection_list(mapped_residues)}",
                )
                cmd.select(
                    f"{region.module_id}",
                    f"{filename} & resi {compress_selection_list(mapped_residues)}",
                )
                cmd.save(f"{PATH}/{region.module_id}.pdb", f"{region.module_id}")
                cmd.save(
                    f"{domains_output_path}/{region.module_id}.pdb",
                    f"{region.module_id}",
                )
                cmd.delete(filename)

    os.chdir(cwd)
    # save the confident regions
    with open(args.detections_output_path, "wb") as f:
        pickle.dump(filename_2_known_regions_completed_confident, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
