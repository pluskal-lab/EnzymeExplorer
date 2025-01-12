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
from terpeneminer.src.structure_processing.structural_algorithms import (
    SUPPORTED_DOMAINS,
    DOMAIN_2_THRESHOLD,
    MappedRegion,
    get_alignments,
    plot_aligned_domains,
    get_mapped_regions_per_file,
    get_remaining_residues,
    find_continuous_segments_longer_than,
    get_mapped_regions_with_surroundings_parallel,
    compress_selection_list,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    parser.add_argument(
        "--detections-output-path",
        help="A path to save a dictionary with the detected domains to",
        type=str,
        default="data/filename_2_detected_domains_completed_confident.pkl",
    )
    parser.add_argument(
        "--store-domains",
        help="A flag to store detected domains",
        action="store_true",
    )
    parser.add_argument("--detected-regions-root-path", type=str, default="data/alphafold_structs/")
    parser.add_argument(
        "--domains-output-path",
        help="A root path for saving the detected domains to",
        type=str,
        default="data/detected_domains",
    )
    parser.add_argument("--is-bfactor-confidence", action="store_true")
    parser.add_argument("--do-not-store-intermediate-files", action="store_true")
    parser.add_argument("--recompute-existing-secondary-structure-residues", action="store_true")
    return parser.parse_args()


def detect_domains_roughly(
    specified_pdb_files: list[Path],
    file_2_all_residues_mapping: dict[str, set[str]],
    domain_2_threshold: dict[str, tuple[float, int]],
    output_root: Path,
    supported_domains: set[str],
    n_jobs: int = 16,
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
    domain_2_possible_regions = {}
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
        if len(file_2_tmscore_residues_domain) and not args.do_not_store_intermediate_files:
            plot_aligned_domains(
                file_2_tmscore_residues_domain,
                title=f"{domain_this} domain detections",
                save_path=output_root
                / f"{domain_this}_detections_{execution_timestamp}.png",
            )
        regions_of_possible_domain = []



        for uniprot_id, current_detections in file_2_tmscore_residues_domain.items():
            logger.info(uniprot_id)
            for i, (tm_score, res_mapping) in enumerate(current_detections):
                logger.info(f'tm_score: {tm_score:.2f}')
                logger.info(f'len of res_mapping: {len(res_mapping)}')
                if tm_score >= domain_2_threshold[domain_this][0] and len(res_mapping) >= domain_2_threshold[domain_this][1]:
                    regions_of_possible_domain.append(
                        (
                            uniprot_id,
                            MappedRegion(
                                module_id=f"{uniprot_id}_{domain_this}_{i}",
                                domain=domain_this,
                                tmscore=tm_score,
                                residues_mapping=res_mapping,
                            ),
                        )
                    )

        logger.info(
            "Detected %d %s domains", len(regions_of_possible_domain), domain_this
        )

        if not args.do_not_store_intermediate_files:
            with open(
                output_root
                / f"final_regions_{domain_this}s_tm_ALL_{execution_timestamp}.pkl",
                "wb",
            ) as result_file:
                pickle.dump(regions_of_possible_domain, result_file)

        domain_2_possible_regions[domain_this] = regions_of_possible_domain

        if "alpha" in domain_this:
            file_2_mapped_regions = get_mapped_regions_per_file(
                {"alpha": file_2_tmscore_residues_domain}, domain_2_threshold
            )
            file_2_remaining_ress = get_remaining_residues(
                file_2_mapped_regions, file_2_all_residues_mapping
            )
            tm_score_threshold, mapping_res_size_threshold = domain_2_threshold["alpha"]
            file_2_tmscore_residues_2nd_alpha = get_alignments(
                specified_pdb_files,
                "alpha",
                file_2_remaining_ress,
                tm_score_threshold,
                mapping_res_size_threshold,
                n_jobs=n_jobs,
            )
            regions_of_possible_2nd_alphas = []
            for (
                uniprot_id,
                current_detections,
            ) in file_2_tmscore_residues_2nd_alpha.items():
                for i, (tm_score, res_mapping) in enumerate(current_detections):
                    new_tuple = (
                        uniprot_id,
                        MappedRegion(
                            module_id=f"{uniprot_id}_alpha_{i + len(file_2_tmscore_residues_domain[uniprot_id])}",
                            domain="alpha",
                            tmscore=tm_score,
                            residues_mapping=res_mapping,
                        ),
                    )
                    regions_of_possible_2nd_alphas.append(new_tuple)
            execution_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if not args.do_not_store_intermediate_files:
                if len(file_2_tmscore_residues_2nd_alpha):
                    plot_aligned_domains(
                        file_2_tmscore_residues_2nd_alpha,
                        title="2nd alpha domain detections",
                        save_path=output_root
                        / f"2nd_alpha_detections_{execution_timestamp}.png",
                    )

                with open(
                    output_root
                    / f"final_regions_2nd_alphas_tm_ALL_{execution_timestamp}.pkl",
                    "wb",
                ) as result_file:
                    pickle.dump(regions_of_possible_2nd_alphas, result_file)
            domain_2_possible_regions[domain_this] += regions_of_possible_2nd_alphas

    file_2_known_regions: dict = defaultdict(list)
    for domain_name_to_include in ["alpha", "epsilon", "delta", "beta", "gamma", "alphaWeird"]:
        potential_regions = domain_2_possible_regions[domain_name_to_include]
        # filter clashes with already loaded domains
        regions_of_possible_domain_to_include = [
            (file_, region)
            for file_, region in potential_regions
            if not is_similar_to_anything_known(file_, region, file_2_known_regions)
        ]
        for file_name, domain_region in regions_of_possible_domain_to_include:
            file_2_known_regions[file_name].append(domain_region)
    return file_2_known_regions


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
        return len(filename_2_remaining_residues_mapping[file_name]) > min_continuous_len
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
    uniprot_id: str, confidence_threshold: int = 70
) -> set[int]:
    """
    Retrieves a set of residues from an AlphaFold PDB file that have a confidence score (B-factor) above the specified threshold.

    :param uniprot_id: The UniProt ID of the protein for which the PDB file is to be parsed
    :param confidence_threshold: The minimum B-factor required for a residue to be considered confident, defaults to 70

    :return: A set of residue numbers that have a confidence score above the specified threshold
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure(uniprot_id, f"{uniprot_id}.pdb")

    confident_residues = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_bfactor() >= confidence_threshold:
                        confident_residues.add(residue.get_id()[1])
                    break
    return confident_residues


def get_all_confidence_values(
    uniprot_id: str
) -> list[int]:
    """
    Retrieves a set of residues from an AlphaFold PDB file that have a confidence score (B-factor) above the specified threshold.

    :param uniprot_id: The UniProt ID of the protein for which the PDB file is to be parsed
    :param confidence_threshold: The minimum B-factor required for a residue to be considered confident, defaults to 70

    :return: A set of residue numbers that have a confidence score above the specified threshold
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure(uniprot_id, f"{uniprot_id}.pdb")

    values = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    values.append(atom.get_bfactor())
                    break
    return values


if __name__ == "__main__":
    args = parse_args()
    # reading the needed proteins
    if args.needed_proteins_csv_path is not None:
        proteins_df = pd.read_csv(args.needed_proteins_csv_path)
        relevant_protein_ids = set(proteins_df[args.csv_id_column].values)

    input_directory = Path(args.input_directory_with_structures)
    all_secondary_structure_residues_path = input_directory / "file_2_all_residues.pkl"
    if not all_secondary_structure_residues_path.exists() or args.recompute_existing_secondary_structure_residues:
        secondary_structure_computation_output = subprocess.check_output(
            f"python -m terpeneminer.src.structure_processing.compute_secondary_structure_residues --input-directory {input_directory} --output-path {all_secondary_structure_residues_path}".split(),
        )
    with open(all_secondary_structure_residues_path, "rb") as file:
        file_2_all_residues = pickle.load(file)

    # getting the files
    cwd = os.getcwd()
    os.chdir(input_directory)
    blacklist_files = (
        {"1ps1.pdb", "5eat.pdb", "3p5r.pdb"}
        .union({f"{domain}.pdb" for domain in SUPPORTED_DOMAINS})
        .union({f"{domain}_object.pdb" for domain in SUPPORTED_DOMAINS})
    )
    pdb_files = [
        filepath
        for filepath in Path(".").glob("*.pdb")
        if str(filepath) not in blacklist_files
        and filepath.stem in file_2_all_residues
        and (
            args.needed_proteins_csv_path is None
            or filepath.stem in relevant_protein_ids
            or "".join(filepath.stem.replace("(", "").replace(")", "").replace("-", ""))
            in relevant_protein_ids
        )
    ]

    # Detecting TPS domains in protein structures
    filename_2_known_regions = detect_domains_roughly(
        pdb_files,
        file_2_all_residues,
        DOMAIN_2_THRESHOLD,
        supported_domains=SUPPORTED_DOMAINS,
        output_root=Path("."),
        n_jobs=args.n_jobs,
    )

    # Assigning missed secondary structure parts to the closest domains
    filename_2_known_regions_completed = get_mapped_regions_with_surroundings_parallel(
        list(filename_2_known_regions.keys()),
        file_2_all_residues,
        filename_2_known_regions,
        n_jobs=args.n_jobs,
    )

    # Get unsegmented parts
    file_2_remaining_residues = get_remaining_residues(
        filename_2_known_regions, file_2_all_residues
    )
    file_2_remaining_residues_unassigned = file_2_remaining_residues.copy()
    for filename in filename_2_known_regions.keys():
        segments = find_continuous_segments_longer_than(
                    file_2_remaining_residues[filename],
                    min_secondary_struct_len=120,
                    max_allowed_gap=25,
                )
        residues = set(map(str, sum(segments, [])))
        if residues:
            file_2_remaining_residues_unassigned[filename] = residues

    # attempt to detect additional domains in the remaining parts
    domain_2_file_2_tmscore_residues = {}
    for domain_type, (
        tmscore_threshold,
        mapping_size_threshold,
    ) in DOMAIN_2_THRESHOLD.items():
        domain_2_file_2_tmscore_residues[domain_type] = get_alignments(
            pdb_files,
            domain_type,
            file_2_remaining_residues_unassigned,
            tmscore_threshold,
            mapping_size_threshold,
            n_jobs=args.n_jobs,
        )

    sorted_additional_detections = sorted(
        list(domain_2_file_2_tmscore_residues.items()),
        key=lambda x: (
            -len(x[1]),
            0 if len(x[1]) == 0 else -list(x[1].items())[0][1][0][0],
        ),
    )
    for domain, file_2_tmscore_residues in sorted_additional_detections:
        for uni_id, detections in file_2_tmscore_residues.items():
            for tmscore, mapping in detections:
                n_domains_of_type = len(
                    [
                        reg
                        for reg in filename_2_known_regions[uni_id]
                        if reg.domain == domain
                    ]
                )
                new_region = MappedRegion(
                    module_id=f"{uni_id}_{domain}_{n_domains_of_type}",
                    domain=domain,
                    tmscore=tmscore,
                    residues_mapping=mapping,
                )
                if not is_similar_to_anything_known(
                    uni_id, new_region, filename_2_known_regions
                ):
                    filename_2_known_regions[uni_id].append(new_region)

    # Getting confident residues
    filename_2_known_regions_completed_confident = {}
    for filename, regions in tqdm(filename_2_known_regions_completed.items()):
        if args.is_bfactor_confidence:
            conf_residues = get_confident_af_residues(filename)
            if len(conf_residues) < 0.6 * len(file_2_all_residues[filename]):
                logger.warning("Too few confident residues, leaving top-80% most confident residues")
                all_confidence_values = get_all_confidence_values(filename)
                conf_residues = get_confident_af_residues(filename, np.percentile(all_confidence_values, 20))
            new_regions = []
            for mapped_region_init in regions:
                new_residues_mapping = {
                    res: res_dom
                    for res, res_dom in mapped_region_init.residues_mapping.items()
                    if res in conf_residues
                }
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

    with open(Path(cwd) / args.detected_regions_root_path / "regions_completed_very_confident_all_ALL.pkl", "wb") as f:
        pickle.dump(domain_2_regions_completed_confident["all"], f)
    for domain_name in SUPPORTED_DOMAINS:
        with open(Path(cwd) / args.detected_regions_root_path / f"regions_completed_very_confident_{domain_name}_ALL.pkl", "wb") as f:
            pickle.dump(domain_2_regions_completed_confident[domain_name], f)

    if args.store_domains:
        domains_output_path = Path(cwd) / args.domains_output_path
        if not domains_output_path.exists():
            domains_output_path.mkdir(parents=True)
        for domain_name in SUPPORTED_DOMAINS:
            PATH = domains_output_path / f"tps_domain_detections_{domain_name}"
            if not PATH.exists():
                PATH.mkdir(parents=True)

        for filename, protein_regions in tqdm(
            filename_2_known_regions_completed_confident.items()
        ):
            for region in protein_regions:
                PATH = Path(domains_output_path / f"tps_domain_detections_{region.domain}")
                mapped_residues = list(set(region.residues_mapping.keys()))
                cmd.delete(filename)
                cmd.load(f"{filename}.pdb")
                print(f"{region.module_id}",
                    f"{filename} & resi {compress_selection_list(mapped_residues)}")
                cmd.select(
                    f"{region.module_id}",
                    f"{filename} & resi {compress_selection_list(mapped_residues)}",
                )
                cmd.save(f"{PATH}/{region.module_id}.pdb", f"{region.module_id}")
                cmd.save(f"{domains_output_path}/{region.module_id}.pdb", f"{region.module_id}")
                cmd.delete(filename)

    os.chdir(cwd)
    # save the confident regions
    with open(args.detections_output_path, "wb") as f:
        pickle.dump(filename_2_known_regions_completed_confident, f)
