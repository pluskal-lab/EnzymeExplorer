# TODO: Refactor the code to use pandas DataFrames instead of dictionaries for mappings. This will make the code cleaner and more efficient.

import logging
import subprocess
from typing import Optional
from enzymeexplorer.src.structure_processing.structural_algorithms import (
    MappedRegion,
    compress_selection_list,
    get_alignments,
    find_continuous_segments_longer_than,
)
from multiprocessing import Pool
from pymol import cmd
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import tempfile
from pathlib import Path
import os
import pickle
from Bio import PDB
from collections import defaultdict
from enzymeexplorer.src.structure_processing.foldseek_wrapper import FoldseekWrapper
from tqdm.auto import tqdm
import re
import time
import subprocess
from datetime import datetime
import configargparse
from functools import partial
import pickle

FEATURE_DOMAIN_TYPES = ["alpha_1", "alpha_2", "beta", "gamma"]

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def __get_domain_names(
    seq_2_regions: dict[str, list[MappedRegion]],
) -> list[str]:
    return [
        region.module_id for regions in seq_2_regions.values() for region in regions
    ]


def __get_domain_2_seq_id_and_domain_type_maps(
    seq_2_regions: dict[str, list[MappedRegion]],
) -> tuple[dict[str, str], dict[str, str]]:
    domain_2_seq_id = {
        region.module_id: seq_id
        for seq_id, regions in seq_2_regions.items()
        for region in regions
    }
    domain_2_domain_type = {
        region.module_id: region.domain
        for regions in seq_2_regions.values()
        for region in regions
    }
    return domain_2_seq_id, domain_2_domain_type


def get_foldseek_alignment_df(
    query_seq_2_regions: dict[str, list[MappedRegion]],
    query_domains_dir: str,
    ref_seq_2_regions: dict[str, list[MappedRegion]],
    reference_domains_dir: str,
    reference_domain_subset_path: str | None = None,
) -> pd.DataFrame:
    """Compute pairwise distance based features between domains.

    Args:
        query_seq_2_regions (dict[str, list[MappedRegion]]): A mapping from query sequence IDs to lists of MappedRegion objects.
        query_domains_dir (str): Directory containing query domain structures.
        ref_seq_2_regions (dict[str, list[MappedRegion]]): A mapping from reference sequence IDs to lists of MappedRegion objects.
        reference_domains_dir (str): Directory containing reference domain structures.
        reference_domain_subset_path (str): Path to the pickled list of reference domains to consider.

    Returns:
        pd.DataFrame: DataFrame containing foldseek alignment results.
    """

    query_domains = __get_domain_names(query_seq_2_regions)
    reference_domains = __get_domain_names(ref_seq_2_regions)

    assert set(
        [file.stem for file in Path(reference_domains_dir).glob("*.pdb")]
    ) >= set(
        reference_domains
    ), "Reference domains directory does not contain all required domain structures."

    assert set([file.stem for file in Path(query_domains_dir).glob("*.pdb")]) >= set(
        query_domains
    ), "Query domains directory does not contain all required domain structures."

    if reference_domain_subset_path:
        # Load reference domain subset
        with open(reference_domain_subset_path, "rb") as f:
            reference_domain_subset, _ = pickle.load(f)

        assert set(reference_domain_subset).issubset(
            set(reference_domains)
        ), "Reference domain subset contains domains not present in the reference regions data."
        reference_domains = reference_domain_subset

    logger.info(
        f"Running Foldseek alignment for {len(query_domains)} query domains against {len(reference_domains)} reference domains... (eg. {query_domains[0]} vs {reference_domains[0]})"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        query_pdbs_dir = os.path.join(tmpdir, "query_pdbs")
        reference_pdbs_dir = os.path.join(tmpdir, "reference_pdbs")
        Path(query_pdbs_dir).mkdir(parents=True, exist_ok=True)
        Path(reference_pdbs_dir).mkdir(parents=True, exist_ok=True)
        # Copy relevant PDB files to temporary directories
        for domain in reference_domains:
            src_path = os.path.join(reference_domains_dir, f"{domain}.pdb")
            dst_path = os.path.join(reference_pdbs_dir, f"{domain}.pdb")
            os.symlink(src_path, dst_path)
        for domain in query_domains:
            src_path = os.path.join(query_domains_dir, f"{domain}.pdb")
            dst_path = os.path.join(query_pdbs_dir, f"{domain}.pdb")
            os.symlink(src_path, dst_path)

        alignment_df = FoldseekWrapper().easy_search(
            query_dir=query_pdbs_dir,
            target_dir=reference_pdbs_dir,
            tmp_dir=os.path.join(tmpdir, "tmp_foldseek"),
            output=os.path.join(tmpdir, "foldseek_output.tsv"),
        )

    query_domain_2_seq_id, query_domain_2_domain_type = (
        __get_domain_2_seq_id_and_domain_type_maps(query_seq_2_regions)
    )
    ref_domain_2_seq_id, ref_domain_2_domain_type = (
        __get_domain_2_seq_id_and_domain_type_maps(ref_seq_2_regions)
    )

    alignment_df = alignment_df.sort_values(
        by="alntmscore", ascending=False, inplace=False
    )
    alignment_df["query_domain_type"] = alignment_df["query"].apply(
        lambda x: query_domain_2_domain_type[x]
    )
    alignment_df["query_seq_id"] = alignment_df["query"].apply(
        lambda x: query_domain_2_seq_id[x]
    )
    alignment_df["target_domain_type"] = alignment_df["target"].apply(
        lambda x: ref_domain_2_domain_type[x]
    )
    alignment_df["target_seq_id"] = alignment_df["target"].apply(
        lambda x: ref_domain_2_seq_id[x]
    )

    return alignment_df


def get_reference_domain_type_2_module_ids(
    reference_seq_2_regions: dict[str, list[MappedRegion]],
) -> dict[str, list[str]]:
    """Get a mapping from reference domain types to module ids.

    Args:
        reference_seq_2_regions (dict[str, list[MappedRegion]]): A dictionary mapping reference sequence ids to lists of reference domains.
    Returns:
        defaultdict[str, list[str]]: A dictionary mapping reference domain types to sorted lists of module ids
    """
    ref_domain_type_2_module_id = defaultdict(list)
    for ref_seq in reference_seq_2_regions:
        for region in reference_seq_2_regions[ref_seq]:
            ref_domain_type_2_module_id[region.domain].append(region.module_id)
    ref_domain_type_2_module_id["alpha_1"] = ref_domain_type_2_module_id["alpha"]
    ref_domain_type_2_module_id["alpha_2"] = ref_domain_type_2_module_id["alpha"]
    del ref_domain_type_2_module_id["alpha"]

    for domain_type in ref_domain_type_2_module_id:
        ref_domain_type_2_module_id[domain_type] = sorted(
            ref_domain_type_2_module_id[domain_type]
        )
    return ref_domain_type_2_module_id


def get_col_idx_for_structural_features(
    ref_domain_type_2_module_ids: dict[str, list[str]],
) -> dict[str, dict[str, int]]:
    """Initialize a numpy array to store structural features.

    Args:
        ref_domain_type_2_module_ids (dict[str, list[str]]): A dictionary mapping reference domain types to lists of module ids.

    Returns:
        dict[str, dict[str, int]]: A dictionary mapping domain types to dictionaries mapping module ids to column indices in the structural features array.
    """
    domain_type_2_module_id_2_col_idx = {}
    idx = 0
    for feature_domain_type in FEATURE_DOMAIN_TYPES:
        domain_type_2_module_id_2_col_idx[feature_domain_type] = {}
        for domain_name in ref_domain_type_2_module_ids[feature_domain_type]:
            domain_type_2_module_id_2_col_idx[feature_domain_type][domain_name] = idx
            idx += 1
    return domain_type_2_module_id_2_col_idx


def get_reference_sequence_col_indices(
    ref_seq_2_regions: dict[str, list[MappedRegion]],
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> dict[str, list[int]]:
    """Get a mapping from reference sequence ids to lists of column indices in the structural features array corresponding to the domains present in the reference sequence.

    Args:
        ref_seq_2_regions (dict[str, list[MappedRegion]]): A dictionary mapping reference sequence ids to lists of reference domains.
        domain_type_2_ref_module_id_2_col_idx (dict[str, dict[str, int]]): A dictionary mapping domain types to dictionaries mapping reference module ids to column indices in the structural features array.

    Returns:
        dict[str, list[int]]: A dictionary mapping reference sequence ids to lists of column indices in the structural features array corresponding to the domains present in the reference sequence.
    """
    ref_seq_2_col_idx = defaultdict(list)
    for ref_seq in ref_seq_2_regions:
        for region in ref_seq_2_regions[ref_seq]:
            domain_type = region.domain
            if domain_type == "alpha":
                ref_seq_2_col_idx[ref_seq].append(
                    domain_type_2_ref_module_id_2_col_idx[domain_type + "_1"][
                        region.module_id
                    ]
                )
                ref_seq_2_col_idx[ref_seq].append(
                    domain_type_2_ref_module_id_2_col_idx[domain_type + "_2"][
                        region.module_id
                    ]
                )
            else:
                ref_seq_2_col_idx[ref_seq].append(
                    domain_type_2_ref_module_id_2_col_idx[domain_type][region.module_id]
                )
    return ref_seq_2_col_idx


def get_reference_domains_col_indices(
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> dict[str, list[int]]:
    """Get a mapping from reference domain ids to column indices in the structural features array corresponding to the domains.

    Args:
        domain_type_2_ref_module_id_2_col_idx (dict[str, dict[str, int]]): A dictionary mapping domain types to dictionaries mapping reference module ids to column indices in the structural features array.

    Returns:
        dict[str, list[int]]: A dictionary mapping reference domain ids to lists of column indices in the structural features array corresponding to the domains.
    """
    ref_domain_module_id_2_col_idx = defaultdict(list)
    for domain_type in domain_type_2_ref_module_id_2_col_idx:
        for module_id in domain_type_2_ref_module_id_2_col_idx[domain_type]:
            ref_domain_module_id_2_col_idx[module_id].append(
                domain_type_2_ref_module_id_2_col_idx[domain_type][module_id]
            )
    return ref_domain_module_id_2_col_idx


def get_domain_type_2_col_idx_range(
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> dict[str, tuple[int, int]]:
    """Get a mapping from domain types to tuples of (min column index, max column index) in the structural features array corresponding to the domains of the given type.

    Args:
        domain_type_2_ref_module_id_2_col_idx (dict[str, dict[str, int]]): A dictionary mapping domain types to dictionaries mapping reference module ids to column indices in the structural features array.

    Returns:
        dict[str, tuple[int, int]]: A dictionary mapping domain types to tuples of (min column index, max column index) in the structural features array corresponding to the domains of the given type.
    """
    domain_type_2_col_idx_range = {}
    for domain_type in domain_type_2_ref_module_id_2_col_idx:
        col_idxs = list(domain_type_2_ref_module_id_2_col_idx[domain_type].values())
        domain_type_2_col_idx_range[domain_type] = (min(col_idxs), max(col_idxs) + 1)
    return domain_type_2_col_idx_range


def get_structural_features(
    alignment_df: pd.DataFrame,
    query_sequence_ids: list[str],
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> np.ndarray:
    """Fill the structural features array based on foldseek alignment results.

    Args:
        structural_features (np.ndarray): The numpy array to fill with structural features.
        alignment_df (pd.DataFrame): DataFrame containing foldseek alignment results.
        query_sequence_ids (list[str]): List of query sequence IDs corresponding to the rows of the structural features array.
        domain_type_2_ref_module_id_2_col_idx (dict[str, dict[str, int]]): A dictionary mapping domain types to dictionaries mapping reference module ids to column indices in the structural features array.

    Returns:
        np.ndarray: The structural features array.
    """
    number_of_features = sum(
        len(module_id_2_col_idx)
        for module_id_2_col_idx in domain_type_2_ref_module_id_2_col_idx.values()
    )
    structural_features = np.ones((len(query_sequence_ids), number_of_features))

    for idx, query_seq_id in tqdm(
        enumerate(query_sequence_ids), total=len(query_sequence_ids)
    ):
        query_alignments = alignment_df[
            (alignment_df["query_seq_id"] == query_seq_id)
            & (alignment_df.query_domain_type == alignment_df.target_domain_type)
        ]
        for _, query_alignment in query_alignments.iterrows():
            domain_type = query_alignment["query_domain_type"]
            if domain_type == "alpha":
                domain_type = (
                    domain_type + f"_{int(query_alignment['query'].split('_')[-1]) + 1}"
                )
            target_module_id = query_alignment["target"]
            try:
                feature_col_idx = domain_type_2_ref_module_id_2_col_idx[domain_type][
                    target_module_id
                ]
            except KeyError:
                raise KeyError(
                    f"Domain {target_module_id} of type {domain_type} not found in reference domain {query_alignment['query']}."
                )
            structural_features[idx, feature_col_idx] = (
                1 - query_alignment["alntmscore"]
            )

    return structural_features


def save_file_to_all_residues(
    secondary_structure_residues_path: Path,
    pdb_files: list[Path],
    domain_templates: list[dict[str, Path]],
):
    logger.info(
        f"Secondary structure residues file not found at {secondary_structure_residues_path}, computing secondary structure residues."
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        sec_str_input_dir = Path(tmpdir)
        for pdb_file in pdb_files:
            if not pdb_file.exists():
                logger.warning(f"PDB file {pdb_file}, skipping this protein.")
                continue
            dst_path = sec_str_input_dir / f"{pdb_file.name}"
            os.symlink(pdb_file, dst_path)

        for domain_template in domain_templates:
            template_dst_path = sec_str_input_dir / f"{domain_template['path'].name}"
            os.symlink(Path(domain_template["path"]), template_dst_path)

        subprocess.check_output(
            f"python -m enzymeexplorer.src.structure_processing.compute_secondary_structure_residues --input-directory {str(sec_str_input_dir)} --output-path {secondary_structure_residues_path}".split(),
        )


def get_pdb_files(
    needed_proteins_csv_path: str, csv_id_column: str, input_directory: Path
) -> list[Path]:
    if needed_proteins_csv_path is not None:
        proteins_df = pd.read_csv(needed_proteins_csv_path)
        relevant_protein_ids = set(proteins_df[csv_id_column].unique())
    else:
        relevant_protein_ids = set(
            [filepath.stem for filepath in input_directory.glob("*.pdb")]
        )

    pdb_files = []
    logger.info(
        f"Filtering PDB files in {input_directory} to only include those specified in {needed_proteins_csv_path}"
    )
    for protein_id in relevant_protein_ids:
        pdb_path = input_directory / f"{protein_id}.pdb"
        if not pdb_path.exists():
            logger.warning(
                f"PDB file for {protein_id} not found at {pdb_path}, skipping this protein."
            )
            continue
        pdb_files.append(pdb_path.absolute())

    for filename in [pdb_file.stem for pdb_file in pdb_files]:
        filename_regex = "[a-zA-Z0-9_]+"
        if not re.fullmatch(filename_regex, filename):
            raise ValueError(
                f"Filename {filename} does not match the expected pattern {filename_regex}, which may cause issues with PyMOL selection syntax. Consider renaming this file."
            )
    return pdb_files


def filter_pdb_files_by_foldseek_alignments(
    pdb_files: list[Path],
    domain_templates: list[dict[str, Path | str]],
    batch_size: int = 1000,
) -> list[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        query_dir = tmpdir_path / "query"
        query_dir.mkdir()
        store_templates(domain_templates, query_dir)

        filtered_pdb_file_names = set()
        target_dir = tmpdir_path / "target"
        target_dir.mkdir()
        for idx in range(0, len(pdb_files), batch_size):
            batch_dir = target_dir / f"batch_{idx}"
            batch_dir.mkdir()
            batch_pdb_files = pdb_files[idx : min(idx + batch_size, len(pdb_files))]
            for pdb_file in batch_pdb_files:
                os.symlink(pdb_file, batch_dir / pdb_file.name)
            alignment_df = FoldseekWrapper().easy_search(
                query_dir=str(query_dir),
                target_dir=str(batch_dir),
                tmp_dir=str(tmpdir_path / "tmp_foldseek"),
                output=str(tmpdir_path / "foldseek_output.tsv"),
                max_seqs=batch_size * 2,
                e_value=10,
                sensitivity=10,
                cov_mode=2,
                coverage=0.6
            )
            filtered_pdb_file_names.update(set(alignment_df["target"].unique()))

        return [
            pdb_file
            for pdb_file in pdb_files
            if pdb_file.stem in filtered_pdb_file_names
        ]


def filter_domains_by_foldseek_alignments(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    domain_templates: list[dict[str, Path | str]],
    domain_pdbs_root: Path,
) -> dict[str, list[MappedRegion]]:
    filtered_domain_pdb_files = set()
    filtered_regions = set()
    filename_2_known_regions_completed_confident_filtered = defaultdict(list)
    for domain in supported_domains:
        domain_pdbs_dir = domain_pdbs_root / domain
        if domain_pdbs_dir.exists():
            domain_pdb_files = [
                path.absolute() for path in domain_pdbs_dir.glob(f"*.pdb")
            ]
            if domain_pdb_files:
                filtered_domains = set(
                    filter_pdb_files_by_foldseek_alignments(
                        domain_pdb_files,
                        [
                            domain_template
                            for domain_template in domain_templates
                            if domain_template["name"] == domain
                        ],
                        batch_size=3000,
                    )
                )
                filtered_domain_regions_ids = set(
                    [filtered_domain.stem for filtered_domain in filtered_domains]
                )
                filtered_domain_pdb_files.update([filtered_domain.name for filtered_domain in filtered_domains])
                filtered_regions.update(filtered_domain_regions_ids)
                for domain_pdb_file in domain_pdb_files:
                    if domain_pdb_file not in filtered_domains:
                        os.remove(domain_pdb_file)

    for domain_pdb_file in domain_pdbs_root.glob(f"*.pdb"):
        if domain_pdb_file.name not in filtered_domain_pdb_files:
            logger.info(f"Removing domain pdb file {domain_pdb_file} due to lack of foldseek alignment.")
            os.remove(domain_pdb_file)

    for filename in filename_2_known_regions_completed_confident:
        filtered_regions_for_filename = sorted(
            [
                region
                for region in filename_2_known_regions_completed_confident[filename]
                if region.module_id in filtered_regions
            ],
            key=lambda r: r.module_id,
        )
        if len(filtered_regions_for_filename) == 0:
            continue
        group_by_domain_type = defaultdict(list)
        for region in filtered_regions_for_filename:
            group_by_domain_type[region.domain].append(copy.deepcopy(region))
        regions_for_file = []
        for domain in group_by_domain_type:
            if (len(group_by_domain_type[domain]) - 1) != int(
                group_by_domain_type[domain][-1].module_id.split("_")[-1]
            ):
                for region in group_by_domain_type[domain]:
                    old_module_id = region.module_id
                    region.module_id = f"{filename}_{region.domain}_{group_by_domain_type[domain].index(region)}"
                    os.rename(
                        domain_pdbs_root / domain / f"{old_module_id}.pdb",
                        domain_pdbs_root / domain / f"{region.module_id}.pdb",
                    )
                    os.rename(
                        domain_pdbs_root / f"{old_module_id}.pkl",
                        domain_pdbs_root / f"{region.module_id}.pkl",
                    )
                    regions_for_file.append(region)
            else:
                for region in group_by_domain_type[domain]:
                    regions_for_file.append(region)

        filename_2_known_regions_completed_confident_filtered[filename] = sorted(
            regions_for_file, key=lambda r: r.module_id
        )
    return filename_2_known_regions_completed_confident_filtered


def store_domain_separately(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    detected_regions_root_path: Path,
):
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
    with open(
        detected_regions_root_path / "regions_completed_very_confident_all_ALL.pkl",
        "wb",
    ) as f:
        pickle.dump(domain_2_regions_completed_confident["all"], f)
    for domain_name in supported_domains:
        with open(
            detected_regions_root_path
            / f"regions_completed_very_confident_{domain_name}_ALL.pkl",
            "wb",
        ) as f:
            pickle.dump(domain_2_regions_completed_confident[domain_name], f)


def store_templates(
    domain_templates: list[dict[str, Path | str]],
    output_path: Path,
):
    for template in domain_templates:
        try:
            output_pdb_path = output_path / f"{template['name']}.pdb"
            cmd.delete(f"{template['name']}")
            cmd.delete(f'{template["name"]}_domain')
            cmd.load(template["path"], str(f'{template["name"]}_domain'))
            cmd.select(
                f'{template["name"]}',
                f"{template['name']}_domain & {template['residues']}",
            )
            cmd.save(f"{output_pdb_path}", f'{template["name"]}')
            cmd.delete(f"{template['name']}")
            cmd.delete(f'{template["name"]}_domain')
        except Exception as e:
            logger.error(
                f"Error storing domain {template['name']} from file {template['path']}: {e}"
            )


def store_domain(
    filename_region: tuple[str, MappedRegion],
    domains_output_path: Path,
):
    try:
        filename, region = filename_region
        PATH = Path(domains_output_path / f"{region.domain}")
        mapped_residues = list(set(region.residues_mapping.keys()))
        cmd.delete(filename)
        cmd.load(f"{filename}.pdb")
        logger.info(
            f"{region.module_id} {filename} & resi {compress_selection_list(mapped_residues)}",
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
        return True
    except Exception as e:
        logger.error(
            f"Error storing domain {region.module_id} from file {filename}: {e}"
        )
        return False


def store_domains(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    domains_output_path: Path,
    n_jobs: int = 1,
):

    if not domains_output_path.exists():
        domains_output_path.mkdir(parents=True)
    for domain_name in supported_domains:
        PATH = domains_output_path / f"{domain_name}"
        if not PATH.exists():
            PATH.mkdir(parents=True)
    store_domain_partial = partial(
        store_domain, domains_output_path=domains_output_path
    )
    filename_domain_tuples = [
        (filename, region)
        for (
            filename,
            regions,
        ) in filename_2_known_regions_completed_confident.items()
        for region in regions
    ]
    with Pool(n_jobs) as pool:
        tqdm(
            pool.map(
                store_domain_partial,
                filename_domain_tuples,
            ),
            desc="Storing detected domains",
            total=len(filename_domain_tuples),
        )


def plot_aligned_domains(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    save_dir: Path,
):
    """
    Helper function plotting TM-scores of detected domains on x-axis and
    the number of residues assigned to the domain object on y-axis
    """
    execution_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for domain_this in supported_domains:
        all_tmscores_and_mappings = [
            (region.tmscore, region.residues_mapping)
            for regions in filename_2_known_regions_completed_confident.values()
            for region in regions
            if region.domain == domain_this
        ]
        if len(all_tmscores_and_mappings) > 0:
            plt.figure(figsize=(17, 9))
            results_of_mapping = [
                (tmscore, len(mapping))
                for tmscore, mapping in all_tmscores_and_mappings
                if mapping is not None
            ]
            mapping_lenghts = list(map(lambda x: x[1], results_of_mapping))
            plt.scatter(list(map(lambda x: x[0], results_of_mapping)), mapping_lenghts)
            plt.xticks(np.arange(0, 1, 0.05), rotation=90)
            plt.yticks(
                np.arange(min(mapping_lenghts) - 10, max(mapping_lenghts) + 10, 5)
            )
            plt.xlabel("TM-score", fontsize=11)
            plt.ylabel("Number of residues assigned to the domain", fontsize=11)
            plt.title(f"{domain_this} domain detections", fontsize=14)
            plt.savefig(
                save_dir / f"{domain_this}_detections_{execution_timestamp}.png"
            )
            plt.show()


def detect_domains_roughly(
    specified_pdb_files: list[Path],
    file_2_all_residues_mapping: dict[str, set[str]],
    domain_templates: list[dict],
    args: configargparse.Namespace,
    iteration: int = 0,
) -> dict[str, list[MappedRegion]]:
    """
    Detects protein domains in multiple structures based on alignment scores and domain-specific thresholds.

    :param file_2_all_residues_mapping: A dictionary mapping file identifiers to sets of residue sequences present in those files
    :param domain_templates: A list of dictionaries containing domain template information
    :param output_root: The root directory where output images and serialized results will be saved
    :param args: arguments containing parameters number of iterations and flags for storing intermediate results

    :return: A dictionary mapping each filename to a list of known MappedRegion objects representing the detected
             reliable domains, while ensuring that no overlaying domains are included.
    """
    file_2_possible_regions: dict = defaultdict(list)
    for domain_template in domain_templates:
        domain_this = domain_template["name"]
        logger.info("Started detection of domain %s", domain_this)
        start_t = time.time()
        file_2_tmscore_residues_domain = get_alignments(
            specified_pdb_files,
            domain_template=domain_template,
            file_2_current_residues=file_2_all_residues_mapping,
            n_jobs=args.n_jobs,
        )
        logger.info(
            "Detection of %s domain. Execution took %d seconds",
            domain_this,
            time.time() - start_t,
        )

        num_of_new_detections = 0
        for sequence_id, current_detections in file_2_tmscore_residues_domain.items():
            logger.info(sequence_id)
            for i, (tm_score, res_mapping) in enumerate(current_detections):
                logger.info(f"tm_score: {tm_score:.2f}")
                logger.info(f"len of res_mapping: {len(res_mapping)}")
                if (
                    len(res_mapping) >= domain_template["thresholds"]["min_align_len"]
                    and tm_score >= domain_template["thresholds"]["tmscore"]
                ):
                    num_of_new_detections += 1
                    file_2_possible_regions[sequence_id].append(
                        MappedRegion(
                            module_id=f"{sequence_id}_{domain_this}_{i}",
                            domain=domain_this,
                            tmscore=tm_score,
                            residues_mapping=res_mapping,
                        ),
                    )

        logger.info(
            "Detected %d potential %s domains in iteration %d",
            num_of_new_detections,
            domain_this,
            iteration,
        )

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


def get_confident_residue_mappings(
    filename_2_known_regions_completed: dict[str, list[MappedRegion]],
    file_2_all_residues: dict[str, set[str]],
    domain_2_threshold: dict[str, dict[str, int]],
) -> dict[str, list[MappedRegion]]:
    filename_2_known_regions_completed_confident = {}
    for filename, regions in tqdm(
        filename_2_known_regions_completed.items(), desc="Filtering confident residues"
    ):
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
                >= domain_2_threshold[mapped_region_init.domain]["min_align_len"]
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
    return filename_2_known_regions_completed_confident
