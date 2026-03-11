# TODO: Refactor the code to use pandas DataFrames instead of dictionaries for mappings. This will make the code cleaner and more efficient.

import logging
import subprocess
from enzymeexplorer.src.structure_processing.structural_algorithms import MappedRegion
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import os
import pickle
from collections import defaultdict
from enzymeexplorer.src.structure_processing.foldseek_wrapper import FoldseekWrapper
from tqdm.auto import tqdm

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
        region.module_id
        for regions in seq_2_regions.values()
        for region in regions
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

    query_domain_2_seq_id, query_domain_2_domain_type = __get_domain_2_seq_id_and_domain_type_maps(
        query_seq_2_regions
    )
    ref_domain_2_seq_id, ref_domain_2_domain_type = __get_domain_2_seq_id_and_domain_type_maps( 
        ref_seq_2_regions
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
        for domain_name in ref_domain_type_2_module_ids[
            feature_domain_type
        ]:
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
        domain_type_2_col_idx_range[domain_type] = (min(col_idxs), max(col_idxs)+1)
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
                raise KeyError(f"Domain {target_module_id} of type {domain_type} not found in reference domain {query_alignment['query']}.")
            structural_features[idx, feature_col_idx] = (
                1 - query_alignment["alntmscore"]
            )

    return structural_features
