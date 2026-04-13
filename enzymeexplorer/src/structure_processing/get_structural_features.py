from datetime import datetime
import configargparse
import logging
import os
import pickle
import numpy as np
from pathlib import Path

from enzymeexplorer.src.structure_processing.structural_algorithms import MappedRegion
from enzymeexplorer.src.structure_processing.utils import (
    get_foldseek_alignment_df,
    get_reference_domain_type_2_module_ids,
    get_reference_sequence_col_indices,
    get_col_idx_for_structural_features,
    get_domain_type_2_col_idx_range,
    get_reference_domains_col_indices,
    get_structural_features,
    preprocess_domains_by_renaming_domain_types,
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
    :return: current configargparse.Namespace
    """
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
        default="configs/structural_features_config.yaml",
    )
    parser.add_argument(
        "--reference-domains-file-path",
        "-refdoms",
        help="A path to a pickled dictionary of reference sequence ids to reference domains",
        type=str,
    )
    parser.add_argument(
        "--reference-domains-structures-directory",
        "-refdomsstructs",
        help="A path to a directory containing reference domain structures",
        type=str,
    )
    parser.add_argument(
        "--query-domains-file-path",
        "-querydoms",
        help="A path to a pickled dictionary of query sequence ids to query domains",
        type=str,
    )
    parser.add_argument(
        "--query-domains-structures-directory",
        "-querydomsstructs",
        help="A path to a directory containing query domain structures",
        type=str,
    )
    parser.add_argument(
        "--store-intermediate-results",
        "-storeintermediates",
        help="Whether to store the intermediate results (eg. foldseek alignments, structural features, ...)",
        action="store_true",
    )
    parser.add_argument(
        "--output-directory",
        "-outputdir",
        help="A path to a directory where the results will be stored. If not provided, results will be stored in the current directory.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--domain-type-preprocessing-config",
        "-domaintypepreprocconfig",
        default={
            "alpha": "alpha",
            "beta": "beta",
            "gamma": "gamma",
            "delta": "delta",
            "epsilon": "epsilon",
            "ids": "alpha",
            "alpha_cls2": "alpha",
        },
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.query_domains_file_path = os.path.abspath(args.query_domains_file_path)
    args.reference_domains_file_path = os.path.abspath(args.reference_domains_file_path)

    args.query_domains_structures_directory = os.path.abspath(
        args.query_domains_structures_directory
    )
    args.reference_domains_structures_directory = os.path.abspath(
        args.reference_domains_structures_directory
    )

    args.output_directory = (
        os.path.abspath(args.output_directory)
        if args.output_directory
        else os.path.abspath(
            "./data/structural_feature_files_"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )

    query_seq_2_regions: dict[str, list[MappedRegion]] = pickle.load(
        open(args.query_domains_file_path, "rb")
    )
    ref_seq_2_regions: dict[str, list[MappedRegion]] = pickle.load(
        open(args.reference_domains_file_path, "rb")
    )

    Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    query_renamed_modules_name_map, ref_renamed_modules_name_map = {}, {}
    if not (
        {"alpha", "beta", "gamma", "delta", "epsilon"}
        >= set(
            q_dom.domain for q_r in query_seq_2_regions.values() for q_dom in q_r
        ).union(
            set(r_dom.domain for r_r in ref_seq_2_regions.values() for r_dom in r_r)
        )
    ):
        query_seq_2_regions, query_renamed_modules_name_map = (
            preprocess_domains_by_renaming_domain_types(
                query_seq_2_regions, args.domain_type_preprocessing_config
            )
        )
        ref_seq_2_regions, ref_renamed_modules_name_map = (
            preprocess_domains_by_renaming_domain_types(
                ref_seq_2_regions, args.domain_type_preprocessing_config
            )
        )
        pickle.dump(
            query_seq_2_regions,
            open(args.output_directory + "/preprocessed_query_seq_2_regions.pkl", "wb"),
        )
        pickle.dump(
            ref_seq_2_regions,
            open(
                args.output_directory + "/preprocessed_reference_seq_2_regions.pkl",
                "wb",
            ),
        )
        logger.info(
            "Domain type preprocessing completed. Preprocessed query and reference sequence to regions mappings saved."
        )

    assert set(
        q_dom.domain for q_r in query_seq_2_regions.values() for q_dom in q_r
    ) <= set(
        r_dom.domain for r_r in ref_seq_2_regions.values() for r_dom in r_r
    ), "After preprocessing, there are still domain types in query regions that are not present in reference regions. Please check the domain type preprocessing configuration and the input data."

    feature_domain_types = ["alpha_1", "alpha_2"] + list(
        set(
            [
                region.domain
                for q, rs in ref_seq_2_regions.items()
                for region in rs
                if not region.domain.startswith("alpha")
            ]
        )
    )

    alignment_df = get_foldseek_alignment_df(
        query_seq_2_regions=query_seq_2_regions,
        query_domains_dir=args.query_domains_structures_directory,
        query_preprocessed_name_map=query_renamed_modules_name_map,
        ref_seq_2_regions=ref_seq_2_regions,
        reference_domains_dir=args.reference_domains_structures_directory,
        ref_preprocessed_name_map=ref_renamed_modules_name_map,
    )
    logger.info(
        f"Foldseek alignment completed. Number of alignments: {len(alignment_df)}"
    )

    query_seq_ids = sorted(list(query_seq_2_regions.keys()))

    ref_domain_type_2_module_id = get_reference_domain_type_2_module_ids(
        ref_seq_2_regions
    )
    assert set(ref_domain_type_2_module_id.keys()) == set(
        feature_domain_types
    ), "Unexpected domain types in reference regions data."

    domain_type_2_ref_module_id_2_col_idx = get_col_idx_for_structural_features(
        ref_domain_type_2_module_id, feature_domain_types
    )

    logger.info(
        f"Number of structural features: {sum(len(module_id_2_col_idx) for module_id_2_col_idx in domain_type_2_ref_module_id_2_col_idx.values())}"
    )

    structural_features = get_structural_features(
        alignment_df,
        query_seq_ids,
        domain_type_2_ref_module_id_2_col_idx,
    )

    logger.info(f"Structural features array shape: {structural_features.shape}")

    ref_seq_2_col_idxs = get_reference_sequence_col_indices(
        ref_seq_2_regions, domain_type_2_ref_module_id_2_col_idx
    )

    logger.info(
        f"Number of reference sequences with at least one aligned domain: {len(ref_seq_2_col_idxs)}"
    )

    ref_domain_id_2_col_idxs = get_reference_domains_col_indices(
        domain_type_2_ref_module_id_2_col_idx
    )

    logger.info(
        f"Number of reference domains with at least one aligned query domain: {len(ref_domain_id_2_col_idxs)}"
    )

    domain_type_2_start_end_cols = get_domain_type_2_col_idx_range(
        domain_type_2_ref_module_id_2_col_idx
    )

    logger.info(f"Domain type to column index range: {domain_type_2_start_end_cols}")

    if args.store_intermediate_results:
        alignment_df.to_csv(
            f"{args.output_directory}/foldseek_alignment_results.csv", index=False
        )
        np.save(f"{args.output_directory}/structural_features.npy", structural_features)

    with open(args.output_directory + "/domain_dist_based_features.pkl", "wb") as f:
        pickle.dump(
            (
                structural_features,
                query_seq_ids,
                ref_seq_2_col_idxs,
                ref_domain_id_2_col_idxs,
            ),
            f,
        )

    with open(args.output_directory + "/domain_2_start_end_cols.pkl", "wb") as f:
        pickle.dump(domain_type_2_start_end_cols, f)

    regions_ids_2_tmscore = {}
    for _, row in alignment_df.iterrows():
        regions_ids_2_tmscore[tuple(sorted([row["query"], row["target"]]))] = float(
            row["alntmscore"]
        )
    with open(
        args.output_directory + "/precomputed_tmscores_foldseek.pkl", "wb"
    ) as file:
        pickle.dump(regions_ids_2_tmscore, file)

    logger.info(
        f"Structural features and related mappings saved to {args.output_directory}"
    )
