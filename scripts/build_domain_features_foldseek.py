"""Build domain distance feature matrix using Foldseek for structural comparison.

This script:
1. Runs foldseek easy-search to compare detected domain PDBs against reference
   domain PDBs (self-comparison or cross-set comparison).
2. Parses the Foldseek alignment output.
3. Assembles a feature matrix in the format expected by PlmDomainsRandomForest:
   (feats_dom_dists, all_ids_list_dom, uniid_2_column_ids,
    domain_module_id_2_dist_matrix_index)
4. Saves the result as a pickle file.

Usage:
    python scripts/build_domain_features_foldseek.py \
        --query-domains-dir data/new_dataset_detected_domains \
        --target-domains-dir data/new_dataset_detected_domains \
        --detections-pkl data/new_dataset_domain_detections.pkl \
        --output-path data/clustering__domain_dist_based_features_new_dataset.pkl
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build domain distance feature matrix using Foldseek"
    )
    parser.add_argument(
        "--query-domains-dir",
        type=str,
        required=True,
        help="Directory containing query domain PDB files",
    )
    parser.add_argument(
        "--target-domains-dir",
        type=str,
        required=True,
        help="Directory containing target/reference domain PDB files",
    )
    parser.add_argument(
        "--detections-pkl",
        type=str,
        required=True,
        help="Pickle with domain detections (protein_id -> list[MappedRegion])",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output pickle path for the feature matrix",
    )
    parser.add_argument(
        "--foldseek-bin",
        type=str,
        default="foldseek",
        help="Path to foldseek binary",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate Foldseek output files",
    )
    return parser.parse_args()


def run_foldseek(
    query_dir: str,
    target_dir: str,
    output_dir: Path,
    foldseek_bin: str = "foldseek",
) -> pd.DataFrame:
    """Run foldseek easy-search and return parsed results."""
    run_id = str(uuid4())[:8]
    tsv_path = output_dir / f"foldseek_domain_comparison_{run_id}.tsv"
    tmp_path = output_dir / f"foldseek_tmp_{run_id}"

    cmd = [
        foldseek_bin,
        "easy-search",
        query_dir,
        target_dir,
        str(tsv_path),
        str(tmp_path),
        "--max-seqs",
        "5000",
        "-e",
        "1",
        "-s",
        "10",
        "--exhaustive-search",
        "--format-output",
        "query,target,fident,alnlen,mismatch,gapopen,"
        "qstart,qend,tstart,tend,evalue,bits,alntmscore",
    ]

    logger.info("Running Foldseek: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    logger.info("Foldseek finished. Parsing results from %s", tsv_path)

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=[
            "query",
            "target",
            "fident",
            "alnlen",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bits",
            "alntmscore",
        ],
    )

    if tmp_path.exists():
        rmtree(tmp_path)

    return df, tsv_path


def build_feature_matrix(
    foldseek_df: pd.DataFrame,
    detections: dict,
) -> tuple:
    """Build the 4-tuple feature matrix from Foldseek comparison results.

    Returns (feats_dom_dists, all_ids_list_dom, uniid_2_column_ids,
             domain_module_id_2_dist_matrix_index)
    """
    all_target_modules = sorted(foldseek_df["target"].unique())
    module_to_col = {mod: i for i, mod in enumerate(all_target_modules)}
    n_cols = len(all_target_modules)
    logger.info("Reference domain modules (columns): %d", n_cols)

    protein_ids = sorted(detections.keys())
    n_proteins = len(protein_ids)
    logger.info("Proteins with domain detections: %d", n_proteins)

    feats = np.zeros((n_proteins, n_cols), dtype=np.float32)

    query_to_protein = {}
    for pid, regions in detections.items():
        for region in regions:
            query_to_protein[region.module_id] = pid

    protein_to_row = {pid: i for i, pid in enumerate(protein_ids)}

    uniid_2_column_ids: dict[str, list[int]] = defaultdict(list)
    domain_module_id_2_dist_matrix_index: dict[str, list[int]] = defaultdict(list)

    for mod_id in all_target_modules:
        col_idx = module_to_col[mod_id]
        parts = mod_id.rsplit("_", 2)
        if len(parts) >= 3:
            protein_id = "_".join(parts[:-2])
        else:
            protein_id = parts[0]

        if protein_id in protein_to_row:
            uniid_2_column_ids[protein_id].append(col_idx)
        domain_module_id_2_dist_matrix_index[mod_id].append(col_idx)

    for _, row in foldseek_df.iterrows():
        query_mod = row["query"]
        target_mod = row["target"]
        tmscore = float(row["alntmscore"])

        if query_mod not in query_to_protein:
            continue
        pid = query_to_protein[query_mod]
        if pid not in protein_to_row:
            continue
        row_idx = protein_to_row[pid]
        if target_mod not in module_to_col:
            continue
        col_idx = module_to_col[target_mod]

        feats[row_idx, col_idx] = max(feats[row_idx, col_idx], tmscore)

    all_ids_list_dom = protein_ids

    logger.info(
        "Feature matrix shape: %s, non-zero: %d (%.1f%%)",
        feats.shape,
        np.count_nonzero(feats),
        100 * np.count_nonzero(feats) / feats.size,
    )

    return (
        feats,
        all_ids_list_dom,
        uniid_2_column_ids,
        domain_module_id_2_dist_matrix_index,
    )


def load_detections(pkl_path: str) -> dict:
    """Load domain detections pickle, handling the MappedRegion import.

    Uses a lightweight mock MappedRegion to avoid importing PyMOL-dependent
    modules that are not needed for feature assembly.
    """
    from dataclasses import dataclass
    from types import ModuleType

    @dataclass(eq=True)
    class MappedRegion:
        module_id: str
        domain: str
        tmscore: float
        residues_mapping: dict

    mock_module = ModuleType(
        "enzymeexplorer.src.structure_processing.structural_algorithms"
    )
    mock_module.MappedRegion = MappedRegion  # type: ignore[attr-defined]
    sys.modules["enzymeexplorer.src.structure_processing.structural_algorithms"] = (
        mock_module
    )

    with open(pkl_path, "rb") as f:
        detections = pickle.load(f)

    logger.info(
        "Loaded detections: %d proteins, %d total domains",
        len(detections),
        sum(len(v) for v in detections.values()),
    )
    return detections


def main():
    args = parse_args()

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = load_detections(args.detections_pkl)

    foldseek_df, tsv_path = run_foldseek(
        args.query_domains_dir,
        args.target_domains_dir,
        output_dir,
        foldseek_bin=args.foldseek_bin,
    )
    logger.info("Foldseek returned %d alignment rows", len(foldseek_df))

    feature_tuple = build_feature_matrix(foldseek_df, detections)

    with open(args.output_path, "wb") as f:
        pickle.dump(feature_tuple, f)
    logger.info("Saved feature matrix to %s", args.output_path)

    if not args.keep_intermediate and tsv_path.exists():
        os.remove(tsv_path)
        logger.info("Removed intermediate TSV")


if __name__ == "__main__":
    main()
