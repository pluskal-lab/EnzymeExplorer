#!/usr/bin/env python3
"""Build combined domain features for cross-dataset experiments (Track D).

Steps:
1. Extract old domain PDBs from AlphaFold structures + residue mappings
2. Run Foldseek: new domain PDBs vs old domain PDBs (cross-comparison)
3. Build a unified feature matrix covering both old and new proteins

The resulting matrix has rows = union of old + new proteins, columns = union
of old + new domain modules, with TM-scores from Foldseek comparisons.
"""

import os
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


_MOCK_MODULES = {
    "src.structure_processing.structural_algorithms",
    "src.utils.final__pymol_regions_detection_used",
    "enzymeexplorer.src.structure_processing.structural_algorithms",
}


class UnpicklerWithMocks(pickle.Unpickler):
    def find_class(self, module, name):
        if any(module.startswith(m) or module == m for m in _MOCK_MODULES):
            return type(name, (), {})
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return type(name, (), {})


def extract_old_domain_pdbs(
    detections_pkl: str,
    pdb_source_dir: str,
    output_dir: str,
) -> dict:
    """Extract domain PDBs from full AlphaFold structures using residue mappings."""
    os.makedirs(output_dir, exist_ok=True)

    with open(detections_pkl, "rb") as f:
        data = UnpicklerWithMocks(f).load()

    protein_to_modules = defaultdict(list)
    extracted = 0
    skipped_no_pdb = 0

    for uid, region in data:
        module_id = region.module_id
        residues = sorted(region.residues_mapping.keys())
        protein_to_modules[uid].append(module_id)

        out_pdb = os.path.join(output_dir, f"{module_id}.pdb")
        if os.path.exists(out_pdb):
            extracted += 1
            continue

        src_pdb = os.path.join(pdb_source_dir, f"{uid}.pdb")
        if not os.path.exists(src_pdb):
            skipped_no_pdb += 1
            continue

        residue_set = set(residues)
        lines = []
        with open(src_pdb) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        resnum = int(line[22:26].strip())
                    except ValueError:
                        continue
                    if resnum in residue_set:
                        lines.append(line)
                elif line.startswith("END"):
                    lines.append(line)

        if lines:
            with open(out_pdb, "w") as fh:
                fh.writelines(lines)
            extracted += 1
        else:
            skipped_no_pdb += 1

    print(f"Extracted {extracted} old domain PDBs, skipped {skipped_no_pdb} (missing source PDB)")
    return dict(protein_to_modules)


def run_foldseek_cross(
    query_dir: str,
    target_dir: str,
    output_dir: str,
    label: str,
    foldseek_bin: str = "foldseek",
) -> pd.DataFrame:
    """Run Foldseek easy-search: query domains vs target domains."""
    os.makedirs(output_dir, exist_ok=True)
    tsv_path = os.path.join(output_dir, f"foldseek_{label}.tsv")
    tmp_dir = os.path.join(output_dir, f"tmp_{label}")

    if os.path.exists(tsv_path) and os.path.getsize(tsv_path) > 0:
        print(f"Reusing existing {tsv_path}")
    else:
        os.makedirs(tmp_dir, exist_ok=True)
        cmd = [
            foldseek_bin, "easy-search",
            query_dir, target_dir, tsv_path, tmp_dir,
            "--format-output", "query,target,alntmscore",
            "-e", "inf",
            "--max-seqs", "100000",
            "--exhaustive-search", "1",
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["query", "target", "tmscore"])
    df["query"] = df["query"].str.replace(r"\.pdb$", "", regex=True)
    df["target"] = df["target"].str.replace(r"\.pdb$", "", regex=True)
    print(f"  {label}: {len(df)} hits")
    return df


def build_combined_matrix(
    old_detections: dict,
    new_detections_pkl: str,
    foldseek_dfs: list[pd.DataFrame],
) -> tuple:
    """Build unified feature matrix from multiple Foldseek comparison results.

    Returns the standard 4-tuple: (feats, all_ids, uniid_2_column_ids, module_2_idx)
    """
    with open(new_detections_pkl, "rb") as f:
        new_det = UnpicklerWithMocks(f).load()

    new_protein_to_modules = defaultdict(list)
    if isinstance(new_det, dict):
        for uid, regions in new_det.items():
            for region in (regions if isinstance(regions, list) else [regions]):
                mid = region.module_id if hasattr(region, "module_id") else f"{uid}_domain"
                new_protein_to_modules[uid].append(mid)
    elif isinstance(new_det, list):
        for uid, region in new_det:
            new_protein_to_modules[uid].append(region.module_id)

    # Combine all Foldseek results
    all_hits = pd.concat(foldseek_dfs, ignore_index=True)

    # Collect all unique query proteins and target modules
    query_to_protein = {}
    for _, row in all_hits.iterrows():
        qmod = row["query"]
        protein_id = "_".join(qmod.rsplit("_", 2)[:-2]) if qmod.count("_") >= 2 else qmod.rsplit("_", 1)[0]
        query_to_protein[qmod] = protein_id

    # All unique target modules (columns)
    all_target_modules = sorted(all_hits["target"].unique())
    module_to_col = {m: i for i, m in enumerate(all_target_modules)}
    n_cols = len(all_target_modules)

    # All unique proteins (from both old and new detections)
    all_protein_ids = sorted(
        set(old_detections.keys()) | set(new_protein_to_modules.keys())
    )
    protein_to_row = {pid: i for i, pid in enumerate(all_protein_ids)}
    n_proteins = len(all_protein_ids)

    print(f"Combined matrix: {n_proteins} proteins × {n_cols} target modules")

    feats = np.zeros((n_proteins, n_cols), dtype=np.float32)

    # Build protein_id → set of its own module_ids
    protein_modules = defaultdict(set)
    for pid, mods in old_detections.items():
        for m in mods:
            protein_modules[pid].add(m)
    for pid, mods in new_protein_to_modules.items():
        for m in mods:
            protein_modules[pid].add(m)

    # Fill matrix from Foldseek results
    for _, row in all_hits.iterrows():
        qmod = row["query"]
        tmod = row["target"]
        tmscore = row["tmscore"]

        # Find which protein this query module belongs to
        pid = query_to_protein.get(qmod)
        if pid is None or pid not in protein_to_row:
            continue
        if tmod not in module_to_col:
            continue

        r = protein_to_row[pid]
        c = module_to_col[tmod]
        feats[r, c] = max(feats[r, c], tmscore)

    # Build uniid_2_column_ids: for each protein, which target module columns
    # correspond to its own domains
    uniid_2_column_ids = defaultdict(list)
    for pid, mods in protein_modules.items():
        for m in mods:
            if m in module_to_col:
                uniid_2_column_ids[pid].append(module_to_col[m])

    domain_module_id_2_dist_matrix_index = defaultdict(list)
    for m, idx in module_to_col.items():
        domain_module_id_2_dist_matrix_index[m].append(idx)

    return (
        feats,
        all_protein_ids,
        uniid_2_column_ids,
        domain_module_id_2_dist_matrix_index,
    )


def main():
    old_detections_pkl = "data/tps_domains_and_comparisons/regions_completed_very_confident_all_ALL.pkl"
    pdb_source_dir = "foldseek_temp"
    old_domain_pdb_dir = "data/old_dataset_detected_domains"
    new_domain_pdb_dir = "data/new_dataset_detected_domains"
    new_detections_pkl = "data/new_dataset_domain_detections.pkl"
    output_dir = "data/combined_domain_foldseek"
    out_pkl = "data/clustering__domain_dist_based_features_combined.pkl"

    # Step 1: Extract old domain PDBs
    print("=== Step 1: Extracting old domain PDBs ===")
    old_det = extract_old_domain_pdbs(old_detections_pkl, pdb_source_dir, old_domain_pdb_dir)
    print(f"  {len(old_det)} proteins with domain detections")

    old_pdb_count = len([f for f in os.listdir(old_domain_pdb_dir) if f.endswith(".pdb")])
    print(f"  {old_pdb_count} old domain PDB files on disk")

    new_pdb_count = len([f for f in os.listdir(new_domain_pdb_dir) if f.endswith(".pdb")])
    print(f"  {new_pdb_count} new domain PDB files on disk")

    # Step 2: Run Foldseek cross-comparisons
    # We need 4 comparisons to get a complete matrix:
    #   old_query vs old_target (old proteins' TM-scores to old modules)
    #   new_query vs old_target (new proteins' TM-scores to old modules)
    #   old_query vs new_target (old proteins' TM-scores to new modules)
    #   new_query vs new_target (new proteins' TM-scores to new modules)
    print("\n=== Step 2: Running Foldseek comparisons ===")

    dfs = []

    df_oo = run_foldseek_cross(old_domain_pdb_dir, old_domain_pdb_dir, output_dir, "old_vs_old")
    dfs.append(df_oo)

    df_no = run_foldseek_cross(new_domain_pdb_dir, old_domain_pdb_dir, output_dir, "new_vs_old")
    dfs.append(df_no)

    df_on = run_foldseek_cross(old_domain_pdb_dir, new_domain_pdb_dir, output_dir, "old_vs_new")
    dfs.append(df_on)

    df_nn = run_foldseek_cross(new_domain_pdb_dir, new_domain_pdb_dir, output_dir, "new_vs_new")
    dfs.append(df_nn)

    # Step 3: Build combined matrix
    print("\n=== Step 3: Building combined feature matrix ===")
    result = build_combined_matrix(old_det, new_detections_pkl, dfs)
    feats, all_ids, uniid_2_col, mod_2_idx = result

    print(f"  Matrix shape: {feats.shape}")
    print(f"  Proteins: {len(all_ids)}")
    print(f"  Non-zero entries: {(feats > 0).sum()} / {feats.size}")

    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved combined domain features to {out_pkl}")


if __name__ == "__main__":
    main()
