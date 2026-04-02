#!/usr/bin/env python3
"""Build the combined domain feature matrix from pre-computed Foldseek TSVs.

Reads the 4 Foldseek cross-comparison TSVs (old×old, new×old, old×new, new×new)
and assembles a unified (proteins × modules) TM-score matrix.
"""

import os
import pickle
from collections import defaultdict

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


def load_detections(pkl_path):
    """Load domain detections, returning protein_id → [module_id] mapping."""
    with open(pkl_path, "rb") as f:
        data = UnpicklerWithMocks(f).load()

    protein_to_modules = defaultdict(list)
    if isinstance(data, list):
        for uid, region in data:
            protein_to_modules[uid].append(region.module_id)
    elif isinstance(data, dict):
        for uid, regions in data.items():
            for r in (regions if isinstance(regions, list) else [regions]):
                mid = r.module_id if hasattr(r, "module_id") else f"{uid}_domain"
                protein_to_modules[uid].append(mid)
    return dict(protein_to_modules)


def module_id_to_protein(module_id: str) -> str:
    """Extract protein ID from module_id like 'F2XF97_alpha_0' → 'F2XF97'."""
    parts = module_id.rsplit("_", 2)
    if len(parts) >= 3:
        return parts[0]
    return module_id.rsplit("_", 1)[0]


def main():
    tsv_dir = "data/combined_domain_foldseek"
    out_pkl = "data/clustering__domain_dist_based_features_combined.pkl"

    old_det_pkl = "data/tps_domains_and_comparisons/regions_completed_very_confident_all_ALL.pkl"
    new_det_pkl = "data/new_dataset_domain_detections.pkl"

    print("Loading detections...")
    old_det = load_detections(old_det_pkl)
    new_det = load_detections(new_det_pkl)
    print(f"  Old: {len(old_det)} proteins, {sum(len(v) for v in old_det.values())} modules")
    print(f"  New: {len(new_det)} proteins, {sum(len(v) for v in new_det.values())} modules")

    # Build module_id → protein_id mapping
    module_to_protein = {}
    for pid, mods in old_det.items():
        for m in mods:
            module_to_protein[m] = pid
    for pid, mods in new_det.items():
        for m in mods:
            module_to_protein[m] = pid

    # All unique proteins and target modules
    all_proteins = sorted(set(old_det.keys()) | set(new_det.keys()))
    protein_to_row = {p: i for i, p in enumerate(all_proteins)}
    n_proteins = len(all_proteins)
    print(f"Total unique proteins: {n_proteins}")

    # Read all TSVs to find all target modules
    tsv_files = [
        os.path.join(tsv_dir, f"foldseek_{label}.tsv")
        for label in ["old_vs_old", "new_vs_old", "old_vs_new", "new_vs_new"]
    ]

    print("Reading TSVs to collect target modules...")
    all_target_modules = set()
    for tsv_path in tsv_files:
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["query", "target", "tmscore"],
                         usecols=[1])
        df["target"] = df["target"].str.replace(r"\.pdb$", "", regex=True)
        all_target_modules.update(df["target"].unique())
        del df

    all_target_modules = sorted(all_target_modules)
    module_to_col = {m: i for i, m in enumerate(all_target_modules)}
    n_cols = len(all_target_modules)
    print(f"Total target modules (columns): {n_cols}")

    # Build feature matrix
    feats = np.zeros((n_proteins, n_cols), dtype=np.float32)

    print("Filling feature matrix from TSVs...")
    for tsv_path in tsv_files:
        label = os.path.basename(tsv_path)
        print(f"  Processing {label}...")
        for chunk in pd.read_csv(
            tsv_path, sep="\t", header=None,
            names=["query", "target", "tmscore"],
            chunksize=2_000_000,
        ):
            chunk["query"] = chunk["query"].str.replace(r"\.pdb$", "", regex=True)
            chunk["target"] = chunk["target"].str.replace(r"\.pdb$", "", regex=True)

            # Map query module → protein
            chunk["protein"] = chunk["query"].map(module_to_protein)
            chunk = chunk.dropna(subset=["protein"])

            # Map protein → row
            chunk["row"] = chunk["protein"].map(protein_to_row)
            chunk = chunk.dropna(subset=["row"])
            chunk["row"] = chunk["row"].astype(int)

            # Map target → col
            chunk["col"] = chunk["target"].map(module_to_col)
            chunk = chunk.dropna(subset=["col"])
            chunk["col"] = chunk["col"].astype(int)

            # Take max TM-score per (row, col)
            rows = chunk["row"].values
            cols = chunk["col"].values
            scores = chunk["tmscore"].values.astype(np.float32)

            np.maximum.at(feats, (rows, cols), scores)

    # Build uniid_2_column_ids
    uniid_2_column_ids = defaultdict(list)
    for pid in all_proteins:
        modules = old_det.get(pid, []) + new_det.get(pid, [])
        for m in modules:
            if m in module_to_col:
                uniid_2_column_ids[pid].append(module_to_col[m])

    domain_module_id_2_dist_matrix_index = defaultdict(list)
    for m, idx in module_to_col.items():
        domain_module_id_2_dist_matrix_index[m].append(idx)

    result = (
        feats,
        all_proteins,
        uniid_2_column_ids,
        domain_module_id_2_dist_matrix_index,
    )

    print(f"\nMatrix shape: {feats.shape}")
    print(f"Non-zero entries: {(feats > 0).sum()} / {feats.size}")
    print(f"Sparsity: {1 - (feats > 0).sum() / feats.size:.4f}")

    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved {out_pkl}")


if __name__ == "__main__":
    main()
