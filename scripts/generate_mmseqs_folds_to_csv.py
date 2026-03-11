#!/usr/bin/env python3
"""
Generate stratified group k-folds using mmseqs clusters and save directly to CSV.

This script creates cross-validation folds where sequences in the same mmseqs cluster
are kept together (no leakage between train/test), and saves the fold assignments
directly to CSV files.
"""

import argparse
import logging
import pickle
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate stratified group k-folds using mmseqs clusters"
    )
    parser.add_argument(
        "--tps-csv-path",
        type=str,
        default="data/TPS-Nov19_2023_verified_all_reactions.csv",
        help="Path to the TPS CSV file",
    )
    parser.add_argument(
        "--negatives-pkl-path",
        type=str,
        default="data/sampled_id_2_seq.pkl",
        help="Path to negative samples pickle file",
    )
    parser.add_argument(
        "--clusters-pkl-path",
        type=str,
        default="data/mmseqs_clusters_30pct_50cov.pkl",
        help="Path to mmseqs clusters pickle file (default uses 30%% identity, 50%% coverage)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_mmseqs",
        help="Suffix to add to output CSV filenames",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


# Major substrate classes used for stratification
MAJOR_CLASSES = {
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # FPP
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GPP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GGPP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",  # squalene epoxide
    "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # copalyl PP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GFPP
    "CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",  # DMAPP+IPP
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",  # GPP+IPP
}


def load_tps_data(csv_path: str) -> pd.DataFrame:
    """Load TPS data from CSV."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def load_negative_samples(pkl_path: str) -> dict[str, str]:
    """Load negative samples from pickle."""
    with open(pkl_path, "rb") as f:
        id_2_seq = pickle.load(f)
    logger.info("Loaded %d negative samples from %s", len(id_2_seq), pkl_path)
    return id_2_seq


def load_clusters(pkl_path: str) -> dict[str, int]:
    """Load mmseqs clusters from pickle."""
    with open(pkl_path, "rb") as f:
        id_2_cluster, _ = pickle.load(f)
    logger.info("Loaded clusters for %d sequences from %s", len(id_2_cluster), pkl_path)
    return id_2_cluster


def add_negatives_to_df(tps_df: pd.DataFrame, neg_id_2_seq: dict[str, str]) -> pd.DataFrame:
    """Add negative samples to the TPS dataframe."""
    # Get existing IDs
    existing_ids = set(tps_df["Uniprot ID"].str.strip())
    
    # Create negative rows
    neg_rows = []
    for neg_id, neg_seq in neg_id_2_seq.items():
        if neg_id not in existing_ids:
            neg_rows.append({
                "Uniprot ID": neg_id,
                "Amino acid sequence": neg_seq,
                "SMILES_substrate_canonical_no_stereo": "Negative",
                "Type (mono, sesq, di, …)": "Unknown",
            })
    
    if neg_rows:
        neg_df = pd.DataFrame(neg_rows)
        combined_df = pd.concat([tps_df, neg_df], ignore_index=True)
        logger.info("Added %d negative samples to dataframe", len(neg_rows))
        return combined_df
    
    return tps_df


def generate_folds(
    tps_df: pd.DataFrame,
    id_2_cluster: dict[str, int],
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[set, set]]:
    """
    Generate stratified group k-folds.
    
    Returns list of (train_ids, val_ids) tuples.
    """
    target_col = "SMILES_substrate_canonical_no_stereo"
    
    # Get unique sequences per ID
    unique_df = tps_df.drop_duplicates("Uniprot ID").copy()
    unique_df["Uniprot ID"] = unique_df["Uniprot ID"].str.strip()
    
    # Separate positives (in clusters) and negatives
    unique_df["in_cluster"] = unique_df["Uniprot ID"].map(lambda x: x in id_2_cluster)
    
    pos_df = unique_df[unique_df["in_cluster"]].copy()
    neg_df = unique_df[~unique_df["in_cluster"]].copy()
    
    logger.info("Positive samples (in clusters): %d", len(pos_df))
    logger.info("Negative samples (not in clusters): %d", len(neg_df))
    
    # Assign cluster groups to positives
    # ALL sequences in the same cluster should stay together to prevent leakage
    def get_group(row):
        uid = row["Uniprot ID"]
        if uid in id_2_cluster:
            return str(id_2_cluster[uid])
        else:
            # Sequence not in any cluster - assign unique ID
            return str(uuid.uuid4())
    
    pos_df["group"] = pos_df.apply(get_group, axis=1)
    
    # Create stratification target (substrate or combined set)
    pos_df["strat_target"] = pos_df[target_col]
    
    # Run stratified group k-fold on positives
    kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    pos_folds = list(kfold.split(
        pos_df,
        pos_df["strat_target"],
        pos_df["group"]
    ))
    
    # Run regular stratified k-fold on negatives (no grouping needed)
    # Handle case when there are too few negatives
    if len(neg_df) >= n_folds:
        neg_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        neg_df["dummy_strat"] = "Negative"
        neg_folds = list(neg_kfold.split(neg_df, neg_df["dummy_strat"]))
    else:
        # Too few negatives - just assign them all to training
        logger.warning("Too few negatives (%d) for %d-fold split, assigning all to training", 
                       len(neg_df), n_folds)
        neg_folds = [(list(range(len(neg_df))), []) for _ in range(n_folds)]
    
    # Combine folds
    all_folds = []
    for fold_idx in range(n_folds):
        pos_train_idx, pos_val_idx = pos_folds[fold_idx]
        neg_train_idx, neg_val_idx = neg_folds[fold_idx]
        
        train_ids = set(pos_df.iloc[pos_train_idx]["Uniprot ID"]) | set(neg_df.iloc[neg_train_idx]["Uniprot ID"])
        val_ids = set(pos_df.iloc[pos_val_idx]["Uniprot ID"]) | set(neg_df.iloc[neg_val_idx]["Uniprot ID"])
        
        all_folds.append((train_ids, val_ids))
        
        logger.info("Fold %d: train=%d (pos=%d, neg=%d), val=%d (pos=%d, neg=%d)",
                    fold_idx,
                    len(train_ids), len(pos_df.iloc[pos_train_idx]), len(neg_df.iloc[neg_train_idx]),
                    len(val_ids), len(pos_df.iloc[pos_val_idx]), len(neg_df.iloc[neg_val_idx]))
    
    return all_folds


def save_folds_to_csv(
    tps_df: pd.DataFrame,
    folds: list[tuple[set, set]],
    output_path: str,
    split_name: str = "stratified_mmseqs_based_split",
):
    """Save fold assignments to CSV."""
    # Add fold column
    tps_df[split_name] = "-1"
    tps_df[f"{split_name}_ignore_in_eval"] = ""
    
    for fold_idx, (_, val_ids) in enumerate(folds):
        tps_df.loc[
            tps_df["Uniprot ID"].isin(val_ids),
            split_name
        ] = f"fold_{fold_idx}"
    
    # Mark substrates that are too concentrated in single clusters
    # (simplified version - mark non-major substrates as ignore_in_eval)
    target_col = "SMILES_substrate_canonical_no_stereo"
    tps_df.loc[
        ~tps_df[target_col].isin(MAJOR_CLASSES) & (tps_df[target_col] != "Negative"),
        f"{split_name}_ignore_in_eval"
    ] = "1"
    
    tps_df.to_csv(output_path, index=False)
    logger.info("Saved folds to %s", output_path)


def main():
    """Main function."""
    args = parse_args()
    
    # Load data
    tps_df = load_tps_data(args.tps_csv_path)
    neg_id_2_seq = load_negative_samples(args.negatives_pkl_path)
    id_2_cluster = load_clusters(args.clusters_pkl_path)
    
    # Generate folds for both with and without negatives
    for include_negatives in [True, False]:
        logger.info("\n" + "=" * 60)
        logger.info("Generating folds %s negatives", "WITH" if include_negatives else "WITHOUT")
        logger.info("=" * 60)
        
        # Prepare dataframe
        if include_negatives:
            df = add_negatives_to_df(tps_df.copy(), neg_id_2_seq)
            suffix = f"_with_neg_with_folds{args.output_suffix}"
        else:
            df = tps_df.copy()
            suffix = f"_with_folds{args.output_suffix}"
        
        # Generate folds
        folds = generate_folds(
            df, 
            id_2_cluster, 
            n_folds=args.n_folds,
            random_state=args.random_state
        )
        
        # Save to CSV
        output_path = args.tps_csv_path.replace(".csv", f"{suffix}.csv")
        split_name = f"stratified_mmseqs_based_split_with_minor_products"
        save_folds_to_csv(df, folds, output_path, split_name)
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
