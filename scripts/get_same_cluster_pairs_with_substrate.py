#!/usr/bin/env python3
"""
Filter high-similarity test-train pairs to only those from the same phylogenetic cluster,
and add substrate class information.

This script:
1. Loads high_similarity_pairs_gt95.csv
2. Checks phylogenetic cluster membership
3. Filters to keep only pairs from the same cluster
4. Adds substrate class for both test and train sequences
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd


# Major substrates used in evaluation metrics (from terpene_miner_main.py)
MAJOR_SUBSTRATES_FOR_METRICS = {
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # FPP (sesquiterpene)
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GPP (monoterpene)
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GGPP (diterpene)
    "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",  # squalene epoxide (triterpene)
    "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # copalyl PP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GFPP (sesterterpene)
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # double FPP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # double GGPP
}


def load_phylogenetic_clusters(clusters_path: str) -> dict:
    """
    Load phylogenetic clusters from pickle file.
    
    Args:
        clusters_path: Path to phylogenetic_clusters.pkl
        
    Returns:
        Dictionary mapping ID to cluster
    """
    with open(clusters_path, "rb") as f:
        id_2_group, _ = pickle.load(f)
    return id_2_group


def load_substrate_mapping(csv_path: str) -> dict:
    """
    Load substrate class mapping from CSV.
    
    Args:
        csv_path: Path to TPS CSV file
        
    Returns:
        Dictionary mapping ID to substrate SMILES
    """
    df = pd.read_csv(csv_path)
    
    # Get unique substrate per ID (take first occurrence)
    id_to_substrate = (
        df.groupby("ID")["SMILES_substrate_canonical_no_stereo"]
        .first()
        .to_dict()
    )
    
    return id_to_substrate


def filter_same_cluster_pairs(
    pairs_csv_path: str,
    clusters_path: str,
    tps_csv_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Filter pairs to same-cluster only and add substrate info.
    
    Args:
        pairs_csv_path: Path to high_similarity_pairs_gt95.csv
        clusters_path: Path to phylogenetic_clusters.pkl
        tps_csv_path: Path to TPS CSV file
        output_path: Path to save output CSV
        
    Returns:
        DataFrame with filtered pairs and substrate info
    """
    # Load data
    print(f"Loading phylogenetic clusters from {clusters_path}...")
    id_2_cluster = load_phylogenetic_clusters(clusters_path)
    print(f"Loaded {len(id_2_cluster)} ID-to-cluster mappings")
    
    print(f"\nLoading substrate mappings from {tps_csv_path}...")
    id_to_substrate = load_substrate_mapping(tps_csv_path)
    print(f"Loaded substrate info for {len(id_to_substrate)} IDs")
    
    print(f"\nLoading high similarity pairs from {pairs_csv_path}...")
    pairs_df = pd.read_csv(pairs_csv_path)
    print(f"Loaded {len(pairs_df)} pairs")
    
    # Add cluster info and filter
    results = []
    for _, row in pairs_df.iterrows():
        test_id = row["test_id"]
        train_id = row["train_id"]
        
        test_cluster = id_2_cluster.get(test_id)
        train_cluster = id_2_cluster.get(train_id)
        
        # Only keep pairs from the same cluster
        if test_cluster is not None and test_cluster == train_cluster:
            test_substrate = id_to_substrate.get(test_id, "Unknown")
            train_substrate = id_to_substrate.get(train_id, "Unknown")
            
            # Check if either substrate is a major substrate used in metrics
            is_major = (
                test_substrate in MAJOR_SUBSTRATES_FOR_METRICS or
                train_substrate in MAJOR_SUBSTRATES_FOR_METRICS
            )
            
            results.append({
                "test_id": test_id,
                "train_id": train_id,
                "percent_identity": row["percent_identity"],
                "test_fold": row["test_fold"],
                "cluster": test_cluster,
                "test_substrate": test_substrate,
                "train_substrate": train_substrate,
                "is_major_substrate_used_in_metrics": is_major,
            })
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Pairs from SAME cluster: {len(results_df)} (out of {len(pairs_df)} total)")
    print("=" * 80)
    
    if len(results_df) > 0:
        print("\nSame-cluster pairs with substrates:")
        for _, row in results_df.iterrows():
            substrate_match = "SAME" if row["test_substrate"] == row["train_substrate"] else "DIFF"
            major_flag = "MAJOR" if row["is_major_substrate_used_in_metrics"] else "minor"
            print(
                f"  {row['test_id']} <-> {row['train_id']} | "
                f"{row['percent_identity']}% | cluster {row['cluster']} | "
                f"substrates: {substrate_match} | {major_flag}"
            )
        
        # Count substrate matches
        same_substrate = (results_df["test_substrate"] == results_df["train_substrate"]).sum()
        print(f"\nPairs with SAME substrate: {same_substrate}")
        print(f"Pairs with DIFFERENT substrate: {len(results_df) - same_substrate}")
        
        # Count major substrates
        major_count = results_df["is_major_substrate_used_in_metrics"].sum()
        print(f"\nPairs with MAJOR substrate (used in metrics): {major_count}")
        print(f"Pairs with minor substrate (not used in metrics): {len(results_df) - major_count}")
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")
    else:
        print("\nNo pairs found from the same cluster.")
    
    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter same-cluster pairs and add substrate class"
    )
    parser.add_argument(
        "--pairs-csv",
        type=str,
        default=None,
        help="Path to high similarity pairs CSV (default: auto-detect)"
    )
    parser.add_argument(
        "--clusters-pkl",
        type=str,
        default=None,
        help="Path to phylogenetic_clusters.pkl (default: data/phylogenetic_clusters.pkl)"
    )
    parser.add_argument(
        "--tps-csv",
        type=str,
        default=None,
        help="Path to TPS CSV file (default: data/EnzymeExplorer_Dataset.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV (default: same dir as input)"
    )
    args = parser.parse_args()
    
    # Set default paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if args.pairs_csv is None:
        pairs_csv = project_root / "outputs" / "fold_similarity_analysis" / "high_similarity_pairs_gt95.csv"
    else:
        pairs_csv = Path(args.pairs_csv)
    
    if args.clusters_pkl is None:
        clusters_pkl = project_root / "data" / "phylogenetic_clusters.pkl"
    else:
        clusters_pkl = Path(args.clusters_pkl)
    
    if args.tps_csv is None:
        tps_csv = project_root / "data" / "EnzymeExplorer_Dataset.csv"
    else:
        tps_csv = Path(args.tps_csv)
    
    if args.output is None:
        output_path = pairs_csv.parent / "same_cluster_pairs_with_substrate.csv"
    else:
        output_path = Path(args.output)
    
    # Run analysis
    filter_same_cluster_pairs(
        str(pairs_csv),
        str(clusters_pkl),
        str(tps_csv),
        str(output_path)
    )


if __name__ == "__main__":
    main()
