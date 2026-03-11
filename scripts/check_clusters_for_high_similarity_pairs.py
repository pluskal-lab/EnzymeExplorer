#!/usr/bin/env python3
"""
Check phylogenetic clusters for high similarity test-train pairs.

This script reads the high similarity pairs CSV and looks up their
corresponding phylogenetic clusters to identify if highly similar
sequences ended up in different folds despite being in the same cluster.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd


def load_phylogenetic_clusters(clusters_path: str) -> dict:
    """
    Load phylogenetic clusters from pickle file.
    
    Args:
        clusters_path: Path to phylogenetic_clusters.pkl
        
    Returns:
        Dictionary mapping Uniprot ID to cluster
    """
    with open(clusters_path, "rb") as f:
        id_2_group, _ = pickle.load(f)
    return id_2_group


def check_clusters_for_pairs(
    pairs_csv_path: str,
    clusters_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Check cluster membership for high similarity pairs.
    
    Args:
        pairs_csv_path: Path to high_similarity_pairs_gt95.csv
        clusters_path: Path to phylogenetic_clusters.pkl
        output_path: Path to save output CSV
        
    Returns:
        DataFrame with cluster information added
    """
    # Load data
    print(f"Loading phylogenetic clusters from {clusters_path}...")
    id_2_group = load_phylogenetic_clusters(clusters_path)
    print(f"Loaded {len(id_2_group)} ID-to-cluster mappings")
    
    print(f"\nLoading high similarity pairs from {pairs_csv_path}...")
    pairs_df = pd.read_csv(pairs_csv_path)
    print(f"Loaded {len(pairs_df)} pairs")
    
    # Look up clusters for each pair
    results = []
    for _, row in pairs_df.iterrows():
        test_id = row["test_id"]
        train_id = row["train_id"]
        
        test_cluster = id_2_group.get(test_id, "NOT_FOUND")
        train_cluster = id_2_group.get(train_id, "NOT_FOUND")
        same_cluster = test_cluster == train_cluster
        
        results.append({
            "test_id": test_id,
            "train_id": train_id,
            "percent_identity": row["percent_identity"],
            "test_fold": row["test_fold"],
            "test_cluster": test_cluster,
            "train_cluster": train_cluster,
            "same_cluster": same_cluster
        })
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("High similarity pairs with their clusters:")
    print("=" * 80)
    for _, row in results_df.iterrows():
        status = "SAME" if row["same_cluster"] else "DIFF"
        print(
            f"{row['test_id']} (cluster {row['test_cluster']}) <-> "
            f"{row['train_id']} (cluster {row['train_cluster']}) | "
            f"{row['percent_identity']}% | {status}"
        )
    
    print("\n" + "=" * 80)
    same_count = results_df["same_cluster"].sum()
    diff_count = len(results_df) - same_count
    print(f"Pairs in SAME cluster: {same_count}")
    print(f"Pairs in DIFFERENT clusters: {diff_count}")
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    
    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check phylogenetic clusters for high similarity pairs"
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
        "--output",
        type=str,
        default=None,
        help="Path to output CSV (default: same dir as input with '_with_clusters' suffix)"
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
    
    if args.output is None:
        output_path = pairs_csv.parent / "high_similarity_pairs_with_clusters.csv"
    else:
        output_path = Path(args.output)
    
    # Run analysis
    check_clusters_for_pairs(
        str(pairs_csv),
        str(clusters_pkl),
        str(output_path)
    )


if __name__ == "__main__":
    main()
