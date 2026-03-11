#!/usr/bin/env python3
"""
Filter problematic negative samples that have high similarity to TPS sequences.

This script removes negative samples that are too similar to known TPS sequences,
which could cause data leakage or indicate annotation errors.

Usage:
    python scripts/filter_problematic_negatives.py
    python scripts/filter_problematic_negatives.py --threshold 50
    python scripts/filter_problematic_negatives.py --dry-run
"""

import argparse
import pickle
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter problematic negatives with high similarity to TPS"
    )
    parser.add_argument(
        "--high-similarity-csv",
        type=str,
        default="outputs/negative_similarity_analysis/high_similarity_negatives_gt40.csv",
        help="Path to CSV with high-similarity negatives (from analyze_negative_similarity.py)",
    )
    parser.add_argument(
        "--negatives-pkl",
        type=str,
        default="data/sampled_id_2_seq.pkl",
        help="Path to the negative samples pickle file",
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default=None,
        help="Path for filtered output (default: overwrite input with backup)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="Minimum percent identity threshold for exclusion (default: 40)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without making changes",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup of the original file",
    )
    return parser.parse_args()


def main():
    """Main function to filter problematic negatives."""
    args = parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    high_sim_path = project_root / args.high_similarity_csv
    negatives_path = project_root / args.negatives_pkl
    
    # Check files exist
    if not high_sim_path.exists():
        print(f"Error: High-similarity CSV not found: {high_sim_path}")
        print("Run analyze_negative_similarity.py first to generate this file.")
        return 1
    
    if not negatives_path.exists():
        print(f"Error: Negatives pickle not found: {negatives_path}")
        return 1
    
    # Load high-similarity negatives
    print(f"Loading high-similarity negatives from {high_sim_path}...")
    exclude_df = pd.read_csv(high_sim_path)
    
    # Filter by threshold
    exclude_df = exclude_df[exclude_df["percent_identity"] >= args.threshold]
    exclude_ids = set(exclude_df["negative_id"].tolist())
    
    print(f"Found {len(exclude_ids)} negatives with >= {args.threshold}% identity to TPS")
    
    if len(exclude_ids) == 0:
        print("No negatives to exclude. Exiting.")
        return 0
    
    # Show what will be excluded
    print("\nNegatives to exclude:")
    print("-" * 60)
    for _, row in exclude_df.iterrows():
        print(f"  {row['negative_id']:15} -> {row['tps_id']:15} ({row['percent_identity']:.1f}%)")
    print("-" * 60)
    
    # Load original negatives
    print(f"\nLoading negatives from {negatives_path}...")
    with open(negatives_path, "rb") as f:
        original_id_2_seq = pickle.load(f)
    
    original_count = len(original_id_2_seq)
    print(f"Original negative count: {original_count}")
    
    # Filter
    filtered_id_2_seq = {
        k: v for k, v in original_id_2_seq.items() 
        if k not in exclude_ids
    }
    filtered_count = len(filtered_id_2_seq)
    removed_count = original_count - filtered_count
    
    print(f"After filtering: {filtered_count}")
    print(f"Removed: {removed_count}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return 0
    
    # Determine output path
    if args.output_pkl:
        output_path = project_root / args.output_pkl
    else:
        output_path = negatives_path
        
        # Create backup
        if not args.no_backup:
            backup_path = negatives_path.with_suffix(".pkl.backup")
            if not backup_path.exists():
                shutil.copy(negatives_path, backup_path)
                print(f"\nBacked up original to {backup_path}")
            else:
                print(f"\nBackup already exists at {backup_path}")
    
    # Save filtered negatives
    with open(output_path, "wb") as f:
        pickle.dump(filtered_id_2_seq, f)
    
    print(f"Saved filtered negatives to {output_path}")
    print(f"\nSummary: {original_count} -> {filtered_count} negatives ({removed_count} removed)")
    
    return 0


if __name__ == "__main__":
    exit(main())
