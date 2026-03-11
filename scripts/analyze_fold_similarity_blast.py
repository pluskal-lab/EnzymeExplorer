#!/usr/bin/env python3
"""
Analyze train-test sequence similarity across folds using BLAST.

This script iterates over cross-validation folds in the TPS dataset,
uses BLAST (blastp) to find the closest train sequence for each test sequence,
and plots a histogram of maximum similarities.

Installation on WSL Ubuntu:
    sudo apt update
    sudo apt install ncbi-blast+
"""

import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add the project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from terpeneminer.src.utils.msa import get_fasta_seqs


def load_tps_data(csv_path: str) -> pd.DataFrame:
    """
    Load and filter TPS data from CSV.
    
    Args:
        csv_path: Path to the CSV file with TPS data
        
    Returns:
        DataFrame with unique TPS sequences per ID
    """
    df = pd.read_csv(csv_path)
    
    fold_col = "Fold"
    ignore_col = f"{fold_col}_ignore_in_eval"
    
    # Filter for TPS sequences (exclude Unknown type which are non-TPS)
    tps_df = df[df["Type"] != "Unknown"].copy()
    
    # Exclude entries marked as ignore_in_eval
    if ignore_col in tps_df.columns:
        tps_df = tps_df[tps_df[ignore_col] != 1].copy()
        print(f"Excluded {len(df) - len(tps_df)} entries marked as ignore_in_eval")
    
    # Get unique sequences per ID
    unique_df = tps_df.drop_duplicates(subset=["ID"])[
        ["ID", "Aminoacid_sequence", fold_col]
    ].copy()
    
    return unique_df


def get_folds(df: pd.DataFrame) -> list[str]:
    """
    Extract unique fold names from the dataset.
    
    Args:
        df: DataFrame with fold column
        
    Returns:
        Sorted list of fold names
    """
    fold_col = "Fold"
    folds = sorted(df[fold_col].dropna().unique().tolist())
    return folds


def write_fasta(ids: list[str], seqs: list[str], output_path: str) -> None:
    """
    Write sequences to a FASTA file.
    
    Args:
        ids: List of sequence identifiers
        seqs: List of Aminoacid_sequences
        output_path: Path to output FASTA file
    """
    fasta_str = get_fasta_seqs(seqs, ids)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fasta_str)


def run_blast_search(
    query_fasta: str, 
    target_fasta: str, 
    output_file: str,
    num_threads: int = 4
) -> None:
    """
    Run BLAST to find similarities.
    
    Args:
        query_fasta: Path to query FASTA file (test sequences)
        target_fasta: Path to target FASTA file (train sequences)
        output_file: Path to output file
        num_threads: Number of threads to use
    """
    # Create BLAST database from train sequences
    db_path = target_fasta + "_db"
    makedb_cmd = [
        "makeblastdb",
        "-in", target_fasta,
        "-dbtype", "prot",
        "-out", db_path
    ]
    subprocess.run(makedb_cmd, check=True, capture_output=True)
    
    # Run blastp search
    # Output format 6: tabular with custom fields
    blast_cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db_path,
        "-out", output_file,
        "-outfmt", "6 qseqid sseqid pident",
        "-max_target_seqs", "100",  # Get top 100 hits to find best
        "-num_threads", str(num_threads)
    ]
    subprocess.run(blast_cmd, check=True, capture_output=True)


def parse_blast_results(output_file: str) -> dict[str, tuple[str, float]]:
    """
    Parse BLAST output file and get best hit per query.
    
    Args:
        output_file: Path to BLAST tabular output file
        
    Returns:
        Dictionary mapping query ID to (target_id, percent_identity)
    """
    best_hits = {}
    
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return best_hits
    
    with open(output_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id = parts[0]
                target_id = parts[1]
                pident = float(parts[2])
                
                if query_id not in best_hits or pident > best_hits[query_id][1]:
                    best_hits[query_id] = (target_id, pident)
    
    return best_hits


def analyze_fold_similarity(csv_path: str, output_dir: str, num_threads: int = 4) -> tuple[list[float], list[dict]]:
    """
    Main function to analyze train-test similarity across all folds.
    
    Args:
        csv_path: Path to TPS CSV file
        output_dir: Directory to save output files
        num_threads: Number of threads to use for BLAST
        
    Returns:
        Tuple of (list of all best-hit similarities, list of high similarity pairs)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading TPS data...")
    df = load_tps_data(csv_path)
    print(f"Loaded {len(df)} unique TPS sequences")
    
    # Get folds
    folds = get_folds(df)
    print(f"Found {len(folds)} folds: {folds}")
    
    fold_col = "Fold"
    all_best_similarities = []
    high_similarity_pairs = []  # Store pairs with >95% identity
    
    with tempfile.TemporaryDirectory() as tmp_base:
        for test_fold in tqdm(folds, desc="Processing folds"):
            print(f"\nProcessing fold: {test_fold}")
            
            # Split into train and test
            test_df = df[df[fold_col] == test_fold]
            train_df = df[df[fold_col] != test_fold]
            
            print(f"  Train sequences: {len(train_df)}")
            print(f"  Test sequences: {len(test_df)}")
            
            if len(test_df) == 0 or len(train_df) == 0:
                print(f"  Skipping fold {test_fold}: empty train or test set")
                continue
            
            # Create temporary files
            train_fasta = os.path.join(tmp_base, f"train_{test_fold}.fasta")
            test_fasta = os.path.join(tmp_base, f"test_{test_fold}.fasta")
            output_blast = os.path.join(tmp_base, f"results_{test_fold}.txt")
            
            # Write FASTA files
            write_fasta(
                train_df["ID"].tolist(),
                train_df["Aminoacid_sequence"].tolist(),
                train_fasta
            )
            write_fasta(
                test_df["ID"].tolist(),
                test_df["Aminoacid_sequence"].tolist(),
                test_fasta
            )
            
            # Run BLAST
            print(f"  Running blastp search...")
            try:
                run_blast_search(test_fasta, train_fasta, output_blast, num_threads)
            except subprocess.CalledProcessError as e:
                print(f"  Error running BLAST: {e}")
                print(f"  stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
                continue
            
            # Parse results
            best_hits = parse_blast_results(output_blast)
            print(f"  Found best hits for {len(best_hits)} test sequences")
            
            # Add to overall results and collect high similarity pairs
            for query_id, (target_id, pident) in best_hits.items():
                all_best_similarities.append(pident)
                if pident > 95.0:
                    high_similarity_pairs.append({
                        "test_id": query_id,
                        "train_id": target_id,
                        "percent_identity": pident,
                        "test_fold": test_fold
                    })
            
            # Also track sequences with no hits (0% similarity)
            no_hits = len(test_df) - len(best_hits)
            if no_hits > 0:
                print(f"  {no_hits} test sequences had no hits")
                all_best_similarities.extend([0.0] * no_hits)
    
    return all_best_similarities, high_similarity_pairs


def plot_histogram(similarities: list[float], output_path: str) -> None:
    """
    Plot histogram of sequence similarities.
    
    Args:
        similarities: List of percent identity values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    
    plt.xlabel("Percent Identity to Closest Train Sequence (BLAST)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Test-Train Sequence Similarities\n(TPS sequences across all folds, BLAST, excluding ignored_in_eval)", fontsize=14)
    
    # Add statistics
    if similarities:
        mean_sim = sum(similarities) / len(similarities)
        plt.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.1f}%')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved histogram to {output_path}")
    plt.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze train-test sequence similarity using BLAST"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Number of threads for BLAST (default: 4)"
    )
    args = parser.parse_args()
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    csv_path = project_root / "data" / "EnzymeExplorer_Dataset.csv"
    output_dir = project_root / "outputs" / "fold_similarity_analysis_blast"
    
    # Run analysis
    similarities, high_similarity_pairs = analyze_fold_similarity(str(csv_path), str(output_dir), args.threads)
    
    if similarities:
        print(f"\nTotal test sequences analyzed: {len(similarities)}")
        print(f"Mean similarity: {sum(similarities)/len(similarities):.2f}%")
        print(f"Min similarity: {min(similarities):.2f}%")
        print(f"Max similarity: {max(similarities):.2f}%")
        
        # Plot histogram
        plot_path = output_dir / "train_test_similarity_histogram_blast_no_ignored.png"
        plot_histogram(similarities, str(plot_path))
        
        # Save high similarity pairs (>95% identity)
        if high_similarity_pairs:
            high_sim_df = pd.DataFrame(high_similarity_pairs)
            high_sim_df = high_sim_df.sort_values("percent_identity", ascending=False)
            high_sim_path = output_dir / "high_similarity_pairs_gt95.csv"
            high_sim_df.to_csv(high_sim_path, index=False)
            print(f"\nFound {len(high_similarity_pairs)} test-train pairs with >95% identity")
            print(f"Saved to {high_sim_path}")
        else:
            print("\nNo test-train pairs with >95% identity found")
    else:
        print("No similarities found!")


if __name__ == "__main__":
    main()
