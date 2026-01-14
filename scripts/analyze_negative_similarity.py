#!/usr/bin/env python3
"""
Analyze similarity between negative samples and TPS sequences using mmseqs.

This script checks if any negative samples (from sampled_id_2_seq.pkl) have
high similarity to known TPS sequences, which would indicate potential
contamination or data leakage.
"""

import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add the project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from enzymeexplorer.src.utils.msa import get_fasta_seqs


def load_tps_data(csv_path: str) -> pd.DataFrame:
    """
    Load TPS data from CSV.
    
    Args:
        csv_path: Path to the CSV file with TPS data
        
    Returns:
        DataFrame with unique TPS sequences per Uniprot ID
    """
    df = pd.read_csv(csv_path)
    
    # Filter for TPS sequences (exclude Unknown type which are non-TPS/negatives)
    tps_df = df[df["Type (mono, sesq, di, …)"] != "Unknown"].copy()
    
    # Get unique sequences per Uniprot ID
    unique_df = tps_df.drop_duplicates(subset=["Uniprot ID"])[
        ["Uniprot ID", "Amino acid sequence"]
    ].copy()
    
    # Clean IDs
    unique_df["Uniprot ID"] = unique_df["Uniprot ID"].str.strip()
    
    return unique_df


def load_negative_samples(pkl_path: str) -> dict[str, str]:
    """
    Load negative samples from pickle file.
    
    Args:
        pkl_path: Path to sampled_id_2_seq.pkl
        
    Returns:
        Dictionary mapping negative sample ID to sequence
    """
    with open(pkl_path, "rb") as f:
        id_2_seq = pickle.load(f)
    return id_2_seq


def write_fasta(ids: list[str], seqs: list[str], output_path: str) -> None:
    """
    Write sequences to a FASTA file.
    
    Args:
        ids: List of sequence identifiers
        seqs: List of amino acid sequences
        output_path: Path to output FASTA file
    """
    fasta_str = get_fasta_seqs(seqs, ids)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fasta_str)


def run_mmseqs_search(
    query_fasta: str, 
    target_fasta: str, 
    output_file: str, 
    tmp_dir: str
) -> None:
    """
    Run mmseqs easy-search to find similarities.
    
    Args:
        query_fasta: Path to query FASTA file (negative samples)
        target_fasta: Path to target FASTA file (TPS sequences)
        output_file: Path to output .m8 file
        tmp_dir: Path to temporary directory for mmseqs
    """
    cmd = [
        "mmseqs", "easy-search",
        query_fasta,
        target_fasta,
        output_file,
        tmp_dir,
        "--format-output", "query,target,pident"
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def parse_m8_results(m8_file: str) -> dict[str, tuple[str, float]]:
    """
    Parse mmseqs output file and get best hit per query.
    
    Args:
        m8_file: Path to .m8 output file
        
    Returns:
        Dictionary mapping query ID to (target_id, percent_identity)
    """
    best_hits = {}
    
    if not os.path.exists(m8_file) or os.path.getsize(m8_file) == 0:
        return best_hits
    
    with open(m8_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id = parts[0]
                target_id = parts[1]
                pident = float(parts[2])
                
                if query_id not in best_hits or pident > best_hits[query_id][1]:
                    best_hits[query_id] = (target_id, pident)
    
    return best_hits


def analyze_negative_similarity(
    tps_csv_path: str, 
    negatives_pkl_path: str, 
    output_dir: str
) -> tuple[list[float], list[dict]]:
    """
    Analyze similarity between negative samples and TPS sequences.
    
    Args:
        tps_csv_path: Path to TPS CSV file
        negatives_pkl_path: Path to negative samples pickle file
        output_dir: Directory to save output files
        
    Returns:
        Tuple of (list of all best-hit similarities, list of high similarity pairs)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading TPS data...")
    tps_df = load_tps_data(tps_csv_path)
    print(f"Loaded {len(tps_df)} unique TPS sequences")
    
    print("Loading negative samples...")
    neg_id_2_seq = load_negative_samples(negatives_pkl_path)
    print(f"Loaded {len(neg_id_2_seq)} negative samples")
    
    all_best_similarities = []
    high_similarity_pairs = []
    
    with tempfile.TemporaryDirectory() as tmp_base:
        # Write FASTA files
        tps_fasta = os.path.join(tmp_base, "tps.fasta")
        neg_fasta = os.path.join(tmp_base, "negatives.fasta")
        output_m8 = os.path.join(tmp_base, "results.m8")
        mmseqs_tmp = os.path.join(tmp_base, "mmseqs_tmp")
        os.makedirs(mmseqs_tmp, exist_ok=True)
        
        write_fasta(
            tps_df["Uniprot ID"].tolist(),
            tps_df["Amino acid sequence"].tolist(),
            tps_fasta
        )
        
        neg_ids = list(neg_id_2_seq.keys())
        neg_seqs = [neg_id_2_seq[nid] for nid in neg_ids]
        write_fasta(neg_ids, neg_seqs, neg_fasta)
        
        # Run mmseqs: query=negatives, target=TPS
        print("Running mmseqs easy-search (negatives vs TPS)...")
        try:
            run_mmseqs_search(neg_fasta, tps_fasta, output_m8, mmseqs_tmp)
        except subprocess.CalledProcessError as e:
            print(f"Error running mmseqs: {e}")
            return [], []
        
        # Parse results
        best_hits = parse_m8_results(output_m8)
        print(f"Found hits for {len(best_hits)} out of {len(neg_ids)} negative samples")
        
        # Collect results
        for neg_id in neg_ids:
            if neg_id in best_hits:
                tps_id, pident = best_hits[neg_id]
                all_best_similarities.append(pident)
                if pident > 40.0:  # Flag anything above 40% as potentially concerning
                    high_similarity_pairs.append({
                        "negative_id": neg_id,
                        "tps_id": tps_id,
                        "percent_identity": pident,
                    })
            else:
                all_best_similarities.append(0.0)
    
    return all_best_similarities, high_similarity_pairs


def plot_histogram(similarities: list[float], output_path: str, title: str) -> None:
    """
    Plot histogram of sequence similarities.
    
    Args:
        similarities: List of percent identity values
        output_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    
    plt.xlabel("Percent Identity to Closest TPS Sequence", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add statistics
    if similarities:
        mean_sim = sum(similarities) / len(similarities)
        plt.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.1f}%')
        
        # Mark threshold
        plt.axvline(40, color='orange', linestyle=':', label='40% threshold')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved histogram to {output_path}")
    plt.close()


def main():
    """Main entry point."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    tps_csv_path = project_root / "data" / "TPS-Nov19_2023_verified_all_reactions_with_folds.csv"
    negatives_pkl_path = project_root / "data" / "sampled_id_2_seq.pkl"
    output_dir = project_root / "outputs" / "negative_similarity_analysis"
    
    # Check if files exist
    if not tps_csv_path.exists():
        print(f"TPS CSV not found: {tps_csv_path}")
        return
    if not negatives_pkl_path.exists():
        print(f"Negatives pickle not found: {negatives_pkl_path}")
        return
    
    # Run analysis
    similarities, high_similarity_pairs = analyze_negative_similarity(
        str(tps_csv_path), str(negatives_pkl_path), str(output_dir)
    )
    
    if similarities:
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print('='*60)
        print(f"Total negative samples analyzed: {len(similarities)}")
        print(f"Mean similarity to closest TPS: {sum(similarities)/len(similarities):.2f}%")
        print(f"Min similarity: {min(similarities):.2f}%")
        print(f"Max similarity: {max(similarities):.2f}%")
        
        # Count by threshold
        above_40 = sum(1 for s in similarities if s > 40)
        above_50 = sum(1 for s in similarities if s > 50)
        above_60 = sum(1 for s in similarities if s > 60)
        above_80 = sum(1 for s in similarities if s > 80)
        
        print(f"\nNegatives with >40% identity to TPS: {above_40} ({100*above_40/len(similarities):.1f}%)")
        print(f"Negatives with >50% identity to TPS: {above_50} ({100*above_50/len(similarities):.1f}%)")
        print(f"Negatives with >60% identity to TPS: {above_60} ({100*above_60/len(similarities):.1f}%)")
        print(f"Negatives with >80% identity to TPS: {above_80} ({100*above_80/len(similarities):.1f}%)")
        
        # Plot histogram
        plot_path = output_dir / "negative_tps_similarity_histogram.png"
        plot_histogram(
            similarities, 
            str(plot_path),
            "Distribution of Negative Sample Similarity to TPS Sequences"
        )
        
        # Save high similarity pairs
        if high_similarity_pairs:
            high_sim_df = pd.DataFrame(high_similarity_pairs)
            high_sim_df = high_sim_df.sort_values("percent_identity", ascending=False)
            high_sim_path = output_dir / "high_similarity_negatives_gt40.csv"
            high_sim_df.to_csv(high_sim_path, index=False)
            print(f"\nFound {len(high_similarity_pairs)} negatives with >40% identity to TPS")
            print(f"Saved to {high_sim_path}")
            
            # Print top concerning cases
            print("\nTop 10 most similar negatives to TPS:")
            print(high_sim_df.head(10).to_string(index=False))
        else:
            print("\nNo negatives with >40% identity to TPS found - data looks clean!")
    else:
        print("No similarities found!")


if __name__ == "__main__":
    main()
