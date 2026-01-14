#!/usr/bin/env python3
"""
Analyze train-test sequence similarity across folds using mmseqs.

This script iterates over cross-validation folds in the TPS dataset,
uses mmseqs to find the closest train sequence for each test sequence,
and plots a histogram of maximum similarities.
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

from terpeneminer.src.utils.msa import get_fasta_seqs


# Major substrates used in mAP evaluation (from terpene_miner_main.py)
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


def load_tps_data(csv_path: str) -> pd.DataFrame:
    """
    Load and filter TPS data from CSV.
    
    Args:
        csv_path: Path to the CSV file with TPS data
        
    Returns:
        DataFrame with unique TPS sequences per Uniprot ID
    """
    df = pd.read_csv(csv_path)
    
    fold_col = "stratified_phylogeny_based_split_with_minor_products"
    ignore_col = f"{fold_col}_ignore_in_eval"
    
    # Filter for TPS sequences (exclude Unknown type which are non-TPS)
    tps_df = df[df["Type (mono, sesq, di, …)"] != "Unknown"].copy()
    
    # Exclude entries marked as ignore_in_eval
    if ignore_col in tps_df.columns:
        tps_df = tps_df[tps_df[ignore_col] != 1].copy()
        print(f"Excluded {len(df) - len(tps_df)} entries marked as ignore_in_eval")
    
    # Get unique sequences per Uniprot ID
    unique_df = tps_df.drop_duplicates(subset=["Uniprot ID"])[
        ["Uniprot ID", "Amino acid sequence", fold_col]
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
    fold_col = "stratified_phylogeny_based_split_with_minor_products"
    folds = sorted(df[fold_col].dropna().unique().tolist())
    return folds


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
        query_fasta: Path to query FASTA file (test sequences)
        target_fasta: Path to target FASTA file (train sequences)
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


def analyze_fold_similarity(
    csv_path: str, 
    output_dir: str, 
    similarity_threshold: float = 95.0
) -> tuple[list[float], list[dict]]:
    """
    Main function to analyze train-test similarity across all folds.
    
    Args:
        csv_path: Path to TPS CSV file
        output_dir: Directory to save output files
        similarity_threshold: Threshold for collecting high-similarity pairs (default 95%)
        
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
    
    fold_col = "stratified_phylogeny_based_split_with_minor_products"
    all_best_similarities = []
    high_similarity_pairs = []
    
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
            output_m8 = os.path.join(tmp_base, f"results_{test_fold}.m8")
            mmseqs_tmp = os.path.join(tmp_base, f"mmseqs_tmp_{test_fold}")
            os.makedirs(mmseqs_tmp, exist_ok=True)
            
            # Write FASTA files
            write_fasta(
                train_df["Uniprot ID"].tolist(),
                train_df["Amino acid sequence"].tolist(),
                train_fasta
            )
            write_fasta(
                test_df["Uniprot ID"].tolist(),
                test_df["Amino acid sequence"].tolist(),
                test_fasta
            )
            
            # Run mmseqs
            print(f"  Running mmseqs easy-search...")
            try:
                run_mmseqs_search(test_fasta, train_fasta, output_m8, mmseqs_tmp)
            except subprocess.CalledProcessError as e:
                print(f"  Error running mmseqs: {e}")
                continue
            
            # Parse results
            best_hits = parse_m8_results(output_m8)
            print(f"  Found best hits for {len(best_hits)} test sequences")
            
            # Add to overall results and collect high similarity pairs
            for query_id, (target_id, pident) in best_hits.items():
                all_best_similarities.append(pident)
                if pident > similarity_threshold:
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


def enrich_pairs_with_cluster_and_substrate(
    pairs_df: pd.DataFrame,
    clusters_path: str,
    tps_csv_path: str
) -> pd.DataFrame:
    """
    Add cluster and substrate information to high-similarity pairs.
    
    Args:
        pairs_df: DataFrame with test_id, train_id, percent_identity, test_fold
        clusters_path: Path to phylogenetic_clusters.pkl
        tps_csv_path: Path to TPS CSV file for substrate info
        
    Returns:
        Enriched DataFrame with cluster and substrate columns
    """
    # Load phylogenetic clusters
    with open(clusters_path, "rb") as f:
        id_2_cluster, _ = pickle.load(f)
    
    # Load substrate mapping
    tps_df = pd.read_csv(tps_csv_path)
    id_to_substrate = (
        tps_df.groupby("Uniprot ID")["SMILES_substrate_canonical_no_stereo"]
        .first()
        .to_dict()
    )
    
    # Enrich each pair
    enriched_data = []
    for _, row in pairs_df.iterrows():
        test_id = row["test_id"]
        train_id = row["train_id"]
        
        test_cluster = id_2_cluster.get(test_id, "N/A")
        train_cluster = id_2_cluster.get(train_id, "N/A")
        same_cluster = test_cluster == train_cluster and test_cluster != "N/A"
        
        test_substrate = id_to_substrate.get(test_id, "Unknown")
        train_substrate = id_to_substrate.get(train_id, "Unknown")
        
        # Check if either substrate is a major one used in metrics
        is_major = (
            test_substrate in MAJOR_SUBSTRATES_FOR_METRICS or
            train_substrate in MAJOR_SUBSTRATES_FOR_METRICS
        )
        
        enriched_data.append({
            "test_id": test_id,
            "train_id": train_id,
            "percent_identity": row["percent_identity"],
            "test_fold": row["test_fold"],
            "test_cluster": test_cluster,
            "train_cluster": train_cluster,
            "same_cluster": same_cluster,
            "test_substrate": test_substrate,
            "train_substrate": train_substrate,
            "is_major_substrate_used_in_metrics": is_major,
        })
    
    return pd.DataFrame(enriched_data)


def plot_histogram(similarities: list[float], output_path: str, title_suffix: str = "") -> None:
    """
    Plot histogram of sequence similarities.
    
    Args:
        similarities: List of percent identity values
        output_path: Path to save the plot
        title_suffix: Additional text to add to the title (e.g., file source info)
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    
    plt.xlabel("Percent Identity to Closest Train Sequence", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    title = "Distribution of Test-Train Sequence Similarities"
    if title_suffix:
        title += f"\n{title_suffix}"
    plt.title(title, fontsize=14)
    
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
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "outputs" / "fold_similarity_analysis"
    clusters_path = project_root / "data" / "phylogenetic_clusters.pkl"
    
    # Define both CSV files to analyze
    csv_files = [
        {
            "path": project_root / "data" / "TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
            "label": "CORRUPTED",
            "title_suffix": "(CORRUPTED folds - with_neg_with_folds.csv)",
            "output_prefix": "train_test_similarity_histogram_CORRUPTED",
            "threshold": 95.0,
        },
        {
            "path": project_root / "data" / "TPS-Nov19_2023_verified_all_reactions_with_folds.csv",
            "label": "CORRECT",
            "title_suffix": "(CORRECT folds - with_folds.csv)",
            "output_prefix": "train_test_similarity_histogram_CORRECT",
            "threshold": 80.0,
        },
    ]
    
    for csv_info in csv_files:
        csv_path = csv_info["path"]
        label = csv_info["label"]
        title_suffix = csv_info["title_suffix"]
        output_prefix = csv_info["output_prefix"]
        threshold = csv_info["threshold"]
        
        print("\n" + "=" * 80)
        print(f"ANALYZING {label} FILE: {csv_path.name}")
        print(f"Using similarity threshold: {threshold}%")
        print("=" * 80)
        
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            continue
        
        # Run analysis
        similarities, high_similarity_pairs = analyze_fold_similarity(
            str(csv_path), str(output_dir), similarity_threshold=threshold
        )
        
        if similarities:
            print(f"\nTotal test sequences analyzed: {len(similarities)}")
            print(f"Mean similarity: {sum(similarities)/len(similarities):.2f}%")
            print(f"Min similarity: {min(similarities):.2f}%")
            print(f"Max similarity: {max(similarities):.2f}%")
            
            # Plot histogram
            plot_path = output_dir / f"{output_prefix}.png"
            plot_histogram(similarities, str(plot_path), title_suffix)
            
            # Save high similarity pairs
            if high_similarity_pairs:
                high_sim_df = pd.DataFrame(high_similarity_pairs)
                high_sim_df = high_sim_df.sort_values("percent_identity", ascending=False)
                
                # Enrich with cluster and substrate info
                print(f"\nEnriching pairs with cluster and substrate info...")
                enriched_df = enrich_pairs_with_cluster_and_substrate(
                    high_sim_df, str(clusters_path), str(csv_path)
                )
                
                # Save enriched results
                threshold_int = int(threshold)
                high_sim_path = output_dir / f"high_similarity_pairs_gt{threshold_int}_{label}.csv"
                enriched_df.to_csv(high_sim_path, index=False)
                print(f"\nFound {len(high_similarity_pairs)} test-train pairs with >{threshold}% identity")
                print(f"Saved to {high_sim_path}")
                
                # Print summary
                same_cluster_count = enriched_df["same_cluster"].sum()
                print(f"  Pairs in same cluster: {same_cluster_count}")
                print(f"  Pairs in different clusters: {len(enriched_df) - same_cluster_count}")
            else:
                print(f"\nNo test-train pairs with >{threshold}% identity found")
        else:
            print("No similarities found!")


if __name__ == "__main__":
    main()
