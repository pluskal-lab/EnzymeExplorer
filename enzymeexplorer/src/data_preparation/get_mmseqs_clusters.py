"""
This script performs grouping of sequences based on mmseqs sequence clustering.

Unlike phylogenetic clustering (which uses MSA + tree), this approach directly
clusters sequences based on sequence identity, which is more robust to N-terminal
extensions and signal peptides.
"""

import argparse
import logging
import os
import pickle
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from enzymeexplorer.src.utils.msa import get_fasta_seqs

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    :return: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate sequence clusters using mmseqs easy-cluster"
    )
    parser.add_argument(
        "--tps-cleaned-csv-path",
        type=str,
        default="data/TPS-Nov19_2023_verified_all_reactions.csv",
        help="Path to the TPS CSV file",
    )
    parser.add_argument(
        "--min-seq-id",
        type=float,
        default=0.3,
        help="Minimum sequence identity for clustering (0-1, default: 0.3)",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.5,
        help="Minimum coverage for clustering (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--coverage-mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Coverage mode: 0=bidirectional, 1=target, 2=query (default: 0)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/mmseqs_clusters.pkl",
        help="Path to save the cluster pickle file",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=8,
        help="Number of threads to use for mmseqs (default: 8)",
    )
    return parser.parse_args()


def run_mmseqs_cluster(
    fasta_path: str,
    output_prefix: str,
    tmp_dir: str,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    coverage_mode: int = 0,
    n_threads: int = 8,
) -> None:
    """
    Run mmseqs easy-cluster on a FASTA file.
    
    :param fasta_path: Path to input FASTA file
    :param output_prefix: Prefix for output files
    :param tmp_dir: Temporary directory for mmseqs
    :param min_seq_id: Minimum sequence identity threshold
    :param coverage: Minimum coverage threshold
    :param coverage_mode: Coverage mode (0=bidirectional, 1=target, 2=query)
    :param n_threads: Number of threads
    """
    cmd = [
        "mmseqs", "easy-cluster",
        fasta_path,
        output_prefix,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(coverage_mode),
        "--threads", str(n_threads),
    ]
    
    logger.info("Running mmseqs command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        logger.error("mmseqs stderr: %s", result.stderr)
        raise RuntimeError(f"mmseqs easy-cluster failed with return code {result.returncode}")
    
    logger.info("mmseqs clustering completed successfully")


def parse_mmseqs_clusters(cluster_tsv_path: str) -> tuple[dict[str, int], dict[int, list[str]]]:
    """
    Parse mmseqs cluster output TSV file.
    
    The TSV file has format: representative_id<tab>member_id
    
    :param cluster_tsv_path: Path to the cluster TSV file
    :return: Tuple of (id_to_cluster, cluster_to_ids) mappings
    """
    # First pass: identify unique representatives and assign cluster IDs
    representative_to_cluster = {}
    cluster_id = 0
    
    with open(cluster_tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                representative = parts[0]
                if representative not in representative_to_cluster:
                    representative_to_cluster[representative] = cluster_id
                    cluster_id += 1
    
    # Second pass: assign all members to their cluster
    id_to_cluster = {}
    cluster_to_ids = defaultdict(list)
    
    with open(cluster_tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                cluster_num = representative_to_cluster[representative]
                id_to_cluster[member] = cluster_num
                cluster_to_ids[cluster_num].append(member)
    
    # Convert defaultdict to regular dict
    cluster_to_ids = dict(cluster_to_ids)
    
    return id_to_cluster, cluster_to_ids


def main():
    """
    Main function to generate mmseqs-based sequence clusters.
    """
    cli_args = parse_args()
    
    logger.info("Loading data from %s...", cli_args.tps_cleaned_csv_path)
    tps_df = pd.read_csv(cli_args.tps_cleaned_csv_path)
    tps_df = tps_df.drop_duplicates("Uniprot ID")
    
    # Clean IDs (strip whitespace) and filter invalid sequences
    tps_df["Uniprot ID"] = tps_df["Uniprot ID"].str.strip()
    tps_df = tps_df[tps_df["Amino acid sequence"].notna()]
    tps_df = tps_df[tps_df["Amino acid sequence"].str.len() > 0]
    
    # Remove any non-standard amino acid characters that might cause issues
    tps_df["Amino acid sequence"] = tps_df["Amino acid sequence"].str.replace(r'[^A-Z]', '', regex=True)
    tps_df = tps_df[tps_df["Amino acid sequence"].str.len() > 0]
    
    logger.info("Loaded %d unique valid sequences", len(tps_df))
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write sequences to FASTA
        fasta_path = os.path.join(tmp_dir, "sequences.fasta")
        fasta_str = get_fasta_seqs(
            tps_df["Amino acid sequence"].tolist(),
            tps_df["Uniprot ID"].tolist()
        )
        with open(fasta_path, "w", encoding="utf-8") as f:
            f.write(fasta_str)
        logger.info("Wrote %d sequences to FASTA file", len(tps_df))
        
        # Run mmseqs clustering
        output_prefix = os.path.join(tmp_dir, "clusters")
        mmseqs_tmp = os.path.join(tmp_dir, "mmseqs_tmp")
        os.makedirs(mmseqs_tmp, exist_ok=True)
        
        logger.info(
            "Running mmseqs clustering with min-seq-id=%.2f, coverage=%.2f...",
            cli_args.min_seq_id,
            cli_args.coverage
        )
        run_mmseqs_cluster(
            fasta_path=fasta_path,
            output_prefix=output_prefix,
            tmp_dir=mmseqs_tmp,
            min_seq_id=cli_args.min_seq_id,
            coverage=cli_args.coverage,
            coverage_mode=cli_args.coverage_mode,
            n_threads=cli_args.n_threads,
        )
        
        # Parse cluster results
        cluster_tsv_path = f"{output_prefix}_cluster.tsv"
        logger.info("Parsing cluster results from %s...", cluster_tsv_path)
        id_to_cluster, cluster_to_ids = parse_mmseqs_clusters(cluster_tsv_path)
        
        logger.info("Found %d clusters for %d sequences", len(cluster_to_ids), len(id_to_cluster))
        
        # Print cluster size distribution
        sizes = [len(members) for members in cluster_to_ids.values()]
        logger.info("Cluster size stats: min=%d, max=%d, mean=%.1f, median=%.1f",
                    min(sizes), max(sizes), sum(sizes)/len(sizes),
                    sorted(sizes)[len(sizes)//2])
        singletons = sum(1 for s in sizes if s == 1)
        logger.info("Singleton clusters: %d (%.1f%%)", singletons, 100*singletons/len(sizes))
    
    # Save clusters
    output_path = Path(cli_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump((id_to_cluster, cluster_to_ids), f)
    
    logger.info("Saved clusters to %s", output_path)


if __name__ == "__main__":
    main()
