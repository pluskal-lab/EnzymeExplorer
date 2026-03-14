"""This script performs grouping of sequences based on clades of phylogenetic tree"""

import argparse
import logging
import os
import pickle
from collections import deque

import pandas as pd  # type: ignore
from Bio import Phylo  # type: ignore

from enzymeexplorer.src.utils.msa import generate_msa_mafft

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    # pylint: disable=R0801
    parser.add_argument(
        "--tps-cleaned-csv-path",
        type=str,
        default="data/TPS-Nov19_2023_verified_all_reactions.csv",
    )
    parser.add_argument("--n-workers", type=int, default=64)
    parser.add_argument(
        "--max-evolutionary-depth-for-cluster-root", type=float, default=0.4
    )
    args = parser.parse_args()
    return args


def main():
    """
    This function performs grouping of sequences based on clades of phylogenetic tree
    @return:
    """
    cli_args = parse_args()
    logger.info("Loading data...")
    tps_df = pd.read_csv(cli_args.tps_cleaned_csv_path)
    tps_df = tps_df.drop_duplicates("Uniprot ID")
    logger.info("Data were loaded! There are %s loaded records in total.", len(tps_df))

    logger.info("Generating MSA...")
    msa_path_working = "data/_mafft_msa_tps_all.fasta"
    generate_msa_mafft(
        seqs=tps_df["Amino acid sequence"],
        ids=tps_df["Uniprot ID"],
        output_name=msa_path_working,
        num_workers=cli_args.n_workers,
    )
    logger.info("MSA is ready!")

    logger.info("Computing the phylogenetic tree...")
    os.system(
        f"iqtree -s {msa_path_working} -st AA -m TEST -bb 1000 -alrt 1000 -T {cli_args.n_workers}"
    )
    logger.info("The phylogenetic tree is ready!")

    logger.info(
        "Performing several traversals of the phylogenetic tree to get clade-based groups..."
    )
    tree = Phylo.read(f"{msa_path_working}.treefile", "newick")
    clade_2_parent = {}
    queue_for_depth_computation = deque()
    queue_for_depth_computation.append(tree.root)
    while len(queue_for_depth_computation):
        parent = queue_for_depth_computation.popleft()
        for child_for_depth in parent.clades:
            clade_2_parent[child_for_depth] = parent
            queue_for_depth_computation.append(child_for_depth)

    clade_2_max_depth = {}
    tree_terminals = set(tree.get_terminals())

    # recursive computation of the maximum evolutionary depth
    def _get_max_evolutionary_depth(clade):
        if clade not in clade_2_max_depth:
            if clade in tree_terminals:
                clade_2_max_depth[clade] = clade.branch_length
            else:
                own_length = clade.branch_length
                if own_length is None:
                    own_length = 0
                max_depth = own_length + max(
                    [
                        _get_max_evolutionary_depth(child_clade)
                        for child_clade in clade.clades
                    ]
                )
                clade_2_max_depth[clade] = max_depth
        return clade_2_max_depth[clade]

    _get_max_evolutionary_depth(tree.root)

    def _split_to_clusters(
        phylogenetic_tree,
        max_evo_depth_for_cluster_root=cli_args.max_evolutionary_depth_for_cluster_root,
    ):
        num_clusters = 0
        _uniid_2_cluster = {}
        _all_nodes_2_cluster = {}
        nodes_closed_set = set()
        all_terminals = deque(phylogenetic_tree.get_terminals())
        next_cluster_seed = all_terminals.pop()
        while len(nodes_closed_set) < phylogenetic_tree.count_terminals():
            while (
                clade_2_max_depth[clade_2_parent[next_cluster_seed]]
                < max_evo_depth_for_cluster_root
            ):
                next_cluster_seed = clade_2_parent[next_cluster_seed]
            for cluster_terminal in next_cluster_seed.get_terminals():
                _uniid_2_cluster[cluster_terminal.name] = num_clusters
                nodes_closed_set.add(cluster_terminal)

            _all_nodes_2_cluster[next_cluster_seed] = num_clusters
            queue = deque()
            queue.append(next_cluster_seed)
            while len(queue):
                next_node = queue.popleft()
                for child in next_node.clades:
                    _all_nodes_2_cluster[child] = num_clusters
                    queue.append(child)
            num_clusters += 1

            if len(all_terminals) == 0:
                assert len(nodes_closed_set) == phylogenetic_tree.count_terminals()
                return _uniid_2_cluster, _all_nodes_2_cluster
            next_cluster_seed = all_terminals.pop()
            while next_cluster_seed in nodes_closed_set and len(all_terminals):
                next_cluster_seed = all_terminals.pop()
        return _uniid_2_cluster, _all_nodes_2_cluster

    uniid_2_cluster, all_nodes_2_cluster = _split_to_clusters(tree)
    logger.info("Clade-based groups are ready!")

    with open("data/phylogenetic_clusters.pkl", "wb") as f:
        pickle.dump((uniid_2_cluster, all_nodes_2_cluster), f)
    logger.info(
        "Clade-based sequence grouped were stored to data/phylogenetic_clusters.pkl."
    )


if __name__ == "__main__":
    main()
