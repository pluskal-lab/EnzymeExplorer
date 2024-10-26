"""This script detects TPS domains in protein structures"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import pickle
import logging
import subprocess
from uuid import uuid4
from shutil import rmtree

from pymol import cmd  # type: ignore
import pandas as pd  # type: ignore
from Bio import PDB  # type: ignore
from tqdm.auto import tqdm  # type: ignore

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to compare detected TPS domains to the known ones"
    )
    parser.add_argument(
        "--known-domain-structures-root",
        help="A directory containing structures of known domains",
        type=str,
        default="data/detected_domains/all",
    )
    parser.add_argument(
        "--detected-domain-structures-root",
        help="A path to new detected domain structures",
        type=str,
        default="_temp/detected_domains",
    )
    parser.add_argument("--path-to-known-domains-subset", type=str, default="data/domains_subset.pkl")
    parser.add_argument("--output-path", type=str, default="_temp/filename_2_regions_vs_known_reg_dists.pkl")
    parser.add_argument("--pdb-id", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    working_dir = Path('_temp')
    if not working_dir.exists():
        working_dir.mkdir()
    tsv_path = working_dir / f'aln_all_domains_vs_all_{uuid4()}.tsv'
    tmp_path = working_dir / f'tmp_all_{uuid4()}'
    foldseek_comparison_output = subprocess.check_output(
        f'foldseek easy-search {args.detected_domain_structures_root} {args.known_domain_structures_root} {tsv_path} {tmp_path} --max-seqs 3000 -e 0.1 --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore'.split())
    df_foldseek = pd.read_csv(tsv_path, sep='\t', header=None,
                              names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend',
                                     'tstart', 'tend', 'evalue', 'bits', 'alntmscore'])

    region_2_known_reg_dists = defaultdict(list)
    with open(args.path_to_known_domains_subset, "rb") as file:
        dom_subset, feat_indices_subset = pickle.load(file)

    for _, row in df_foldseek.iterrows():
        if row['target'] in dom_subset:
            region_2_known_reg_dists[row['query']].append([row['target'], float(row['alntmscore'])])
    filename_2_regions_vs_known_reg_dists = {args.pdb_id: region_2_known_reg_dists}

    os.remove(tsv_path)
    rmtree(tmp_path)

    with open(args.output_path, "wb") as file:
        pickle.dump(filename_2_regions_vs_known_reg_dists, file)
