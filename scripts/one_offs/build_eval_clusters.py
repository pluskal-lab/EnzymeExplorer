"""Cluster the eval dataset at 50% sequence identity for block bootstrapping.

Runs MMseqs2 easy-cluster on unique (ID, Aminoacid_sequence) pairs of the
full eval dataset (positives + SwissProt negatives together), then
writes a two-column TSV ``<Representative>\t<Member>`` that the eval
pipeline consumes as the cluster-block map.

Match the training-time defaults so the resulting groups are consistent
with the fold split logic: ``--min-seq-id 0.5 -c 0.8 --cov-mode 0``.

Usage::

    python -m scripts.evaluation.build_eval_clusters \\
        --dataset data/EnzymeExplorer_Dataset.csv \\
        --out data/EnzymeExplorer_Dataset_clusters_50.tsv
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd  # type: ignore


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from enzymeexplorer.src.data_preparation.mmseqs2_wrapper import (  # noqa: E402
    MMSeqs2Wrapper,
)


def _write_fasta(id2seq: dict[str, str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for uid, seq in id2seq.items():
            fh.write(f">{uid}\n{seq}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset", type=Path,
        default=REPO / "data" / "EnzymeExplorer_Dataset.csv",
    )
    ap.add_argument(
        "--out", type=Path,
        default=REPO / "data" / "EnzymeExplorer_Dataset_clusters_50.tsv",
    )
    ap.add_argument("--min-seq-id", type=float, default=0.5)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--mmseqs", type=str, default="mmseqs")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, usecols=["ID", "Aminoacid_sequence"])
    df = df.drop_duplicates(subset=["ID"]).dropna(subset=["Aminoacid_sequence"])
    id2seq = dict(zip(df["ID"].astype(str), df["Aminoacid_sequence"].astype(str)))
    print(f"unique IDs: {len(id2seq)}")

    mmseqs = MMSeqs2Wrapper(mmseqs_path=args.mmseqs, threads=args.threads)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta = td / "input.fasta"
        _write_fasta(id2seq, fasta)
        out_prefix = td / "clu"
        tmp_dir = td / "tmp"
        tmp_dir.mkdir()
        clusters_df, _ = mmseqs.easy_cluster(
            input_fasta=str(fasta),
            output=str(out_prefix),
            tmp=str(tmp_dir),
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
        )

    n_clusters = clusters_df["Representative"].nunique()
    n_members = len(clusters_df)
    missing = set(id2seq) - set(clusters_df["Member"])
    if missing:
        print(f"WARNING: {len(missing)} IDs missing from cluster output "
              f"(likely dropped by mmseqs); assigning each to its own cluster")
        singletons = pd.DataFrame({
            "Representative": sorted(missing),
            "Member": sorted(missing),
        })
        clusters_df = pd.concat([clusters_df, singletons], ignore_index=True)
        n_clusters = clusters_df["Representative"].nunique()
        n_members = len(clusters_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clusters_df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}: {n_clusters} clusters over {n_members} members")


if __name__ == "__main__":
    main()
