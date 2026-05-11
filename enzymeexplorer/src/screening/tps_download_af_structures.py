"""CPU-only AF-DB structure download for one FASTA slice.

Called by the per-batch download SLURM job
(``scripts/tps_screening_download_one_batch.sh``) — that wrapper
turns ``$SLURM_ARRAY_TASK_ID`` and ``batch_size`` into the
``[start_i, end_i)`` window passed below and routes outputs to a
``batch_<idx>/`` sub-directory under the manager's structures root.

Can also be driven manually on a non-SLURM host: pass ``--start-i 0``
and omit ``--end-i`` to drain the entire FASTA in one process.

Outputs:
  <output-dir>/<uid>.pdb       one per successfully downloaded structure
  <missing-csv>                IDs with no AF-DB entry (one ``id`` column)
"""

from __future__ import annotations

import argparse
import logging
from itertools import islice
from pathlib import Path

import pandas as pd  # type: ignore
from Bio import SeqIO  # type: ignore

from enzymeexplorer.src.screening.af_db import download_many

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta-path", required=True, type=Path,
        help="Input FASTA; record IDs are treated as UniProt accessions.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to drop downloaded <uid>.pdb files into.",
    )
    parser.add_argument(
        "--missing-csv", required=True, type=Path,
        help="CSV path for IDs with no AF-DB entry (one ``id`` column).",
    )
    parser.add_argument(
        "--start-i", type=int, default=0,
        help="Index of the first sequence (inclusive) to process.",
    )
    parser.add_argument(
        "--end-i", type=int, default=None,
        help="Index of the last sequence (exclusive). Default: end of FASTA.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=16,
        help="Concurrent HTTP downloads (AF-DB ingress is the bottleneck).",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even when a non-empty PDB is already on disk.",
    )
    return parser.parse_args()


def _shard_ids(path: Path, start_i: int, end_i: int | None) -> list[str]:
    records = SeqIO.parse(str(path), "fasta")
    sliced = islice(records, start_i, end_i)
    seen: set[str] = set()
    ids: list[str] = []
    for r in sliced:
        if r.id and r.id not in seen:
            seen.add(r.id)
            ids.append(r.id)
    return ids


def main(args: argparse.Namespace) -> None:
    uids = _shard_ids(args.fasta_path, args.start_i, args.end_i)
    logger.info(
        "AF-DB shard [%d, %s) from %s — %d unique UniProt IDs",
        args.start_i,
        args.end_i if args.end_i is not None else "end",
        args.fasta_path,
        len(uids),
    )
    if not uids:
        # Empty shard — still write an empty missing.csv with the schema
        # so the gather/launcher steps don't have to special-case absent
        # files.
        args.missing_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["id"]).to_csv(args.missing_csv, index=False)
        return

    _, missing = download_many(
        uids,
        args.output_dir,
        n_workers=args.n_workers,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )

    args.missing_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": sorted(missing)}).to_csv(args.missing_csv, index=False)
    logger.info(
        "Wrote %d missing IDs to %s; %d PDBs in %s",
        len(missing), args.missing_csv,
        len(uids) - len(missing), args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
