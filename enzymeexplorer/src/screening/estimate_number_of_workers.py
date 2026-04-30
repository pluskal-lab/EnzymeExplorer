"""Count sequences in a FASTA and report shard sizing for the screening launcher."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from Bio import SeqIO  # type: ignore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-path", required=True, type=Path)
    parser.add_argument(
        "--shard-size", type=int, default=40_000,
        help="Sequences per shard; must match the launcher's --shard-size.",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=8,
        help="GPUs (and parallel workers) per session; must match the launcher.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    total = sum(1 for _ in SeqIO.parse(str(args.fasta_path), "fasta"))
    per_session = args.shard_size * args.n_gpus
    n_sessions = math.ceil(total / per_session) if per_session else 0
    logger.info("Total sequences: %d", total)
    logger.info("Sequences per session (shard_size × n_gpus): %d", per_session)
    logger.info(
        "Sessions required (set SLURM --array=1-%d): %d",
        n_sessions,
        n_sessions,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(parse_args())
