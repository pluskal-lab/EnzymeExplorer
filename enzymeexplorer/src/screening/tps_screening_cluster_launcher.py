"""Multi-GPU launcher for FASTA screening.

Splits the input FASTA into ``n_gpus`` contiguous shards and spawns one
:mod:`tps_predict_fasta` worker per shard. Each worker is given a disjoint
``[start_i, end_i)`` slice and a per-shard CSV path; the launcher pins the
worker to a GPU via ``CUDA_VISIBLE_DEVICES`` (also set as
``HIP_VISIBLE_DEVICES`` so the same launcher works on AMD ROCm without
modification).

For SLURM array jobs, ``--session-i`` lets one array task pick up its own
contiguous chunk of the FASTA: each session processes
``n_gpus * shard_size`` sequences, so set ``--array=1-K`` where
``K = ceil(N / (n_gpus * shard_size))``.

After all workers finish, aggregate the per-shard CSVs with
``python -m enzymeexplorer.src.screening.gather_detections_to_csv``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from enzymeexplorer.src.prediction.pipeline import DEFAULT_PLM_MODEL

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta-path", required=True, type=Path,
        help="Input FASTA to screen.",
    )
    parser.add_argument(
        "--output-root", required=True, type=Path,
        help="Per-shard CSVs are written under <output-root>/shards/.",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=8,
        help="Number of parallel workers; one GPU each, round-robin.",
    )
    parser.add_argument(
        "--shard-size", type=int, default=40_000,
        help="Number of sequences per shard.",
    )
    parser.add_argument(
        "--session-i", type=int, default=1,
        help=(
            "1-indexed session number for SLURM array jobs. Session i "
            "processes sequences "
            "[(i-1)*n_gpus*shard_size, i*n_gpus*shard_size)."
        ),
    )
    parser.add_argument(
        "--plm-model", type=str, default=DEFAULT_PLM_MODEL,
    )
    parser.add_argument("--plm-batch-size", type=int, default=32)
    parser.add_argument(
        "--keep-negatives", action="store_true",
        help="Forwarded to each worker; see tps_predict_fasta --help.",
    )
    parser.add_argument(
        "--workdir", type=Path, default=None,
        help=(
            "Parent directory for per-worker scratch. Each spawned shard "
            "creates its own subdir under this parent and removes it on "
            "exit. Defaults to the system tmp directory."
        ),
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Forwarded to each worker; don't delete their scratch dirs.",
    )
    return parser.parse_args()


def _spawn_worker(
    *,
    gpu_id: int,
    fasta_path: Path,
    output_csv: Path,
    start_i: int,
    end_i: int,
    plm_model: str,
    plm_batch_size: int,
    keep_negatives: bool,
    workdir: Path | None,
    keep_intermediate: bool,
) -> subprocess.Popen:
    """Spawn one screening worker pinned to the given GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        "-m",
        "enzymeexplorer.src.screening.tps_predict_fasta",
        "--fasta-path", str(fasta_path),
        "--output-csv", str(output_csv),
        "--start-i", str(start_i),
        "--end-i", str(end_i),
        "--plm-model", plm_model,
        "--plm-batch-size", str(plm_batch_size),
    ]
    if keep_negatives:
        cmd.append("--keep-negatives")
    if workdir is not None:
        cmd += ["--workdir", str(workdir)]
    if keep_intermediate:
        cmd.append("--keep-intermediate")
    logger.info("Launching worker on GPU %d → %s", gpu_id, output_csv)
    return subprocess.Popen(cmd, env=env)


def main(args: argparse.Namespace) -> None:
    shards_dir = args.output_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    session_offset = (args.session_i - 1) * args.n_gpus * args.shard_size

    processes: list[subprocess.Popen] = []
    for gpu_i in range(args.n_gpus):
        start_i = session_offset + gpu_i * args.shard_size
        end_i = start_i + args.shard_size
        output_csv = (
            shards_dir
            / f"session_{args.session_i:03d}_gpu_{gpu_i}_{start_i:09d}_{end_i:09d}.csv"
        )
        processes.append(
            _spawn_worker(
                gpu_id=gpu_i,
                fasta_path=args.fasta_path,
                output_csv=output_csv,
                start_i=start_i,
                end_i=end_i,
                plm_model=args.plm_model,
                plm_batch_size=args.plm_batch_size,
                keep_negatives=args.keep_negatives,
                workdir=args.workdir,
                keep_intermediate=args.keep_intermediate,
            )
        )

    failures = 0
    for proc in processes:
        rc = proc.wait()
        if rc != 0:
            logger.error("Worker pid=%d exited with code %d", proc.pid, rc)
            failures += 1
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
