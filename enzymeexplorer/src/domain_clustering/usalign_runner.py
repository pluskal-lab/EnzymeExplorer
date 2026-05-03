"""All-vs-all USalign batch runner for the domain-clustering pipeline.

USalign is invoked in batch mode (``-dir1 folder list1 -dir2 folder list2
-outfmt 2``) on N parallel chunks. The full output is the cross-product of
chunk-queries × all-targets — pair duplicates and self-pairs are dropped
during parsing. The ``-fast`` flag is on by default (~2.3× speedup vs the
rigorous mode at the cost of slightly less precise TM-scores; perfectly
adequate for a clustering distance matrix).

Output of ``-outfmt 2`` (tab-separated, one header line):
    PDBchain1  PDBchain2  TM1  TM2  RMSD  ID1  ID2  IDali  L1  L2  Lali

PDBchain identifiers are like ``marts_E00000_alpha_0.pdb:A``; we strip the
``.pdb:<chain>`` suffix on parse. Symmetric TM is stored as
``max(TM1, TM2)`` — the more permissive of the two normalisations, so
similar substructures aren't penalised by length mismatch (the user's
explicit goal: "do not miss alignments").
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

USALIGN_OUTFMT2_COLUMNS = [
    "PDBchain1", "PDBchain2", "TM1", "TM2", "RMSD",
    "ID1", "ID2", "IDali", "L1", "L2", "Lali",
]
_NAME_SUFFIX_RE = re.compile(r"\.pdb(:[A-Za-z0-9])?$")


def _resolve_usalign() -> str:
    """Locate the USalign binary using the same resolution rules as
    structure_processing's USALIGN_PATH."""
    p = os.environ.get("ENZYMEEXPLORER_USALIGN") or shutil.which("USalign")
    if p and Path(p).exists():
        return p
    import sys
    fallback = Path(sys.prefix) / "bin" / "USalign"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("USalign not found (set ENZYMEEXPLORER_USALIGN or add to PATH)")


def _stage_top_level_pdbs(pdb_dir: Path, work_dir: Path) -> tuple[Path, list[str]]:
    """Stage symlinks to top-level *.pdb files (no subdir recursion)."""
    pdb_dir = Path(pdb_dir).resolve()
    staged = work_dir / "input_pdbs"
    staged.mkdir(parents=True, exist_ok=True)
    pdbs = sorted(pdb_dir.glob("*.pdb"))
    stems = [p.stem for p in pdbs]
    existing = {p.name for p in staged.iterdir()}
    wanted = {f"{s}.pdb" for s in stems}
    for name in existing - wanted:
        (staged / name).unlink(missing_ok=True)
    for src in pdbs:
        link = staged / src.name
        if not link.exists():
            link.symlink_to(src.resolve())
    logger.info("Staged %d PDBs under %s", len(stems), staged)
    return staged, stems


def _run_chunk(args: tuple) -> Path:
    """Run one USalign batch invocation and return the output TSV path.

    Per-chunk idempotency: if ``output_path`` already exists and is
    non-empty (i.e. a previous run produced it), skip the USalign call
    and reuse the cached chunk. This means partial-failure restarts pick
    up where they left off — only missing chunks get recomputed.
    """
    chunk_idx, query_list, target_list, staged_dir, output_path, fast_mode, usalign = args
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        n_lines = sum(1 for _ in open(output_path))
        logger.info(
            "chunk %d: reusing cached %s (%d lines)",
            chunk_idx, output_path, n_lines,
        )
        return output_path

    cmd = [
        usalign,
        "-dir1", str(staged_dir) + "/", str(query_list),
        "-dir2", str(staged_dir) + "/", str(target_list),
        "-suffix", ".pdb",
        "-outfmt", "2",
    ]
    if fast_mode:
        cmd.append("-fast")
    t0 = time.perf_counter()
    with open(output_path, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)
    elapsed = time.perf_counter() - t0
    n_lines = sum(1 for _ in open(output_path))
    logger.info(
        "chunk %d done in %.0fs → %d lines", chunk_idx, elapsed, n_lines,
    )
    return output_path


def run_all_vs_all(
    pdb_dir: str | Path,
    output_dir: str | Path,
    *,
    n_jobs: int = 8,
    fast_mode: bool = True,
    chunk_size: int | None = None,
) -> Path:
    """Run all-vs-all USalign on ``pdb_dir``'s top-level *.pdb files.

    Returns the path to the merged output TSV. Workers run as parallel
    subprocesses; each does ``chunk_size`` queries × all targets (full
    cross-product) and the merger drops duplicates / self-pairs in the
    parsing step. Default ``-fast`` mode.
    """
    pdb_dir = Path(pdb_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persistent chunks dir — kept across runs. Each chunk's TSV is the
    # ground truth for that subset of (query, target) alignments; never
    # deleted. Re-runs of run_all_vs_all() detect cached chunks and skip
    # the corresponding USalign subprocess (see _run_chunk).
    work_dir = output_dir / "chunks"
    work_dir.mkdir(parents=True, exist_ok=True)
    # Migrate the original "tmp_chunks" directory if present (older runs
    # used that name before we made the persistence intent explicit).
    legacy_dir = output_dir / "tmp_chunks"
    if legacy_dir.exists() and legacy_dir != work_dir:
        for f in legacy_dir.iterdir():
            target = work_dir / f.name
            if not target.exists():
                f.rename(target)
        try:
            legacy_dir.rmdir()
        except OSError:
            pass  # not empty (rare); leave it

    usalign = _resolve_usalign()
    staged_dir, stems = _stage_top_level_pdbs(pdb_dir, output_dir)
    n = len(stems)
    if chunk_size is None:
        chunk_size = max(1, (n + n_jobs - 1) // n_jobs)
    n_chunks = (n + chunk_size - 1) // chunk_size

    full_list = work_dir / "all.lst"
    full_list.write_text("\n".join(stems) + "\n")

    chunk_args = []
    for i in range(n_chunks):
        chunk_stems = stems[i * chunk_size : (i + 1) * chunk_size]
        if not chunk_stems:
            continue
        chunk_list = work_dir / f"chunk_{i:03d}.lst"
        chunk_list.write_text("\n".join(chunk_stems) + "\n")
        chunk_out = work_dir / f"chunk_{i:03d}.tsv"
        chunk_args.append(
            (i, chunk_list, full_list, staged_dir, chunk_out, fast_mode, usalign)
        )

    logger.info(
        "Running USalign on %d PDBs in %d parallel chunks of %d "
        "(fast_mode=%s, binary=%s)",
        n, len(chunk_args), chunk_size, fast_mode, usalign,
    )
    t0 = time.perf_counter()
    with mp.Pool(processes=min(n_jobs, len(chunk_args))) as pool:
        chunk_outputs = pool.map(_run_chunk, chunk_args)
    elapsed = time.perf_counter() - t0
    logger.info("All chunks done in %.1f min", elapsed / 60)

    # Concatenate (dropping per-chunk header lines except the first).
    merged_path = output_dir / "alignment_usalign.tsv"
    with open(merged_path, "w") as out:
        wrote_header = False
        for chunk_out in chunk_outputs:
            with open(chunk_out) as f:
                for line in f:
                    if line.startswith("#"):
                        if not wrote_header:
                            out.write(line)
                            wrote_header = True
                        continue
                    out.write(line)
    logger.info("Merged TSV → %s", merged_path)
    return merged_path


def _strip_chain_suffix(s: str) -> str:
    return _NAME_SUFFIX_RE.sub("", s)


def parse_alignment_tsv(
    tsv_path: str | Path,
    *,
    tm_aggregator: str = "max",
) -> dict[tuple[str, str], float]:
    """Parse USalign ``-outfmt 2`` output → ``{(min(a,b), max(a,b)): TM}``.

    ``tm_aggregator``:
      * ``"max"``: ``max(TM1, TM2)`` — best of both normalisations
      * ``"avg"``: ``(TM1 + TM2) / 2`` — symmetric average
      * ``"min"``: ``min(TM1, TM2)`` — penalise size mismatch
    """
    df = pd.read_csv(
        tsv_path, sep="\t", comment="#", header=None,
        names=USALIGN_OUTFMT2_COLUMNS,
    )
    df["q"] = df["PDBchain1"].astype(str).map(_strip_chain_suffix)
    df["t"] = df["PDBchain2"].astype(str).map(_strip_chain_suffix)
    if tm_aggregator == "max":
        df["TM"] = np.maximum(df["TM1"].astype(float), df["TM2"].astype(float))
    elif tm_aggregator == "avg":
        df["TM"] = (df["TM1"].astype(float) + df["TM2"].astype(float)) / 2.0
    elif tm_aggregator == "min":
        df["TM"] = np.minimum(df["TM1"].astype(float), df["TM2"].astype(float))
    else:
        raise ValueError(f"unknown tm_aggregator: {tm_aggregator}")
    df = df[df["q"] != df["t"]]

    pairs: dict[tuple[str, str], float] = {}
    for q, t, tm in zip(df["q"], df["t"], df["TM"]):
        a, b = (q, t) if q <= t else (t, q)
        prev = pairs.get((a, b))
        if prev is None or tm > prev:
            pairs[(a, b)] = float(tm)
    logger.info(
        "Parsed %s → %d unique unordered pairs (%s aggregation)",
        tsv_path, len(pairs), tm_aggregator,
    )
    return pairs
