"""Compute secondary-structure residues for every PDB in a directory.

Importable API: :func:`compute_secondary_structure_residues` runs the
extraction in-process and returns the resulting mapping. Internally it
delegates to :func:`save_file_to_all_residues`, which routes parallel
work through the centralized pool service. The CLI ``main`` opens a
one-shot pool session and forwards to the same function — there is no
separate subprocess fork path.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from shutil import copyfile
from typing import Optional

from enzymeexplorer.src.structure_processing._pool_service import (
    get_active_service,
    pool_session,
)
from enzymeexplorer.src.structure_processing.utils import (
    SSR_PARALLEL_THRESHOLD,
    save_file_to_all_residues,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


def _collect_pdb_files(input_directory: Path) -> list[Path]:
    """Glob *.pdb files, replacing PyMOL-incompatible names by sanitised copies.

    PyMOL cannot select objects whose names contain spaces or
    parentheses, so any matching files are copied to a sanitised name
    in-place and the sanitised path is what's passed downstream. (The
    original file is left intact.)
    """
    pdb_files_raw = list(input_directory.glob("*.pdb"))
    pdb_files: list[Path] = []
    for filepath in pdb_files_raw:
        name = filepath.name
        if "(" in name or ")" in name or len(name.split()) > 1:
            sanitised = filepath.with_name(
                "".join(name.replace("(", "").replace(")", "").split())
            )
            if not sanitised.exists():
                copyfile(filepath, sanitised)
            pdb_files.append(sanitised)
        else:
            pdb_files.append(filepath)
    return pdb_files


def compute_secondary_structure_residues(
    input_directory: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    n_jobs: int = 20,
) -> dict[str, set[str]]:
    """Compute and persist the SS-residues map for every PDB in ``input_directory``.

    Returns the mapping ``{pdb_stem: set_of_ss_residue_ids}`` and writes
    it to ``output_path`` as a pickle. The serial in-process path is
    used for inputs below ``SSR_PARALLEL_THRESHOLD``; otherwise parallel
    SSR runs through the active pool service. If no session is open and
    parallelism is needed, this function opens a one-shot session.
    """
    input_directory = Path(input_directory).absolute()
    output_path = Path(output_path).absolute()

    pdb_files = _collect_pdb_files(input_directory)
    logger.info("SSR: %d PDB files in %s", len(pdb_files), input_directory)

    needs_parallel = len(pdb_files) >= SSR_PARALLEL_THRESHOLD
    if needs_parallel and get_active_service() is None:
        with pool_session(n_jobs=n_jobs, working_dir=str(input_directory)):
            save_file_to_all_residues(
                secondary_structure_residues_path=output_path,
                pdb_files=pdb_files,
                domain_templates=[],
            )
    else:
        save_file_to_all_residues(
            secondary_structure_residues_path=output_path,
            pdb_files=pdb_files,
            domain_templates=[],
        )

    with open(output_path, "rb") as f:
        return pickle.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute secondary-structure residues for every PDB in a directory."
        )
    )
    parser.add_argument(
        "--input-directory",
        type=str,
        default="data/alphafold_structs/",
        help="Directory containing PDB structures.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/alphafold_structs/file_2_all_residues.pkl",
        help="Path to write the SSR pickle to.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=20,
        help="Pool size used when input ≥ SSR_PARALLEL_THRESHOLD.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    compute_secondary_structure_residues(
        input_directory=args.input_directory,
        output_path=args.output_path,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
