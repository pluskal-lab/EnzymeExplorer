"""Thin wrapper around the foldseek CLI.

All foldseek subprocess invocations in the codebase should go through this
class — direct ``subprocess.run([..., "foldseek", ...])`` calls in domain
detection / structural-feature code were consolidated here so that the
command line, defaults, and error handling live in one place.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Iterable

import pandas as pd  # type: ignore

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FoldseekWrapper:
    def __init__(self, foldseek_path: str = "foldseek", threads: int = 8):
        self.foldseek_path = foldseek_path
        self.threads = threads

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _run(self, cmd: list[str], op: str) -> None:
        """Invoke a single foldseek command and raise on non-zero exit."""
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("foldseek %s stderr: %s", op, result.stderr)
            raise RuntimeError(
                f"foldseek {op} failed (rc={result.returncode}): "
                f"{result.stderr[:400]}"
            )

    def _resolved_threads(self, threads: int | None) -> int:
        return self.threads if threads is None else threads

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def createdb(
        self,
        input_dir: str,
        db_path: str,
        *,
        threads: int | None = None,
    ) -> None:
        """Build a foldseek database from a directory of structures."""
        cmd = [
            self.foldseek_path,
            "createdb",
            str(input_dir),
            str(db_path),
            "--threads", str(self._resolved_threads(threads)),
        ]
        self._run(cmd, "createdb")

    def search(
        self,
        query_db: str,
        target_db: str,
        result_db: str,
        tmp_dir: str,
        *,
        max_seqs: int = 5000,
        e_value: float = 100.0,
        sensitivity: float = 9.5,
        write_alignments: bool = False,
        cov_mode: int | None = None,
        coverage: float | None = None,
        threads: int | None = None,
    ) -> None:
        """Run ``foldseek search`` between two pre-built databases.

        ``write_alignments=True`` adds the ``-a`` flag so that
        :meth:`convertalis` can later format CIGAR/start/end columns.
        """
        cmd = [
            self.foldseek_path,
            "search",
            str(query_db),
            str(target_db),
            str(result_db),
            str(tmp_dir),
            "-e", str(e_value),
            "--max-seqs", str(max_seqs),
            "-s", str(sensitivity),
            "--threads", str(self._resolved_threads(threads)),
        ]
        if write_alignments:
            cmd.append("-a")
        if cov_mode is not None:
            cmd += ["--cov-mode", str(cov_mode)]
        if coverage is not None:
            cmd += ["-c", str(coverage)]
        self._run(cmd, "search")

    def convertalis(
        self,
        query_db: str,
        target_db: str,
        result_db: str,
        output_tsv: str,
        *,
        format_output: Iterable[str] | str = ("query", "target", "alntmscore"),
        threads: int | None = None,
    ) -> pd.DataFrame:
        """Run ``foldseek convertalis`` and return the parsed TSV.

        ``format_output`` may be a comma-joined string or any iterable of
        column names; the returned DataFrame has those columns in the
        same order.
        """
        if isinstance(format_output, str):
            cols = [c.strip() for c in format_output.split(",") if c.strip()]
            format_str = format_output
        else:
            cols = list(format_output)
            format_str = ",".join(cols)
        cmd = [
            self.foldseek_path,
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(output_tsv),
            "--format-output", format_str,
            "--threads", str(self._resolved_threads(threads)),
        ]
        self._run(cmd, "convertalis")
        return pd.read_csv(output_tsv, sep="\t", header=None, names=cols)

    def easy_search(
        self,
        query_dir: str,
        target_dir: str,
        output: str,
        tmp_dir: str,
        max_seqs: int = 5000,
        e_value: float = 100,
        sensitivity: float = 9.5,
        cov_mode: int | None = None,
        coverage: float | None = None,
        format_output: Iterable[str] | str = ("query", "target", "alntmscore"),
    ) -> pd.DataFrame:
        if isinstance(format_output, str):
            cols = [c.strip() for c in format_output.split(",") if c.strip()]
            format_str = format_output
        else:
            cols = list(format_output)
            format_str = ",".join(cols)
        cmd = [
            self.foldseek_path,
            "easy-search",
            str(query_dir),
            str(target_dir),
            str(output),
            str(tmp_dir),
            "-e", str(e_value),
            "--max-seqs", str(max_seqs),
            "-s", str(sensitivity),
            "--format-output", format_str,
        ]
        if cov_mode is not None:
            cmd += ["--cov-mode", str(cov_mode)]
        if coverage is not None:
            cmd += ["-c", str(coverage)]
        self._run(cmd, "easy-search")
        return pd.read_csv(output, sep="\t", header=None, names=cols)
