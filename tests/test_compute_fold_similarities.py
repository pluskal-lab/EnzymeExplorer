"""Tests for scripts/compute_fold_similarities.py — best-hit selection logic."""

import os

from scripts.compute_fold_similarities import (
    get_fold_names,
    select_best_hits,
)

import pandas as pd

# ── select_best_hits ─────────────────────────────────────────────────


class TestSelectBestHits:
    """Test the post-hoc best-hit selection from MMseqs .m8 output."""

    def _write_m8(self, tmp_dir: str, lines: list[str]) -> str:
        path = os.path.join(tmp_dir, "test.m8")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def test_best_hit_by_pident(self, tmp_path: str) -> None:
        lines = [
            "Q1\tT1\t50.0\t0.9\t1e-10",
            "Q1\tT2\t80.0\t0.8\t1e-5",
        ]
        path = self._write_m8(str(tmp_path), lines)
        result = select_best_hits(path, min_qcov=0.5)
        assert result["Q1"]["pident"] == 80.0
        assert result["Q1"]["has_hit"] is True

    def test_tiebreak_by_evalue(self, tmp_path: str) -> None:
        lines = [
            "Q1\tT1\t90.0\t0.9\t1e-5",
            "Q1\tT2\t90.0\t0.85\t1e-20",
        ]
        path = self._write_m8(str(tmp_path), lines)
        result = select_best_hits(path, min_qcov=0.5)
        assert result["Q1"]["pident"] == 90.0
        assert result["Q1"]["evalue"] == 1e-20

    def test_qcov_filter(self, tmp_path: str) -> None:
        lines = [
            "Q1\tT1\t99.0\t0.3\t1e-20",
            "Q1\tT2\t40.0\t0.8\t1e-5",
        ]
        path = self._write_m8(str(tmp_path), lines)
        result = select_best_hits(path, min_qcov=0.5)
        assert result["Q1"]["pident"] == 40.0
        assert result["Q1"]["has_hit"] is True

    def test_all_below_qcov(self, tmp_path: str) -> None:
        lines = [
            "Q1\tT1\t99.0\t0.2\t1e-20",
            "Q1\tT2\t80.0\t0.1\t1e-5",
        ]
        path = self._write_m8(str(tmp_path), lines)
        result = select_best_hits(path, min_qcov=0.5)
        assert result["Q1"]["has_hit"] is False
        assert result["Q1"]["pident"] == 0.0

    def test_empty_file(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "empty.m8")
        with open(path, "w"):
            pass
        result = select_best_hits(path, min_qcov=0.5)
        assert result == {}

    def test_missing_file(self) -> None:
        result = select_best_hits("/nonexistent/path.m8", min_qcov=0.5)
        assert result == {}

    def test_multiple_queries(self, tmp_path: str) -> None:
        lines = [
            "Q1\tT1\t70.0\t0.9\t1e-10",
            "Q2\tT2\t85.0\t0.7\t1e-15",
            "Q2\tT3\t60.0\t0.95\t1e-8",
        ]
        path = self._write_m8(str(tmp_path), lines)
        result = select_best_hits(path, min_qcov=0.5)
        assert result["Q1"]["pident"] == 70.0
        assert result["Q2"]["pident"] == 85.0

    def test_qcov_percentage_normalization(self, tmp_path: str) -> None:
        """qcov values > 1.0 are treated as percentages and divided by 100."""
        lines = [
            "Q1\tT1\t80.0\t90.0\t1e-10",
        ]
        path = self._write_m8(str(tmp_path), lines)
        result = select_best_hits(path, min_qcov=0.5)
        assert result["Q1"]["qcov"] == 0.9
        assert result["Q1"]["has_hit"] is True


# ── get_fold_names ───────────────────────────────────────────────────


class TestGetFoldNames:
    def test_extracts_fold_names(self) -> None:
        df = pd.DataFrame({"fold": ["fold_0", "fold_1", "fold_2", "fold_0", "fold_1"]})
        names = get_fold_names(df, "fold")
        assert names == ["fold_0", "fold_1", "fold_2"]

    def test_ignores_non_fold_values(self) -> None:
        df = pd.DataFrame({"fold": ["fold_0", "fold_1", "other", None, "fold_0"]})
        names = get_fold_names(df, "fold")
        assert names == ["fold_0", "fold_1"]

    def test_sorted_order(self) -> None:
        df = pd.DataFrame({"fold": ["fold_2", "fold_0", "fold_1"]})
        names = get_fold_names(df, "fold")
        assert names == ["fold_0", "fold_1", "fold_2"]
