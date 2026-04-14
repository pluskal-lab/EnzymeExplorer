"""Tests for evaluation.py similarity artifact loading and bin helpers."""

import pickle

from enzymeexplorer.src.evaluation.evaluation import (
    _build_bin_label,
    _get_pident,
    _has_hit,
    _make_record_key,
    load_similarity_artifact,
)


class TestLoadSimilarityArtifact:
    """Verify normalisation of legacy BLAST and rich MMseqs pickles."""

    def test_legacy_blast_format(self, tmp_path) -> None:
        legacy = {0: {"A1": 85.0, "A2": 42.0}, 1: {"B1": 99.0}}
        path = str(tmp_path / "blast.pkl")
        with open(path, "wb") as f:
            pickle.dump(legacy, f)

        result = load_similarity_artifact(path)
        assert set(result.keys()) == {0, 1}
        rec = result[0]["A1"]
        assert rec["pident"] == 85.0
        assert rec["has_hit"] is True
        assert rec["is_synthetic"] is True

    def test_mmseqs_format_passthrough(self, tmp_path) -> None:
        rich = {
            0: {"A1": {"pident": 75.0, "qcov": 0.9, "evalue": 1e-10, "has_hit": True}}
        }
        path = str(tmp_path / "mmseqs.pkl")
        with open(path, "wb") as f:
            pickle.dump(rich, f)

        result = load_similarity_artifact(path)
        rec = result[0]["A1"]
        assert rec["pident"] == 75.0
        assert rec["qcov"] == 0.9
        assert "is_synthetic" not in rec

    def test_string_fold_keys_converted_to_int(self, tmp_path) -> None:
        data = {"0": {"A1": 50.0}, "1": {"B1": 60.0}}
        path = str(tmp_path / "str_keys.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)

        result = load_similarity_artifact(path)
        assert set(result.keys()) == {0, 1}


class TestHelpers:
    def test_get_pident(self) -> None:
        assert _get_pident({"pident": 42.5, "has_hit": True}) == 42.5

    def test_get_pident_missing(self) -> None:
        assert _get_pident({}) == 0.0

    def test_has_hit_true(self) -> None:
        assert _has_hit({"pident": 50.0, "has_hit": True}) is True

    def test_has_hit_false(self) -> None:
        assert _has_hit({"pident": 0.0, "has_hit": False}) is False

    def test_has_hit_missing(self) -> None:
        assert _has_hit({}) is True

    def test_build_bin_label(self) -> None:
        assert _build_bin_label(30.0, 50.0) == "30-50"

    def test_make_record_key_no_category(self) -> None:
        assert _make_record_key("isTPS", "", "all") == "isTPS"

    def test_make_record_key_with_category(self) -> None:
        assert _make_record_key("isTPS", "Fungi", "all") == "Fungi_|_isTPS"

    def test_make_record_key_with_bin(self) -> None:
        assert _make_record_key("isTPS", "", "30-50") == "30-50_||_isTPS"

    def test_make_record_key_with_category_and_bin(self) -> None:
        assert _make_record_key("isTPS", "Fungi", "30-50") == "30-50_||_Fungi_|_isTPS"

    def test_make_record_key_no_hit(self) -> None:
        assert _make_record_key("isTPS", "", "no_hit") == "no_hit_||_isTPS"
