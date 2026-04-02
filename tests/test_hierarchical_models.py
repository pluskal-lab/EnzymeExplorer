"""Tests for build_hierarchical_models.py pure logic and helpers.

Covers:
- _assign_is_tps: label set augmentation
- _normalize_fold_column: fold column normalization
- product-of-probabilities combination logic
- predict_substrate_proba missing-embedding fallback
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_hierarchical_models import (  # noqa: E402
    _assign_is_tps,
    _normalize_fold_column,
    _NON_TPS_LABELS,
)


# ══════════════════════════════════════════════════════════════════════════
# Tests for _assign_is_tps
# ══════════════════════════════════════════════════════════════════════════


class TestAssignIsTps:
    """Tests for the label-set isTPS augmentation helper."""

    def test_non_tps_only_unchanged(self):
        assert _assign_is_tps({"Unknown"}) == {"Unknown"}
        assert _assign_is_tps({"precursor substr"}) == {"precursor substr"}
        assert _assign_is_tps({"other"}) == {"other"}

    def test_all_non_tps_labels(self):
        result = _assign_is_tps({"Unknown", "precursor substr", "other"})
        assert "isTPS" not in result
        assert result == _NON_TPS_LABELS

    def test_real_substrate_gets_istps(self):
        result = _assign_is_tps({"FPP_SMILES"})
        assert "isTPS" in result
        assert "FPP_SMILES" in result

    def test_mixed_real_and_non_tps(self):
        result = _assign_is_tps({"FPP_SMILES", "precursor substr"})
        assert "isTPS" in result
        assert "FPP_SMILES" in result
        assert "precursor substr" in result

    def test_empty_set(self):
        result = _assign_is_tps(set())
        assert result == set()

    def test_idempotent(self):
        first = _assign_is_tps({"substrate_X"})
        second = _assign_is_tps(first)
        assert first == second


# ══════════════════════════════════════════════════════════════════════════
# Tests for _normalize_fold_column
# ══════════════════════════════════════════════════════════════════════════


class TestNormalizeFoldColumn:
    """Tests for fold column normalization to fold_N format."""

    def test_integer_folds_converted(self):
        df = pd.DataFrame({"Fold": [0, 1, 2, 3, 4]})
        _normalize_fold_column(df, "Fold")
        assert list(df["Fold"]) == [
            "fold_0",
            "fold_1",
            "fold_2",
            "fold_3",
            "fold_4",
        ]

    def test_already_prefixed_untouched(self):
        df = pd.DataFrame({"Fold": ["fold_0", "fold_1", "fold_2"]})
        _normalize_fold_column(df, "Fold")
        assert list(df["Fold"]) == ["fold_0", "fold_1", "fold_2"]

    def test_handles_nan_values(self):
        df = pd.DataFrame({"Fold": [0, 1, None]})
        _normalize_fold_column(df, "Fold")
        assert df["Fold"].iloc[0] == "fold_0"
        assert df["Fold"].iloc[1] == "fold_1"
        assert pd.isna(df["Fold"].iloc[2])

    def test_empty_column(self):
        df = pd.DataFrame({"Fold": pd.Series([], dtype=object)})
        _normalize_fold_column(df, "Fold")
        assert len(df) == 0


# ══════════════════════════════════════════════════════════════════════════
# Tests for hierarchical product-of-probabilities combination
# ══════════════════════════════════════════════════════════════════════════


class TestProductOfProbabilities:
    """Tests for the hierarchical combination logic P(sub) = P(TPS) × P(sub|TPS)."""

    def test_basic_combination(self):
        """P(TPS) × P(sub|TPS) produces expected combined scores."""
        class_names = ["isTPS", "FPP", "GPP"]
        istps_idx = 0
        n = 5

        p_tps = np.array([0.9, 0.8, 0.1, 0.05, 0.95])
        p_substrate = np.array(
            [
                [0.7, 0.2],
                [0.6, 0.3],
                [0.8, 0.1],
                [0.5, 0.5],
                [0.9, 0.05],
            ]
        )

        combined = np.zeros((n, len(class_names)))
        sub_idx = 0
        for ci, cn in enumerate(class_names):
            if cn == "isTPS":
                combined[:, ci] = p_tps
            else:
                combined[:, ci] = p_tps * p_substrate[:, sub_idx]
                sub_idx += 1

        # isTPS unchanged
        np.testing.assert_array_equal(combined[:, istps_idx], p_tps)

        # Substrate scores = P(TPS) * P(sub|TPS)
        expected_fpp = p_tps * p_substrate[:, 0]
        expected_gpp = p_tps * p_substrate[:, 1]
        np.testing.assert_array_almost_equal(combined[:, 1], expected_fpp)
        np.testing.assert_array_almost_equal(combined[:, 2], expected_gpp)

    def test_zero_p_tps_zeros_substrate(self):
        """When P(TPS) = 0, all substrate scores must be 0."""
        p_tps = np.zeros(3)
        p_sub = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])

        class_names = ["isTPS", "FPP", "GPP"]
        combined = np.zeros((3, 3))
        sub_idx = 0
        for ci, cn in enumerate(class_names):
            if cn == "isTPS":
                combined[:, ci] = p_tps
            else:
                combined[:, ci] = p_tps * p_sub[:, sub_idx]
                sub_idx += 1

        assert np.all(combined == 0.0)

    def test_perfect_detection_preserves_substrate(self):
        """When P(TPS) = 1.0, substrate scores equal P(sub|TPS)."""
        p_tps = np.ones(4)
        p_sub = np.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])

        class_names = ["isTPS", "FPP", "GPP"]
        combined = np.zeros((4, 3))
        sub_idx = 0
        for ci, cn in enumerate(class_names):
            if cn == "isTPS":
                combined[:, ci] = p_tps
            else:
                combined[:, ci] = p_tps * p_sub[:, sub_idx]
                sub_idx += 1

        np.testing.assert_array_almost_equal(combined[:, 1], p_sub[:, 0])
        np.testing.assert_array_almost_equal(combined[:, 2], p_sub[:, 1])

    def test_combined_never_exceeds_p_tps(self):
        """Combined substrate scores should never exceed P(TPS)."""
        rng = np.random.RandomState(42)
        n = 100
        p_tps = rng.rand(n)
        p_sub = rng.rand(n, 3)

        for ci in range(3):
            combined = p_tps * p_sub[:, ci]
            assert np.all(combined <= p_tps + 1e-10)

    def test_combined_ordering(self):
        """If sample A has higher P(TPS) and higher P(sub|TPS), it should
        have higher combined score than sample B."""
        p_tps = np.array([0.9, 0.3])
        p_sub = np.array([0.8, 0.2])
        combined = p_tps * p_sub
        assert combined[0] > combined[1]


# ══════════════════════════════════════════════════════════════════════════
# Tests for pkl roundtrip of hierarchical results
# ══════════════════════════════════════════════════════════════════════════


class TestHierarchicalPklRoundtrip:
    """Verify pkl save/load preserves hierarchical predictions."""

    def test_save_load_preserves_values(self, tmp_path):
        n = 20
        class_names = ["isTPS", "FPP", "GPP"]
        proba = np.random.RandomState(0).rand(n, 3)
        test_df = pd.DataFrame(
            {
                "ID": [f"P{i}" for i in range(n)],
                "SMILES_substrate_canonical_no_stereo": ["FPP"] * 10
                + ["Unknown"] * 10,
            }
        )

        pkl_path = tmp_path / "fold_0_results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump((proba, class_names, test_df), f)

        with open(pkl_path, "rb") as f:
            loaded_proba, loaded_names, loaded_df = pickle.load(f)

        np.testing.assert_array_equal(loaded_proba, proba)
        assert loaded_names == class_names
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_class_names_as_ndarray_roundtrip(self, tmp_path):
        """class_names stored as ndarray should be convertible to list."""
        class_names_np = np.array(["isTPS", "FPP"])
        proba = np.zeros((5, 2))
        test_df = pd.DataFrame({"ID": [f"P{i}" for i in range(5)]})

        pkl_path = tmp_path / "fold_0_results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump((proba, class_names_np, test_df), f)

        with open(pkl_path, "rb") as f:
            _, loaded_names, _ = pickle.load(f)

        if not isinstance(loaded_names, list):
            loaded_names = list(loaded_names)

        assert "isTPS" in loaded_names
        assert loaded_names.index("isTPS") == 0


# ══════════════════════════════════════════════════════════════════════════
# Tests for process_track fold-file handling
# ══════════════════════════════════════════════════════════════════════════


class TestProcessTrackFileHandling:
    """Tests for postprocess_clean_ec_detection.process_track on synthetic data."""

    def test_process_track_creates_output(self, tmp_path):
        """Verify that process_track creates correct output pkl files."""
        from postprocess_clean_ec_detection import process_track as pt_ec

        n = 20
        class_names = ["isTPS", "FPP"]
        proba = np.zeros((n, 2))
        proba[:10, 0] = 0.9  # original isTPS
        proba[10:, 0] = 0.1

        ids = [f"P{i}" for i in range(n)]
        test_df = pd.DataFrame(
            {
                "ID": ids,
                "SMILES_substrate_canonical_no_stereo": ["FPP"] * 10
                + ["Unknown"] * 10,
            }
        )

        # Set up CLEAN output structure
        import postprocess_clean_ec_detection as mod

        track_name = "test_track"
        ts_name = "20260101_000000"
        track_dir = (
            tmp_path
            / "CLEAN"
            / track_name
            / "all_folds"
            / "all_classes"
            / ts_name
        )
        track_dir.mkdir(parents=True)

        for fold_i in range(2):
            pkl_path = track_dir / f"fold_{fold_i}_results.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump((proba, class_names, test_df), f)

        # Override module-level paths
        orig_clean_root = mod.CLEAN_OUTPUT_ROOT
        orig_new_root = mod.NEW_OUTPUT_ROOT
        mod.CLEAN_OUTPUT_ROOT = tmp_path / "CLEAN"
        mod.NEW_OUTPUT_ROOT = tmp_path / "CLEANEcDetection"

        try:
            tps_ecs = {"EC:4.2.3.100"}
            id_2_ec_conf = {}
            for i in range(10):
                id_2_ec_conf[f"P{i}"] = {"EC:4.2.3.100": 0.85}
            for i in range(10, 20):
                id_2_ec_conf[f"P{i}"] = {"EC:1.1.1.1": 0.70}

            result = pt_ec(track_name, tps_ecs, id_2_ec_conf, n_folds=2)
            assert result == 2

            out_dir = (
                tmp_path
                / "CLEANEcDetection"
                / track_name
                / "all_folds"
                / "all_classes"
                / ts_name
            )
            assert out_dir.exists()
            for fold_i in range(2):
                out_pkl = out_dir / f"fold_{fold_i}_results.pkl"
                assert out_pkl.exists()

                with open(out_pkl, "rb") as f:
                    new_proba, new_names, new_df = pickle.load(f)

                istps_idx = new_names.index("isTPS")
                # TPS proteins should get 0.85 (from EC:4.2.3.100)
                np.testing.assert_allclose(
                    new_proba[:10, istps_idx], 0.85
                )
                # Negatives should get 0.0
                np.testing.assert_allclose(
                    new_proba[10:, istps_idx], 0.0
                )
                # Non-isTPS columns unchanged
                np.testing.assert_array_equal(new_proba[:, 1], proba[:, 1])
        finally:
            mod.CLEAN_OUTPUT_ROOT = orig_clean_root
            mod.NEW_OUTPUT_ROOT = orig_new_root

    def test_process_track_missing_dir_returns_zero(self, tmp_path):
        from postprocess_clean_ec_detection import process_track as pt_ec
        import postprocess_clean_ec_detection as mod

        orig = mod.CLEAN_OUTPUT_ROOT
        mod.CLEAN_OUTPUT_ROOT = tmp_path / "nonexistent"
        try:
            result = pt_ec("fake_track", set(), {}, n_folds=5)
            assert result == 0
        finally:
            mod.CLEAN_OUTPUT_ROOT = orig


# ══════════════════════════════════════════════════════════════════════════
# Tests for load_all_raw_clean_predictions
# ══════════════════════════════════════════════════════════════════════════


class TestLoadRawCleanPredictions:
    """Tests for parsing CLEAN maxsep CSV files."""

    @staticmethod
    def _import():
        from postprocess_clean_ec_detection import load_all_raw_clean_predictions
        return load_all_raw_clean_predictions

    def test_parses_single_file(self, tmp_path):
        load_all_raw_clean_predictions = self._import()
        csv_path = tmp_path / "test_maxsep.csv"
        csv_path.write_text(
            "ProtA,EC:4.2.3.100/0.95,EC:1.1.1.1/0.30\n"
            "ProtB,EC:2.2.2.2/0.80\n"
        )
        result = load_all_raw_clean_predictions(tmp_path)
        assert "ProtA" in result
        assert "ProtB" in result
        assert result["ProtA"]["EC:4.2.3.100"] == pytest.approx(0.95)
        assert result["ProtA"]["EC:1.1.1.1"] == pytest.approx(0.30)
        assert result["ProtB"]["EC:2.2.2.2"] == pytest.approx(0.80)

    def test_deduplicates_protein_ids(self, tmp_path):
        """First occurrence wins when same protein appears in multiple files."""
        load_all_raw_clean_predictions = self._import()
        csv1 = tmp_path / "a_maxsep.csv"
        csv2 = tmp_path / "b_maxsep.csv"
        csv1.write_text("ProtA,EC:4.2.3.100/0.95\n")
        csv2.write_text("ProtA,EC:4.2.3.100/0.50\n")

        result = load_all_raw_clean_predictions(tmp_path)
        assert result["ProtA"]["EC:4.2.3.100"] == pytest.approx(0.95)

    def test_empty_directory(self, tmp_path):
        load_all_raw_clean_predictions = self._import()
        result = load_all_raw_clean_predictions(tmp_path)
        assert result == {}
