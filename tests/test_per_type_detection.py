"""Tests for per-TPS-type detection evaluation and analysis.

Covers:
- scripts/evaluate_per_type_tps_detection.py (compute_per_type_ap, macro-avg)
- scripts/analyze_per_type_tps_detection.py (get_types_for_fold)
- scripts/postprocess_clean_ec_detection.py (build_tps_ec_set, compute_ec_based_istps)
- scripts/postprocess_clean_tps_detection.py (compute_ptps_proportion)
- scripts/evaluate_new_models.py (compute_istps_ap, compute_substrate_map)
- scripts/extend_ec_mapping.py (collect_tps_substrates)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Imports from scripts ─────────────────────────────────────────────────
# We import the pure functions that don't require external data.

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from postprocess_clean_ec_detection import (  # noqa: E402
    build_tps_ec_set,
    compute_ec_based_istps,
)
from postprocess_clean_tps_detection import (  # noqa: E402
    compute_ptps_proportion,
)
from evaluate_per_type_tps_detection import (  # noqa: E402
    compute_per_type_ap,
    _get_type_for_row,
)
from analyze_per_type_tps_detection import (  # noqa: E402
    get_types_for_fold,
)
from evaluate_new_models import (  # noqa: E402
    compute_istps_ap,
    compute_substrate_map,
    SUBSTRATE_CLASSES,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

FPP = "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
GPP = "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"


def _make_fold_data(
    n_tps: int = 20,
    n_neg: int = 80,
    tps_score: float = 0.9,
    neg_score: float = 0.1,
    tps_types: list[str] | None = None,
    substrate: str = FPP,
    id_prefix: str = "P",
    fold_i: int = 0,
):
    """Create synthetic fold data (proba, class_names, test_df)."""
    if tps_types is None:
        tps_types = ["sesq"] * n_tps
    assert len(tps_types) == n_tps

    n = n_tps + n_neg
    class_names = ["isTPS", substrate, "precursor substr"]
    proba = np.zeros((n, len(class_names)))

    # isTPS scores
    proba[:n_tps, 0] = tps_score
    proba[n_tps:, 0] = neg_score

    # substrate scores
    proba[:n_tps, 1] = 0.8
    proba[n_tps:, 1] = 0.05

    # precursor scores
    proba[:n_tps, 2] = 0.1
    proba[n_tps:, 2] = 0.02

    ids = [f"{id_prefix}{i}" for i in range(n)]
    targets = [substrate] * n_tps + ["Unknown"] * n_neg
    types = list(tps_types) + ["Unknown"] * n_neg

    test_df = pd.DataFrame(
        {
            "ID": ids,
            "SMILES_substrate_canonical_no_stereo": targets,
            "Type": types,
        }
    )
    return fold_i, (proba, class_names, test_df)


# ══════════════════════════════════════════════════════════════════════════
# Tests for compute_ec_based_istps
# ══════════════════════════════════════════════════════════════════════════


class TestComputeEcBasedIstps:
    """Tests for the EC-based isTPS scoring function."""

    def test_no_tps_ec_returns_zero(self):
        ec_confs = {"EC:1.1.1.1": 0.9, "EC:2.2.2.2": 0.8}
        tps_ecs = {"EC:3.3.3.3", "EC:4.4.4.4"}
        assert compute_ec_based_istps(ec_confs, tps_ecs) == 0.0

    def test_single_tps_ec(self):
        ec_confs = {"EC:1.1.1.1": 0.9, "EC:2.2.2.2": 0.8}
        tps_ecs = {"EC:1.1.1.1"}
        assert compute_ec_based_istps(ec_confs, tps_ecs) == 0.9

    def test_multiple_tps_ecs_returns_max(self):
        ec_confs = {"EC:1.1.1.1": 0.7, "EC:2.2.2.2": 0.9, "EC:3.3.3.3": 0.5}
        tps_ecs = {"EC:1.1.1.1", "EC:2.2.2.2"}
        assert compute_ec_based_istps(ec_confs, tps_ecs) == 0.9

    def test_empty_ec_confs_returns_zero(self):
        assert compute_ec_based_istps({}, {"EC:1.1.1.1"}) == 0.0

    def test_empty_tps_ecs_returns_zero(self):
        ec_confs = {"EC:1.1.1.1": 0.9}
        assert compute_ec_based_istps(ec_confs, set()) == 0.0

    def test_all_ecs_are_tps(self):
        ec_confs = {"EC:1.1.1.1": 0.6, "EC:2.2.2.2": 0.8}
        tps_ecs = {"EC:1.1.1.1", "EC:2.2.2.2"}
        assert compute_ec_based_istps(ec_confs, tps_ecs) == 0.8

    def test_confidence_preserved_exactly(self):
        ec_confs = {"EC:4.2.3.100": 0.12345}
        tps_ecs = {"EC:4.2.3.100"}
        assert compute_ec_based_istps(ec_confs, tps_ecs) == pytest.approx(
            0.12345
        )


# ══════════════════════════════════════════════════════════════════════════
# Tests for compute_ptps_proportion
# ══════════════════════════════════════════════════════════════════════════


class TestComputePtpsProportion:
    """Tests for the proportion-based P(TPS) scoring function."""

    def test_all_tps_returns_one(self):
        ec_confs = {"EC:1.1.1.1": 0.5, "EC:2.2.2.2": 0.5}
        tps_ecs = {"EC:1.1.1.1", "EC:2.2.2.2"}
        assert compute_ptps_proportion(ec_confs, tps_ecs) == pytest.approx(1.0)

    def test_no_tps_returns_zero(self):
        ec_confs = {"EC:1.1.1.1": 0.5, "EC:2.2.2.2": 0.5}
        tps_ecs = {"EC:3.3.3.3"}
        assert compute_ptps_proportion(ec_confs, tps_ecs) == pytest.approx(0.0)

    def test_half_tps_equal_confidence(self):
        ec_confs = {"EC:1.1.1.1": 0.5, "EC:2.2.2.2": 0.5}
        tps_ecs = {"EC:1.1.1.1"}
        assert compute_ptps_proportion(ec_confs, tps_ecs) == pytest.approx(0.5)

    def test_weighted_proportion(self):
        ec_confs = {"EC:1.1.1.1": 0.8, "EC:2.2.2.2": 0.2}
        tps_ecs = {"EC:1.1.1.1"}
        expected = 0.8 / (0.8 + 0.2)
        assert compute_ptps_proportion(ec_confs, tps_ecs) == pytest.approx(
            expected
        )

    def test_empty_ec_confs_returns_zero(self):
        assert compute_ptps_proportion({}, {"EC:1.1.1.1"}) == 0.0

    def test_zero_total_confidence_returns_zero(self):
        assert compute_ptps_proportion({"EC:1.1.1.1": 0.0}, {"EC:1.1.1.1"}) == 0.0

    def test_multiple_tps_ecs(self):
        ec_confs = {"EC:1.1.1.1": 0.3, "EC:2.2.2.2": 0.2, "EC:3.3.3.3": 0.5}
        tps_ecs = {"EC:1.1.1.1", "EC:2.2.2.2"}
        expected = (0.3 + 0.2) / (0.3 + 0.2 + 0.5)
        assert compute_ptps_proportion(ec_confs, tps_ecs) == pytest.approx(
            expected
        )


# ══════════════════════════════════════════════════════════════════════════
# Tests for build_tps_ec_set (from postprocess_clean_ec_detection)
# ══════════════════════════════════════════════════════════════════════════


class TestBuildTpsEcSet:
    """Tests for build_tps_ec_set using a temporary JSON mapping."""

    def test_basic_mapping(self, tmp_path):
        mapping = {
            "4.2.3.100": [FPP],
            "4.2.3.200": [GPP],
            "1.1.1.1": ["precursor substr"],
            "2.2.2.2": [],
        }
        mapping_path = tmp_path / "ec_map.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)

        result = build_tps_ec_set(mapping_path)
        assert "EC:4.2.3.100" in result
        assert "EC:4.2.3.200" in result
        assert "EC:1.1.1.1" not in result
        assert "EC:2.2.2.2" not in result

    def test_precursor_only_excluded(self, tmp_path):
        mapping = {"1.1.1.1": ["precursor substr"]}
        mapping_path = tmp_path / "ec_map.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)

        result = build_tps_ec_set(mapping_path)
        assert len(result) == 0

    def test_mixed_precursor_and_real(self, tmp_path):
        mapping = {"4.2.3.100": [FPP, "precursor substr"]}
        mapping_path = tmp_path / "ec_map.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)

        result = build_tps_ec_set(mapping_path)
        assert "EC:4.2.3.100" in result

    def test_empty_mapping(self, tmp_path):
        mapping_path = tmp_path / "ec_map.json"
        with open(mapping_path, "w") as f:
            json.dump({}, f)

        result = build_tps_ec_set(mapping_path)
        assert len(result) == 0


# ══════════════════════════════════════════════════════════════════════════
# Tests for _get_type_for_row
# ══════════════════════════════════════════════════════════════════════════


class TestGetTypeForRow:
    """Tests for the type extraction helper."""

    def test_normal_type(self):
        row = {"Type": "sesq"}
        assert _get_type_for_row(row, "Type", "SMILES") == "sesq"

    def test_unknown_returns_unknown(self):
        row = {"Type": "Unknown"}
        assert _get_type_for_row(row, "Type", "SMILES") == "Unknown"

    def test_nan_returns_unknown(self):
        row = {"Type": float("nan")}
        assert _get_type_for_row(row, "Type", "SMILES") == "Unknown"

    def test_empty_string_returns_unknown(self):
        row = {"Type": "  "}
        assert _get_type_for_row(row, "Type", "SMILES") == "Unknown"

    def test_case_normalized(self):
        row = {"Type": " Sesq "}
        assert _get_type_for_row(row, "Type", "SMILES") == "sesq"

    def test_missing_col_returns_unknown(self):
        row = {"OtherCol": "sesq"}
        assert _get_type_for_row(row, "Type", "SMILES") == "Unknown"


# ══════════════════════════════════════════════════════════════════════════
# Tests for get_types_for_fold (analyze script)
# ══════════════════════════════════════════════════════════════════════════


class TestGetTypesForFold:
    """Tests for the fold-level type extraction."""

    def test_uses_eval_df_mapping(self):
        test_df = pd.DataFrame({"ID": ["A", "B", "C"]})
        eval_df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Type": ["sesq", "mono", "Unknown"],
            }
        )
        result = get_types_for_fold(test_df, eval_df, "Type", "ID")
        assert list(result) == ["sesq", "mono", "unknown"]

    def test_missing_id_becomes_unknown(self):
        test_df = pd.DataFrame({"ID": ["A", "X"]})
        eval_df = pd.DataFrame(
            {"ID": ["A"], "Type": ["sesq"]}
        )
        result = get_types_for_fold(test_df, eval_df, "Type", "ID")
        assert result.iloc[0] == "sesq"
        assert result.iloc[1] == "unknown"

    def test_uses_test_df_type_column_when_no_eval(self):
        test_df = pd.DataFrame(
            {
                "ID": ["A", "B"],
                "Type": ["Sesq", "Unknown"],
            }
        )
        result = get_types_for_fold(test_df, None, "Type", "ID")
        assert list(result) == ["sesq", "unknown"]

    def test_falls_back_to_type_like_column(self):
        test_df = pd.DataFrame(
            {
                "ID": ["A"],
                "Type (mono, sesq, di, …)": ["di"],
            }
        )
        result = get_types_for_fold(
            test_df, None, "NotPresent", "ID"
        )
        assert result.iloc[0] == "di"


# ══════════════════════════════════════════════════════════════════════════
# Tests for compute_per_type_ap
# ══════════════════════════════════════════════════════════════════════════


class TestComputePerTypeAp:
    """Tests for per-type AP computation."""

    def test_perfect_separation(self):
        """TPS perfectly separated from negatives → AP = 1.0."""
        fold_data = _make_fold_data(
            n_tps=10,
            n_neg=50,
            tps_score=1.0,
            neg_score=0.0,
            tps_types=["sesq"] * 10,
        )
        result = compute_per_type_ap(
            [fold_data], type_col="Type"
        )
        assert "sesq" in result
        assert result["sesq"][0] == pytest.approx(1.0)

    def test_random_scores_ap_below_one(self):
        """Random scores → AP strictly below 1.0."""
        rng = np.random.RandomState(42)
        fold_i = 0
        n_tps, n_neg = 20, 80
        n = n_tps + n_neg
        class_names = ["isTPS", FPP]
        proba = np.zeros((n, 2))
        proba[:, 0] = rng.rand(n)  # random isTPS scores
        proba[:, 1] = rng.rand(n)

        test_df = pd.DataFrame(
            {
                "ID": [f"P{i}" for i in range(n)],
                "SMILES_substrate_canonical_no_stereo": [FPP] * n_tps
                + ["Unknown"] * n_neg,
                "Type": ["sesq"] * n_tps + ["Unknown"] * n_neg,
            }
        )

        result = compute_per_type_ap(
            [(fold_i, (proba, class_names, test_df))], type_col="Type"
        )
        assert "sesq" in result
        assert 0.0 < result["sesq"][0] < 1.0

    def test_multiple_types(self):
        """Two TPS types are evaluated separately."""
        fold_data = _make_fold_data(
            n_tps=20,
            n_neg=60,
            tps_score=0.95,
            neg_score=0.05,
            tps_types=["sesq"] * 10 + ["mono"] * 10,
        )
        result = compute_per_type_ap(
            [fold_data], type_col="Type"
        )
        assert "sesq" in result
        assert "mono" in result

    def test_min_count_filtering(self):
        """Types with fewer than MIN_TYPE_COUNT positives are excluded."""
        fold_data = _make_fold_data(
            n_tps=4,
            n_neg=50,
            tps_score=0.9,
            neg_score=0.1,
            tps_types=["sesq", "sesq", "rare", "rare"],
        )
        result = compute_per_type_ap(
            [fold_data], type_col="Type"
        )
        # "rare" has only 2 samples, below MIN_TYPE_COUNT=3
        assert "rare" not in result

    def test_multiple_folds_aggregated(self):
        """AP values from multiple folds are collected into a list."""
        folds = [
            _make_fold_data(
                n_tps=10,
                n_neg=50,
                tps_score=0.9,
                neg_score=0.1,
                tps_types=["sesq"] * 10,
                fold_i=i,
            )
            for i in range(3)
        ]
        result = compute_per_type_ap(folds, type_col="Type")
        assert "sesq" in result
        assert len(result["sesq"]) == 3

    def test_eval_dataset_overrides_test_df(self):
        """When eval_dataset is provided, types come from it."""
        fold_i, (proba, class_names, test_df) = _make_fold_data(
            n_tps=10, n_neg=40, tps_types=["sesq"] * 10
        )
        test_df["Type"] = "wrong_type"

        eval_df = pd.DataFrame(
            {
                "ID": test_df["ID"].values,
                "Type": ["correct"] * 10 + ["Unknown"] * 40,
            }
        )

        result = compute_per_type_ap(
            [(fold_i, (proba, class_names, test_df))],
            type_col="Type",
            eval_dataset=eval_df,
        )
        assert "correct" in result
        assert "wrong_type" not in result


# ══════════════════════════════════════════════════════════════════════════
# Tests for compute_istps_ap and compute_substrate_map (evaluate_new_models)
# ══════════════════════════════════════════════════════════════════════════


class TestComputeIstpsAp:
    """Tests for isTPS AP computation from evaluate_new_models."""

    def test_perfect_detection(self):
        _, data = _make_fold_data(
            n_tps=20, n_neg=80, tps_score=1.0, neg_score=0.0
        )
        proba, class_names, test_df = data
        test_df["SMILES_substrate_canonical_no_stereo"] = (
            [f"isTPS_{FPP}"] * 20 + ["Unknown"] * 80
        )
        aps = compute_istps_ap([(proba, class_names, test_df)])
        assert len(aps) == 1
        assert aps[0] == pytest.approx(1.0)

    def test_no_istps_class_returns_empty(self):
        _, data = _make_fold_data(n_tps=10, n_neg=40)
        proba, _, test_df = data
        class_names_no_tps = ["substrate1", "substrate2", "substrate3"]
        proba_trimmed = proba[:, :3]
        aps = compute_istps_ap(
            [(proba_trimmed, class_names_no_tps, test_df)]
        )
        assert aps == []

    def test_too_few_positives_skipped(self):
        _, data = _make_fold_data(
            n_tps=2, n_neg=80, tps_types=["sesq", "sesq"]
        )
        proba, class_names, test_df = data
        test_df["SMILES_substrate_canonical_no_stereo"] = (
            [f"isTPS_{FPP}"] * 2 + ["Unknown"] * 80
        )
        aps = compute_istps_ap([(proba, class_names, test_df)])
        assert aps == []

    def test_multiple_folds(self):
        fold_results = []
        for _ in range(5):
            _, data = _make_fold_data(
                n_tps=20, n_neg=80, tps_score=0.9, neg_score=0.1
            )
            proba, class_names, test_df = data
            test_df["SMILES_substrate_canonical_no_stereo"] = (
                [f"isTPS_{FPP}"] * 20 + ["Unknown"] * 80
            )
            fold_results.append((proba, class_names, test_df))
        aps = compute_istps_ap(fold_results)
        assert len(aps) == 5
        for ap in aps:
            assert 0.0 < ap <= 1.0


class TestComputeSubstrateMap:
    """Tests for substrate mAP computation."""

    def test_with_known_substrate(self):
        substrate = SUBSTRATE_CLASSES[0]  # FPP
        n_tps, n_neg = 20, 80
        n = n_tps + n_neg
        class_names = ["isTPS", substrate, "precursor substr"]
        proba = np.zeros((n, 3))
        proba[:n_tps, 1] = 0.9  # high for substrate
        proba[n_tps:, 1] = 0.05
        proba[:, 2] = 0.1

        test_df = pd.DataFrame(
            {
                "ID": [f"P{i}" for i in range(n)],
                "SMILES_substrate_canonical_no_stereo": [substrate] * n_tps
                + ["Unknown"] * n_neg,
            }
        )

        maps, class_aps = compute_substrate_map(
            [(proba, class_names, test_df)]
        )
        assert len(maps) == 1
        assert maps[0] > 0.5

    def test_no_matching_substrate_class(self):
        class_names = ["isTPS", "some_other_smiles"]
        proba = np.zeros((10, 2))
        test_df = pd.DataFrame(
            {
                "ID": [f"P{i}" for i in range(10)],
                "SMILES_substrate_canonical_no_stereo": ["Unknown"] * 10,
            }
        )
        maps, _ = compute_substrate_map([(proba, class_names, test_df)])
        assert maps == []


# ══════════════════════════════════════════════════════════════════════════
# Tests for extend_ec_mapping.collect_tps_substrates
# ══════════════════════════════════════════════════════════════════════════


class TestCollectTpsSubstrates:
    """Tests for TPS substrate collection from dataset CSVs."""

    def test_collects_from_csv(self, tmp_path):
        from extend_ec_mapping import collect_tps_substrates

        csv_path = tmp_path / "dataset.csv"
        df = pd.DataFrame(
            {
                "SMILES_substrate_canonical_no_stereo": [
                    FPP,
                    GPP,
                    "Unknown",
                    "Negative",
                    FPP,
                ],
            }
        )
        df.to_csv(csv_path, index=False)

        result = collect_tps_substrates([str(csv_path)])
        assert FPP in result
        assert GPP in result
        assert "Unknown" not in result
        assert "Negative" not in result
        assert len(result) == 2

    def test_empty_csv(self, tmp_path):
        from extend_ec_mapping import collect_tps_substrates

        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame(
            {"SMILES_substrate_canonical_no_stereo": ["Unknown", "Negative"]}
        )
        df.to_csv(csv_path, index=False)

        result = collect_tps_substrates([str(csv_path)])
        assert len(result) == 0

    def test_multiple_csvs_merged(self, tmp_path):
        from extend_ec_mapping import collect_tps_substrates

        csv1 = tmp_path / "d1.csv"
        csv2 = tmp_path / "d2.csv"
        pd.DataFrame(
            {"SMILES_substrate_canonical_no_stereo": [FPP]}
        ).to_csv(csv1, index=False)
        pd.DataFrame(
            {"SMILES_substrate_canonical_no_stereo": [GPP]}
        ).to_csv(csv2, index=False)

        result = collect_tps_substrates([str(csv1), str(csv2)])
        assert FPP in result
        assert GPP in result
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════
# Integration-style tests: end-to-end fold processing
# ══════════════════════════════════════════════════════════════════════════


class TestFoldProcessingIntegration:
    """Integration tests that create synthetic fold pkl files and process them."""

    def test_ec_detection_replaces_istps_column(self, tmp_path):
        """Verify that EC-based post-processing modifies only the isTPS column."""
        n_tps, n_neg = 10, 40
        _, data = _make_fold_data(n_tps=n_tps, n_neg=n_neg)
        proba, class_names, test_df = data

        original_substrate_scores = proba[:, 1].copy()

        tps_ecs = {"EC:4.2.3.100", "EC:4.2.3.200"}
        id_2_ec_conf = {}
        for i in range(n_tps):
            id_2_ec_conf[f"P{i}"] = {"EC:4.2.3.100": 0.95}
        for i in range(n_tps, n_tps + n_neg):
            id_2_ec_conf[f"P{i}"] = {"EC:1.1.1.1": 0.80}

        # Simulate the logic from process_track
        istps_idx = class_names.index("isTPS")
        new_istps = np.zeros(len(test_df))
        for i, pid in enumerate(test_df["ID"].values):
            if pid in id_2_ec_conf:
                new_istps[i] = compute_ec_based_istps(
                    id_2_ec_conf[pid], tps_ecs
                )

        new_proba = proba.copy()
        new_proba[:, istps_idx] = new_istps

        # TPS should get high scores, negatives should get 0
        assert all(new_proba[:n_tps, istps_idx] == 0.95)
        assert all(new_proba[n_tps:, istps_idx] == 0.0)

        # Substrate scores unchanged
        np.testing.assert_array_equal(
            new_proba[:, 1], original_substrate_scores
        )

    def test_proportion_detection_replaces_istps_column(self, tmp_path):
        """Verify proportion-based post-processing computes weighted P(TPS)."""
        n_tps, n_neg = 10, 40
        _, data = _make_fold_data(n_tps=n_tps, n_neg=n_neg)
        proba, class_names, test_df = data

        tps_ecs = {"EC:4.2.3.100"}
        id_2_ec_conf = {}
        for i in range(n_tps):
            id_2_ec_conf[f"P{i}"] = {
                "EC:4.2.3.100": 0.8,
                "EC:1.1.1.1": 0.2,
            }
        for i in range(n_tps, n_tps + n_neg):
            id_2_ec_conf[f"P{i}"] = {"EC:1.1.1.1": 0.9}

        new_istps = np.zeros(len(test_df))
        for i, pid in enumerate(test_df["ID"].values):
            if pid in id_2_ec_conf:
                new_istps[i] = compute_ptps_proportion(
                    id_2_ec_conf[pid], tps_ecs
                )

        # TPS: 0.8 / (0.8 + 0.2) = 0.8
        expected_tps_score = 0.8 / 1.0
        assert all(
            np.isclose(new_istps[:n_tps], expected_tps_score)
        )

        # Negatives: 0 / 0.9 = 0
        assert all(new_istps[n_tps:] == 0.0)

    def test_per_type_ap_with_pkl_roundtrip(self, tmp_path):
        """Create pkl files, load them, compute per-type AP."""
        folds = []
        for fold_i in range(3):
            _, data = _make_fold_data(
                n_tps=15,
                n_neg=50,
                tps_score=0.95,
                neg_score=0.05,
                tps_types=["sesq"] * 8 + ["mono"] * 7,
                fold_i=fold_i,
                id_prefix=f"F{fold_i}_P",
            )
            proba, class_names, test_df = data
            pkl_path = tmp_path / f"fold_{fold_i}_results.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump((proba, class_names, test_df), f)

            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)
            folds.append((fold_i, loaded))

        result = compute_per_type_ap(folds, type_col="Type")
        assert "sesq" in result
        assert "mono" in result
        assert len(result["sesq"]) == 3
        assert len(result["mono"]) == 3
        for ap in result["sesq"] + result["mono"]:
            assert 0.5 < ap <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# Edge case / regression tests
# ══════════════════════════════════════════════════════════════════════════


class TestPrenyltransferaseExclusion:
    """Prenyltransferases (pt, ggpps, fpps, etc.) must be excluded from
    per-type TPS detection and treated as negatives."""

    def test_pt_excluded_from_per_type_ap(self):
        """Proteins with type 'pt' must not appear as a TPS type."""
        fold_data = _make_fold_data(
            n_tps=20,
            n_neg=50,
            tps_score=0.9,
            neg_score=0.1,
            tps_types=["sesq"] * 10 + ["pt"] * 10,
        )
        result = compute_per_type_ap([fold_data], type_col="Type")
        assert "pt" not in result
        assert "sesq" in result

    def test_ggpps_fpps_excluded_from_per_type_ap(self):
        """Old-dataset precursor types are also excluded."""
        fold_data = _make_fold_data(
            n_tps=20,
            n_neg=50,
            tps_score=0.9,
            neg_score=0.1,
            tps_types=["sesq"] * 10 + ["ggpps"] * 5 + ["fpps"] * 5,
        )
        result = compute_per_type_ap([fold_data], type_col="Type")
        assert "ggpps" not in result
        assert "fpps" not in result
        assert "sesq" in result

    def test_pt_proteins_counted_as_negatives(self):
        """PT proteins should increase the negative pool, not be ignored."""
        fold_no_pt = _make_fold_data(
            n_tps=10,
            n_neg=50,
            tps_score=0.9,
            neg_score=0.1,
            tps_types=["sesq"] * 10,
        )
        # Add 10 "pt" proteins that look like TPS (high scores)
        fold_with_pt = _make_fold_data(
            n_tps=20,
            n_neg=50,
            tps_score=0.9,
            neg_score=0.1,
            tps_types=["sesq"] * 10 + ["pt"] * 10,
        )
        result_no_pt = compute_per_type_ap([fold_no_pt], type_col="Type")
        result_with_pt = compute_per_type_ap(
            [fold_with_pt], type_col="Type"
        )
        # Both should have sesq
        assert "sesq" in result_no_pt
        assert "sesq" in result_with_pt
        # pt should not appear
        assert "pt" not in result_with_pt

    def test_precursor_types_in_experiment_runner(self):
        """Verify _PRECURSOR_TYPES includes 'pt'."""
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _PRECURSOR_TYPES,
        )

        assert "pt" in _PRECURSOR_TYPES
        assert "ggpps" in _PRECURSOR_TYPES
        assert "fpps" in _PRECURSOR_TYPES
        assert "gpps" in _PRECURSOR_TYPES
        assert "gfpps" in _PRECURSOR_TYPES
        assert "hsqs" in _PRECURSOR_TYPES


class TestEdgeCases:
    """Regression and edge case tests."""

    def test_ec_based_istps_with_ec_prefix_matching(self):
        """Ensure EC: prefix is required for matching."""
        ec_confs = {"EC:4.2.3.100": 0.9}
        tps_ecs_with_prefix = {"EC:4.2.3.100"}
        tps_ecs_without_prefix = {"4.2.3.100"}

        assert compute_ec_based_istps(ec_confs, tps_ecs_with_prefix) == 0.9
        assert compute_ec_based_istps(ec_confs, tps_ecs_without_prefix) == 0.0

    def test_proportion_with_single_ec(self):
        """Single EC that is TPS → P(TPS) = 1.0."""
        ec_confs = {"EC:4.2.3.100": 0.95}
        tps_ecs = {"EC:4.2.3.100"}
        assert compute_ptps_proportion(ec_confs, tps_ecs) == pytest.approx(
            1.0
        )

    def test_proportion_with_single_non_tps_ec(self):
        """Single EC that is NOT TPS → P(TPS) = 0.0."""
        ec_confs = {"EC:1.1.1.1": 0.95}
        tps_ecs = {"EC:4.2.3.100"}
        assert compute_ptps_proportion(ec_confs, tps_ecs) == pytest.approx(
            0.0
        )

    def test_per_type_ap_with_no_negatives(self):
        """No negatives → type is skipped (n_neg < MIN_TYPE_COUNT)."""
        fold_i = 0
        n = 10
        class_names = ["isTPS", FPP]
        proba = np.ones((n, 2)) * 0.9
        test_df = pd.DataFrame(
            {
                "ID": [f"P{i}" for i in range(n)],
                "SMILES_substrate_canonical_no_stereo": [FPP] * n,
                "Type": ["sesq"] * n,
            }
        )
        result = compute_per_type_ap(
            [(fold_i, (proba, class_names, test_df))], type_col="Type"
        )
        assert result == {}

    def test_per_type_ap_all_unknown(self):
        """All proteins are Unknown → no types to evaluate."""
        fold_i = 0
        n = 20
        class_names = ["isTPS", FPP]
        proba = np.ones((n, 2)) * 0.5
        test_df = pd.DataFrame(
            {
                "ID": [f"P{i}" for i in range(n)],
                "SMILES_substrate_canonical_no_stereo": ["Unknown"] * n,
                "Type": ["Unknown"] * n,
            }
        )
        result = compute_per_type_ap(
            [(fold_i, (proba, class_names, test_df))], type_col="Type"
        )
        assert result == {}

    def test_class_names_as_ndarray(self):
        """class_names as numpy array (common in pkl files) should work."""
        fold_i = 0
        n_tps, n_neg = 10, 40
        class_names_np = np.array(["isTPS", FPP])
        proba = np.zeros((n_tps + n_neg, 2))
        proba[:n_tps, 0] = 0.9
        proba[n_tps:, 0] = 0.1

        test_df = pd.DataFrame(
            {
                "ID": [f"P{i}" for i in range(n_tps + n_neg)],
                "SMILES_substrate_canonical_no_stereo": [FPP] * n_tps
                + ["Unknown"] * n_neg,
                "Type": ["sesq"] * n_tps + ["Unknown"] * n_neg,
            }
        )

        result = compute_per_type_ap(
            [(fold_i, (proba, class_names_np, test_df))], type_col="Type"
        )
        assert "sesq" in result

    def test_uniprot_id_column_fallback(self):
        """test_df with 'Uniprot ID' instead of 'ID' should work."""
        fold_i = 0
        n_tps, n_neg = 10, 40
        class_names = ["isTPS", FPP]
        proba = np.zeros((n_tps + n_neg, 2))
        proba[:n_tps, 0] = 0.9

        test_df = pd.DataFrame(
            {
                "Uniprot ID": [f"P{i}" for i in range(n_tps + n_neg)],
                "SMILES_substrate_canonical_no_stereo": [FPP] * n_tps
                + ["Unknown"] * n_neg,
                "Type (mono, sesq, di, \u2026)": ["di"] * n_tps
                + ["Unknown"] * n_neg,
            }
        )

        result = compute_per_type_ap(
            [(fold_i, (proba, class_names, test_df))],
            type_col="Type (mono, sesq, di, \u2026)",
        )
        assert "di" in result
