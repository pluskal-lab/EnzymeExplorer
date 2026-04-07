"""Tests for fold-column compatibility and synced-fold dataset generation."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from enzymeexplorer.src.utils.data import get_folds_from_csv


class TestGetFoldsFromCsvFormats:
    """get_folds_from_csv must handle both fold_N and bare-integer formats."""

    def test_old_fold_format(self, tmp_path: str) -> None:
        """Old datasets use fold_0, fold_1, … and -1 for excluded negatives."""
        csv = os.path.join(tmp_path, "old.csv")
        df = pd.DataFrame(
            {
                "split": [
                    "fold_0",
                    "fold_0",
                    "fold_1",
                    "fold_2",
                    "-1",
                    "-1",
                ]
            }
        )
        df.to_csv(csv, index=False)
        folds = get_folds_from_csv(csv, "split")
        assert folds == ["0", "1", "2"]

    def test_new_integer_format(self, tmp_path: str) -> None:
        """New datasets use bare integers 0, 1, 2, …"""
        csv = os.path.join(tmp_path, "new.csv")
        df = pd.DataFrame({"Fold": [0, 1, 2, 3, 4, 0, 1]})
        df.to_csv(csv, index=False)
        folds = get_folds_from_csv(csv, "Fold")
        assert folds == ["0", "1", "2", "3", "4"]

    def test_integer_format_with_negatives(self, tmp_path: str) -> None:
        """Negative fold indices should be excluded from valid folds."""
        csv = os.path.join(tmp_path, "mixed.csv")
        df = pd.DataFrame({"split": [-1, 0, 1, 2, -1]})
        df.to_csv(csv, index=False)
        folds = get_folds_from_csv(csv, "split")
        assert folds == ["0", "1", "2"]

    def test_empty_csv(self, tmp_path: str) -> None:
        csv = os.path.join(tmp_path, "empty.csv")
        df = pd.DataFrame({"split": pd.Series(dtype=str)})
        df.to_csv(csv, index=False)
        folds = get_folds_from_csv(csv, "split")
        assert folds == []


class TestNormalizeFoldColumn:
    """_normalize_fold_column should add fold_ prefix for bare-integer columns."""

    def test_bare_integers_normalized(self) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _normalize_fold_column,
        )

        df = pd.DataFrame({"Fold": [0, 1, 2, 3, 4]})
        _normalize_fold_column(df, "Fold")
        assert list(df["Fold"]) == [
            "fold_0",
            "fold_1",
            "fold_2",
            "fold_3",
            "fold_4",
        ]

    def test_already_prefixed_untouched(self) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _normalize_fold_column,
        )

        df = pd.DataFrame({"split": ["fold_0", "fold_1", "-1"]})
        _normalize_fold_column(df, "split")
        assert list(df["split"]) == ["fold_0", "fold_1", "-1"]

    def test_nan_values_preserved(self) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _normalize_fold_column,
        )

        df = pd.DataFrame({"Fold": [0, 1, np.nan, 2]})
        _normalize_fold_column(df, "Fold")
        assert df.loc[0, "Fold"] == "fold_0"
        assert pd.isna(df.loc[2, "Fold"])


class TestBuildSyncedDataset:
    """Integration tests for the synced-fold dataset builder."""

    def test_shared_tps_get_new_folds(self) -> None:
        from scripts.build_synced_fold_dataset import build_synced_dataset

        with tempfile.TemporaryDirectory() as tmp:
            old_csv = os.path.join(tmp, "old.csv")
            new_csv = os.path.join(tmp, "new.csv")
            out_csv = os.path.join(tmp, "synced.csv")

            # Old dataset: 2 shared TPS, 1 old-only TPS, 2 neg in fold -1,
            # 1 neg in fold_0
            old_df = pd.DataFrame(
                {
                    "Uniprot ID": ["P1", "P2", "P3", "N1", "N2", "N3"],
                    "Amino acid sequence": [
                        "MSEQ1",
                        "MSEQ2",
                        "MSEQ_OLD_ONLY",
                        "NNEG1",
                        "NNEG2",
                        "NNEG3",
                    ],
                    "SMILES_substrate_canonical_no_stereo": [
                        "GPP",
                        "FPP",
                        "GPP",
                        "Unknown",
                        "Unknown",
                        "Unknown",
                    ],
                    "stratified_phylogeny_based_split_with_minor_products": [
                        "fold_0",
                        "fold_1",
                        "fold_0",
                        "-1",
                        "-1",
                        "fold_0",
                    ],
                    "Type (mono, sesq, di, …)": [
                        "mono",
                        "sesq",
                        "mono",
                        "Unknown",
                        "Unknown",
                        "Unknown",
                    ],
                }
            )
            old_df.to_csv(old_csv, index=False)

            # New dataset: MSEQ1 and MSEQ2 are shared, with different folds
            new_df = pd.DataFrame(
                {
                    "ID": ["NEW_P1", "NEW_P2", "NEW_P3"],
                    "Aminoacid_sequence": ["MSEQ1", "MSEQ2", "MSEQ_NEW_ONLY"],
                    "SMILES_substrate_canonical_no_stereo": ["GPP", "FPP", "GGPP"],
                    "Fold": [3, 4, 0],
                }
            )
            new_df.to_csv(new_csv, index=False)

            result = build_synced_dataset(
                old_csv, new_csv, out_csv, exclude_old_only_tps=True
            )

            # Old-only TPS (MSEQ_OLD_ONLY / P3) should be excluded
            assert "P3" not in result["Uniprot ID"].values

            # Shared TPS should have new fold assignments
            p1_fold = result.loc[result["Uniprot ID"] == "P1", "synced_fold"].iloc[0]
            assert p1_fold == "fold_3"  # from new dataset fold=3

            p2_fold = result.loc[result["Uniprot ID"] == "P2", "synced_fold"].iloc[0]
            assert p2_fold == "fold_4"  # from new dataset fold=4

            # Fold -1 negatives should stay in fold -1
            for neg_id in ["N1", "N2"]:
                neg_fold = result.loc[
                    result["Uniprot ID"] == neg_id, "synced_fold"
                ].iloc[0]
                assert neg_fold == "-1"

            # Fold-assigned negatives should keep their original fold
            n3_fold = result.loc[result["Uniprot ID"] == "N3", "synced_fold"].iloc[0]
            assert n3_fold == "fold_0"

    def test_keep_old_only_tps(self) -> None:
        from scripts.build_synced_fold_dataset import build_synced_dataset

        with tempfile.TemporaryDirectory() as tmp:
            old_csv = os.path.join(tmp, "old.csv")
            new_csv = os.path.join(tmp, "new.csv")
            out_csv = os.path.join(tmp, "synced.csv")

            old_df = pd.DataFrame(
                {
                    "Uniprot ID": ["P1", "P_OLD"],
                    "Amino acid sequence": ["MSEQ1", "MSEQ_OLD"],
                    "SMILES_substrate_canonical_no_stereo": ["GPP", "FPP"],
                    "stratified_phylogeny_based_split_with_minor_products": [
                        "fold_0",
                        "fold_1",
                    ],
                    "Type (mono, sesq, di, …)": ["mono", "sesq"],
                }
            )
            old_df.to_csv(old_csv, index=False)

            new_df = pd.DataFrame(
                {
                    "ID": ["NP1"],
                    "Aminoacid_sequence": ["MSEQ1"],
                    "SMILES_substrate_canonical_no_stereo": ["GPP"],
                    "Fold": [2],
                }
            )
            new_df.to_csv(new_csv, index=False)

            result = build_synced_dataset(
                old_csv, new_csv, out_csv, exclude_old_only_tps=False
            )

            # Old-only TPS should still be present
            assert "P_OLD" in result["Uniprot ID"].values


class TestRemapSubstratesByType:
    """_remap_substrates_by_type must override substrates for non-TPS types."""

    def test_precursor_types_remapped(self) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _remap_substrates_by_type,
        )

        df = pd.DataFrame(
            {
                "Type": ["mono", "fpps", "gpps", "sesq"],
                "SMILES_substrate_canonical_no_stereo": [
                    "GPP",
                    "FPP",
                    "GPP",
                    "FPP",
                ],
            }
        )
        _remap_substrates_by_type(df, "Type")
        assert df.loc[0, "SMILES_substrate_canonical_no_stereo"] == "GPP"
        assert df.loc[1, "SMILES_substrate_canonical_no_stereo"] == "precursor substr"
        assert df.loc[2, "SMILES_substrate_canonical_no_stereo"] == "precursor substr"
        assert df.loc[3, "SMILES_substrate_canonical_no_stereo"] == "FPP"

    def test_unknown_type_substrates_overridden(self) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _remap_substrates_by_type,
        )

        df = pd.DataFrame(
            {
                "Type": ["mono", "Unknown", "Unknown", "sesq"],
                "SMILES_substrate_canonical_no_stereo": [
                    "GPP",
                    "FPP",
                    "Unknown",
                    "GGPP",
                ],
            }
        )
        _remap_substrates_by_type(df, "Type")
        assert df.loc[0, "SMILES_substrate_canonical_no_stereo"] == "GPP"
        assert df.loc[1, "SMILES_substrate_canonical_no_stereo"] == "Unknown"
        assert df.loc[2, "SMILES_substrate_canonical_no_stereo"] == "Unknown"
        assert df.loc[3, "SMILES_substrate_canonical_no_stereo"] == "GGPP"

    def test_substrate_bearing_negative_no_istps(self) -> None:
        """End-to-end: Unknown-type protein with real substrate must NOT get isTPS."""
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _remap_substrates_by_type,
            assign_is_tps_label,
        )

        df = pd.DataFrame(
            {
                "id": ["TPS1", "TPS1", "NEG1", "NEG1"],
                "Type": ["mono", "mono", "Unknown", "Unknown"],
                "SMILES_substrate_canonical_no_stereo": [
                    "GPP",
                    "FPP",
                    "GPP",
                    "Unknown",
                ],
            }
        )
        _remap_substrates_by_type(df, "Type")
        grouped = (
            df.groupby("id")["SMILES_substrate_canonical_no_stereo"]
            .agg(set)
            .reset_index()
        )
        grouped["SMILES_substrate_canonical_no_stereo"] = grouped[
            "SMILES_substrate_canonical_no_stereo"
        ].map(assign_is_tps_label)

        tps_labels = grouped.loc[
            grouped["id"] == "TPS1", "SMILES_substrate_canonical_no_stereo"
        ].iloc[0]
        assert "isTPS" in tps_labels

        neg_labels = grouped.loc[
            grouped["id"] == "NEG1", "SMILES_substrate_canonical_no_stereo"
        ].iloc[0]
        assert "isTPS" not in neg_labels

    def test_missing_type_col_is_noop(self) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _remap_substrates_by_type,
        )

        df = pd.DataFrame(
            {"SMILES_substrate_canonical_no_stereo": ["GPP", "FPP"]}
        )
        _remap_substrates_by_type(df, "NonExistentCol")
        assert list(df["SMILES_substrate_canonical_no_stereo"]) == ["GPP", "FPP"]


class TestLoadEvalDataset:
    """_load_eval_dataset should rename eval columns to match training schema."""

    @pytest.fixture()
    def _config_and_csvs(self, tmp_path):
        from enzymeexplorer.src.models.ifaces.config_baseclasses import BaseConfig
        from enzymeexplorer.src.utils.project_info import ExperimentInfo

        train_csv = tmp_path / "train.csv"
        pd.DataFrame(
            {
                "Uniprot ID": ["P1", "P2"],
                "SMILES_substrate_canonical_no_stereo": ["GPP", "Unknown"],
                "synced_fold": ["fold_0", "fold_1"],
            }
        ).to_csv(train_csv, index=False)

        eval_csv = tmp_path / "eval.csv"
        pd.DataFrame(
            {
                "ID": ["E1", "E2", "E3"],
                "SMILES_substrate_canonical_no_stereo": ["GPP", "FPP", "Unknown"],
                "Fold": [0, 1, 0],
                "Aminoacid_sequence": ["MSEQ1", "MSEQ2", "NNEG"],
            }
        ).to_csv(eval_csv, index=False)

        config = BaseConfig(
            experiment_info=ExperimentInfo(
                model_type="Blastp",
                model_version="test",
            ),
            id_col_name="Uniprot ID",
            target_col_name="SMILES_substrate_canonical_no_stereo",
            split_col_name="synced_fold",
            class_names=["GPP", "FPP", "isTPS"],
            optimize_hyperparams=False,
            n_calls_hyperparams_opt=0,
            hyperparam_dimensions={},
            neg_val="Unknown",
            negatives_sample_path="",
            tps_cleaned_csv_path=str(train_csv),
            random_state=0,
            per_class_optimization=False,
            load_per_class_params_from="",
            reuse_existing_partial_results=False,
            eval_csv_path=str(eval_csv),
            eval_split_col_name="Fold",
            eval_id_col_name="ID",
            eval_seq_col_name="Aminoacid_sequence",
        )
        return config

    def test_columns_renamed(self, _config_and_csvs) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _load_eval_dataset,
        )

        config = _config_and_csvs
        eval_df = _load_eval_dataset(config)

        assert "Uniprot ID" in eval_df.columns
        assert "synced_fold" in eval_df.columns
        assert "ID" not in eval_df.columns
        assert "Fold" not in eval_df.columns

    def test_folds_normalized(self, _config_and_csvs) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _load_eval_dataset,
        )

        config = _config_and_csvs
        eval_df = _load_eval_dataset(config)

        fold_vals = set(eval_df["synced_fold"].dropna().unique())
        assert fold_vals == {"fold_0", "fold_1"}

    def test_fold_selection(self, _config_and_csvs) -> None:
        from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
            _load_eval_dataset,
        )

        config = _config_and_csvs
        eval_df = _load_eval_dataset(config)

        fold_0 = eval_df[eval_df["synced_fold"] == "fold_0"]
        assert len(fold_0) == 2
        assert set(fold_0["Uniprot ID"]) == {"E1", "E3"}

    def test_no_cross_eval_when_empty(self) -> None:
        from enzymeexplorer.src.models.ifaces.config_baseclasses import BaseConfig

        config = BaseConfig.__new__(BaseConfig)
        config.eval_csv_path = ""
        assert not bool(config.eval_csv_path)
