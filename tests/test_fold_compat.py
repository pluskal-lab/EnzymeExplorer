"""Tests for fold-column compatibility and synced-fold dataset generation."""

import os
import tempfile

import numpy as np
import pandas as pd

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

            # Old dataset: 2 shared TPS, 1 old-only TPS, 2 negatives
            old_df = pd.DataFrame(
                {
                    "Uniprot ID": ["P1", "P2", "P3", "N1", "N2"],
                    "Amino acid sequence": [
                        "MSEQ1",
                        "MSEQ2",
                        "MSEQ_OLD_ONLY",
                        "NNEG1",
                        "NNEG2",
                    ],
                    "SMILES_substrate_canonical_no_stereo": [
                        "GPP",
                        "FPP",
                        "GPP",
                        "Unknown",
                        "Unknown",
                    ],
                    "stratified_phylogeny_based_split_with_minor_products": [
                        "fold_0",
                        "fold_1",
                        "fold_0",
                        "-1",
                        "-1",
                    ],
                    "Type (mono, sesq, di, …)": [
                        "mono",
                        "sesq",
                        "mono",
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
                old_csv, new_csv, out_csv, exclude_old_only_tps=True, n_folds=5
            )

            # Old-only TPS (MSEQ_OLD_ONLY / P3) should be excluded
            assert "P3" not in result["Uniprot ID"].values

            # Shared TPS should have new fold assignments
            p1_fold = result.loc[result["Uniprot ID"] == "P1", "synced_fold"].iloc[0]
            assert p1_fold == "fold_3"  # from new dataset fold=3

            p2_fold = result.loc[result["Uniprot ID"] == "P2", "synced_fold"].iloc[0]
            assert p2_fold == "fold_4"  # from new dataset fold=4

            # Negatives should have fold assignments (redistributed)
            for neg_id in ["N1", "N2"]:
                neg_fold = result.loc[
                    result["Uniprot ID"] == neg_id, "synced_fold"
                ].iloc[0]
                assert neg_fold.startswith("fold_")

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
