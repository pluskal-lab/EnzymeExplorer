#!/usr/bin/env python3
"""Build a synchronized-fold dataset for cross-dataset comparison.

This script takes the **old** phylo dataset and remaps fold assignments so
that **shared TPS sequences** (present in both old and new datasets) receive
the fold assignments from the **new** (EnzymeExplorer) dataset.  The result
is a CSV that can be used in native-CV evaluation with the existing
experiment runner, enabling direct fold-by-fold comparison with the new
dataset evaluation.

Approach (Approach B from the evaluation protocol):
  - Shared positive sequences: receive the new dataset's fold assignment.
  - Old-only positive sequences (128 low-quality entries): excluded.
  - Old negatives: **keep their original fold assignments** (all negatives
    are distributed across folds 0-4 in the source phylo CSV).
  - A companion ``_ignore_in_eval`` column is preserved from the original
    dataset.

Output CSV has the same schema as the old dataset, with an additional
``synced_fold`` column.  The experiment configs point
``split_col_name`` at this new column.
"""

import argparse
import logging

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_OLD_FOLD_COL = "stratified_phylogeny_based_split_with_minor_products"
_SYNCED_FOLD_COL = "synced_fold"
_NEG_LABEL = "Unknown"


def _build_seq_to_new_fold(new_csv: str) -> dict[str, int]:
    """Map sequence -> fold from the new dataset."""
    new_df = pd.read_csv(new_csv)
    # Drop duplicates: one fold per sequence
    deduped = new_df.drop_duplicates(subset=["Aminoacid_sequence"])
    return dict(zip(deduped["Aminoacid_sequence"], deduped["Fold"].astype(int)))


def build_synced_dataset(
    old_csv: str,
    new_csv: str,
    output_csv: str,
    *,
    exclude_old_only_tps: bool = True,
) -> pd.DataFrame:
    """Build the synchronized-fold dataset.

    Returns the resulting DataFrame (also written to *output_csv*).
    """
    old_df = pd.read_csv(old_csv)
    seq_to_new_fold = _build_seq_to_new_fold(new_csv)

    old_seq_col = "Amino acid sequence"
    target_col = "SMILES_substrate_canonical_no_stereo"

    is_neg = old_df[target_col] == _NEG_LABEL
    is_tps = ~is_neg

    # --- Shared TPS: remap folds ---
    old_df["_new_fold"] = old_df[old_seq_col].map(seq_to_new_fold)
    shared_tps_mask = is_tps & old_df["_new_fold"].notna()
    old_only_tps_mask = is_tps & old_df["_new_fold"].isna()

    logger.info(
        "Shared TPS sequences: %d rows (%d unique seqs)",
        shared_tps_mask.sum(),
        old_df.loc[shared_tps_mask, old_seq_col].nunique(),
    )
    logger.info(
        "Old-only TPS sequences: %d rows (%d unique seqs)",
        old_only_tps_mask.sum(),
        old_df.loc[old_only_tps_mask, old_seq_col].nunique(),
    )

    if exclude_old_only_tps:
        old_df = old_df[~old_only_tps_mask].copy()
        logger.info("Excluded old-only TPS. Remaining rows: %d", len(old_df))
        # Recompute masks after dropping
        is_neg = old_df[target_col] == _NEG_LABEL
        shared_tps_mask = ~is_neg

    # Assign synced fold for shared TPS
    old_df[_SYNCED_FOLD_COL] = pd.array([pd.NA] * len(old_df), dtype="string")

    old_df.loc[shared_tps_mask, _SYNCED_FOLD_COL] = "fold_" + old_df.loc[
        shared_tps_mask, "_new_fold"
    ].astype(int).astype(str)

    # --- Negatives: keep original fold assignments ---
    # All negatives should already be in fold_0..fold_4 (redistributed).
    logger.info("Negatives total: %d", is_neg.sum())
    old_df.loc[is_neg, _SYNCED_FOLD_COL] = old_df.loc[is_neg, _OLD_FOLD_COL]

    # --- ignore_in_eval column ---
    ignore_col = f"{_SYNCED_FOLD_COL}_ignore_in_eval"
    orig_ignore_col = f"{_OLD_FOLD_COL}_ignore_in_eval"
    if orig_ignore_col in old_df.columns:
        old_df[ignore_col] = old_df[orig_ignore_col]
    else:
        old_df[ignore_col] = np.nan

    # Clean up temp column
    old_df.drop(columns=["_new_fold"], inplace=True, errors="ignore")

    # --- Summary ---
    fold_counts = old_df[_SYNCED_FOLD_COL].value_counts().sort_index()
    logger.info("Synced fold distribution:\n%s", fold_counts.to_string())
    tps_fold = (
        old_df[old_df[target_col] != _NEG_LABEL][_SYNCED_FOLD_COL]
        .value_counts()
        .sort_index()
    )
    neg_fold = (
        old_df[old_df[target_col] == _NEG_LABEL][_SYNCED_FOLD_COL]
        .value_counts()
        .sort_index()
    )
    logger.info("TPS per fold:\n%s", tps_fold.to_string())
    logger.info("Negatives per fold:\n%s", neg_fold.to_string())

    unassigned = old_df[_SYNCED_FOLD_COL].isna().sum()
    if unassigned:
        logger.warning("%d rows have no synced fold assignment!", unassigned)

    old_df.to_csv(output_csv, index=False)
    logger.info("Wrote synced dataset to %s (%d rows)", output_csv, len(old_df))
    return old_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-csv",
        default="data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
        help="Path to the old phylo-fold dataset CSV",
    )
    parser.add_argument(
        "--new-csv",
        default="data/EnzymeExplorer_Dataset.csv",
        help="Path to the new EnzymeExplorer dataset CSV",
    )
    parser.add_argument(
        "--output-csv",
        default="data/TPS-Nov19_2023_with_synced_folds.csv",
        help="Output path for the synchronized dataset",
    )
    parser.add_argument(
        "--keep-old-only-tps",
        action="store_true",
        help="Keep old-only TPS instead of excluding them",
    )
    args = parser.parse_args()
    build_synced_dataset(
        args.old_csv,
        args.new_csv,
        args.output_csv,
        exclude_old_only_tps=not args.keep_old_only_tps,
    )


if __name__ == "__main__":
    main()
