#!/usr/bin/env python3
"""Cross-negative stress test for Blastp models.

Measures how well models trained on one dataset's negatives handle the
other dataset's negatives.  All four cells of the evaluation matrix use
the same 1,035 shared TPS as positives, so any difference in metrics is
directly attributable to the negatives.

Rows:
  1. Old-trained model  +  OLD negatives  (baseline own-neg)
  2. Old-trained model  +  NEW negatives  (cross-neg stress)
  3. New-trained model  +  NEW negatives  (baseline own-neg)
  4. New-trained model  +  OLD negatives  (cross-neg stress)

Models are trained on ALL folds of their source dataset (no held-out fold).
"""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd  # type: ignore
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from enzymeexplorer.src.experiments_orchestration.experiment_runner import (  # noqa: E402
    _normalize_fold_column,
    assign_is_tps_label,
)
from enzymeexplorer.src.models.baselines.blastp import Blastp, BlastConfig  # noqa: E402
from enzymeexplorer.src.utils.project_info import ExperimentInfo  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_PRECURSOR_TYPES = frozenset({"ggpps", "fpps", "gpps", "gfpps", "hsqs"})
_TARGET_COL = "SMILES_substrate_canonical_no_stereo"
_CLASS_NAME = "isTPS"
_CLASS_NAMES = [
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "precursor substr",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
    "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    (
        "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
        "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    ),
    (
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    ),
    "isTPS",
]


def _load_dataset(
    csv_path: str,
    split_col: str,
    id_col: str,
    seq_col: str,
    type_col: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _normalize_fold_column(df, split_col)
    if type_col and type_col in df.columns:
        df.loc[df[type_col].isin(_PRECURSOR_TYPES), _TARGET_COL] = "precursor substr"
    return df


def _build_aggregated_df(
    raw_df: pd.DataFrame,
    id_col: str,
    seq_col: str,
    ignore_col: str | None,
) -> pd.DataFrame:
    sub = raw_df.copy()
    if ignore_col and ignore_col in sub.columns:
        sub.loc[sub[ignore_col] == 1, _TARGET_COL] = "other"
    grouped = sub.groupby(id_col)[_TARGET_COL].agg(set).reset_index()
    grouped[_TARGET_COL] = grouped[_TARGET_COL].map(assign_is_tps_label)
    id_seq = raw_df[[id_col, seq_col]].drop_duplicates(id_col)
    grouped = grouped.merge(id_seq, on=id_col)
    return grouped


def _get_shared_tps_seqs(old_df: pd.DataFrame, new_df: pd.DataFrame) -> set:
    old_tps = old_df[old_df[_TARGET_COL] != "Unknown"]
    new_tps = new_df[new_df[_TARGET_COL] != "Unknown"]
    return set(old_tps["Amino acid sequence"].unique()) & set(
        new_tps["Aminoacid_sequence"].unique()
    )


def _train_blastp(train_df: pd.DataFrame, id_col: str, seq_col: str) -> Blastp:
    experiment_info = ExperimentInfo(
        model_type="Blastp", model_version="cross_neg_stress"
    )
    config = BlastConfig(
        experiment_info=experiment_info,
        id_col_name=id_col,
        target_col_name=_TARGET_COL,
        split_col_name="dummy",
        class_names=_CLASS_NAMES,
        optimize_hyperparams=False,
        random_state=0,
        n_calls_hyperparams_opt=0,
        n_neighbours=1,
        e_threshold=0.001,
        seq_col_name=seq_col,
        neg_val="Unknown",
        negatives_sample_path="data/sampled_id_2_seq.pkl",
        tps_cleaned_csv_path="data/EnzymeExplorer_Dataset.csv",
        per_class_optimization=False,
        reuse_existing_partial_results=False,
        load_per_class_params_from="",
        n_jobs=64,
        pred_batch_size=32,
        hyperparam_dimensions=None,
    )
    model = Blastp(config=config)
    model.fit(train_df)
    return model


def _score(model: Blastp, test_df: pd.DataFrame) -> dict:
    proba = model.predict_proba(test_df)
    class_idx = _CLASS_NAMES.index(_CLASS_NAME)
    y_pred = proba[:, class_idx]
    y_true = test_df[_TARGET_COL].map(lambda x: _CLASS_NAME in x).values
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    y_bin = y_pred >= 0.5
    mcc = matthews_corrcoef(y_true, y_bin)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    fpr = ((y_bin == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
    return {
        "ap": ap,
        "auc": auc,
        "mcc": mcc,
        "f1": f1,
        "fpr": fpr,
        "n_pos": int(y_true.sum()),
        "n_neg": int((~y_true.astype(bool)).sum()),
    }


def run_stress_test(
    synced_csv: str = "data/TPS-Nov19_2023_with_synced_folds.csv",
    old_csv: str = "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    new_csv: str = "data/EnzymeExplorer_Dataset.csv",
) -> dict:
    # Load datasets
    old_df = _load_dataset(
        old_csv,
        "stratified_phylogeny_based_split_with_minor_products",
        "Uniprot ID",
        "Amino acid sequence",
        "Type (mono, sesq, di, …)",
    )
    new_df = _load_dataset(new_csv, "Fold", "ID", "Aminoacid_sequence", "Type")
    synced_df = _load_dataset(
        synced_csv, "synced_fold", "Uniprot ID", "Amino acid sequence", None
    )

    shared_tps_seqs = _get_shared_tps_seqs(old_df, new_df)
    logger.info("Shared TPS sequences: %d", len(shared_tps_seqs))

    # Build full datasets (all entries, including fold -1 negatives)
    old_all = _build_aggregated_df(
        old_df,
        "Uniprot ID",
        "Amino acid sequence",
        "ignore_in_eval",
    )
    new_all = _build_aggregated_df(
        new_df, "ID", "Aminoacid_sequence", "Fold_ignore_in_eval"
    )

    # Shared TPS test set (from old dataset IDs)
    shared_tps_old = old_all[
        old_all["Amino acid sequence"].isin(shared_tps_seqs)
        & old_all[_TARGET_COL].map(lambda s: "isTPS" in s)
    ].copy()
    # Shared TPS test set (from new dataset IDs)
    shared_tps_new = new_all[
        new_all["Aminoacid_sequence"].isin(shared_tps_seqs)
        & new_all[_TARGET_COL].map(lambda s: "isTPS" in s)
    ].copy()

    # Negatives
    old_neg = old_all[old_all[_TARGET_COL].map(lambda s: "isTPS" not in s)].copy()
    new_neg = new_all[new_all[_TARGET_COL].map(lambda s: "isTPS" not in s)].copy()

    logger.info(
        "Shared TPS (old IDs): %d, (new IDs): %d",
        len(shared_tps_old),
        len(shared_tps_new),
    )
    logger.info("Old negatives: %d, New negatives: %d", len(old_neg), len(new_neg))

    # Train models
    logger.info("Training old-data model (synced all folds)...")
    model_old = _train_blastp(
        old_all.rename(
            columns={"Uniprot ID": "_id", "Amino acid sequence": "_seq"}
        ).rename(columns={"_id": "Uniprot ID", "_seq": "Amino acid sequence"}),
        "Uniprot ID",
        "Amino acid sequence",
    )

    logger.info("Training new-data model (all folds)...")
    model_new = _train_blastp(new_all, "ID", "Aminoacid_sequence")

    # Build test sets
    # Row 1: Old model + OLD negatives
    test_1 = pd.concat([shared_tps_old, old_neg], ignore_index=True)
    # Row 2: Old model + NEW negatives (rename cols to match old model expectation)
    new_neg_renamed = new_neg.rename(
        columns={"ID": "Uniprot ID", "Aminoacid_sequence": "Amino acid sequence"}
    )
    test_2 = pd.concat([shared_tps_old, new_neg_renamed], ignore_index=True)
    # Row 3: New model + NEW negatives
    shared_tps_new_renamed = shared_tps_old.rename(
        columns={"Uniprot ID": "ID", "Amino acid sequence": "Aminoacid_sequence"}
    )
    test_3 = pd.concat([shared_tps_new_renamed, new_neg], ignore_index=True)
    # Row 4: New model + OLD negatives
    old_neg_renamed = old_neg.rename(
        columns={"Uniprot ID": "ID", "Amino acid sequence": "Aminoacid_sequence"}
    )
    test_4 = pd.concat([shared_tps_new_renamed, old_neg_renamed], ignore_index=True)

    results = {}
    for label, model, test_df in [
        ("old_model_+_old_neg", model_old, test_1),
        ("old_model_+_new_neg", model_old, test_2),
        ("new_model_+_new_neg", model_new, test_3),
        ("new_model_+_old_neg", model_new, test_4),
    ]:
        logger.info("Scoring: %s (%d samples)...", label, len(test_df))
        metrics = _score(model, test_df)
        results[label] = metrics
        logger.info(
            "  %s: AP=%.4f AUC=%.4f MCC=%.4f F1=%.4f FPR=%.4f (pos=%d neg=%d)",
            label,
            metrics["ap"],
            metrics["auc"],
            metrics["mcc"],
            metrics["f1"],
            metrics["fpr"],
            metrics["n_pos"],
            metrics["n_neg"],
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synced-csv",
        default="data/TPS-Nov19_2023_with_synced_folds.csv",
    )
    parser.add_argument(
        "--old-csv",
        default="data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    )
    parser.add_argument(
        "--new-csv",
        default="data/EnzymeExplorer_Dataset.csv",
    )
    args = parser.parse_args()
    results = run_stress_test(
        synced_csv=args.synced_csv,
        old_csv=args.old_csv,
        new_csv=args.new_csv,
    )

    print("\n" + "=" * 80)
    print("CROSS-NEGATIVE STRESS TEST RESULTS")
    print("=" * 80)
    print(
        f"{'Row':<35} {'AP':>7} {'AUC':>7} {'MCC':>7} {'F1':>7} {'FPR':>7} {'pos':>6} {'neg':>6}"
    )
    print("-" * 80)
    for label, m in results.items():
        print(
            f"{label:<35} {m['ap']:7.4f} {m['auc']:7.4f} {m['mcc']:7.4f} "
            f"{m['f1']:7.4f} {m['fpr']:7.4f} {m['n_pos']:6d} {m['n_neg']:6d}"
        )

    # Save results
    out_path = Path("outputs/cross_negative_stress_test_results.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
