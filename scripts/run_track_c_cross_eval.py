#!/usr/bin/env python3
"""Track C cross-evaluation: train on synced old dataset, test on new dataset.

For each fold *k* (0-4):
  - **Train** on the synced-fold old dataset (shared TPS from folds ≠ k,
    plus the sparse original old negatives from folds ≠ k).
  - **Test** on the new dataset's fold *k* (all TPS + all negatives).

Because shared TPS share fold assignments between the synced and new
datasets, the test TPS from the new dataset have never been seen during
training — no leakage.  The test negatives are almost entirely disjoint
(< 2 % sequence overlap, and the few overlapping negatives in the same
fold are negligible).

This experiment answers: *If the model is trained with sparse / "dummy"
negatives from the old pipeline, how much worse is its performance
compared to a model trained on the new dataset's richer negatives?*

Results are saved in the standard ``fold_{k}_results.pkl`` format under
``outputs/Blastp/cross_synced_to_new/all_classes/{timestamp}/`` so the
existing evaluation infrastructure can consume them.
"""

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore

# ---------------------------------------------------------------------------
# Bootstrap project paths so that imports work from the repo root.
# ---------------------------------------------------------------------------
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from enzymeexplorer.src.experiments_orchestration.experiment_runner import (  # noqa: E402
    _normalize_fold_column,
    assign_is_tps_label,
)
from enzymeexplorer.src.models.baselines.blastp import Blastp, BlastConfig  # noqa: E402
from enzymeexplorer.src.utils.project_info import (  # noqa: E402
    ExperimentInfo,
    get_output_root,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_PRECURSOR_TYPES = frozenset({"ggpps", "fpps", "gpps", "gfpps", "hsqs"})


# -- helpers ----------------------------------------------------------------


def _load_and_prep(
    csv_path: str,
    split_col: str,
    id_col: str,
    seq_col: str,
    target_col: str,
    type_col: str | None,
) -> pd.DataFrame:
    """Load a dataset CSV, normalise folds, reclassify precursor substrates."""
    df = pd.read_csv(csv_path)
    _normalize_fold_column(df, split_col)
    if type_col and type_col in df.columns:
        df.loc[
            df[type_col].isin(_PRECURSOR_TYPES),
            target_col,
        ] = "precursor substr"
    return df


def _build_fold_df(
    raw_df: pd.DataFrame,
    fold_mask: pd.Series,
    id_col: str,
    target_col: str,
    seq_col: str,
    ignore_col: str | None,
) -> pd.DataFrame:
    """Aggregate raw rows into one-row-per-ID with label-sets, attach seq."""
    sub = raw_df.loc[fold_mask].copy()
    if ignore_col and ignore_col in sub.columns:
        sub.loc[sub[ignore_col] == 1, target_col] = "other"
    grouped = sub.groupby(id_col)[target_col].agg(set).reset_index()
    grouped[target_col] = grouped[target_col].map(assign_is_tps_label)
    # re-attach sequence column
    id_seq = raw_df[[id_col, seq_col]].drop_duplicates(id_col)
    grouped = grouped.merge(id_seq, on=id_col)
    return grouped


# ---------------------------------------------------------------------------


def run_cross_eval(
    synced_csv: str,
    new_csv: str,
    n_folds: int = 5,
    output_version: str = "cross_synced_to_new",
    n_neighbours: int = 1,
    e_threshold: float = 0.001,
    n_jobs: int = 64,
    pred_batch_size: int = 32,
) -> Path:
    """Run the Track C cross-evaluation and return the output directory."""

    target_col = "SMILES_substrate_canonical_no_stereo"
    class_names = [
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

    # --- synced old dataset (source: training) ---
    synced_df = _load_and_prep(
        synced_csv,
        split_col="synced_fold",
        id_col="Uniprot ID",
        seq_col="Amino acid sequence",
        target_col=target_col,
        type_col="Type (mono, sesq, di, …)",
    )
    synced_ignore = "synced_fold_ignore_in_eval"

    # --- new dataset (target: testing) ---
    new_df = _load_and_prep(
        new_csv,
        split_col="Fold",
        id_col="ID",
        seq_col="Aminoacid_sequence",
        target_col=target_col,
        type_col="Type",
    )
    new_ignore = "Fold_ignore_in_eval"

    # --- output directory ---
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = get_output_root() / "Blastp" / output_version / "all_classes" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_root)

    for test_fold_i in tqdm(range(n_folds), desc="Cross-eval folds"):
        test_fold_name = f"fold_{test_fold_i}"
        trn_fold_names = {f"fold_{i}" for i in range(n_folds) if i != test_fold_i}

        # --- Training data from synced old dataset ---
        trn_mask = synced_df["synced_fold"].isin(trn_fold_names)
        trn_df = _build_fold_df(
            synced_df,
            trn_mask,
            id_col="Uniprot ID",
            target_col=target_col,
            seq_col="Amino acid sequence",
            ignore_col=synced_ignore,
        )

        # --- Test data from NEW dataset ---
        test_mask = new_df["Fold"] == test_fold_name
        test_df = _build_fold_df(
            new_df,
            test_mask,
            id_col="ID",
            target_col=target_col,
            seq_col="Aminoacid_sequence",
            ignore_col=new_ignore,
        )

        n_trn_tps = sum(trn_df[target_col].map(lambda s: "isTPS" in s))
        n_trn_neg = len(trn_df) - n_trn_tps
        n_test_tps = sum(test_df[target_col].map(lambda s: "isTPS" in s))
        n_test_neg = len(test_df) - n_test_tps
        logger.info(
            "Fold %d: train=%d TPS + %d neg (synced) | test=%d TPS + %d neg (new)",
            test_fold_i,
            n_trn_tps,
            n_trn_neg,
            n_test_tps,
            n_test_neg,
        )

        # --- Instantiate and train model ---
        experiment_info = ExperimentInfo(
            model_type="Blastp",
            model_version=output_version,
        )
        config = BlastConfig(
            experiment_info=experiment_info,
            id_col_name="ID",  # test data uses new-dataset IDs
            target_col_name=target_col,
            split_col_name="Fold",
            class_names=class_names,
            optimize_hyperparams=False,
            random_state=0,
            n_calls_hyperparams_opt=0,
            n_neighbours=n_neighbours,
            e_threshold=e_threshold,
            seq_col_name="Aminoacid_sequence",  # test data col name
            neg_val="Unknown",
            negatives_sample_path="data/sampled_id_2_seq.pkl",
            tps_cleaned_csv_path=new_csv,
            per_class_optimization=False,
            reuse_existing_partial_results=False,
            load_per_class_params_from="",
            n_jobs=n_jobs,
            pred_batch_size=pred_batch_size,
            hyperparam_dimensions=None,
        )
        model = Blastp(config=config)

        # Rename training columns to match what Blastp expects for fit
        # (the model reads id_col_name and seq_col_name from config)
        trn_renamed = trn_df.rename(
            columns={
                "Uniprot ID": "ID",
                "Amino acid sequence": "Aminoacid_sequence",
            }
        )

        model.fit(trn_renamed)
        logger.info("Trained Blastp on fold %d", test_fold_i)

        val_proba_np = model.predict_proba(test_df)
        logger.info(
            "Predicted on fold %d test set (%d samples)", test_fold_i, len(test_df)
        )

        # Save in standard format
        results_path = out_root / f"fold_{test_fold_i}_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump((val_proba_np, class_names, test_df), f)
        logger.info("Saved results to %s", results_path)

    logger.info("Track C cross-evaluation complete. Results in %s", out_root)
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synced-csv",
        default="data/TPS-Nov19_2023_with_synced_folds.csv",
        help="Synced-fold old dataset (training source)",
    )
    parser.add_argument(
        "--new-csv",
        default="data/EnzymeExplorer_Dataset.csv",
        help="New EnzymeExplorer dataset (testing target)",
    )
    parser.add_argument(
        "--output-version",
        default="cross_synced_to_new",
        help="Model version name for output directory",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=64, help="Number of BLAST parallel jobs"
    )
    args = parser.parse_args()
    run_cross_eval(
        synced_csv=args.synced_csv,
        new_csv=args.new_csv,
        output_version=args.output_version,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
