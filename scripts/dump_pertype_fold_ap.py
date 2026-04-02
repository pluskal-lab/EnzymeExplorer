"""Dump raw per-TPS-type, per-fold Average Precision for all models
on Tracks A and B, with and without excluding substrate-bearing negatives.

Also prints isTPS score distributions for substrate-bearing vs true-
unknown negatives.

Usage:
    python scripts/dump_pertype_fold_ap.py
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

MIN_TYPE_COUNT = 3

EVAL_CSV_A = "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv"
EVAL_CSV_B = "data/EnzymeExplorer_Dataset.csv"
OUTPUT_ROOT = Path("outputs")

TRACK_A_CONFIGS: dict[str, tuple[str, str]] = {
    "CLEAN": ("CLEAN", "with_minor_reactions"),
    "PlmRF": (
        "PlmRandomForest",
        "tps_esm-1v-subseq_with_minor_reactions_phylo_folds",
    ),
}

TRACK_B_CONFIGS: dict[str, tuple[str, str]] = {
    "CLEAN": ("CLEAN", "new_dataset"),
    "PlmRF": (
        "PlmRandomForest",
        "tps_esm-1v-subseq_new_dataset",
    ),
}


def _load_folds(
    model_type: str, version: str, n: int = 5
) -> list[tuple[int, tuple]]:
    base = OUTPUT_ROOT / model_type / version / "all_folds" / "all_classes"
    if not base.exists():
        return []
    ts_dirs = sorted(base.iterdir(), reverse=True)
    for ts_dir in ts_dirs:
        if (ts_dir / "fold_0_results.pkl").exists():
            base = ts_dir
            break
    out: list[tuple[int, tuple]] = []
    for i in range(n):
        pkl = base / f"fold_{i}_results.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                out.append((i, pickle.load(f)))
    return out


def _get_sub_neg_ids() -> frozenset[str]:
    df = pd.read_csv(EVAL_CSV_B)
    neg = df[df["Type"].str.lower() == "unknown"]
    sc = "SMILES_substrate_canonical_no_stereo"
    return frozenset(neg[neg[sc] != "Unknown"]["ID"].values)


def _dump(
    model_label: str,
    model_type: str,
    version: str,
    eval_csv: str,
    track_label: str,
    excl_ids: frozenset[str] | None = None,
) -> None:
    eval_df = pd.read_csv(eval_csv)
    tc = (
        "Type"
        if "Type" in eval_df.columns
        else "Type (mono, sesq, di, \u2026)"
    )
    idc = "ID" if "ID" in eval_df.columns else "Uniprot ID"
    id_to_type = dict(zip(eval_df[idc], eval_df[tc].fillna("Unknown")))

    fold_results = _load_folds(model_type, version)
    if not fold_results:
        print(f"\n  No fold results found for {model_label} on {track_label}")
        return

    per_type_fold: dict[str, list[tuple[int, float, int, int]]] = (
        defaultdict(list)
    )

    for fold_i, (val_proba, class_names, test_df) in fold_results:
        cn = list(class_names)
        if "isTPS" not in cn:
            continue
        tps_idx = cn.index("isTPS")
        y_pred = val_proba[:, tps_idx]

        id_col = "ID" if "ID" in test_df.columns else "Uniprot ID"
        ids = test_df[id_col].values
        types = pd.Series(
            [
                str(id_to_type.get(pid, "Unknown")).lower().strip()
                for pid in ids
            ]
        )

        if excl_ids:
            keep = pd.Series([pid not in excl_ids for pid in ids])
        else:
            keep = pd.Series([True] * len(ids))

        _non_tps = {"unknown", "negative", "ggpps", "fpps", "gpps",
                    "gfpps", "hsqs", "pt"}
        neg_mask = types.isin({"unknown", "negative"}) & keep
        n_neg = int(neg_mask.sum())
        all_types = [
            t for t in types[keep].unique() if t not in _non_tps
        ]

        for tps_type in sorted(all_types):
            type_mask = (types == tps_type) & keep
            n_pos = int(type_mask.sum())
            if n_pos < MIN_TYPE_COUNT or n_neg < MIN_TYPE_COUNT:
                per_type_fold[tps_type].append(
                    (fold_i, float("nan"), n_pos, n_neg)
                )
                continue
            subset = type_mask | neg_mask
            y_true = type_mask[subset].astype(int).values
            y_p = y_pred[subset.values]
            ap = average_precision_score(y_true, y_p)
            per_type_fold[tps_type].append((fold_i, ap, n_pos, n_neg))

    excl_tag = " (excl sub-negs)" if excl_ids else ""
    print(f"\n{'=' * 80}")
    print(f"{model_label} — {track_label}{excl_tag}")
    print(f"{'=' * 80}")
    print(
        f"{'Type':>12s}  "
        + "  ".join(f"{'Fold ' + str(i):>12s}" for i in range(5))
        + f"  {'mean(AP)':>9s}  {'n_pos/fold':>12s}"
    )
    print("-" * 110)

    type_means: list[float] = []
    for tps_type in sorted(per_type_fold.keys()):
        entries = per_type_fold[tps_type]
        aps = [e[1] for e in entries]
        n_pos_vals = [e[2] for e in entries]

        cells = []
        for ap_val, n_p in zip(aps, n_pos_vals):
            if np.isnan(ap_val):
                cells.append(f"  skip(n={n_p})")
            else:
                cells.append(f"{ap_val:.4f}(n={n_p})")

        valid_aps = [a for a in aps if not np.isnan(a)]
        if valid_aps:
            mean_ap = float(np.mean(valid_aps))
            type_means.append(mean_ap)
            mean_str = f"{mean_ap:.4f}"
        else:
            mean_str = "n/a"

        avg_n = int(np.mean(n_pos_vals))
        print(
            f"{tps_type:>12s}  "
            + "  ".join(f"{c:>12s}" for c in cells)
            + f"  {mean_str:>9s}  {avg_n:>12d}"
        )

    if type_means:
        macro = float(np.mean(type_means))
        print("-" * 110)
        print(
            f"{'MACRO':>12s}"
            + " " * 72
            + f"{macro:.4f}"
            + f"  ({len(type_means)} types)"
        )

    for fold_i, (_, class_names, test_df) in fold_results:
        cn = list(class_names)
        if "isTPS" not in cn:
            continue
        id_col = "ID" if "ID" in test_df.columns else "Uniprot ID"
        ids = test_df[id_col].values
        types_f = pd.Series(
            [
                str(id_to_type.get(pid, "Unknown")).lower().strip()
                for pid in ids
            ]
        )
        if excl_ids:
            keep_f = pd.Series([pid not in excl_ids for pid in ids])
        else:
            keep_f = pd.Series([True] * len(ids))
        n_neg_f = int(
            (types_f.isin({"unknown", "negative"}) & keep_f).sum()
        )
        n_excl_f = int((~keep_f).sum()) if excl_ids else 0
        print(
            f"  Fold {fold_i}: {n_neg_f} negatives"
            + (f" ({n_excl_f} excluded)" if n_excl_f else "")
        )


def _score_distributions(
    model_label: str, model_type: str, version: str
) -> None:
    eval_df = pd.read_csv(EVAL_CSV_B)
    neg_df = eval_df[eval_df["Type"].str.lower() == "unknown"]
    sc = "SMILES_substrate_canonical_no_stereo"
    sub_neg_ids = frozenset(neg_df[neg_df[sc] != "Unknown"]["ID"].values)
    true_unk_ids = frozenset(neg_df[neg_df[sc] == "Unknown"]["ID"].values)

    fold_results = _load_folds(model_type, version)

    sub_scores: list[float] = []
    unk_scores: list[float] = []

    for _, (val_proba, class_names, test_df) in fold_results:
        cn = list(class_names)
        if "isTPS" not in cn:
            continue
        tps_idx = cn.index("isTPS")
        y_pred = val_proba[:, tps_idx]
        id_col = "ID" if "ID" in test_df.columns else "Uniprot ID"
        ids = test_df[id_col].values
        for pid, sc_val in zip(ids, y_pred):
            if pid in sub_neg_ids:
                sub_scores.append(float(sc_val))
            elif pid in true_unk_ids:
                unk_scores.append(float(sc_val))

    sub_arr = np.array(sub_scores)
    unk_arr = np.array(unk_scores)
    print(f"\n--- {model_label} isTPS score distributions (Track B) ---")
    print(
        f"  Substrate-bearing negs: n={len(sub_arr)}, "
        f"median={np.median(sub_arr):.4f}, "
        f"mean={np.mean(sub_arr):.4f}, "
        f"p75={np.percentile(sub_arr, 75):.4f}, "
        f"p90={np.percentile(sub_arr, 90):.4f}, "
        f"max={np.max(sub_arr):.4f}"
    )
    print(
        f"  True-unknown negs:     n={len(unk_arr)}, "
        f"median={np.median(unk_arr):.4f}, "
        f"mean={np.mean(unk_arr):.4f}, "
        f"p75={np.percentile(unk_arr, 75):.4f}, "
        f"p90={np.percentile(unk_arr, 90):.4f}, "
        f"max={np.max(unk_arr):.4f}"
    )


def main() -> None:
    excl_ids = _get_sub_neg_ids()
    print(f"Substrate-bearing negative IDs: {len(excl_ids)}")

    print("\n" + "#" * 80)
    print("# TRACK A (phylogenetic folds)")
    print("#" * 80)
    for label, (mtype, ver) in TRACK_A_CONFIGS.items():
        _dump(label, mtype, ver, EVAL_CSV_A, "Track A")

    print("\n" + "#" * 80)
    print("# TRACK B (new dataset)")
    print("#" * 80)
    for label, (mtype, ver) in TRACK_B_CONFIGS.items():
        _dump(label, mtype, ver, EVAL_CSV_B, "Track B")
        _dump(label, mtype, ver, EVAL_CSV_B, "Track B", excl_ids=excl_ids)

    print("\n" + "#" * 80)
    print("# SCORE DISTRIBUTIONS")
    print("#" * 80)
    for label, (mtype, ver) in TRACK_B_CONFIGS.items():
        _score_distributions(label, mtype, ver)


if __name__ == "__main__":
    main()
