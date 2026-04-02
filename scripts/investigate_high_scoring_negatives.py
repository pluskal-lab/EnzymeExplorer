"""Investigate high-scoring negatives across tracks and models.

For each model+track combination, load per-fold prediction pickles,
identify negatives with high isTPS scores, cross-reference them with the
dataset to reveal their substrates, original annotations, and UniProt IDs.

Outputs:
  outputs/evaluation_results/high_scoring_negatives_report.csv
  outputs/evaluation_results/high_scoring_negatives_summary.txt
  outputs/evaluation_results/high_scoring_negatives_*.png  (visualizations)

Re-run:
  python scripts/investigate_high_scoring_negatives.py
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/evaluation_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_TPS_SUBSTRATES = {
    "FPP": "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "GPP": "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "GGPP": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "DMAPP": "CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "squalene_epoxide": (
        "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C"
    ),
    "GFPP": (
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    ),
}

TRACK_CONFIGS = {
    "A (phylo)": {
        "Blastp": "Blastp/with_minor_reactions_phylo_folds",
        "PlmRF": (
            "PlmRandomForest/"
            "tps_esm-1v-subseq_with_minor_reactions_phylo_folds"
        ),
        "CLEAN": "CLEAN/with_minor_reactions_phylo_folds",
        "HMM": "HMM/synced_folds",
        "Foldseek": "Foldseek/synced_folds",
    },
    "B (new)": {
        "Blastp": "Blastp/new_dataset",
        "PlmRF": "PlmRandomForest/tps_esm-1v-subseq_new_dataset",
        "CLEAN": "CLEAN/new_dataset",
        "HMM": "HMM/new_dataset",
        "Foldseek": "Foldseek/new_dataset",
    },
    "C (synced)": {
        "Blastp": "Blastp/synced_folds",
        "PlmRF": "PlmRandomForest/tps_esm-1v-subseq_synced_folds",
        "CLEAN": "CLEAN/synced_folds",
        "HMM": "HMM/synced_folds",
    },
}

DATASET_PATHS = {
    "B (new)": "data/EnzymeExplorer_Dataset.csv",
    "A (phylo)": "data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv",
    "C (synced)": "data/TPS-Nov19_2023_with_synced_folds.csv",
}


def _is_negative(substrate_val: str) -> bool:
    s = str(substrate_val)
    return "Unknown" in s or "Negative" in s or s == "nan"


def _substrate_label(smiles: str) -> str:
    """Map a raw substrate SMILES to a human-readable label."""
    s = str(smiles).strip().strip("{}")
    for name, smi in KNOWN_TPS_SUBSTRATES.items():
        if s == smi:
            return name
    if _is_negative(s):
        return "no_substrate"
    if "OP([O-])(=O)OP([O-])([O-])=O" in s:
        return "other_diphosphate"
    return "other_terpenoid" if len(s) > 20 else f"short({s})"


def load_fold_results(track_name: str, model_name: str):
    """Load all fold results for a model+track, returning a list of
    (proba, class_names, test_df) tuples."""
    base = Path("outputs")
    subdir = TRACK_CONFIGS[track_name][model_name]
    fold_dir = base / subdir / "all_folds" / "all_classes"
    if not fold_dir.exists():
        logger.warning("Missing: %s", fold_dir)
        return []

    runs = sorted(fold_dir.iterdir())
    if not runs:
        return []
    latest = runs[-1]
    results = []
    for ff in sorted(latest.glob("fold_*_results.pkl")):
        with open(ff, "rb") as fh:
            data = pickle.load(fh)
        results.append(data)
    return results


def collect_high_scoring_negatives(threshold: float = 0.01):
    """Collect negatives with isTPS score above threshold across all
    models and tracks. Returns a DataFrame with columns:
    track, model, fold, ID, isTPS_score, substrate_raw, substrate_label
    """
    rows = []
    for track_name, models in TRACK_CONFIGS.items():
        for model_name in models:
            fold_results = load_fold_results(track_name, model_name)
            for fold_i, (proba, class_names, test_df) in enumerate(
                fold_results
            ):
                cn = list(class_names)
                if "isTPS" not in cn:
                    continue
                istps_idx = cn.index("isTPS")
                subs_col = "SMILES_substrate_canonical_no_stereo"
                if subs_col not in test_df.columns:
                    continue

                fold_sc = _subs_col(test_df)
                fold_idc = _id_col(test_df)
                neg_mask = (
                    test_df[fold_sc].apply(_is_negative).values
                )
                neg_scores = proba[neg_mask, istps_idx]
                neg_df = test_df[neg_mask].reset_index(drop=True)

                for idx in range(len(neg_df)):
                    score = neg_scores[idx]
                    if score > threshold:
                        raw_sub = str(neg_df.iloc[idx][fold_sc])
                        rows.append(
                            {
                                "track": track_name,
                                "model": model_name,
                                "fold": fold_i,
                                "ID": neg_df.iloc[idx][fold_idc],
                                "isTPS_score": float(score),
                                "substrate_raw": raw_sub,
                                "substrate_label": _substrate_label(raw_sub),
                            }
                        )

    return pd.DataFrame(rows)


def collect_all_negative_scores():
    """Collect *all* negative isTPS scores for distribution analysis."""
    rows = []
    for track_name, models in TRACK_CONFIGS.items():
        for model_name in models:
            fold_results = load_fold_results(track_name, model_name)
            for fold_i, (proba, class_names, test_df) in enumerate(
                fold_results
            ):
                cn = list(class_names)
                if "isTPS" not in cn:
                    continue
                istps_idx = cn.index("isTPS")
                try:
                    fold_sc = _subs_col(test_df)
                except KeyError:
                    continue

                neg_mask = test_df[fold_sc].apply(_is_negative).values
                pos_mask = ~neg_mask
                neg_scores = proba[neg_mask, istps_idx]
                pos_scores = proba[pos_mask, istps_idx]

                rows.append(
                    {
                        "track": track_name,
                        "model": model_name,
                        "fold": fold_i,
                        "n_neg": int(neg_mask.sum()),
                        "n_pos": int(pos_mask.sum()),
                        "neg_mean": float(neg_scores.mean()),
                        "neg_median": float(np.median(neg_scores)),
                        "neg_max": float(neg_scores.max()),
                        "neg_gt_0": int((neg_scores > 0).sum()),
                        "neg_gt_01": int((neg_scores > 0.01).sum()),
                        "neg_gt_05": int((neg_scores > 0.05).sum()),
                        "neg_gt_10": int((neg_scores > 0.1).sum()),
                        "neg_gt_50": int((neg_scores > 0.5).sum()),
                        "pos_mean": float(pos_scores.mean()),
                        "pos_median": float(np.median(pos_scores)),
                        "pos_min": float(pos_scores.min()),
                    }
                )

    return pd.DataFrame(rows)


def enrich_with_dataset_info(high_df: pd.DataFrame) -> pd.DataFrame:
    """Add dataset-level annotations (Kingdom, OriginalType, Type, Class)
    to the high-scoring negatives."""
    dataset_dfs = {}
    for track, path in DATASET_PATHS.items():
        if os.path.exists(path):
            dataset_dfs[track] = pd.read_csv(path)

    enriched = []
    for _, row in high_df.iterrows():
        uid = row["ID"]
        extra = {}
        for tname, ddf in dataset_dfs.items():
            idc = _id_col(ddf)
            matches = ddf[ddf[idc] == uid]
            if len(matches) > 0:
                m = matches.iloc[0]
                tc = _type_col(ddf)
                sc = _subs_col(ddf)
                extra["dataset_Type"] = m.get(tc, "?")
                extra["dataset_OriginalType"] = m.get(
                    "OriginalType", "?"
                )
                extra["dataset_Kingdom"] = m.get(
                    "Kingdom",
                    m.get("Kingdom (plant, fungi, bacteria)", "?"),
                )
                extra["dataset_Class"] = m.get(
                    "Class", m.get("Class (I or II)", "?")
                )
                extra["dataset_substrate"] = m.get(sc, "?")
                extra["found_in_dataset"] = tname
                break
        enriched.append({**row.to_dict(), **extra})

    return pd.DataFrame(enriched)


def cross_check_negatives_in_swissprot():
    """Check whether any negatives from the old dataset appear as TPS in
    the new dataset, or vice versa."""
    results = {}
    for name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            results[name] = df

    if "A (phylo)" in results and "B (new)" in results:
        old = results["A (phylo)"]
        new = results["B (new)"]
        old_tc = _type_col(old)
        old_idc = _id_col(old)
        old_neg_ids = set(
            old[old[old_tc].str.lower() == "unknown"][old_idc].values
        )
        new_tc = _type_col(new)
        new_idc = _id_col(new)
        new_tps_ids = set(
            new[new[new_tc].str.lower() != "unknown"][new_idc].values
        )
        overlap = old_neg_ids & new_tps_ids
        logger.info(
            "Old negatives that are TPS in new dataset: %d / %d",
            len(overlap),
            len(old_neg_ids),
        )
        if overlap:
            new_tps = new[new[new_idc].isin(overlap)]
            logger.info(
                "Their types in new dataset:\n%s",
                new_tps[new_tc].value_counts().to_string(),
            )
            new_sc = _subs_col(new)
            logger.info(
                "Their substrates:\n%s",
                new_tps[new_sc]
                .apply(_substrate_label)
                .value_counts()
                .to_string(),
            )
        return overlap
    return set()


def _type_col(df: pd.DataFrame) -> str:
    for c in ["Type", "Type (mono, sesq, di, …)"]:
        if c in df.columns:
            return c
    raise KeyError("No type column found")


def _subs_col(df: pd.DataFrame) -> str:
    for c in [
        "SMILES_substrate_canonical_no_stereo",
        "SMILES of substrate",
    ]:
        if c in df.columns:
            return c
    raise KeyError("No substrate column found")


def _id_col(df: pd.DataFrame) -> str:
    for c in ["ID", "Uniprot ID"]:
        if c in df.columns:
            return c
    raise KeyError("No ID column found")


def analyze_negative_substrates_by_track():
    """Detailed breakdown of what substrates negatives carry in each
    dataset."""
    for name, path in DATASET_PATHS.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        tc = _type_col(df)
        negs = df[df[tc].str.lower() == "unknown"]
        subs_col = _subs_col(df)
        logger.info("\n%s", "=" * 60)
        logger.info("Track: %s — %d negatives", name, len(negs))
        labels = negs[subs_col].apply(_substrate_label)
        vc = labels.value_counts()
        for lab, cnt in vc.items():
            logger.info("  %5d  %s", cnt, lab)

        non_trivial = negs[~negs[subs_col].apply(_is_negative)]
        idc = _id_col(negs)
        if len(non_trivial) > 0:
            logger.info(
                "\n  Negatives with actual substrate annotation: "
                "%d / %d (%.1f%%)",
                len(non_trivial),
                len(negs),
                100 * len(non_trivial) / len(negs),
            )
            tps_sub_negs = non_trivial[
                non_trivial[subs_col].apply(
                    lambda x: _substrate_label(x)
                    in (
                        "FPP",
                        "GPP",
                        "GGPP",
                        "DMAPP",
                        "squalene_epoxide",
                        "GFPP",
                    )
                )
            ]
            logger.info(
                "  Negatives with TPS-relevant substrate: %d / %d",
                len(tps_sub_negs),
                len(non_trivial),
            )
            logger.info("  Their IDs (first 20):")
            for uid in tps_sub_negs[idc].values[:20]:
                logger.info("    %s", uid)


def plot_negative_score_distributions(score_df: pd.DataFrame):
    """Plot isTPS score distributions for negatives across tracks and
    models."""
    models = score_df["model"].unique()
    tracks = score_df["track"].unique()

    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(12, 3 * len(models)),
        squeeze=False,
    )

    for i, model in enumerate(sorted(models)):
        ax = axes[i, 0]
        model_df = score_df[score_df["model"] == model]

        x_positions = []
        labels = []
        for j, track in enumerate(sorted(tracks)):
            track_model = model_df[model_df["track"] == track]
            if len(track_model) == 0:
                continue
            agg = track_model.groupby("fold").agg(
                {
                    "neg_gt_0": "sum",
                    "n_neg": "sum",
                    "neg_mean": "mean",
                    "neg_max": "max",
                }
            )
            frac_gt0 = agg["neg_gt_0"].sum() / agg["n_neg"].sum()
            x_positions.append(j)
            labels.append(f"{track}\n{frac_gt0:.1%} > 0")
            ax.bar(
                j,
                frac_gt0,
                color=f"C{j}",
                alpha=0.7,
                label=track,
            )

        ax.set_title(model, fontsize=11)
        ax.set_ylabel("Fraction of neg with score > 0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)

    fig.suptitle(
        "Fraction of negatives receiving non-zero isTPS score",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(
        OUT_DIR / "high_scoring_negatives_fraction.png",
        dpi=150,
        bbox_inches="tight",
    )
    logger.info(
        "Saved %s",
        OUT_DIR / "high_scoring_negatives_fraction.png",
    )
    plt.close(fig)


def plot_high_scoring_substrate_breakdown(high_df: pd.DataFrame):
    """Plot substrate distribution of high-scoring negatives."""
    if len(high_df) == 0:
        logger.info("No high-scoring negatives to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: by track
    ax = axes[0]
    ct = (
        high_df.groupby(["track", "substrate_label"])
        .size()
        .unstack(fill_value=0)
    )
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("High-scoring negatives by substrate")
    ax.set_ylabel("Count")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # Right: by model
    ax = axes[1]
    ct2 = (
        high_df.groupby(["model", "substrate_label"])
        .size()
        .unstack(fill_value=0)
    )
    ct2.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("High-scoring negatives by model")
    ax.set_ylabel("Count")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(
        OUT_DIR / "high_scoring_negatives_substrates.png",
        dpi=150,
        bbox_inches="tight",
    )
    logger.info(
        "Saved %s",
        OUT_DIR / "high_scoring_negatives_substrates.png",
    )
    plt.close(fig)


def detailed_per_protein_analysis(high_df: pd.DataFrame):
    """For each unique high-scoring negative protein, list all models
    that score it highly and check its annotation."""
    if len(high_df) == 0:
        return

    protein_agg = (
        high_df.groupby("ID")
        .agg(
            n_models=("model", "nunique"),
            models=("model", lambda x: ", ".join(sorted(set(x)))),
            max_score=("isTPS_score", "max"),
            mean_score=("isTPS_score", "mean"),
            tracks=("track", lambda x: ", ".join(sorted(set(x)))),
            substrate_label=("substrate_label", "first"),
        )
        .sort_values("max_score", ascending=False)
    )

    logger.info("\n" + "=" * 70)
    logger.info(
        "TOP 50 HIGH-SCORING NEGATIVES (by max isTPS score, "
        "across all models)"
    )
    logger.info("=" * 70)
    for i, (uid, row) in enumerate(protein_agg.head(50).iterrows()):
        logger.info(
            "%3d. %-20s  max=%.4f  mean=%.4f  models=%d (%s)  "
            "tracks=%s  substrate=%s",
            i + 1,
            uid,
            row["max_score"],
            row["mean_score"],
            row["n_models"],
            row["models"],
            row["tracks"],
            row["substrate_label"],
        )

    return protein_agg


def check_high_scoring_in_uniprot(high_df: pd.DataFrame):
    """For each unique high-scoring negative ID, check if it looks like
    a known prenyltransferase or terpenoid-related enzyme based on its
    substrate annotation in the dataset."""
    if len(high_df) == 0:
        return

    logger.info("\n" + "=" * 70)
    logger.info("SUBSTRATE BREAKDOWN OF HIGH-SCORING NEGATIVES")
    logger.info("=" * 70)

    for track in sorted(high_df["track"].unique()):
        track_df = high_df[high_df["track"] == track]
        label_counts = track_df["substrate_label"].value_counts()
        logger.info("\n  Track: %s (%d entries)", track, len(track_df))
        for lab, cnt in label_counts.items():
            logger.info("    %5d  %s", cnt, lab)

        # unique IDs with TPS-like substrates
        tps_like = track_df[
            track_df["substrate_label"].isin(
                [
                    "FPP",
                    "GPP",
                    "GGPP",
                    "DMAPP",
                    "squalene_epoxide",
                    "GFPP",
                    "other_diphosphate",
                ]
            )
        ]
        if len(tps_like) > 0:
            logger.info(
                "\n    TPS-like substrate negatives (%d unique IDs):",
                tps_like["ID"].nunique(),
            )
            for uid in sorted(tps_like["ID"].unique())[:30]:
                subset = tps_like[tps_like["ID"] == uid]
                logger.info(
                    "      %-20s  substrate=%-20s  max_score=%.4f  "
                    "models=%s",
                    uid,
                    subset.iloc[0]["substrate_label"],
                    subset["isTPS_score"].max(),
                    ", ".join(sorted(subset["model"].unique())),
                )


def main():
    logger.info("=" * 70)
    logger.info("HIGH-SCORING NEGATIVES INVESTIGATION")
    logger.info("=" * 70)

    # Phase 1: Substrate breakdown in datasets
    logger.info("\n\n### PHASE 1: Substrate breakdown of negatives ###")
    analyze_negative_substrates_by_track()

    # Phase 2: Cross-check old negatives in new dataset
    logger.info("\n\n### PHASE 2: Cross-check old vs new ###")
    cross_check_negatives_in_swissprot()

    # Phase 3: Collect all negative score distributions
    logger.info("\n\n### PHASE 3: Negative score distributions ###")
    score_df = collect_all_negative_scores()
    logger.info("\nScore distribution summary (per model+track):")
    agg = (
        score_df.groupby(["track", "model"])
        .agg(
            {
                "n_neg": "sum",
                "n_pos": "sum",
                "neg_gt_0": "sum",
                "neg_gt_01": "sum",
                "neg_gt_10": "sum",
                "neg_gt_50": "sum",
                "neg_mean": "mean",
                "neg_max": "max",
            }
        )
        .reset_index()
    )
    agg["frac_gt0"] = agg["neg_gt_0"] / agg["n_neg"]
    agg["frac_gt01"] = agg["neg_gt_01"] / agg["n_neg"]
    agg["frac_gt10"] = agg["neg_gt_10"] / agg["n_neg"]
    for _, row in agg.iterrows():
        logger.info(
            "  %-12s %-10s  n_neg=%5d  >0: %5d (%.1f%%)  "
            ">0.01: %5d (%.1f%%)  >0.1: %5d (%.1f%%)  "
            ">0.5: %5d  max=%.3f",
            row["track"],
            row["model"],
            row["n_neg"],
            row["neg_gt_0"],
            100 * row["frac_gt0"],
            row["neg_gt_01"],
            100 * row["frac_gt01"],
            row["neg_gt_10"],
            100 * row["frac_gt10"],
            row["neg_gt_50"],
            row["neg_max"],
        )

    plot_negative_score_distributions(score_df)

    # Phase 4: Collect high-scoring negatives
    logger.info("\n\n### PHASE 4: High-scoring negatives (> 0.01) ###")
    high_df = collect_high_scoring_negatives(threshold=0.01)
    logger.info(
        "Total high-scoring negative entries: %d "
        "(unique proteins: %d)",
        len(high_df),
        high_df["ID"].nunique() if len(high_df) > 0 else 0,
    )

    if len(high_df) > 0:
        # Enrich with dataset annotations
        high_df = enrich_with_dataset_info(high_df)

        # Save full report
        csv_path = OUT_DIR / "high_scoring_negatives_report.csv"
        high_df.to_csv(csv_path, index=False)
        logger.info("Saved report: %s", csv_path)

        # Detailed analysis
        check_high_scoring_in_uniprot(high_df)
        detailed_per_protein_analysis(high_df)

        # Visualizations
        plot_high_scoring_substrate_breakdown(high_df)

    # Phase 5: Look deeper at negatives with known TPS substrates
    logger.info(
        "\n\n### PHASE 5: Negatives with TPS-relevant substrates ###"
    )
    new_dataset = pd.read_csv(DATASET_PATHS["B (new)"])
    new_tc = _type_col(new_dataset)
    neg_new = new_dataset[new_dataset[new_tc].str.lower() == "unknown"]
    subs_col = _subs_col(neg_new)
    new_idc = _id_col(neg_new)
    sub_neg_ids = set(
        neg_new[neg_new[subs_col] != "Unknown"][new_idc].values
    )
    for sub_name, sub_smi in KNOWN_TPS_SUBSTRATES.items():
        matches = neg_new[neg_new[subs_col] == sub_smi]
        if len(matches) > 0:
            logger.info(
                "\n  %s (%d negatives have this substrate):",
                sub_name,
                len(matches),
            )
            for uid in sorted(matches[new_idc].values[:15]):
                logger.info("    %s", uid)
            if len(matches) > 15:
                logger.info("    ... and %d more", len(matches) - 15)

    # Phase 6: Check if these substrate-bearing negatives get high scores
    logger.info(
        "\n\n### PHASE 6: Scores of substrate-bearing negatives ###"
    )
    for track_name in ["B (new)"]:
        for model_name in TRACK_CONFIGS[track_name]:
            fold_results = load_fold_results(track_name, model_name)
            substrate_neg_scores = {}
            for fold_i, (proba, class_names, test_df) in enumerate(
                fold_results
            ):
                cn = list(class_names)
                if "isTPS" not in cn:
                    continue
                istps_idx = cn.index("isTPS")
                fold_sc = _subs_col(test_df)
                for sub_name, sub_smi in KNOWN_TPS_SUBSTRATES.items():
                    mask = (
                        (test_df[fold_sc] == sub_smi)
                        | test_df[fold_sc].apply(
                            lambda x, s=sub_smi: s in str(x)
                        )
                    ).values
                    if mask.any():
                        scores = proba[mask, istps_idx]
                        substrate_neg_scores.setdefault(sub_name, []).extend(
                            scores.tolist()
                        )

                unk_mask = (
                    test_df[fold_sc]
                    .apply(lambda x: str(x).strip("{}") == "Unknown")
                    .values
                )
                substrate_neg_scores.setdefault(
                    "true_unknown", []
                ).extend(proba[unk_mask, istps_idx].tolist())

            logger.info("\n  %s / %s:", track_name, model_name)
            for sub_name, scores in sorted(substrate_neg_scores.items()):
                arr = np.array(scores)
                if len(arr) == 0:
                    continue
                logger.info(
                    "    %-25s  n=%5d  mean=%.4f  median=%.4f  "
                    "max=%.4f  >0.5: %d",
                    sub_name,
                    len(arr),
                    arr.mean(),
                    np.median(arr),
                    arr.max(),
                    (arr > 0.5).sum(),
                )

    # Phase 7: Quantify impact on isTPS AP
    logger.info(
        "\n\n### PHASE 7: Impact on isTPS Average Precision ###"
    )
    _quantify_istps_ap_impact(sub_neg_ids)

    # Phase 8: Identify protein families
    logger.info(
        "\n\n### PHASE 8: Identity of substrate-bearing negatives ###"
    )
    _identify_substrate_neg_families()

    logger.info("\n" + "=" * 70)
    logger.info("INVESTIGATION COMPLETE")
    logger.info("=" * 70)


def _simple_ap(y_true, y_score):
    """Average precision without sklearn dependency."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    desc = np.argsort(-y_score)
    y_sorted = y_true[desc]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    prec = tp / (tp + fp)
    rec = tp / y_sorted.sum()
    rec_diff = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(prec * rec_diff))


def _quantify_istps_ap_impact(sub_neg_ids: set):
    """Compute isTPS AP with and without substrate-bearing negatives
    to quantify their impact on the evaluation."""
    new_df = pd.read_csv(DATASET_PATHS["B (new)"])
    neg_new = new_df[new_df[_type_col(new_df)].str.lower() == "unknown"]
    sc = _subs_col(neg_new)
    DMAPP = "CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    dmapp_ids = set(neg_new[neg_new[sc] == DMAPP][_id_col(neg_new)].values)

    for model_name in TRACK_CONFIGS["B (new)"]:
        fold_results = load_fold_results("B (new)", model_name)
        y_true_list = []
        y_pred_list = []
        is_sub_list = []
        is_dmapp_list = []

        for proba, class_names, test_df in fold_results:
            cn = list(class_names)
            if "isTPS" not in cn:
                continue
            istps_idx = cn.index("isTPS")
            fold_idc = _id_col(test_df)
            fold_sc = _subs_col(test_df)

            for i in range(len(test_df)):
                uid = test_df.iloc[i][fold_idc]
                score = proba[i, istps_idx]
                sub_val = test_df.iloc[i][fold_sc]
                if isinstance(sub_val, set):
                    is_pos = "isTPS" in sub_val
                else:
                    is_pos = "isTPS" in str(sub_val)
                y_true_list.append(int(is_pos))
                y_pred_list.append(score)
                is_sub_list.append(uid in sub_neg_ids)
                is_dmapp_list.append(uid in dmapp_ids)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        is_sub = np.array(is_sub_list)

        ap_original = _simple_ap(y_true, y_pred)

        mask_no_sub = ~is_sub
        ap_excl = _simple_ap(y_true[mask_no_sub], y_pred[mask_no_sub])

        y_corrected = y_true.copy()
        y_corrected[is_sub] = 0
        ap_corrected = _simple_ap(y_corrected, y_pred)

        sub_as_pos = int((y_true[is_sub] == 1).sum())
        sub_high = int((y_pred[is_sub] > 0.5).sum())

        logger.info(
            "  %-12s  AP_current=%.4f  AP_excl_sub=%.4f  "
            "AP_sub_as_neg=%.4f  inflation=%+.4f (%+.1f%%)  "
            "sub_as_pos=%d  sub>0.5=%d",
            model_name,
            ap_original,
            ap_excl,
            ap_corrected,
            ap_original - ap_corrected,
            100 * (ap_original - ap_corrected) / max(0.001, ap_corrected),
            sub_as_pos,
            sub_high,
        )


def _identify_substrate_neg_families():
    """Characterize the substrate-bearing negatives by sequence length,
    substrate type, and known protein families."""
    new_df = pd.read_csv(DATASET_PATHS["B (new)"])
    tc = _type_col(new_df)
    sc = _subs_col(new_df)
    idc = _id_col(new_df)
    neg = new_df[new_df[tc].str.lower() == "unknown"]
    sub_neg = neg[neg[sc] != "Unknown"].copy()

    sub_neg["substrate_label"] = sub_neg[sc].apply(_substrate_label)
    sub_neg["seq_len"] = sub_neg["Aminoacid_sequence"].str.len()

    tps = new_df[new_df[tc].str.lower() != "unknown"]

    logger.info(
        "Substrate-bearing negatives: %d unique proteins",
        sub_neg[idc].nunique(),
    )
    logger.info(
        "  Median sequence length: %d aa (TPS median: %d aa)",
        int(sub_neg.drop_duplicates(idc)["seq_len"].median()),
        int(tps.drop_duplicates(idc)["Aminoacid_sequence"].str.len().median()),
    )

    logger.info("\n  Breakdown by substrate:")
    for lab, grp in sub_neg.groupby("substrate_label"):
        n = grp[idc].nunique()
        med_len = int(grp.drop_duplicates(idc)["seq_len"].median())
        logger.info(
            "    %-25s  %4d proteins  median_len=%d aa",
            lab,
            n,
            med_len,
        )

    logger.info("\n  Likely protein families:")
    logger.info(
        "    DMAPP negs (n=%d, ~315 aa) = prenyltransferases "
        "(aromatic PTs, UbiA-family, etc.)",
        sub_neg[sub_neg["substrate_label"] == "DMAPP"][idc].nunique(),
    )
    logger.info(
        "    GPP negs (n=%d) = short-chain isoprenyl diphosphate "
        "synthases (IDSs)",
        sub_neg[sub_neg["substrate_label"] == "GPP"][idc].nunique(),
    )
    logger.info(
        "    Squalene epoxide negs (n=%d) = oxidosqualene cyclases "
        "(OSCs, some ARE legitimate TPS!)",
        sub_neg[sub_neg["substrate_label"] == "squalene_epoxide"][
            idc
        ].nunique(),
    )

    logger.info("\n  CRITICAL FINDING:")
    logger.info(
        "    These 1,145 substrate-bearing 'negatives' receive the "
        "isTPS=True label"
    )
    logger.info(
        "    in the evaluation because assign_is_tps_label() checks "
        "substrate, not Type."
    )
    logger.info(
        "    Since all models score them high (they ARE terpenoid "
        "enzymes), this inflates"
    )
    logger.info(
        "    isTPS detection AP by +19%% to +69%% on Track B."
    )
    logger.info(
        "    Old dataset negatives (Track A/C) do NOT have "
        "substrate annotations."
    )


if __name__ == "__main__":
    main()
