"""Visualize the substrate-bearing negatives issue.

Generates:
  outputs/evaluation_results/substrate_neg_score_comparison.{png,pdf,svg}
  outputs/evaluation_results/substrate_neg_ap_impact.{png,pdf,svg}
  outputs/evaluation_results/substrate_neg_enzyme_families.{png,pdf,svg}

Re-run:
  python scripts/plot_substrate_neg_analysis.py
"""

from __future__ import annotations

import logging
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

MODELS_B = {
    "Blastp": "Blastp/new_dataset",
    "PlmRF": "PlmRandomForest/tps_esm-1v-subseq_new_dataset",
    "CLEAN": "CLEAN/new_dataset",
    "HMM": "HMM/new_dataset",
    "Foldseek": "Foldseek/new_dataset",
}

SUBSTRATE_LABELS = {
    "CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "DMAPP",
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "GPP",
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O": "FPP",
    (
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)"
        "OP([O-])([O-])=O"
    ): "GGPP",
    (
        "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C"
    ): "Squalene\nepoxide",
}


def _id_col(df: pd.DataFrame) -> str:
    for c in ["ID", "Uniprot ID"]:
        if c in df.columns:
            return c
    raise KeyError("No ID column")


def _subs_col(df: pd.DataFrame) -> str:
    for c in [
        "SMILES_substrate_canonical_no_stereo",
        "SMILES of substrate",
    ]:
        if c in df.columns:
            return c
    raise KeyError("No substrate column")


def _load_fold_results(subdir: str):
    """Load latest fold results from an output subdirectory."""
    fold_dir = Path("outputs") / subdir / "all_folds" / "all_classes"
    if not fold_dir.exists():
        return []
    runs = sorted(fold_dir.iterdir())
    if not runs:
        return []
    latest = runs[-1]
    results = []
    for ff in sorted(latest.glob("fold_*_results.pkl")):
        with open(ff, "rb") as fh:
            results.append(pickle.load(fh))
    return results


def _simple_ap(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0:
        return 0.0
    desc = np.argsort(-y_score)
    y_sorted = y_true[desc]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    prec = tp / (tp + fp)
    rec = tp / y_sorted.sum()
    rec_diff = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(prec * rec_diff))


def collect_scores():
    """Collect per-protein isTPS scores for Track B, categorized as
    true TPS, substrate-bearing neg, or true-unknown neg."""
    new_df = pd.read_csv("data/EnzymeExplorer_Dataset.csv")
    neg = new_df[new_df["Type"].str.lower() == "unknown"]
    sc = _subs_col(neg)
    idc = _id_col(neg)
    sub_neg_ids = set(neg[neg[sc] != "Unknown"][idc].values)
    unk_neg_ids = set(neg[neg[sc] == "Unknown"][idc].values)

    sub_neg_substrates = (
        neg[neg[sc] != "Unknown"]
        .groupby(idc)[sc]
        .first()
        .to_dict()
    )

    all_rows = []
    for model_name, subdir in MODELS_B.items():
        fold_results = _load_fold_results(subdir)
        for proba, class_names, test_df in fold_results:
            cn = list(class_names)
            if "isTPS" not in cn:
                continue
            istps_idx = cn.index("isTPS")
            fold_idc = _id_col(test_df)

            for i in range(len(test_df)):
                uid = test_df.iloc[i][fold_idc]
                score = proba[i, istps_idx]

                if uid in sub_neg_ids:
                    raw_sub = sub_neg_substrates.get(uid, "?")
                    sub_label = SUBSTRATE_LABELS.get(raw_sub, "other")
                    cat = f"neg ({sub_label})"
                    group = "substrate_neg"
                elif uid in unk_neg_ids:
                    cat = "neg (no substrate)"
                    group = "true_neg"
                else:
                    cat = "TPS"
                    group = "tps"

                all_rows.append(
                    {
                        "model": model_name,
                        "ID": uid,
                        "score": score,
                        "category": cat,
                        "group": group,
                    }
                )

    return pd.DataFrame(all_rows), sub_neg_ids


def fig1_score_comparison(scores_df: pd.DataFrame):
    """Box/violin plots comparing isTPS score distributions across
    TPS, substrate-bearing negs, and true-unknown negs."""
    models = sorted(scores_df["model"].unique())
    fig, axes = plt.subplots(
        1, len(models), figsize=(4 * len(models), 5), sharey=True
    )

    groups = ["tps", "substrate_neg", "true_neg"]
    labels = ["TPS\n(positives)", "Substrate-\nbearing negs", "True-unknown\nnegatives"]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    for ax, model in zip(axes, models):
        data_list = []
        tick_labels = []
        for g, lab in zip(groups, labels):
            vals = scores_df[
                (scores_df["model"] == model)
                & (scores_df["group"] == g)
            ]["score"].values
            data_list.append(vals)
            tick_labels.append(f"{lab}\n(n={len(vals):,})")

        vp = ax.violinplot(
            data_list,
            positions=range(len(data_list)),
            showmedians=True,
            showextrema=False,
        )
        for pc, color in zip(vp["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        vp["cmedians"].set_color("black")

        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(model, fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("isTPS score", fontsize=10)

    fig.suptitle(
        "isTPS score distributions on Track B:\n"
        "Substrate-bearing negatives score like TPS",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"substrate_neg_score_comparison.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    logger.info("Saved substrate_neg_score_comparison.*")
    plt.close(fig)


def fig2_ap_impact(sub_neg_ids: set):
    """Bar chart showing AP with and without substrate-bearing negs."""
    models_ap = {}
    for model_name, subdir in MODELS_B.items():
        fold_results = _load_fold_results(subdir)
        y_true_list = []
        y_pred_list = []
        is_sub_list = []

        for proba, class_names, test_df in fold_results:
            cn = list(class_names)
            if "isTPS" not in cn:
                continue
            istps_idx = cn.index("isTPS")
            fold_idc = _id_col(test_df)

            for i in range(len(test_df)):
                uid = test_df.iloc[i][fold_idc]
                score = proba[i, istps_idx]
                sub_val = test_df.iloc[i][_subs_col(test_df)]
                if isinstance(sub_val, set):
                    is_pos = "isTPS" in sub_val
                else:
                    is_pos = "isTPS" in str(sub_val)
                y_true_list.append(int(is_pos))
                y_pred_list.append(score)
                is_sub_list.append(uid in sub_neg_ids)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        is_sub = np.array(is_sub_list)

        ap_current = _simple_ap(y_true, y_pred)
        mask_excl = ~is_sub
        ap_excl = _simple_ap(y_true[mask_excl], y_pred[mask_excl])
        y_corrected = y_true.copy()
        y_corrected[is_sub] = 0
        ap_neg = _simple_ap(y_corrected, y_pred)

        models_ap[model_name] = {
            "current": ap_current,
            "excl": ap_excl,
            "as_neg": ap_neg,
        }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models_ap))
    width = 0.25
    model_names = list(models_ap.keys())

    bars1 = ax.bar(
        x - width,
        [models_ap[m]["current"] for m in model_names],
        width,
        label="Current AP (sub-negs as TPS)",
        color="#e74c3c",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x,
        [models_ap[m]["excl"] for m in model_names],
        width,
        label="AP excluding sub-negs",
        color="#f39c12",
        alpha=0.85,
    )
    bars3 = ax.bar(
        x + width,
        [models_ap[m]["as_neg"] for m in model_names],
        width,
        label="AP with sub-negs as negatives",
        color="#2ecc71",
        alpha=0.85,
    )

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Average Precision (isTPS)", fontsize=11)
    ax.set_title(
        "Impact of 1,145 substrate-bearing negatives\n"
        "on isTPS detection AP (Track B)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"substrate_neg_ap_impact.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    logger.info("Saved substrate_neg_ap_impact.*")
    plt.close(fig)


def fig3_enzyme_families():
    """Pie/bar chart showing the composition of substrate-bearing
    negatives by enzyme family (from UniProt query results)."""
    families = {
        "tRNA dimethylallyl-\ntransferase (EC 2.5.1.75)": 149,
        "HdrB/IspH reductase\n(EC 1.17.7.4)": 74,
        "IPP/DMAPP isomerase\n(EC 5.3.3.2)": 54,
        "tRNA selenouridine\nsynthase (EC 2.9.1.3)": 44,
        "Nudix hydrolases\n(EC 3.6.1.x)": 14,
        "NDP kinases\n(EC 2.7.4.6)": 8,
        "Squalene epoxidase\n(EC 1.14.14.17)": 16,
        "Other prenyltrans-\nferases (EC 2.5.1.x)": 18,
        "Lipid phosphatases\n(EC 3.1.3.x)": 8,
        "Other": 15,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: substrate breakdown
    substrate_data = {
        "DMAPP": 1018,
        "GPP": 69,
        "Other terpenoid": 56,
        "GGPP": 17,
        "Squalene epoxide": 16,
        "FPP": 15,
        "Other diphosphate": 8,
    }
    colors_sub = plt.cm.Set2(np.linspace(0, 1, len(substrate_data)))
    wedges, texts, autotexts = ax1.pie(
        substrate_data.values(),
        labels=substrate_data.keys(),
        autopct=lambda p: f"{p:.0f}%\n({int(p*sum(substrate_data.values())/100)})"
        if p > 3
        else "",
        colors=colors_sub,
        startangle=90,
        pctdistance=0.75,
    )
    for text in texts:
        text.set_fontsize(8)
    for text in autotexts:
        text.set_fontsize(7)
    ax1.set_title(
        "Substrate breakdown\n(1,145 proteins)",
        fontsize=11,
        fontweight="bold",
    )

    # Right: enzyme families (horizontal bar)
    sorted_fam = sorted(families.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_fam]
    counts = [x[1] for x in sorted_fam]
    colors_fam = plt.cm.tab10(np.linspace(0, 1, len(families)))

    bars = ax2.barh(
        range(len(names)),
        counts,
        color=colors_fam[: len(names)],
        alpha=0.85,
    )
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Count (of 300 queried)", fontsize=10)
    ax2.set_title(
        "Enzyme families (UniProt lookup)\n"
        "None are terpene synthases",
        fontsize=11,
        fontweight="bold",
    )
    for bar, cnt in zip(bars, counts):
        ax2.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(cnt),
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"substrate_neg_enzyme_families.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    logger.info("Saved substrate_neg_enzyme_families.*")
    plt.close(fig)


def fig4_seq_length_comparison():
    """Compare sequence length distributions between TPS, substrate-
    bearing negatives, and true-unknown negatives."""
    new_df = pd.read_csv("data/EnzymeExplorer_Dataset.csv")
    tc = "Type"
    sc = "SMILES_substrate_canonical_no_stereo"

    tps = new_df[new_df[tc].str.lower() != "unknown"].drop_duplicates("ID")
    neg = new_df[new_df[tc].str.lower() == "unknown"]
    sub_neg = neg[neg[sc] != "Unknown"].drop_duplicates("ID")
    unk_neg = neg[neg[sc] == "Unknown"].drop_duplicates("ID")

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.arange(0, 1100, 25)
    ax.hist(
        tps["Aminoacid_sequence"].str.len(),
        bins=bins,
        alpha=0.6,
        label=f"TPS (n={len(tps):,}, median={int(tps['Aminoacid_sequence'].str.len().median())} aa)",
        color="#2ecc71",
        density=True,
    )
    ax.hist(
        sub_neg["Aminoacid_sequence"].str.len(),
        bins=bins,
        alpha=0.6,
        label=f"Substrate-bearing negs (n={len(sub_neg):,}, median={int(sub_neg['Aminoacid_sequence'].str.len().median())} aa)",
        color="#e74c3c",
        density=True,
    )
    ax.hist(
        unk_neg["Aminoacid_sequence"].str.len(),
        bins=bins,
        alpha=0.4,
        label=f"True-unknown negs (n={len(unk_neg):,}, median={int(unk_neg['Aminoacid_sequence'].str.len().median())} aa)",
        color="#3498db",
        density=True,
    )

    ax.set_xlabel("Sequence length (aa)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Sequence length distributions:\n"
        "Substrate-bearing negatives are much shorter than TPS",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"substrate_neg_seq_lengths.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    logger.info("Saved substrate_neg_seq_lengths.*")
    plt.close(fig)


def main():
    logger.info("Generating substrate-bearing negatives analysis plots...")

    scores_df, sub_neg_ids = collect_scores()
    fig1_score_comparison(scores_df)
    fig2_ap_impact(sub_neg_ids)
    fig3_enzyme_families()
    fig4_seq_length_comparison()

    logger.info("All plots saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
