"""Compare model variants on the hard-candidate set.

Runs four PlmRandomForest / PlmDomainsRandomForest × {ankh_large, esm-2}
predictions on ``data/hard_candidates/candidates.fasta`` using the
PDB structures under ``data/hard_candidates/afdb/`` for the
structure-aware variants, and produces three grouped bar plots of
``TPS_p_calibrated`` per candidate:

1. PLM-axis comparison for PlmDomainsRF — ESM-2 vs Ankh-large.
2. Model-axis comparison on Ankh-large — PlmRF vs PlmDomainsRF.
3. Model-axis comparison on ESM-2 — PlmRF vs PlmDomainsRF.

Inputs:
    candidates.fasta (16 IDs, 16 sequences) + afdb/ (14 PDB structures —
    two candidates lack an AF-DB entry; the pipeline falls back to
    PLM-only for those, so the structure-aware bars show their PLM-only
    fallback score there).

Outputs (under ``data/hard_candidates/comparison/``):
    scores.csv               — long-form (id, model, plm, TPS_p_calibrated)
    plot_plm_axis_pdomrf.png — comparison #1
    plot_model_axis_ankh.png — comparison #2
    plot_model_axis_esm2.png — comparison #3
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from Bio import SeqIO  # type: ignore

from enzymeexplorer.src.prediction.pipeline import (
    DEFAULT_CALIBRATION_CSV,
    DEFAULT_REFERENCE_DOMAINS_PICKLE,
    DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
    predict_sequences_only,
    predict_with_structures,
)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
HARD_DIR = REPO / "data/hard_candidates"
FASTA = HARD_DIR / "candidates.fasta"
STRUCTURES_DIR = HARD_DIR / "afdb"
BUNDLES_DIR = HARD_DIR / "bundles"
OUT_DIR = HARD_DIR / "comparison"

# Per-PLM calibration CSV. ankh_large's calibration is the project default
# (data/calibration_fit_summary.csv); esm-2's was fit on its OOF predictions
# via ``scripts/fit_calibration_for_plm.py``. Each prediction run uses the
# calibration table matching ITS PLM so the TPS_p_calibrated values are
# comparable.
CALIBRATION_CSVS = {
    "ankh_large": Path(DEFAULT_CALIBRATION_CSV),
    "esm-2-t36-L33": BUNDLES_DIR / "calibration_esm-2-t36-L33.csv",
    "esm-2-t36-L34": BUNDLES_DIR / "calibration_esm-2-t36-L34.csv",
}

# (label-in-plots, model-name, plm-name, bundle-path).
# Per the ESM-2 layer ablation, the winning PlmRF layer is L33 and the
# winning PlmDomRF layer is L34 (both on the 3B t36 backbone). Comparing
# each against ankh_large with the SAME RF head — i.e. the only variable
# in each plot is the PLM. PlmDomRF's PLM-only fallback uses the same
# L34 bundle so the structure-aware run stays internally consistent.
RUNS = [
    ("PlmRF / Ankh-large",     "PlmRandomForest",        "ankh_large",       BUNDLES_DIR / "plm_ankh_large.pkl"),
    ("PlmRF / ESM-2-L33",      "PlmRandomForest",        "esm-2-t36-L33",    BUNDLES_DIR / "plm_esm-2-t36-L33.pkl"),
    ("PlmDomRF / Ankh-large",  "PlmDomainsRandomForest", "ankh_large",       BUNDLES_DIR / "plm_domains_ankh_large.pkl"),
    ("PlmDomRF / ESM-2-L34",   "PlmDomainsRandomForest", "esm-2-t36-L34",    BUNDLES_DIR / "plm_domains_esm-2-t36-L34.pkl"),
]


def load_fasta() -> pd.DataFrame:
    rows = [
        {"id": r.id, "sequence": str(r.seq).upper().replace(" ", "")}
        for r in SeqIO.parse(str(FASTA), "fasta")
    ]
    return pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)


def run_one(seq_df: pd.DataFrame, label: str, model: str, plm: str, bundle: Path) -> pd.DataFrame:
    """Return per-candidate ``TPS_p_calibrated`` for one model/PLM combo.

    Structure-aware runs use the ``afdb`` directory; rows whose PDB is
    missing land in the PLM-only fallback table, which we union back so
    every hard candidate has exactly one score per run. The non-structure
    runs just call ``predict_sequences_only``.
    """
    logger.info("=== %s ===", label)
    cal_csv = CALIBRATION_CSVS[plm]
    if model == "PlmRandomForest":
        out = predict_sequences_only(
            seq_df.copy(),
            plm_only_bundle_path=bundle,
            calibration_csv_path=cal_csv,
            plm_model=plm,
        )
        scores = out[["id", "TPS_p_calibrated"]].copy()
    else:
        plm_dom, plm_fallback = predict_with_structures(
            seq_df.copy(),
            structures_dir=STRUCTURES_DIR,
            reference_domains_pickle=DEFAULT_REFERENCE_DOMAINS_PICKLE,
            reference_domains_structures_dir=DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
            plm_domains_bundle_path=bundle,
            # For the PLM-only fallback inside predict_with_structures we
            # reuse the corresponding PlmRF bundle (same PLM family) so
            # the fallback rows match the comparison's PLM axis.
            plm_only_bundle_path=BUNDLES_DIR / f"plm_{plm}.pkl",
            calibration_csv_path=cal_csv,
            plm_model=plm,
            plm_only_model=plm,
        )
        primary = plm_dom[["id", "TPS_p_calibrated"]] if len(plm_dom) else pd.DataFrame(columns=["id", "TPS_p_calibrated"])
        fb = plm_fallback[["id", "TPS_p_calibrated"]] if len(plm_fallback) else pd.DataFrame(columns=["id", "TPS_p_calibrated"])
        # Concat; if a candidate appears in both (shouldn't), prefer the
        # structure-aware row.
        scores = pd.concat([primary, fb], ignore_index=True).drop_duplicates("id", keep="first")
    scores["method_label"] = label
    scores["model"] = model
    scores["plm"] = plm
    return scores


def collect_all_scores(seq_df: pd.DataFrame) -> pd.DataFrame:
    parts = [run_one(seq_df, *r) for r in RUNS]
    return pd.concat(parts, ignore_index=True)


def plot_grouped(scores_wide: pd.DataFrame, methods: list[str], colors: list[str],
                 title: str, out_path: Path) -> None:
    """One grouped bar chart, x-axis = candidate id, one bar per method."""
    candidates = scores_wide.index.tolist()
    n_cand = len(candidates)
    n_meth = len(methods)
    bar_w = 0.8 / n_meth
    x = np.arange(n_cand)

    # Wider figure to make room for the legend outside the axes; constrained_layout
    # keeps the legend from clipping at the right edge.
    fig, ax = plt.subplots(
        figsize=(max(14, 0.7 * n_cand + 2.5), 5.2),
        constrained_layout=True,
    )
    for i, m in enumerate(methods):
        vals = scores_wide[m].fillna(0).to_numpy()
        offsets = (i - (n_meth - 1) / 2) * bar_w
        ax.bar(x + offsets, vals, bar_w, label=m, color=colors[i], edgecolor="black", linewidth=0.4)
    ax.axhline(0.9, color="#888", lw=1.0, ls="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(candidates, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("TPS_p_calibrated")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
        borderaxespad=0.0,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seq_df = load_fasta()
    logger.info("Hard candidates: %d", len(seq_df))

    # Reuse cached predictions if present — the 4 runs take a few minutes
    # of model + embedder load, so iterating on plot styling shouldn't
    # re-pay that cost. Delete scores.csv to force a re-predict.
    scores_path = OUT_DIR / "scores.csv"
    if scores_path.exists():
        logger.info("Reusing cached predictions at %s", scores_path)
        scores = pd.read_csv(scores_path)
    else:
        scores = collect_all_scores(seq_df)
        scores.to_csv(scores_path, index=False)
        logger.info("Wrote %s", scores_path)

    wide = scores.pivot_table(
        index="id", columns="method_label", values="TPS_p_calibrated",
    ).reindex(seq_df["id"].tolist())

    # PlmRF: Ankh-large vs ESM-2-t36-L33 (winning PlmRF layer from the layer ablation).
    plot_grouped(
        wide, ["PlmRF / Ankh-large", "PlmRF / ESM-2-L33"],
        colors=["#ff7f0e", "#1f77b4"],
        title="PlmRandomForest — Ankh-large vs ESM-2-t36-L33 (TPS_p_calibrated)",
        out_path=OUT_DIR / "plot_plm_axis_prf.png",
    )
    # PlmDomRF: Ankh-large vs ESM-2-t36-L34 (winning PlmDomRF layer).
    plot_grouped(
        wide, ["PlmDomRF / Ankh-large", "PlmDomRF / ESM-2-L34"],
        colors=["#ff7f0e", "#1f77b4"],
        title="PlmDomainsRandomForest — Ankh-large vs ESM-2-t36-L34 (TPS_p_calibrated)",
        out_path=OUT_DIR / "plot_plm_axis_pdomrf.png",
    )


if __name__ == "__main__":
    main()
