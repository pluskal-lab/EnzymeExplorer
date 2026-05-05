"""Plot theme and colorblind-safe palette helpers.

Centralises seaborn / matplotlib defaults and exposes palette builders so
every chart in the evaluation pipeline draws from the same colour vocabulary.
The default per-class palette is Wong's 8-colour set (extended to 10 via
seaborn's ``colorblind`` palette); model-family palettes use sequential
single-hue ramps (``Blues`` for HBI baselines, ``Greens`` for the
EnzymeExplorer family) which remain distinguishable under deuteranopia and
protanopia.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

# Wong's colorblind-safe 8-colour set, ordered for high contrast.
WONG_COLORS: list[str] = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

DEFAULT_HBI_BASELINES: list[str] = [
    "HMM", "SUPFAM", "PFAM", "FoldSeek", "Foldseek", "BLAST", "Blastp"
]
DEFAULT_EE_FAMILY: list[str] = [
    "Domains", "PLM", "PLM_Domains",
    "Domains_no_distractors", "PLM_no_distractors", "PLM_Domains_no_distractors",
]

DEFAULT_CLASSIFIER_DISPLAY: dict[str, str] = {
    "PLM_Domains": "EnzymeExplorer",
    "Domains": "EnzymeExplorer\nDomains",
    "PLM": "EnzymeExplorer\nPLM",
    "BLAST": "BLASTp",
    "Blastp": "BLASTp",
    "FoldSeek": "FoldSeek",
    "Foldseek": "FoldSeek",
    "HMM": "pHMM",
    "PFAM": "Pfam",
    "SUPFAM": "SUPFAM",
    "CLEAN": "CLEAN",
    # No-distractor variants render the same as their with-distractor parent;
    # the universe is communicated by the output dir / plot title, not the
    # axis labels.
    "PLM_Domains_no_distractors": "EnzymeExplorer",
    "PLM_no_distractors": "EnzymeExplorer\nPLM",
    "Domains_no_distractors": "EnzymeExplorer\nDomains",
    "BLAST_no_distractors": "BLASTp",
    "HMM_no_distractors": "pHMM",
    "Foldseek_no_distractors": "FoldSeek",
    "PFAM_no_distractors": "Pfam",
    "SUPFAM_no_distractors": "SUPFAM",
    # Ablation-only siblings.
    "PLM_Domains_LR": "Logistic Regression",
    "PLM_Domains_MLP": "MLP",
    "PLM_Domains_RF": "Random Forest",
    "PLM_RF": "Random Forest",
    "PLM_Xgb": "XGBoost",
    "PLM_MLP": "MLP",
    # PLM ablation entries — the YAML uses these labels.
    "PLM_AnkhBase": "Ankh Base",
    "PLM_AnkhLarge": "Ankh Large",
    "PLM_Esm1v": "ESM-1v",
    "PLM_Esm2": "ESM-2",
    "PLM_TpsAnkhBase": "Ankh Base (TPS)",
    "PLM_TpsEsm1vSubseq": "ESM-1v subseq (TPS)",
    "PLM_Domains_AnkhBase": "Ankh Base",
    "PLM_Domains_AnkhLarge": "Ankh Large",
    "PLM_Domains_Esm1v": "ESM-1v",
    "PLM_Domains_Esm2": "ESM-2",
    "PLM_Domains_TpsAnkhBase": "Ankh Base (TPS)",
    "PLM_Domains_TpsEsm1vSubseq": "ESM-1v subseq (TPS)",
}


def apply_theme(context: str = "paper", base_font_size: float = 12.0) -> None:
    """Set seaborn + matplotlib defaults used by every plot.

    PDF/PS font types are forced to TrueType (``42``) so vectors embed text
    rather than rasterising it for paper figures.
    """
    sns.set_theme(style="whitegrid", context=context)
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 4,
            "axes.labelsize": base_font_size + 1,
            "legend.fontsize": base_font_size - 1,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
        }
    )


def class_palette(class_names: list[str]) -> dict[str, tuple[float, float, float]]:
    """Stable per-class colour mapping drawn from Wong + seaborn colorblind."""
    cb = sns.color_palette("colorblind", n_colors=max(10, len(class_names)))
    base = [mpl.colors.to_rgb(c) for c in WONG_COLORS] + list(cb)
    return {name: base[idx % len(base)] for idx, name in enumerate(class_names)}


def model_family_palette(
    classifiers: list[str],
    *,
    hbi: list[str] | None = None,
    ee_family: list[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Per-classifier palette: HBI baselines get a Blues ramp, the
    EnzymeExplorer family gets a Greens ramp, anything else falls back to
    Wong's set."""
    hbi = list(hbi if hbi is not None else DEFAULT_HBI_BASELINES)
    ee_family = list(ee_family if ee_family is not None else DEFAULT_EE_FAMILY)

    out: dict[str, tuple[float, float, float]] = {}
    hbi_present = [c for c in classifiers if c in hbi]
    ee_present = [c for c in classifiers if c in ee_family]
    other_present = [c for c in classifiers if c not in hbi and c not in ee_family]

    if hbi_present:
        ramp = sns.color_palette("Blues", n_colors=max(3, len(hbi_present) + 1))[1:]
        for idx, name in enumerate(hbi_present):
            out[name] = ramp[idx % len(ramp)]
    if ee_present:
        ramp = sns.color_palette("Greens", n_colors=max(3, len(ee_present) + 1))[1:]
        for idx, name in enumerate(ee_present):
            out[name] = ramp[idx % len(ramp)]
    if other_present:
        for idx, name in enumerate(other_present):
            out[name] = mpl.colors.to_rgb(WONG_COLORS[idx % len(WONG_COLORS)])
    return out


def comparison_palette(
    classifiers: list[str],
    *,
    ee_family: list[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Two-family palette for headline bar plots.

    Non-EnzymeExplorer methods get colorblind-safe shades of blue (sequential
    ``Blues`` ramp). EnzymeExplorer family methods get colorblind-safe shades
    of green (sequential ``Greens`` ramp). Both ramps are extracted skipping
    the lightest entries so all bars remain readable on a white background.
    """
    ee_family = list(ee_family if ee_family is not None else DEFAULT_EE_FAMILY)
    others = [c for c in classifiers if c not in ee_family]
    ee_present = [c for c in classifiers if c in ee_family]

    out: dict[str, tuple[float, float, float]] = {}
    if others:
        ramp = sns.color_palette("Blues", n_colors=max(3, len(others)) + 2)[2:]
        for idx, name in enumerate(others):
            out[name] = ramp[idx % len(ramp)]
    if ee_present:
        ramp = sns.color_palette("Greens", n_colors=max(3, len(ee_present)) + 2)[2:]
        for idx, name in enumerate(ee_present):
            out[name] = ramp[idx % len(ramp)]
    return out


def ee_ablation_palette(
    classifiers: list[str],
) -> dict[str, tuple[float, float, float]]:
    """Sequential Greens ramp for EnzymeExplorer-family ablation plots.

    Classifiers in canonical ablation order ``["Domains", "PLM",
    "PLM_Domains"]`` get progressively darker shades of green so the
    ablation plots share the EnzymeExplorer hue used in headline bars.
    Variants outside that list fall back to Wong's set.
    """
    canonical = ["Domains", "PLM", "PLM_Domains"]
    in_canonical = [c for c in canonical if c in classifiers]
    others = [c for c in classifiers if c not in canonical]
    out: dict[str, tuple[float, float, float]] = {}
    if in_canonical:
        ramp = sns.color_palette(
            "Greens", n_colors=max(3, len(in_canonical)) + 2
        )[2:]
        for idx, name in enumerate(in_canonical):
            out[name] = ramp[idx]
    for idx, name in enumerate(others):
        out[name] = mpl.colors.to_rgb(WONG_COLORS[idx % len(WONG_COLORS)])
    return out


def categorical_palette(
    classifiers: list[str],
) -> dict[str, tuple[float, float, float]]:
    """One distinct, colorblind-safe colour per classifier — for boxplots
    where each method needs its own hue against the same y-axis."""
    base = sns.color_palette("colorblind", n_colors=max(10, len(classifiers)))
    return {clf: base[idx % len(base)] for idx, clf in enumerate(classifiers)}


def display_name(classifier: str) -> str:
    """Human-friendly classifier label for plot ticks/legends."""
    return DEFAULT_CLASSIFIER_DISPLAY.get(classifier, classifier)


def save_figure(
    fig: plt.Figure, out_path: Path | str, *, formats=("png",), close: bool = True
) -> None:
    """Save a figure to multiple formats using its stem as a base.

    Closes the figure by default so long visualize runs don't leak memory or
    trip matplotlib's max-open-figures warning.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"))
    if close:
        plt.close(fig)
