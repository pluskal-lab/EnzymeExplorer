"""Plot theme and colorblind-safe palette helpers.

Centralises seaborn / matplotlib defaults and exposes palette builders so
every chart in the evaluation pipeline draws from the same colour vocabulary.
The default per-class palette is Wong's 8-colour set (extended to 10 via
seaborn's ``colorblind`` palette); model-family palettes use sequential
single-hue ramps (``Blues`` for homology baselines, ``Greens`` for the
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

DEFAULT_HOMOLOGY_BASELINES: list[str] = [
    "HMM", "SUPFAM", "PFAM", "FoldSeek", "Foldseek", "BLAST", "Blastp"
]
DEFAULT_EE_FAMILY: list[str] = [
    "Domains", "PLM", "PLM_Domains",
]

DEFAULT_CLASSIFIER_DISPLAY: dict[str, str] = {
    "PLM_Domains": "Enzyme\nExplorer",
    "Domains": "Enzyme\nExplorer\nDomains",
    "PLM": "Enzyme\nExplorer\nPLM",
    "BLAST": "BLASTp",
    "Blastp": "BLASTp",
    "FoldSeek": "Foldseek",
    "Foldseek": "Foldseek",
    "HMM": "pHMM",
    "PFAM": "Pfam",
    "SUPFAM": "SUPFAM",
    "CLEAN": "CLEAN",
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


# Master colour map shared across every config so the same method
# always renders in the same colour. `_render_v4_scenarios` uses this
# as the fallback palette; YAML ``palette:`` entries take precedence
# only for keys explicitly listed there.
UNIVERSAL_PALETTE: dict[str, str] = {
    # Okabe-Ito (Wong 2011) colorblind-safe palette throughout. Each
    # canonical method gets a uniquely identifiable hue so 5+ methods
    # can be told apart in print, in greyscale, and by red-green
    # colorblind viewers.
    # Ablation triplet (Domains/PLM/PLM_Domains).
    "Domains": "#A2F1DC",        # blue
    "PLM": "#74D3BA",            # vermillion
    "PLM_Domains": "#009E73",    # green
    # homology baselines & comparators.
    "BLAST": "#0072B2",          # blue (sequence search)
    "Blastp": "#0072B2",
    "Foldseek": "#56B4E9",       # sky blue (structure search)
    "FoldSeek": "#56B4E9",
    "HMM": "#E69F00",            # orange (HMM)
    "PFAM": "#D55E00",           # vermillion (Pfam DB)
    "SUPFAM": "#999999",         # neutral grey
    "CLEAN": "#CC79A7",          # pink (ML)
    # ML-head ablation siblings.
    "PLM_RF": "#0072B2",
    "PLM_Xgb": "#D55E00",
    "PLM_MLP": "#009E73",
    "PLM_Domains_LR": "#0072B2",
    "PLM_Domains_MLP": "#D55E00",
    "PLM_Domains_RF": "#009E73",
    # PLM-head encoder ablations (6 variants → full Okabe-Ito set).
    "PLM_AnkhBase": "#0072B2",
    "PLM_AnkhLarge": "#56B4E9",
    "PLM_Esm1v": "#E69F00",
    "PLM_Esm2": "#D55E00",
    "PLM_TpsAnkhBase": "#009E73",
    "PLM_TpsEsm1vSubseq": "#CC79A7",
    # PLM_Domains-head encoder ablations.
    "PLM_Domains_AnkhBase": "#0072B2",
    "PLM_Domains_AnkhLarge": "#56B4E9",
    "PLM_Domains_Esm1v": "#E69F00",
    "PLM_Domains_Esm2": "#D55E00",
    "PLM_Domains_TpsAnkhBase": "#009E73",
    "PLM_Domains_TpsEsm1vSubseq": "#CC79A7",
}


# Module-level poster flag. Set by ``apply_theme(poster=True)`` so plot
# helpers in bars.py / deltas.py / curves.py can scale their figsize and
# hardcoded stroke/font sizes coherently with the rc theme.
POSTER_MODE: bool = False

# Layout scale factors for poster mode. Targets a 15 cm × 8 cm
# (~5.9 × 3.1") displayed tile on a 1.1 m poster viewed from ~1–1.5 m.
# Figures are rendered only modestly larger than the target physical
# size so the matplotlib font.size in points survives the shrink
# without becoming unreadable — fonts hit the page at ~14 pt visible,
# error bars at ~2 pt, axes at ~1.5 pt, grid clearly perceptible.
#
# Height gets a larger multiplier than width so rotated x-tick labels
# (35°, long classifier names) do not steal vertical space from the
# axes/bars themselves.
POSTER_FIGSIZE_W_SCALE: float = 1.5
POSTER_FIGSIZE_H_SCALE: float = 1.9
POSTER_STROKE_SCALE: float = 4.0
POSTER_FONTSIZE_SCALE: float = 2.75


def is_poster() -> bool:
    """Return ``True`` when the current theme was set up via
    :func:`apply_theme(poster=True)`."""
    return POSTER_MODE


def scale_figsize(width: float, height: float) -> tuple[float, float]:
    """Scale a paper-mode figsize for poster mode (no-op otherwise)."""
    if POSTER_MODE:
        return (
            width * POSTER_FIGSIZE_W_SCALE,
            height * POSTER_FIGSIZE_H_SCALE,
        )
    return width, height


def scale_stroke(value: float) -> float:
    """Scale a paper-mode line/edge width for poster mode."""
    return value * POSTER_STROKE_SCALE if POSTER_MODE else value


def scale_fontsize(value: float) -> float:
    """Scale a paper-mode hardcoded fontsize for poster mode."""
    return value * POSTER_FONTSIZE_SCALE if POSTER_MODE else value


def grid_color() -> str:
    """Grid colour for explicit ``ax.yaxis.grid(color=…)`` calls. Darker
    in poster mode so the grid stays perceptible after the figure is
    shrunk to its final 15 cm × 8 cm tile."""
    return "0.70" if POSTER_MODE else "0.88"


# Poster-mode two-tone blue palette: highlight one "target" classifier
# (typically the headline method, e.g. ``PLM_Domains`` / EnzymeExplorer)
# in dark blue and render every other classifier in light blue. Used by
# the all-methods-comparison bar plots and the delta boxplots when the
# user wants a single-method-vs-baselines visual story.
POSTER_PRIMARY_BLUE: str = "#08519c"
POSTER_SECONDARY_BLUE: str = "#9ecae1"


# ---------------------------------------------------------------------------
# Nature Chemical Biology palettes (colorblind-safe, Wong 2011)
# ---------------------------------------------------------------------------

# All-methods bars: EnzymeExplorer green, BLAST sky, Foldseek blue,
# everything else neutral grey. Delta bars: uniform sky blue everywhere
# (both all-methods and ablation).
NCB_GREEN:      str = "#009E73"   # Wong "bluish green" — EnzymeExplorer
NCB_SKY:        str = "#56B4E9"   # Wong "sky blue"     — BLAST + all deltas
NCB_BLUE:       str = "#0072B2"   # Wong "blue"         — Foldseek
NCB_GREY:       str = "#8C8C8C"   # neutral grey        — every other baseline


def ncb_all_methods_palette(classifiers: list[str]) -> dict[str, str]:
    """Bar palette for the all-methods NCB figure.

    EnzymeExplorer (``PLM_Domains`` and its Enzyme_Explorer aliases)
    always renders green, BLAST/BLASTp light blue, Foldseek dark blue,
    and every other method neutral grey. Colorblind-safe (Wong).
    """
    ee_family  = {"PLM_Domains", "Domains", "PLM"}
    blast_set  = {"BLAST", "Blastp"}
    fs_set     = {"Foldseek", "FoldSeek"}
    out: dict[str, str] = {}
    for c in classifiers:
        if c in ee_family:
            out[c] = NCB_GREEN
        elif c in blast_set:
            out[c] = NCB_SKY
        elif c in fs_set:
            out[c] = NCB_BLUE
        else:
            out[c] = NCB_GREY
    return out


def ee_green_ramp(n: int, *, lightest_frac: float = 0.75) -> list[str]:
    """Return ``n`` sequential shades ending at :data:`NCB_GREEN`.

    Every shade is a lightness-only mix between ``NCB_GREEN`` (Wong
    bluish green, ``#009E73``) and white. Because all shades share the
    same chromatic direction and only differ in lightness, they are
    colorblind-safe under every dichromacy — sequential single-hue
    ramps are the standard accessibility recommendation for ordinal
    data (ColorBrewer, Wong 2011).

    ``lightest_frac`` controls how close to white the FIRST shade sits;
    at ``0.75`` the leftmost shade is a very pale green while the
    rightmost is ``NCB_GREEN`` itself.
    """
    if n <= 0:
        return []
    if n == 1:
        return [NCB_GREEN]
    base = mpl.colors.to_rgb(NCB_GREEN)
    white = (1.0, 1.0, 1.0)
    ramp: list[str] = []
    for i in range(n):
        # i = 0 → most white; i = n-1 → pure NCB_GREEN.
        t = i / (n - 1)
        mix = lightest_frac * (1 - t)  # fraction of white in the mix
        rgb = tuple(base[k] * (1 - mix) + white[k] * mix for k in range(3))
        ramp.append(mpl.colors.to_hex(rgb))
    return ramp


# Ablation uses shades of the same EnzymeExplorer green from the
# all-methods plot: the darkest (== NCB_GREEN) at the RIGHT end of the
# classifier order (typically the pinned "final" model), fading toward
# white on the LEFT. Callers pass ``classifiers`` in plot order.
NCB_ABLATION_LIGHT_GREEN: str = ee_green_ramp(4)[1]  # pale but distinct green


def ncb_ablation_palette(classifiers: list[str]) -> dict[str, str]:
    """EnzymeExplorer-green ramp for ablation bar plots.

    Lightest green is assigned to the *first* classifier in the input
    list (typically the worst-performing method), pure ``NCB_GREEN``
    to the last (typically the pinned "final" model). Same colorblind-
    safe accessibility guarantee as :func:`ee_green_ramp`.
    """
    return dict(zip(classifiers, ee_green_ramp(len(classifiers))))


def ncb_all_methods_curve_palette(classifiers: list[str]) -> dict[str, str]:
    """Curve palette for the NCB all-methods figure.

    EnzymeExplorer green / BLAST sky / Foldseek blue keep their identity
    colours; every OTHER method (currently grouped as neutral grey in
    the bars) gets a distinct shade of grey so the lines can be told
    apart on a shared-axis curve panel.
    """
    ee_family = {"PLM_Domains", "Domains", "PLM"}
    blast_set = {"BLAST", "Blastp"}
    fs_set    = {"Foldseek", "FoldSeek"}
    out: dict[str, str] = {}
    others: list[str] = []
    for c in classifiers:
        if c in ee_family:
            out[c] = NCB_GREEN
        elif c in blast_set:
            out[c] = NCB_SKY
        elif c in fs_set:
            out[c] = NCB_BLUE
        else:
            others.append(c)
    if others:
        # Distinct shades of grey for the "everything-else" bucket.
        grey_shades = ncb_curve_shades(others, hue="grey")
        for c, g in zip(others, grey_shades):
            out[c] = g
    return out


def ncb_curve_shades(classifiers: list[str], *, hue: str = "blue") -> list[str]:
    """Return sequential shades of one hue for curve panels.

    * ``hue="green"`` → Wong-derived ramp ending at :data:`NCB_GREEN`
      (via :func:`ee_green_ramp`) so ablation curves use the same
      EnzymeExplorer green family as the all-methods figure. Colorblind-
      safe (single-hue lightness ramp, Wong 2011).
    * Any other hue → the ColorBrewer sequential ramp for that hue
      (still single-hue, still colorblind-safe).
    """
    if hue == "green":
        return ee_green_ramp(max(1, len(classifiers)))
    import matplotlib as _mpl
    n = max(1, len(classifiers))
    cmap = _mpl.colormaps.get_cmap({
        "blue": "Blues", "green": "Greens", "orange": "Oranges", "grey": "Greys",
    }.get(hue, "Blues"))
    stops = _np_linspace(0.30, 0.95, n)
    return [_mpl.colors.to_hex(cmap(t)) for t in stops]


def _np_linspace(a: float, b: float, n: int) -> list[float]:
    """Tiny helper to avoid a numpy import inside theme.py."""
    if n <= 1:
        return [b]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def poster_two_tone_palette(
    classifiers: list[str], target: str | None,
) -> dict[str, str]:
    """Build a ``{classifier: hex}`` palette where ``target`` gets the
    dark "primary" blue and every other entry gets the light
    "secondary" blue."""
    return {
        c: POSTER_PRIMARY_BLUE if c == target else POSTER_SECONDARY_BLUE
        for c in classifiers
    }


def xtick_rotation(n_labels: int) -> tuple[float, str]:
    """Return ``(rotation, ha)`` for x-tick labels in classifier-axis
    bar/box plots. Poster mode rotates 45° so the long classifier names
    ("EnzymeExplorer", "FoldSeek", …) stop colliding in the smaller
    canvas. Paper mode keeps them horizontal."""
    if POSTER_MODE and n_labels >= 4:
        return 35.0, "right"
    return 0.0, "center"


def apply_theme(
    context: str = "paper",
    base_font_size: float = 8.0,
    *,
    poster: bool = False,
) -> None:
    """Apply Nature Chemical Biology-style figure defaults.

    Conventions enforced here:

    * **Sans-serif**, falling back from Helvetica/Arial to whatever's
      available — matches NCB body text.
    * 8 pt base, 9 pt axis labels, 9 pt titles. Single-column figures
      should sit comfortably at ~3.4″.
    * Hairline (0.6 pt) axes, top/right spines off, ticks pointing
      outward.
    * No grid by default — turn it on per-axis when the data demands it.
    * Vector-safe (TrueType) fonts so saved PDFs are searchable.
    * 600 DPI for raster fallbacks.

    When ``poster=True``, fonts, line widths, tick sizes and markers
    are scaled up for legibility at ~1–2 m viewing distance on a
    large poster (~1 m+). Plot helpers in ``bars.py`` / ``deltas.py``
    / ``curves.py`` consult the module-level :data:`POSTER_MODE` via
    :func:`scale_figsize`, :func:`scale_stroke`, and
    :func:`scale_fontsize` to grow the canvas and stroke/label sizes
    coherently, so xtick labels do not collide and CI whiskers stay
    visible against the larger bars.
    """
    global POSTER_MODE
    POSTER_MODE = poster
    if poster:
        context = "poster"
        base_font_size = 22.0
        axes_lw = 2.0
        line_lw = 3.0
        markersize = 9.0
        patch_lw = 2.0
        tick_size = 8.0
        tick_width = 2.0
        grid_lw = 1.6
        grid_color = "0.70"
    else:
        axes_lw = 0.6
        line_lw = 1.2
        markersize = 4.0
        patch_lw = 0.6
        tick_size = 3.0
        tick_width = 0.6
        grid_lw = 0.4
        grid_color = "0.85"
    sns.set_theme(style="ticks", context=context)
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.dpi": 600,
            "savefig.transparent": False,
            "figure.dpi": 150,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica", "Arial", "DejaVu Sans", "Liberation Sans",
            ],
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 3,
            "axes.titleweight": "regular",
            "axes.labelsize": base_font_size + 1,
            "axes.labelweight": "regular",
            "figure.titlesize": base_font_size + 3,
            "figure.titleweight": "regular",
            "axes.linewidth": axes_lw,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "0.15",
            "axes.titlepad": 6.0,
            "axes.labelpad": 3.0,
            "axes.grid": False,
            "grid.linewidth": grid_lw,
            "grid.color": grid_color,
            "grid.alpha": 1.0,
            "legend.fontsize": base_font_size - 1,
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.4,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": tick_size,
            "ytick.major.size": tick_size,
            "xtick.major.width": tick_width,
            "ytick.major.width": tick_width,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "lines.linewidth": line_lw,
            "lines.markersize": markersize,
            "patch.linewidth": patch_lw,
            "patch.edgecolor": "0.15",
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
    homology: list[str] | None = None,
    ee_family: list[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Per-classifier palette: homology baselines get a Blues ramp, the
    EnzymeExplorer family gets a Greens ramp, anything else falls back to
    Wong's set."""
    homology = list(homology if homology is not None else DEFAULT_HOMOLOGY_BASELINES)
    ee_family = list(ee_family if ee_family is not None else DEFAULT_EE_FAMILY)

    out: dict[str, tuple[float, float, float]] = {}
    homology_present = [c for c in classifiers if c in homology]
    ee_present = [c for c in classifiers if c in ee_family]
    other_present = [c for c in classifiers if c not in homology and c not in ee_family]

    if homology_present:
        ramp = sns.color_palette("Blues", n_colors=max(3, len(homology_present) + 1))[1:]
        for idx, name in enumerate(homology_present):
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
    out: dict[str, tuple[float, float, float]] = {}
    ramp = sns.color_palette(
        "Greens", n_colors=max(3, len(classifiers)) + 2
    )[2:]
    for idx, name in enumerate(classifiers):
        out[name] = ramp[idx]
    return out


# Semantic palettes for category rings (Kingdom + TPS type) used by
# ``plot_category_boxplot`` / ``plot_category_heatmap``. Keyed by the
# category label as it appears in ``bootstrap_long_categorical_ap.csv``
# (``Kingdom`` column values and ``OriginalType`` codes).
NCB_CATEGORY_PALETTE: dict[str, str] = {
    # Kingdoms — user-supplied RGB tuples, converted to hex.
    "Bacteria": mpl.colors.to_hex(
        (0.00392156862745098, 0.45098039215686275, 0.6980392156862745)),
    "Fungi":    mpl.colors.to_hex(
        (0.8705882352941177, 0.5607843137254902, 0.0196078431372549)),
    "Plants":   mpl.colors.to_hex(
        (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)),
    "Animals":  mpl.colors.to_hex((0.8352941176470589, 0.3686274509803922, 0.0)),
    "Protists": mpl.colors.to_hex(
        (0.8, 0.47058823529411764, 0.7372549019607844)),
    "Viruses":  mpl.colors.to_hex((0.9254901960784314, 0.8823529411764706, 0.2)),
    "Archaea":  mpl.colors.to_hex(
        (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)),
    # TPS_Type — user-supplied hex. ``OriginalType`` codes in the CSV are
    # ``mono / sesq / di / sester / tri / sqs / psy / pt / tetra / hemi``.
    "mono":     "#de8f05",
    "sesq":     "#016c45",   # "sesqui" in the paper text
    "tri":      "#a23900",
    "di":       "#be53b3",
    "sester":   "#96a861",
    "psy":      "#0173b2",   # phytoene synthase (raw code)
    "phytoene": "#0173b2",   # pretty label used by the categorical bootstrap
    "sqs":      "#1b84fe",   # squalene synthase (raw code)
    "squalene": "#1b84fe",   # pretty label used by the categorical bootstrap
}


def categorical_palette(
    classifiers: list[str],
) -> dict[str, tuple[float, float, float]]:
    """One distinct, colorblind-safe colour per classifier / category.

    Semantic overrides in :data:`NCB_CATEGORY_PALETTE` win for any key
    that matches (kingdom names, TPS-type codes). Everything else falls
    back to the seaborn ``colorblind`` ramp so brand-new categories keep
    getting a usable distinct hue.
    """
    base = sns.color_palette("colorblind", n_colors=max(10, len(classifiers)))
    out: dict[str, tuple[float, float, float]] = {}
    for idx, key in enumerate(classifiers):
        hex_override = NCB_CATEGORY_PALETTE.get(key)
        out[key] = (mpl.colors.to_rgb(hex_override) if hex_override
                    else base[idx % len(base)])
    return out


def display_name(classifier: str) -> str:
    """Human-friendly classifier label for plot ticks/legends."""
    return DEFAULT_CLASSIFIER_DISPLAY.get(classifier, classifier)


def save_figure(
    fig: plt.Figure, out_path: Path | str, *,
    formats=("png", "svg"), close: bool = True,
) -> None:
    """Save a figure to multiple formats using its stem as a base.

    Defaults to PNG (raster preview) + SVG (post-editable in Illustrator
    for the Nature Chemical Biology submission). The ``pdf.fonttype``
    and ``svg.fonttype`` rc keys are set by :func:`apply_theme` so text
    stays as editable glyphs in both.

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
