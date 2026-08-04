"""Build 4 iTOL datasets for the GTDB ar53 archaeal phylogeny:

  1. ``itol_archaea_phylum_colorstrip.txt``
       DATASET_COLORSTRIP with per-genome Phylum colour matched to the
       swatches in ``data/archaeal_screening/colors.svg``.

  2. ``itol_archaea_C95_bar.txt``    — SIMPLE_BAR, TPS hits at p ≥ 0.95    (light green).
  3. ``itol_archaea_C99_bar.txt``    — SIMPLE_BAR, TPS hits at p ≥ 0.99    (mid green).
  4. ``itol_archaea_C99.95_bar.txt`` — SIMPLE_BAR, TPS hits at p ≥ 0.9995  (Wong green).

The three bar datasets share the same axis (``DATASET_SCALE`` at 1, 3, 6)
and identical width/height so they overlap perfectly when placed in the
same iTOL ring (drag them to the same track in the iTOL control panel;
draw order lightest → darkest so the highest-confidence dark bar sits on
top of the lower-confidence ones).

Reads:  data/archaeal_screening/gtdb_genome_TPS_hits.csv (per-genome hit counts)
Writes: outputs/archaeal_screening/itol_archaea_*.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "data" / "archaeal_screening"
OUT_DIR = REPO / "outputs" / "archaeal_screening"

CSV = IN_DIR / "gtdb_genome_TPS_hits.csv"

# Phylum -> hex, taken directly from colors.svg swatches.  The CSV uses
# "Nanohalarchaeota" (missing the second 'o' from the SVG's
# "Nanohaloarchaeota"); both map to the same colour swatch.
PHYLUM_COLORS = {
    "Methanobacteriota":    "#5954d6",
    "Halobacteriota":       "#00bbad",
    "Thermoproteota":       "#c0affb",
    "Thermoplasmatota":     "#56641a",
    "Micrarchaeota":        "#00c6f8",
    "Nanoarchaeota":        "#878500",
    "Other":                "#cdcdcd",
    "Huberarchaeota":       "#d163e6",
    "Asgardarchaeota":      "#008cf9",
    "Aenigmatarchaeota":    "#ebac23",
    "Altiarchaeota":        "#b80058",
    "Iainarchaeota":        "#ff9287",
    "Hydrothermarchaeota":  "#b24502",
    "Nanohalarchaeota":     "#00a76c",  # CSV spelling
    "Nanohaloarchaeota":    "#00a76c",  # SVG spelling
    "Undinarchaeota":       "#e6a176",
    "Hadarchaeota":         "#006e00",
}

# Confidence-threshold green ramp — light → dark, colorblind-safe mono-hue.
# The dark end is Wong palette green (#009e73). The lighter shades are
# derived by mixing with white so the family stays cb-safe.
GREEN_C95    = "#a6d9c0"  # light
GREEN_C99    = "#4bb491"  # mid
GREEN_C99_95 = "#009e73"  # Wong green (dark)

# Y-axis ticks — user-specified, common to all three bars.
BAR_MAX = 6.0
BAR_TICKS = [1.0, 3.0, 6.0]


def _write_colorstrip(df: pd.DataFrame) -> Path:
    present = set(df["phylum"].unique())
    kingdoms_present = sorted(p for p in present if p != "Other")
    if "Other" in present:
        kingdoms_present.append("Other")
    header = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        "DATASET_LABEL\tArchaeal phylum",
        "COLOR\t#000000",
        "STRIP_WIDTH\t35",
        "MARGIN\t2",
        "BORDER_WIDTH\t0.25",
        "SHOW_INTERNAL\t0",
        "LEGEND_TITLE\tArchaeal phylum",
        "LEGEND_SHAPES\t" + "\t".join(["1"] * len(kingdoms_present)),
        "LEGEND_COLORS\t" + "\t".join(PHYLUM_COLORS[k] for k in kingdoms_present),
        "LEGEND_LABELS\t" + "\t".join(kingdoms_present),
        "LEGEND_SHAPE_SCALES\t" + "\t".join(["1"] * len(kingdoms_present)),
        "LEGEND_GRADIENT\t0",
        "DATA",
    ]
    lines = list(header)
    for acc, phylum in zip(df["accession"], df["phylum"]):
        color = PHYLUM_COLORS.get(phylum)
        if not color:
            continue
        lines.append(f"{acc}\t{color}")
    out = OUT_DIR / "itol_archaea_phylum_colorstrip.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


def _write_bar(df: pd.DataFrame, col: str, label: str, color: str, tag: str) -> Path:
    scale_pieces = []
    for i, v in enumerate(BAR_TICKS):
        dashed = "1" if 0 < i < len(BAR_TICKS) - 1 else "0"
        scale_pieces.append(f"{v:g}-{v:g}-#808080-1-{dashed}-8")
    header = [
        "DATASET_SIMPLEBAR",
        "SEPARATOR TAB",
        f"DATASET_LABEL\t{label}",
        f"COLOR\t{color}",
        "WIDTH\t120",
        "MARGIN\t2",
        "SHOW_INTERNAL\t0",
        "BAR_ZERO\t0",
        "DATASET_SCALE\t" + "\t".join(scale_pieces),
        f"DATASET_SCALE_MAX\t{BAR_MAX:g}",
        "HEIGHT_FACTOR\t1",
        "DATA",
    ]
    lines = list(header)
    for acc, v in zip(df["accession"], df[col]):
        if pd.isna(v):
            continue
        lines.append(f"{acc}\t{float(v):.4g}")
    out = OUT_DIR / f"itol_archaea_{tag}_bar.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    print(f"loaded {len(df)} genomes; {df['phylum'].nunique()} phyla")
    unknown = set(df["phylum"]) - set(PHYLUM_COLORS)
    if unknown:
        print(f"⚠ phyla missing from palette (will be uncolored): {unknown}")

    _write_colorstrip(df)
    _write_bar(df, "C95_count",    "TPS hits @ p≥0.95",   GREEN_C95,    "C95")
    _write_bar(df, "C99_count",    "TPS hits @ p≥0.99",   GREEN_C99,    "C99")
    _write_bar(df, "C99.95_count", "TPS hits @ p≥0.9995", GREEN_C99_95, "C99.95")


if __name__ == "__main__":
    main()
