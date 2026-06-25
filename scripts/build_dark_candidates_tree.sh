#!/bin/bash
# Build the dark-candidates phylogeny end-to-end.
#
# Stage 1: MAFFT --auto on combined.fasta              -> alignment.fasta
# Stage 2: IQ-TREE -fast (LG+G4) on alignment.fasta    -> tree.nwk
#
# Both stages skip work whose output already exists, so re-running is
# safe: deleting a single product re-triggers only the stages that
# depend on it.
#
# Inputs are produced by ``scripts/build_dark_candidates_combined_fasta.py``;
# run that first.
#
# Usage:
#   bash scripts/build_dark_candidates_tree.sh \
#       [--workdir data/dark_candidates] \
#       [--threads 16]

set -euo pipefail

workdir="data/dark_candidates"
threads=16

while (( $# > 0 )); do
    case "$1" in
        --workdir) workdir="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" >&2; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

combined="$workdir/combined.fasta"
alignment="$workdir/alignment.fasta"
tree_prefix="$workdir/iqtree"
tree="$workdir/tree.nwk"

[[ -f "$combined" ]] || { echo "Missing $combined — run build_dark_candidates_combined_fasta.py first." >&2; exit 1; }

# ---- MAFFT --------------------------------------------------------------
if [[ -s "$alignment" ]]; then
    echo "[tree] alignment exists at $alignment — skipping MAFFT"
else
    echo "[tree] running MAFFT --auto --thread $threads on $combined"
    mafft --auto --thread "$threads" --anysymbol "$combined" > "$alignment.tmp"
    mv "$alignment.tmp" "$alignment"
    echo "[tree] wrote $alignment"
fi

# ---- IQ-TREE -fast ------------------------------------------------------
# -fast: greedy NJ+NNI, no bootstrap. ~10x faster than the default ML
# search and good enough for "which clade is far from which" questions.
# LG+G4: standard amino-acid substitution model with gamma rate var.
# -nt AUTO lets IQ-TREE pick a sensible thread count up to ``$threads``.
if [[ -s "$tree" ]]; then
    echo "[tree] tree exists at $tree — skipping IQ-TREE"
else
    echo "[tree] running IQ-TREE -fast (LG+G4) on $alignment"
    iqtree2 \
        -s "$alignment" \
        -m LG+G4 \
        -fast \
        -nt AUTO -ntmax "$threads" \
        -pre "$tree_prefix" \
        -redo
    cp "${tree_prefix}.treefile" "$tree"
    echo "[tree] wrote $tree"
fi

echo "[tree] done."
