#!/bin/bash
# Build the dark-candidates-selection phylogeny (MARTS-DB TPS-only +
# dark putatives filtered at TPS_p > 0.95). Mirrors
# scripts/build_martsdb_tree.sh:
#   Stage 1: MAFFT --auto           -> alignment.fasta
#   Stage 2: IQ-TREE -fast (LG+G4)  -> tree.nwk
# Re-runnable: each stage skips work whose output already exists.
#
# Assumes the ``enzyme_explorer`` conda env is active before invocation
# (mafft + iqtree2 resolvable on $PATH).

set -euo pipefail

workdir="data/dark_proteome_screening/candidate_selection/phylo_tree"
threads="${1:-16}"

fasta="$workdir/combined.fasta"
alignment="$workdir/alignment.fasta"
tree_prefix="$workdir/iqtree"
tree="$workdir/tree.nwk"

[[ -f "$fasta" ]] || { echo "Missing $fasta" >&2; exit 1; }

if [[ -s "$alignment" ]]; then
    echo "[tree] alignment exists — skipping MAFFT"
else
    echo "[tree] MAFFT --auto --thread $threads --anysymbol"
    mafft --auto --thread "$threads" --anysymbol "$fasta" > "$alignment.tmp"
    mv "$alignment.tmp" "$alignment"
fi

if [[ -s "$tree" ]]; then
    echo "[tree] tree exists — skipping IQ-TREE"
else
    echo "[tree] IQ-TREE -fast LG+G4"
    iqtree2 \
        -s "$alignment" \
        -m LG+G4 \
        -fast \
        -nt AUTO -ntmax "$threads" \
        -pre "$tree_prefix" \
        -redo
    cp "${tree_prefix}.treefile" "$tree"
fi

echo "[tree] done: $tree"
