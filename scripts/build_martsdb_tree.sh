#!/bin/bash
# Build the MARTS-DB phylogeny end-to-end (all 1,374 sequences,
# keyed by marts_E… IDs).
#
# Stage 1: MAFFT --auto on martsdb.fasta  -> alignment.fasta
# Stage 2: IQ-TREE -fast (LG+G4)          -> tree.nwk
#
# Re-running is safe: each stage skips work whose output already exists.

set -euo pipefail

# Ensure the active conda env's binaries are found even under nohup
# (no `conda activate` in the child shell). CONDA_PREFIX is set by
# `conda activate`; fall back to $PATH otherwise.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

workdir="outputs/martsdb/phylogeny"
threads="${1:-16}"

fasta="$workdir/martsdb.fasta"
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
