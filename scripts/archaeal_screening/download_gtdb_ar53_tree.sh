#!/usr/bin/env bash
# Download the GTDB ar53 (release 232) archaeal phylogeny and post-process
# it into an iTOL-loadable Newick.
#
# Called from setup_dev.sh; also invocable standalone from a checkout.
# Idempotent: skips the download if ar53.tree is already present; skips
# cleaning if ar53_clean.tree already exists.
#
# Inputs : (URL only)
# Outputs:
#   data/archaeal_screening/ar53.tree        (raw GTDB drop)
#   data/archaeal_screening/ar53_clean.tree  (iTOL-loadable, see
#                                             clean_gtdb_tree.py)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

URL="https://data.gtdb.ecogenomic.org/releases/release232/232.0/ar53_r232.tree.gz"
DEST_DIR="data/archaeal_screening"
RAW="$DEST_DIR/ar53.tree"
CLEAN="$DEST_DIR/ar53_clean.tree"

mkdir -p "$DEST_DIR"

if [ -s "$RAW" ]; then
    echo "[archaeal_tree] $RAW exists — skipping download"
else
    echo "[archaeal_tree] curl $URL"
    curl -fSL "$URL" -o "$RAW.gz"
    gunzip -f "$RAW.gz"        # produces $DEST_DIR/ar53.tree (gunzip renames by stripping .gz)
    # ``ar53_r232.tree.gz`` extracts to ``ar53_r232.tree`` locally, but
    # curl -o put it at ar53.tree.gz which gunzip → ar53.tree directly.
fi

if [ -s "$CLEAN" ]; then
    echo "[archaeal_tree] $CLEAN exists — skipping clean"
else
    echo "[archaeal_tree] clean_gtdb_tree: strip quoted internal labels"
    python -m scripts.archaeal_screening.clean_gtdb_tree \
        --input "$RAW" --output "$CLEAN"
fi

echo "[archaeal_tree] Done."
