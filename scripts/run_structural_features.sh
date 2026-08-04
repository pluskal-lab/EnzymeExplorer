#!/usr/bin/env bash
# Section 4 — Structural feature extraction driver.
# Runs foldseek all-vs-all query→reference alignment and produces the
# domain_dist_based_features.pkl consumed by every Section-5 model.
#
# Usage:
#   scripts/run_structural_features.sh                     # canonical run
#   scripts/run_structural_features.sh --sanity <suffix>   # side-by-side outputs under data/_rerun_<suffix>/structural_features/
#
# Sanity mode reroutes the output-directory so the canonical artefact
# stays untouched. The foldseek reference DB cache
# (data/foldseek_cache/<hash>/) is READ-only in either mode.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="normal"
SUFFIX=""
if [ "${1-}" = "--sanity" ]; then
    MODE="sanity"
    SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"
fi

if [ "$MODE" = "sanity" ]; then
    OUT_DIR="data/_rerun_${SUFFIX}/structural_features"
    mkdir -p "$OUT_DIR"
    echo "[run_structural_features] Sanity re-run — output under $OUT_DIR"
    python -m enzymeexplorer.src.structure_processing.get_structural_features \
        --config enzymeexplorer/configs/enzyme_explorer_structural_features_config.yaml \
        --output-directory "$OUT_DIR"
else
    echo "[run_structural_features] Canonical run"
    python -m enzymeexplorer.src.structure_processing.get_structural_features \
        --config enzymeexplorer/configs/enzyme_explorer_structural_features_config.yaml
fi

echo "[run_structural_features] Done."
