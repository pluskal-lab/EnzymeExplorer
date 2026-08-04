#!/usr/bin/env bash
# Section 1 — Dataset preparation driver.
# Usage:
#   scripts/run_dataprep.sh                          # full run, writes to canonical paths in the YAML
#   scripts/run_dataprep.sh --sanity <suffix>        # side-by-side re-run under data/_rerun_<suffix>/
#   scripts/run_dataprep.sh --sanity <suffix> --full # sanity re-run WITHOUT the cached preprocessed swissprot shortcut
#
# In sanity mode we point every writable output at the isolated dir so
# canonical outputs are never overwritten; by default we load the cached
# preprocessed SwissProt frame (data/non_tpss_swissprot.tsv) to skip the
# multi-hour Pfam+SUPFAM hmmscan step. Use --full to force the full re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="normal"
SUFFIX=""
USE_CACHED_SWISSPROT=1
if [ "${1-}" = "--sanity" ]; then
    MODE="sanity"
    SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"
    if [ "${3-}" = "--full" ]; then
        USE_CACHED_SWISSPROT=0
    fi
fi

if [ "$MODE" = "sanity" ]; then
    OUT_DIR="data/_rerun_${SUFFIX}"
    mkdir -p "$OUT_DIR"
    echo "[run_dataprep] Sanity re-run — outputs under $OUT_DIR (canonical files untouched)."

    EXTRA_ARGS=()
    if [ "$USE_CACHED_SWISSPROT" = "1" ] && [ -f "data/non_tpss_swissprot.tsv" ]; then
        echo "[run_dataprep] Loading cached preprocessed SwissProt (skips Pfam+SUPFAM hmmscan)."
        EXTRA_ARGS+=(--preprocessed-swissprot-load-path "data/non_tpss_swissprot.tsv")
    fi

    # --structures-root left at YAML default (data/enzyme_explorer_pdbs). Dir is already
    # populated, so download_af_structure calls short-circuit as existence checks — this
    # keeps the negatives filter (rows dropped for missing structures) byte-identical.
    python -m enzymeexplorer.src.data_preparation.prepare_dataset \
        --config enzymeexplorer/configs/dataprep_config.yaml \
        --presplit-martsdb-clusters-csv-path "$OUT_DIR/EnzymeExplorer_Dataset_TPS.csv" \
        --preprocessed-swissprot-save-path "$OUT_DIR/non_tpss_swissprot.tsv" \
        --dataset-output-path "$OUT_DIR/EnzymeExplorer_Dataset.csv" \
        "${EXTRA_ARGS[@]}"

    # Regenerate MartsDB figures into the isolated dir.
    echo "[run_dataprep] Generating MartsDB paper figures into outputs/martsdb/stats"
    python scripts/generate_martsdb_statistics_figure.py"
else
    echo "[run_dataprep] Full run using canonical output paths from dataprep_config.yaml."
    python -m enzymeexplorer.src.data_preparation.prepare_dataset \
        --config enzymeexplorer/configs/dataprep_config.yaml

    echo "[run_dataprep] Generating MartsDB paper figures..."
    python scripts/generate_martsdb_statistics_figure.py
fi

echo "[run_dataprep] Done."
