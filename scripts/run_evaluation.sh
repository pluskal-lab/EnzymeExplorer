#!/usr/bin/env bash
# Section 7 — Evaluation driver.
# Runs `enzyme_explorer_main evaluate` + `visualize` for one or more
# evaluation configs.
#
# Usage:
#   scripts/run_evaluation.sh --config <path> --output-name <label>
#   scripts/run_evaluation.sh --all       # run all four canonical eval configs
#   scripts/run_evaluation.sh --sanity <suffix>   # side-by-side outputs under outputs/evaluation_results/_rerun_<suffix>/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG=""
OUTPUT_NAME=""
RUN_ALL=0
SANITY_SUFFIX=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --all)         RUN_ALL=1; shift ;;
        --sanity)      SANITY_SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

run_one() {
    local cfg="$1" name="$2"
    local out="$name"
    if [ -n "$SANITY_SUFFIX" ]; then
        out="_rerun_${SANITY_SUFFIX}/${name}"
    fi
    echo "[run_evaluation] evaluate config=$cfg output=$out"
    enzyme_explorer_main evaluate --config "$cfg" --output-name "$out"
    enzyme_explorer_main visualize --eval-output-name "$out"
}

if [ "$RUN_ALL" = "1" ]; then
    # run_one enzymeexplorer/configs/evaluation/all_methods_comparison/main.yaml all_methods_comparison
    # run_one enzymeexplorer/configs/evaluation/ablation/features.yaml           ablation/features
    run_one enzymeexplorer/configs/evaluation/ablation/plm_plm_domains.yaml    ablation/plm_plm_domains
    run_one enzymeexplorer/configs/evaluation/ablation/ml_plm_domains.yaml     ablation/ml_plm_domains
    run_one enzymeexplorer/configs/evaluation/homology_sweeps/main.yaml        homology_sweeps
    # Calibration lives in its own driver — see scripts/run_calibration.sh.
else
    if [ -z "$CONFIG" ] || [ -z "$OUTPUT_NAME" ]; then
        echo "ERROR: --config and --output-name are required (or use --all)." >&2
        exit 2
    fi
    run_one "$CONFIG" "$OUTPUT_NAME"
fi

echo "[run_evaluation] Done."
