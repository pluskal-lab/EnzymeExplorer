#!/usr/bin/env bash
# Section 8 — Calibration driver.
# Fits per-(classifier, class) calibrators from the latest OOF fold
# predictions and publishes the resulting fit_summary to both:
#
#   outputs/evaluation_results/<output_name>/calibration/fit_summary.csv
#   data/calibration_fit_summary.csv        (default read path for the
#                                            prediction pipeline)
#
# Followed by `visualize` to render the reliability / metrics-grid /
# score-distribution figures under
#   outputs/evaluation_results/<output_name>/plots/calibration/.
#
# Usage:
#   scripts/run_calibration.sh                                    # canonical
#   scripts/run_calibration.sh --config <path> --output-name <name>
#   scripts/run_calibration.sh --sanity <suffix>                  # side-by-side
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="enzymeexplorer/configs/calibration/main.yaml"
OUTPUT_NAME="calibration"
SANITY_SUFFIX=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --sanity)      SANITY_SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

OUT="$OUTPUT_NAME"
CALIBRATE_EXTRA=()
if [ -n "$SANITY_SUFFIX" ]; then
    OUT="_rerun_${SANITY_SUFFIX}/${OUTPUT_NAME}"
    # Sanity runs must not clobber the deploy-side
    # data/calibration_fit_summary.csv — leave the canonical live table
    # alone; the sanity artefacts still land under the _rerun_* dir for
    # side-by-side diffing.
    CALIBRATE_EXTRA+=(--no-publish)
fi

echo "[run_calibration] calibrate config=$CONFIG output=$OUT"
enzyme_explorer_main calibrate --config "$CONFIG" --output-name "$OUT" "${CALIBRATE_EXTRA[@]}"
enzyme_explorer_main visualize --eval-output-name "$OUT"
echo "[run_calibration] Done."
