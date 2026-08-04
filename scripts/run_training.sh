#!/usr/bin/env bash
# Section 6 — Model training driver.
# Trains one or more model families via `enzyme_explorer_main run`.
# Outputs land under outputs/models/<ModelType>/<version>/<fold>/<class>/<TS>/.
#
# Usage:
#   scripts/run_training.sh --families <name>,<name>,...          # train specific families
#   scripts/run_training.sh --families all                        # train every family
#   scripts/run_training.sh --sanity --families PlmRandomForest,PlmDomainsRandomForest
#
# Sanity mode runs on the full dataset (no subset), but is intended for
# lightweight verification: use --families to restrict which model families
# actually train. Reruns are safe because `experiment_runner` skips already-
# completed folds when outputs match the version directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="normal"
FAMILIES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sanity)   MODE="sanity"; shift ;;
        --families) FAMILIES="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$FAMILIES" ]; then
    echo "ERROR: --families is required (comma-separated names or 'all')." >&2
    exit 2
fi

ALL_FAMILIES=(
    PlmRandomForest PlmDomainsRandomForest PlmDomainsMLP PlmDomainsLogisticRegression
    DomainsRandomForest Blastp HMM Foldseek PfamSUPFAM CLEAN
)

if [ "$FAMILIES" = "all" ]; then
    SELECTED=("${ALL_FAMILIES[@]}")
else
    IFS=',' read -ra SELECTED <<< "$FAMILIES"
fi

echo "[run_training] Mode=$MODE, families=${SELECTED[*]}"
for family in "${SELECTED[@]}"; do
    # Discover every non-`.ignore` version under configs/<family>/ and train it.
    if [ ! -d "enzymeexplorer/configs/$family" ]; then
        echo "  SKIP $family: no configs/$family/ directory"
        continue
    fi
    for version_dir in enzymeexplorer/configs/"$family"/*/; do
        [ -d "$version_dir" ] || continue
        version=$(basename "$version_dir")
        # Skip disabled versions
        [[ "$version" == *.ignore ]] && continue
        [ ! -f "$version_dir/config.yaml" ] && continue

        echo "[run_training] training $family/$version"
        enzyme_explorer_main --select-single-experiment run \
            --model "$family" --model-version "$version" || {
            echo "  FAILED: $family/$version (continuing)"
        }
    done
done

echo "[run_training] Done."
