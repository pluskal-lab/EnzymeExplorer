#!/bin/bash
# Drive the 30 ESM-2 ablation trainings (15 variants × {PlmRandomForest,
# PlmDomainsRandomForest}). Sequential — RF training uses every CPU core
# via n_jobs=-1, so parallelising would just thrash. Restart-safe: any
# (Model, variant) whose latest timestamp dir already contains 5
# fold_results.pkl files is treated as done and skipped, so an
# interrupted run can resume by re-invoking this script.

set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="outputs/logs/esm2_train"
mkdir -p "$LOG_DIR"

BASE_LAYERS=(
    "t30:26,27,28,29,30"
    "t33:29,30,31,32,33"
    "t36:32,33,34,35,36"
)

is_done() {
    # Args: <Model> <variant>. Returns 0 if a complete trained ensemble
    # exists, non-zero otherwise. "Complete" means at least one
    # timestamp dir under all_folds/all_classes/ contains 5
    # fold_results.pkl files.
    local model="$1"
    local variant="$2"
    local root="outputs/${model}/${variant}/all_folds/all_classes"
    [ -d "$root" ] || return 1
    for ts_dir in "$root"/*/; do
        [ -d "$ts_dir" ] || continue
        local n=$(find "$ts_dir" -maxdepth 1 -name "fold_*_results.pkl" 2>/dev/null | wc -l)
        if [ "$n" -ge 5 ]; then
            return 0
        fi
    done
    return 1
}

source /opt/conda/etc/profile.d/conda.sh
conda activate enzyme_explorer

total_done=0
total_skipped=0
total_runs=0
total_failed=0
for entry in "${BASE_LAYERS[@]}"; do
    base="${entry%%:*}"
    layers="${entry##*:}"
    IFS=',' read -ra LAYER_LIST <<< "$layers"
    for L in "${LAYER_LIST[@]}"; do
        variant="esm-2-${base}-L${L}"
        for model in PlmRandomForest PlmDomainsRandomForest; do
            total_runs=$((total_runs+1))
            if is_done "$model" "$variant"; then
                echo "[$(date -Iseconds)] SKIP (already trained): $model / $variant"
                total_skipped=$((total_skipped+1))
                continue
            fi
            log_file="$LOG_DIR/${model}_${variant}.log"
            echo "[$(date -Iseconds)] RUN: $model / $variant -> $log_file"
            t0=$(date +%s)
            if enzyme_explorer_main --select-single-experiment run \
                    --model "$model" --model-version "$variant" \
                    > "$log_file" 2>&1; then
                dt=$(( $(date +%s) - t0 ))
                echo "[$(date -Iseconds)] DONE in ${dt}s: $model / $variant"
                total_done=$((total_done+1))
            else
                rc=$?
                dt=$(( $(date +%s) - t0 ))
                echo "[$(date -Iseconds)] FAIL (rc=$rc, ${dt}s): $model / $variant — see $log_file"
                total_failed=$((total_failed+1))
            fi
        done
    done
done

echo ""
echo "[$(date -Iseconds)] SUMMARY: total=$total_runs skipped=$total_skipped done=$total_done failed=$total_failed"
