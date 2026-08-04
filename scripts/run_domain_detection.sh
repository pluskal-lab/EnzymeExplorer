#!/usr/bin/env bash
# Section 2 — Domain detection driver.
# Runs the three canonical domain-detection configs (initial pass, final
# martsDB, full EnzymeExplorer). Also converts the pkl outputs to JSON
# sidecars — post-8a903e3 detection writes both natively, so this only
# matters for historical runs.
#
# Usage:
#   scripts/run_domain_detection.sh                       # canonical run
#   scripts/run_domain_detection.sh --sanity <suffix>     # side-by-side re-run under data/_rerun_<suffix>/
#   scripts/run_domain_detection.sh --sanity <suffix> --only <config-name>   # run just one config
#
# In sanity mode we route --detections-output-path, --detected-regions-root-path,
# --domains-output-path, --secondary-structure-residues-path into an isolated
# per-config subdir so canonical outputs are never overwritten.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="normal"
SUFFIX=""
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sanity) MODE="sanity"; SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"; shift 2 ;;
        --only)   ONLY="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# name | config path | run dir stem (used for sanity subdir + output filenames)
CONFIGS=(
    "martsdb_initial|enzymeexplorer/configs/martsDB_initial_domain_detection_config.yaml|martsDB_detected_domains_initial"
    "martsdb_final|enzymeexplorer/configs/martsDB_domain_detection_config.yaml|martsDB_detected_domains"
    "enzyme_explorer|enzymeexplorer/configs/enzyme_explorer_domain_detection_config.yaml|enzyme_explorer_detected_domains"
)

run_one() {
    local name="$1" cfg="$2" stem="$3"
    if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
        return 0
    fi
    if [ "$MODE" = "sanity" ]; then
        local out_dir="data/_rerun_${SUFFIX}/${stem}"
        mkdir -p "$out_dir/detections" "$out_dir/domains"
        echo "[run_domain_detection][$name] Sanity re-run — outputs under $out_dir"
        python -m enzymeexplorer.src.structure_processing.domain_detections \
            --config "$cfg" \
            --detections-output-path "$out_dir/${stem}.pkl" \
            --detected-regions-root-path "$out_dir/detections" \
            --domains-output-path "$out_dir/domains" \
            --secondary-structure-residues-path "$out_dir/secondary_structure_residues.pkl"
    else
        echo "[run_domain_detection][$name] Canonical run using paths from $cfg"
        python -m enzymeexplorer.src.structure_processing.domain_detections --config "$cfg"
    fi
}

for row in "${CONFIGS[@]}"; do
    IFS='|' read -r name cfg stem <<< "$row"
    run_one "$name" "$cfg" "$stem"
done

echo "[run_domain_detection] Done."
