#!/usr/bin/env bash
# Section 5 — PLM embedding extraction driver.
# Runs plm_embeddings for each model in EMBEDDING_MODELS and then
# gathers the per-batch pickles into a single data/gathered_embs_<model>_embs_avg.h5.
#
# Usage:
#   scripts/run_plm_embeddings.sh                     # canonical run (all models)
#   scripts/run_plm_embeddings.sh --sanity <suffix>   # side-by-side re-run on a 20-row subset
#   scripts/run_plm_embeddings.sh --model <name>      # extract just one model
#
# Full canonical extraction takes many GPU hours — sanity mode runs a
# tiny subset per model to verify the pipeline; it does NOT reproduce
# byte-identical embeddings (PyTorch/CUDA are not deterministic at the
# fp precision needed for that).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# One canonical model list — matches the paper + rebuttal ablation set.
EMBEDDING_MODELS=(
    "ankh_base:4"
    "ankh_large:4"
    "ankh_tps:4"
    "esm-1v:4:33"
    "esm-1v-finetuned-subseq:8:33"
    "esm-2-t36:1:36"
)

MODE="normal"
SUFFIX=""
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sanity) MODE="sanity"; SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"; shift 2 ;;
        --model)  ONLY="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ "$MODE" = "sanity" ]; then
    OUT_DIR="data/_rerun_${SUFFIX}/embeddings"
    GATHER_DIR="data/_rerun_${SUFFIX}"
    mkdir -p "$OUT_DIR"
    echo "[run_plm_embeddings] Sanity re-run — outputs under $OUT_DIR"
    # Sanity uses a 20-row subset of the dataset to make the pipeline
    # completeable in minutes rather than hours.
    HEAD_ARGS=("--end-index" "20")
else
    OUT_DIR="outputs"
    GATHER_DIR="data"
    echo "[run_plm_embeddings] Canonical run"
    HEAD_ARGS=()
fi

extract_one() {
    local model="$1" bs="$2" layer="${3:-}"
    if [ -n "$ONLY" ] && [ "$ONLY" != "$model" ]; then return 0; fi

    echo "[run_plm_embeddings][$model] extracting (batch_size=$bs${layer:+, layer=$layer})"
    local layer_args=()
    [ -n "$layer" ] && layer_args=(--model-repr-layer "$layer")

    python -m enzymeexplorer.src.embeddings_extraction.transformer_embs \
        --csv-path data/EnzymeExplorer_Dataset.csv \
        --id-column ID \
        --seq-column Aminoacid_sequence \
        --model "$model" \
        --batch-size "$bs" \
        --output-root-path "$OUT_DIR/embeddings_${model}${layer:+-L$layer}" \
        --gpu 0 \
        "${layer_args[@]}" \
        "${HEAD_ARGS[@]}"

    # Gather per-batch pickles into a single .h5.
    local subdir="$OUT_DIR/embeddings_${model}${layer:+-L$layer}/uniprot_embs_${model}${layer:+-L$layer}"
    if [ -d "$subdir" ]; then
        python -m enzymeexplorer.src.embeddings_extraction.gather_required_embs \
            --input-root-path "$subdir/" \
            --csv-path data/EnzymeExplorer_Dataset.csv \
            --id-column ID \
            --output-dir "$GATHER_DIR"
    fi
}

for row in "${EMBEDDING_MODELS[@]}"; do
    IFS=':' read -r model bs layer <<< "$row"
    extract_one "$model" "$bs" "$layer"
done

echo "[run_plm_embeddings] Done."
