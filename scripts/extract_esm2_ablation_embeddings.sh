#!/bin/bash
# Extract embeddings for the ESM-2 size + layer ablation.
#
# Three base models — t30 (150M), t33 (650M), t36 (3B) — each runs a SINGLE
# forward pass per batch that emits the final five layers' hidden states at
# once (see scripts/extract_plm_embeddings.sh and esm_transformer_utils.py).
#
# Per base model the layer set is the final 5 layers:
#   t30: 26, 27, 28, 29, 30
#   t33: 29, 30, 31, 32, 33
#   t36: 32, 33, 34, 35, 36
#
# Outputs land under:
#   outputs/embeddings_esm-2-t<N>/uniprot_embs_esm-2-t<N>-L<M>/
#   data/gathered_embs_esm-2-t<N>-L<M>_embs_avg.h5
#
# Resume-friendly: each (base, layer) gather is idempotent over the gathered
# .h5 path; the per-batch pickles act as a cache so a rerun continues where
# it stopped on OOM / interruption.

set -euo pipefail

id_column_name="ID"
sequence_column_name="Aminoacid_sequence"
input_csv_path="data/EnzymeExplorer_Dataset.csv"
num_gpus="${NUM_GPUS:-1}"

# Per-base config: model_name, batch_size, comma-separated repr layers.
# Batch sizes shrink for larger models (3B model is memory-heavy).
# Batch sizes shrink with model size; the per-batch OOM fallback in
# transformer_embs.py auto-retries sequence-by-sequence on failure, so a
# single 2.7k-aa entry can't take its whole batch down. t36 stays at 1
# because the 3B model is the binding memory case.
declare -A BATCH_SIZE=( [esm-2-t30]=8 [esm-2-t33]=2 [esm-2-t36]=1 )
declare -A LAYERS=(
    [esm-2-t30]="26,27,28,29,30"
    [esm-2-t33]="29,30,31,32,33"
    [esm-2-t36]="32,33,34,35,36"
)

for base in esm-2-t30 esm-2-t33 esm-2-t36; do
    bsz="${BATCH_SIZE[$base]}"
    layers="${LAYERS[$base]}"
    out_root="outputs/embeddings_${base}"
    echo "=============================================="
    echo "Extracting $base (batch=$bsz, layers=$layers)"
    echo "Output root: $out_root"
    echo "=============================================="
    scripts/extract_plm_embeddings.sh \
        "$base" \
        "$bsz" \
        "$num_gpus" \
        "$layers" \
        "$input_csv_path" \
        "$id_column_name" \
        "$sequence_column_name" \
        "$out_root"
done

echo "All ESM-2 ablation embeddings extracted + gathered."
