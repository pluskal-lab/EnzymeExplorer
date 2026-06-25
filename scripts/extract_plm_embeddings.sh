#!/bin/bash
#
# Extract PLM embeddings for one model.
#
# Usage:
#   extract_plm_embeddings.sh \
#       <model_name> <batch_size> <num_gpus> <repr_layer(s)> \
#       <input_csv_path> <id_column_name> <sequence_column_name> \
#       <output_root_path>
#
# <repr_layer(s)> is either a single int (legacy) or a comma-separated list
# (e.g. ``29,30,31,32,33``). For ESM models the list triggers a single
# forward pass per batch with all layers' hidden states extracted at once;
# each layer lands in ``<output_root_path>/uniprot_embs_<model>-L<layer>/``
# and is gathered into its own ``data/gathered_embs_<model>-L<layer>_embs_avg.h5``.
# Ankh models ignore this argument (kept positional for backwards compat).

if [ $# -ne 8 ]; then
    echo "Usage: $0 <model_name> <batch_size> <num_gpus> <model_representations_layer(s)> <input_csv_path> <id_column_name> <sequence_column_name> <output_root_path>"
    exit 1
fi
echo "Extracting embeddings from model $1 with batch size $2 using $3 gpus (from PLM layer(s) $4). Input CSV: $5, id-column: $6 seq-column: $7, output-root-path: $8"
model_name="$1"
batch_size=$2
gpu_count=$3
model_representations_layer="$4"
csv_path="$5"
id_column_name="$6"
sequence_column_name="$7"
output_root_path="$8"

############ computing number of all samples ############
if [ ! -f "$csv_path" ]; then
    echo "Error: File '$csv_path' not found!"
    exit 1
fi
id_column_number=$(awk -F ',' -v column_name="$id_column_name" 'NR==1 {for (i=1; i<=NF; i++) if ($i == column_name) {print i; exit}}' "$csv_path")
if [ -z "$id_column_number" ]; then
    echo "Error: Column '$id_column_name' not found in the CSV file."
    exit 1
fi
number_of_samples=$(awk -F ',' -v col_num="$id_column_number" 'NR > 1 {print $col_num}' "$csv_path" | uniq -c | wc -l)

start_index=0
samples_per_gpu=$(($number_of_samples/gpu_count))
end_index=$samples_per_gpu

############ running in parallel on all gpu's ############
pids=()
gpu_count_remaining=$gpu_count
while [ $gpu_count_remaining -gt 0 ]
do
  gpu_count_remaining=$((gpu_count_remaining-1))
  echo "gpu: $gpu_count_remaining, start index: $start_index, end index: $end_index (count: $((end_index-start_index)))"
  plm_embeddings \
      --start-index $start_index --end-index $end_index --gpu $gpu_count_remaining \
      --model "$model_name" \
      --model-repr-layer "$model_representations_layer" \
      --batch-size $batch_size --csv-path "$csv_path" \
      --id-column "$id_column_name" --seq-column "$sequence_column_name" \
      --output-root-path "$output_root_path" &
  pids+=($!)
  start_index=$end_index
  end_index=$((end_index+samples_per_gpu))
done

for pid in ${pids[*]}
do
  wait $pid
done

############ gathering: one .h5 per output dir ############
# For ESM with a multi-layer list, transformer_embs writes one dir per
# layer (uniprot_embs_<model>-L<layer>); for ankh / single-layer ESM
# there's exactly one dir. Iterate every uniprot_embs_<model>* dir the
# python step produced and gather each into its own .h5.
shopt -s nullglob
for sub_dir in "$output_root_path"/uniprot_embs_"$model_name"*; do
    [ -d "$sub_dir" ] || continue
    echo "Gathering embeddings from $sub_dir"
    gather_plm_embeddings --input-root-path "$sub_dir" --csv-path "$csv_path" --id-column "$id_column_name"
done
