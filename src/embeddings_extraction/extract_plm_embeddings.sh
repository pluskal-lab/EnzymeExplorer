#!/bin/bash

# check that all arguments are provided
if [ $# -ne 8 ]; then
    echo "Usage: $0 <model_name> <batch_size> <num_gpus> <model_representations_layer> <input_csv_path> <id_column_name> <sequence_column_name> <output_root_path>"
    exit 1
fi
echo "Extracting embeddings from model $1 with batch size $2 using $3 gpus (from PLM layer $4). Input CSV: $5, id-column: $6 seq-column: $7, output-root-path: $8"
model_name="$1"
batch_size=$2
gpu_count=$3
model_representations_layer=$4
csv_path="$5"
id_column_name="$6"
sequence_column_name="$7"
output_root_path="$8"

############ computing number of all samples ############
# check if the file exists
if [ ! -f "$csv_path" ]; then
    echo "Error: File '$csv_path' not found!"
    exit 1
fi
id_column_number=$(awk -F ',' -v column_name="$id_column_name" 'NR==1 {for (i=1; i<=NF; i++) if ($i == column_name) {print i; exit}}' "$csv_path")
# check if ID column name exists
if [ -z "$id_column_number" ]; then
    echo "Error: Column '$id_column_name' not found in the CSV file."
    exit 1
fi
number_of_samples=$(awk -F ',' -v col_num="$id_column_number" 'NR > 1 {print $col_num}' "$csv_path" | uniq -c | wc -l)

# per-gpu load
start_index=0
samples_per_gpu=$(($number_of_samples/gpu_count))
end_index=$samples_per_gpu
batches_per_epoch=$((samples_per_gpu/batch_size))

############ running in parallel on all gpu's ############
pids=()
while [ $gpu_count -gt 0 ]
do
  gpu_count=$((gpu_count-1))
  echo "gpu: $gpu_count, start index: $start_index, end index: $end_index (count: $((end_index-start_index)))"
  python -m src.embeddings_extraction.transformer_embs --start-index $start_index --end-index $end_index --gpu $gpu_count --model "$model_name" --model-repr-layer "$model_representations_layer" --batch-size $batch_size --csv-path "$csv_path" --id-column "$id_column_name" --seq-column "$sequence_column_name" --output-root-path "$output_root_path" &
  pids+=($!)
  numbering_offset=$((numbering_offset+batches_per_epoch))
  start_index=$end_index
  end_index=$((end_index+samples_per_gpu))
done

############ waiting for all gpu's to complete computation ############
for pid in ${pids[*]}
do
  wait $pid
done

############ computing number of all samples ############
python -m src.embeddings_extraction.gather_required_embs --input-root-path "$output_root_path/uniprot_embs_$model_name" --csv-path "$csv_path" --id-column "$id_column_name"