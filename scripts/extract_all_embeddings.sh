#!/bin/bash

id_column_name="ID"
sequence_column_name="Aminoacid_sequence"
input_csv_path=data/EnzymeExplorer_Dataset.csv
num_gpus=1

# plm models — the ESM-2 family (incl. the layer-ablation variants) is
# handled by the dedicated ``scripts/extract_esm2_ablation_embeddings.sh``
# so a single forward pass per base model emits all five ablation layers.
model_parameters=(
    "ankh_base 4 -1"
    "ankh_large 4 -1"
    "esm-1v 4 33"
    "ankh_tps 4 -1"
    "esm-1v-finetuned-subseq 8 33"
)

# loop over each model and extract embeddings
for model_param in "${model_parameters[@]}"; do
    read -r model_name batch_size model_representations_layer <<< "$model_param"
    output_root_path="outputs/embeddings_${model_name}"

    scripts/extract_plm_embeddings.sh \
        "$model_name" \
        "$batch_size" \
        "$num_gpus" \
        "$model_representations_layer" \
        "$input_csv_path" \
        "$id_column_name" \
        "$sequence_column_name" \
        "$output_root_path"
done