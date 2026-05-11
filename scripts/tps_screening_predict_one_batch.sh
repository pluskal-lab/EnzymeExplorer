#!/bin/bash
# Per-batch TPS prediction worker — single node, single GPU, single
# process. Spawned as a SLURM array task by tps_screening_manager.sh;
# $SLURM_ARRAY_TASK_ID picks which batch this task handles.
#
# Usage (positional, set by the manager):
#   tps_screening_predict_one_batch.sh \
#       <fasta> <batch_size> <shards_root> <classifier> \
#       [--structures-root <root> | --structures-dir <dir>]
#
# Slice covered by this task: [batch * batch_size, (batch+1) * batch_size).
# Output: <shards_root>/<classifier>/batch_<idx>.csv (+ companion
# sub-dirs depending on classifier; see tps_predict_fasta docstring).
#
# Structure resolution for plm_domains:
#   --structures-root <root>  → look at <root>/batch_<idx>/ (per-batch
#                                downloads from the manager's AF-DB step)
#   --structures-dir <dir>    → use the same flat directory of <uid>.pdb
#                                files for every batch (manager was given a
#                                pre-populated structures_dir argument)
# When neither flag is set and the classifier needs structures, the
# worker downloads from AF-DB inline.

#SBATCH --job-name=tps_screen_predict
#SBATCH --partition=standard-g
#SBATCH --account=project_465000660
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

set -euo pipefail

source ~/.bashrc
conda activate enzyme_explorer

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ---- positional args -----------------------------------------------------

fasta="$1"
batch_size="$2"
shards_root="$3"
classifier="$4"
shift 4

# ---- optional structure flags --------------------------------------------

structures_root=""
structures_dir=""
while (( $# > 0 )); do
    case "$1" in
        --structures-root) structures_root="$2"; shift 2 ;;
        --structures-dir)  structures_dir="$2";  shift 2 ;;
        *) echo "[predict] unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- compute this task's batch window ------------------------------------

batch_idx="${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array task}"
start_i=$(( batch_idx * batch_size ))
end_i=$(( start_i + batch_size ))
batch_name=$(printf "batch_%06d" "$batch_idx")

echo "[predict] batch $batch_idx  range [$start_i, $end_i)  classifier=$classifier"

# ---- resolve --structures-* for tps_predict_fasta ------------------------

predict_struct_args=()
if [[ -n "$structures_root" ]]; then
    # Manager downloaded per-batch into $structures_root/batch_<idx>/.
    # If the dir is missing entirely (download failed for this batch),
    # let the worker fall through with no --structures-dir — every ID
    # in the batch will end up in the no_structure shard.
    batch_struct_dir="$structures_root/$batch_name"
    if [[ -d "$batch_struct_dir" ]]; then
        predict_struct_args+=(--structures-dir "$batch_struct_dir")
    else
        echo "[predict] WARN: $batch_struct_dir missing; running without structures"
    fi
elif [[ -n "$structures_dir" ]]; then
    predict_struct_args+=(--structures-dir "$structures_dir")
fi

# ---- run -----------------------------------------------------------------

python -m enzymeexplorer.src.screening.tps_predict_fasta \
    --fasta-path "$fasta" \
    --output-dir "$shards_root" \
    --shard-name "$batch_name" \
    --start-i "$start_i" \
    --end-i   "$end_i" \
    --classifier "$classifier" \
    "${predict_struct_args[@]}"
