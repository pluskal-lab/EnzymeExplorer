#!/bin/bash
# Per-batch TPS prediction worker — single node, single GPU, single
# process. Spawned as a SLURM array task by tps_screening_manager.sh;
# $SLURM_ARRAY_TASK_ID picks which batch this task handles.
#
# Usage (positional, set by the manager):
#   tps_screening_predict_one_batch.sh \
#       <fasta> <batch_size> <shards_root> <classifier> <n_jobs> \
#       [--structures-root <root> | --structures-dir <dir>]
#
# Slice covered by this task: [batch * batch_size, (batch+1) * batch_size).
# Output: <shards_root>/<classifier>/batch_<idx>.csv (+ companion
# sub-dirs depending on classifier; see tps_predict_fasta docstring).
#
# Structure resolution for plm_domains:
#   --structures-root <root>  → look at <root>/batch_<idx>/ (per-batch
#                                downloads from the manager's AF-DB step).
#                                The worker takes OWNERSHIP of that
#                                directory and removes it via an EXIT
#                                trap so accumulating PDBs don't pile
#                                up across batches. Works on success,
#                                Python exception, SLURM cancel
#                                (SIGTERM → graceful_shutdown → trap),
#                                or set-e shell failure — not on
#                                SIGKILL (the gather job's final sweep
#                                cleans those leftovers).
#   --structures-dir <dir>    → use the same flat directory of <uid>.pdb
#                                files for every batch (manager was given
#                                a pre-populated structures_dir). NOT
#                                deleted by the trap — external state.
#   neither                   → worker falls back to inline AF-DB
#                                downloads under its managed_workdir.
#
# ``n_jobs`` flows to ``predict_with_structures(n_jobs=...)`` which in
# turn drives the domain-detection multiprocessing pool. The manager
# submits this script with ``sbatch --cpus-per-task=$n_jobs`` so the
# allocation matches.

#SBATCH --job-name=tps_screen_predict
#SBATCH --partition=standard-g
#SBATCH --account=project_465000660
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

# Sourcing bashrc with -u disabled (some cluster /etc/bashrc files
# reference unbound vars and would otherwise abort us before the
# pipeline runs).
set -eo pipefail
source ~/.bashrc
conda activate enzyme_explorer
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ---- positional args -----------------------------------------------------

fasta="$1"
batch_size="$2"
shards_root="$3"
classifier="$4"
n_jobs="$5"
shift 5

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

echo "[predict] batch $batch_idx range [$start_i, $end_i)  classifier=$classifier  n_jobs=$n_jobs"

# ---- ownership-aware cleanup -----------------------------------------------
# If we got --structures-root, the manager downloaded into a batch_<idx>/
# sub-directory that this task OWNS. Register an EXIT trap that removes
# it regardless of whether prediction succeeded — this is what keeps the
# disk footprint of a screen of millions bounded by one batch's worth
# of PDBs at a time. ``set -e`` ensures the trap fires on shell failure
# too; ``graceful_shutdown`` in the Python entry point converts SIGTERM
# into a clean exception that unwinds back to here so the trap fires
# on SLURM ``scancel`` as well.

owned_batch_struct_dir=""
predict_struct_args=()
if [[ -n "$structures_root" ]]; then
    owned_batch_struct_dir="$structures_root/$batch_name"
    if [[ -d "$owned_batch_struct_dir" ]]; then
        predict_struct_args+=(--structures-dir "$owned_batch_struct_dir")
    else
        echo "[predict] WARN: $owned_batch_struct_dir missing; running without structures"
    fi
elif [[ -n "$structures_dir" ]]; then
    predict_struct_args+=(--structures-dir "$structures_dir")
fi

cleanup_owned() {
    local rc=$?
    if [[ -n "$owned_batch_struct_dir" && -d "$owned_batch_struct_dir" ]]; then
        echo "[predict] removing per-batch structures: $owned_batch_struct_dir"
        rm -rf "$owned_batch_struct_dir"
    fi
    return $rc
}
trap cleanup_owned EXIT

# ---- run -----------------------------------------------------------------

python -m enzymeexplorer.src.screening.tps_predict_fasta \
    --fasta-path "$fasta" \
    --output-dir "$shards_root" \
    --shard-name "$batch_name" \
    --start-i "$start_i" \
    --end-i   "$end_i" \
    --classifier "$classifier" \
    --n-jobs "$n_jobs" \
    "${predict_struct_args[@]}"
