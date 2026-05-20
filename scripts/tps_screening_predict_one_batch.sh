#!/bin/bash
# Per-batch TPS prediction worker — single node, single GPU, single
# process. Spawned as a SLURM array task by tps_screening_manager.sh;
# $SLURM_ARRAY_TASK_ID picks which batch this task handles.
#
# Usage (positional, set by the manager):
#   tps_screening_predict_one_batch.sh \
#       <fasta> <batch_size> <shards_root> <classifier> <n_jobs> \
#       <plm_batch_size> \
#       [--structures-root <root> | --structures-dir <dir>] \
#       [--workdir <dir>]
#
# Slice covered by this task: [batch * batch_size, (batch+1) * batch_size).
# Output: <shards_root>/<classifier>/batch_<idx>.csv (+ companion
# sub-dirs depending on classifier; see tps_predict_fasta docstring).
#
# Structure resolution for plm_domains:
#   --structures-root <root>  → look at <root>/batch_<idx>/ (per-batch
#                                downloads from the manager's AF-DB step).
#                                The worker takes OWNERSHIP of that
#                                directory and removes it AFTER the
#                                Python entry point exits with status 0.
#                                Any non-success exit (Python exception,
#                                SLURM ``scancel`` SIGTERM, SIGKILL,
#                                ``set -e`` shell failure) leaves the
#                                directory intact so a re-run can skip
#                                already-downloaded PDBs and pick up
#                                where the previous attempt died.
#   --structures-dir <dir>    → use the same flat directory of <uid>.pdb
#                                files for every batch (manager was given
#                                a pre-populated structures_dir). Never
#                                touched — external state.
#   neither                   → worker falls back to inline AF-DB
#                                downloads under its managed_workdir.
#
# ``n_jobs`` flows to ``predict_with_structures(n_jobs=...)`` which in
# turn drives the domain-detection multiprocessing pool. The manager
# submits this script with ``sbatch --cpus-per-task=$n_jobs`` so the
# allocation matches.

#SBATCH --job-name=tps_screen_predict
#SBATCH --account=project_465000660
#SBATCH --partition qgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16GB
#SBATCH --time 4:00:00

# Sourcing bashrc with -u disabled (some cluster /etc/bashrc files
# reference unbound vars and would otherwise abort us before the
# pipeline runs).
set -eo pipefail
source ~/.bashrc
conda activate enzyme_explorer_prod
set -u

# No ``cd`` needed: every path the manager passes is absolute, and
# ``python -m enzymeexplorer.src.screening.tps_predict_fasta`` finds
# the package via sys.path (installed by ``pip install -e .``).
# ``${BASH_SOURCE[0]}`` here resolves to a SLURM spool path, so any
# cwd derived from it would be wrong anyway.

# ---- positional args -----------------------------------------------------

fasta="$1"
batch_size="$2"
shards_root="$3"
classifier="$4"
n_jobs="$5"
plm_batch_size="$6"
shift 6

# ---- optional structure flags --------------------------------------------

structures_root=""
structures_dir=""
workdir_args=()
min_tps_p_args=()
while (( $# > 0 )); do
    case "$1" in
        --structures-root) structures_root="$2"; shift 2 ;;
        --structures-dir)  structures_dir="$2";  shift 2 ;;
        --workdir)         workdir_args=(--workdir "$2"); shift 2 ;;
        --min-tps-p)       min_tps_p_args=(--min-tps-p "$2"); shift 2 ;;
        *) echo "[predict] unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- compute this task's batch window ------------------------------------

batch_idx="${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array task}"
start_i=$(( batch_idx * batch_size ))
end_i=$(( start_i + batch_size ))
batch_name=$(printf "batch_%06d" "$batch_idx")

echo "[predict] batch $batch_idx range [$start_i, $end_i)  classifier=$classifier  n_jobs=$n_jobs  plm_batch_size=$plm_batch_size"

# ---- structure-dir resolution + ownership tracking -----------------------

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

# ---- run -----------------------------------------------------------------
# The embeddings cache lives one level up from the shards root so it
# survives the gather job's ``--delete-shards`` sweep. Persisted PLM
# embeddings let a rerun of this batch skip the expensive
# ankh_large/embedding step and go straight to classifier inference.

embeddings_cache_dir="$(dirname "$shards_root")/embeddings_cache"

python -m enzymeexplorer.src.screening.tps_predict_fasta \
    --fasta-path "$fasta" \
    --output-dir "$shards_root" \
    --shard-name "$batch_name" \
    --start-i "$start_i" \
    --end-i   "$end_i" \
    --classifier "$classifier" \
    --n-jobs "$n_jobs" \
    --plm-batch-size "$plm_batch_size" \
    --embeddings-cache-dir "$embeddings_cache_dir" \
    "${predict_struct_args[@]}" \
    "${workdir_args[@]}" \
    "${min_tps_p_args[@]}"

# Cleanup runs ONLY when Python exited 0. Any other exit (Python crash,
# SLURM ``scancel`` SIGTERM, SIGKILL, ``set -e`` shell failure) bails
# out of the script before reaching this line, leaving the per-batch
# structures dir intact so a rerun can pick up where this attempt died
# (af_db.download_one skips PDBs that already exist on disk).
if [[ -n "$owned_batch_struct_dir" && -d "$owned_batch_struct_dir" ]]; then
    echo "[predict] prediction succeeded; removing per-batch structures: $owned_batch_struct_dir"
    rm -rf "$owned_batch_struct_dir"
fi
