#!/bin/bash
# Top-level orchestrator for FASTA TPS screening on a SLURM cluster.
#
# Submits three dependent SLURM array jobs and a final gather job, then
# exits. All actual screening happens in the spawned arrays.
#
# Per-batch flow (one task = one batch = one node = one GPU):
#
#   (optional)              (1 GPU per task)               (CPU, once)
#   download_one_batch  →   predict_one_batch          →   gather
#       AF-DB ingress       PlmRandomForest +              merge per-batch
#       (CPU only)          PlmDomainsRandomForest         shards into one
#                           on the per-batch slice,        CSV per classifier;
#                           writes per-batch shards,       final sweep of
#                           cleans its own PDBs on         <output>/structures/
#                           exit (trap, not a separate     for any leftovers
#                           SLURM job).
#
# Dependencies:
#   - predict array depends on download array via ``aftercorr`` (task i
#     waits only for download task i to succeed; a single failed download
#     doesn't block the rest of the screen).
#   - gather depends on the predict array via ``afterany`` so partial
#     results are still aggregated even if some batches failed.
#
# Usage (sbatch):
#   sbatch scripts/tps_screening_manager.sh \
#       --fasta <fasta> \
#       --output-root <dir> \
#       [--classifier {plm,plm_domains,both}]  (default: both)
#       [--batch-size <int>]                   (default: 40000)
#       [--n-jobs <int>]                       (default: 10)
#       [--structures-dir <dir>]               (default: download per-batch)
#
# When ``--structures-dir`` is OMITTED (the default behaviour), the
# manager submits a CPU-only download array that fetches PDBs from
# AF-DB into ``<output_root>/structures/batch_<idx>/`` per batch.
# Each predict task scrubs its own batch dir on exit (bash trap), so
# accumulating PDBs never pile up across batches. The final gather
# job sweeps ``<output_root>/structures/`` to remove any leftover
# batch dirs from SIGKILLed prediction tasks.
#
# ``--n-jobs`` controls the multiprocessing pool size for domain
# detection inside ``predict_with_structures`` AND is also passed to
# SLURM as ``--cpus-per-task`` for the predict array, so the worker
# never oversubscribes its CPU allocation.
#
# Output layout under <output_root>:
#   shards/<classifier>/batch_<idx>.csv      per-batch raw predictions
#   structures/batch_<idx>/<uid>.pdb         only while a batch is in flight
#   missing_csvs/batch_<idx>.csv             AF-DB 404 list per batch (kept)
#   plm.csv  plm_domains.csv
#   plm_domains_fallback.csv  no_structure.csv      final unified outputs
#
# Logs (all under <output_root>/logs/):
#   manager_<jobid>.log                      this orchestrator job
#   download_<arrayjobid>_batch_<idx>.log    one per AF-DB download task
#   predict_<arrayjobid>_batch_<idx>.log     one per GPU prediction task
#   gather_<jobid>.log                       the final unification job
# Every SLURM stdout AND stderr (Python logger output, foldseek/USalign
# subprocess output, the script's own ``echo`` lines) is routed into
# the corresponding log file via ``sbatch --output=…``, so post-mortem
# on any batch is one ``less`` away.

#SBATCH --job-name=tps_screening_mgr
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=01:00:00

# ``set -u`` is incompatible with cluster bashrc files that read
# unset variables (e.g. ``BASHRCSOURCED``). Sourcing happens with -u
# off so a 3rd-party bashrc bug doesn't take the screening down
# before it starts. We re-enable strict-unbound-var checking for our
# own code right after.
set -eo pipefail
source ~/.bashrc
conda activate enzyme_explorer_prod
set -u

# SLURM copies the submitted batch script to its spool directory before
# running it — ``${BASH_SOURCE[0]}`` therefore resolves to a spool path
# (e.g. /var/spool/slurmd/...) and cannot be used to find sibling
# scripts. Anchor to ``$SLURM_SUBMIT_DIR`` (the directory the user ran
# ``sbatch`` from) instead so relative paths the user typed
# (``--fasta data/foo.fasta --output-root screening_results``) resolve
# the way they expect. Falls back to the current cwd for ad-hoc runs
# outside SLURM.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ---- args ----------------------------------------------------------------

fasta=""
output_root=""
classifier="both"
batch_size=40000
n_jobs=10
structures_dir=""

usage() { sed -n '2,46p' "$0" >&2; exit 1; }

while (( $# > 0 )); do
    case "$1" in
        --fasta)            fasta="$2";          shift 2 ;;
        --output-root)      output_root="$2";    shift 2 ;;
        --classifier)       classifier="$2";     shift 2 ;;
        --batch-size)       batch_size="$2";     shift 2 ;;
        --n-jobs)           n_jobs="$2";         shift 2 ;;
        --structures-dir)   structures_dir="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *) echo "[mgr] unknown arg: $1" >&2; usage ;;
    esac
done

[[ -n "$fasta"        ]] || { echo "[mgr] --fasta is required" >&2; exit 1; }
[[ -n "$output_root"  ]] || { echo "[mgr] --output-root is required" >&2; exit 1; }
# The existence-check on $fasta runs AFTER realpath below, so the
# error message reports the absolute path the script actually tried.
case "$classifier" in
    plm|plm_domains|both) ;;
    *) echo "[mgr] --classifier must be plm|plm_domains|both, got '$classifier'" >&2; exit 1 ;;
esac
[[ "$batch_size" =~ ^[0-9]+$ && "$batch_size" -ge 1 ]] \
    || { echo "[mgr] --batch-size must be positive int (got '$batch_size')" >&2; exit 1; }
[[ "$n_jobs"     =~ ^[0-9]+$ && "$n_jobs"     -ge 1 ]] \
    || { echo "[mgr] --n-jobs must be positive int (got '$n_jobs')" >&2; exit 1; }

# Locate the sibling SLURM scripts via the installed ``enzymeexplorer``
# package — robust to wherever SLURM copied this script (spool) and
# wherever the user submitted from. ``pip install -e .`` makes
# ``enzymeexplorer.__file__`` point inside the repo's package dir;
# its grandparent is the repo root.
scripts_dir="$(python - <<'PY'
import enzymeexplorer, pathlib
print(pathlib.Path(enzymeexplorer.__file__).resolve().parent.parent / "scripts")
PY
)"
[[ -d "$scripts_dir" ]] \
    || { echo "[mgr] ERROR: could not locate scripts/ via the installed enzymeexplorer package (got '$scripts_dir')" >&2; exit 1; }
[[ -f "$scripts_dir/tps_screening_predict_one_batch.sh" ]] \
    || { echo "[mgr] ERROR: missing sibling script under $scripts_dir" >&2; exit 1; }

# Resolve user-supplied paths to absolute, against the cwd we just
# anchored ($SLURM_SUBMIT_DIR). All sbatch'd children receive absolute
# paths so they don't depend on whatever cwd SLURM gives them.
fasta="$(realpath -m "$fasta")"
output_root="$(realpath -m "$output_root")"
[[ -f "$fasta" ]] \
    || { echo "[mgr] ERROR: fasta not found at $fasta" >&2; exit 1; }

mkdir -p "$output_root/shards" "$output_root/logs"
logs_dir="$output_root/logs"

# Redirect this manager job's remaining stdout/stderr into the per-run
# logs directory so the manager's bookkeeping lives next to the batch
# logs. The few lines emitted before this redirect (arg parsing,
# directory creation) still land in the operator's default SLURM file
# — they're trivial and the manager_${jobid}.log file gets every
# subsequent line.
exec >>"$logs_dir/manager_${SLURM_JOB_ID:-local}.log" 2>&1
echo "[mgr] manager log opened at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[mgr] fasta=$fasta  output_root=$output_root"
echo "[mgr] classifier=$classifier  batch_size=$batch_size  n_jobs=$n_jobs"
echo "[mgr] structures_dir=${structures_dir:-<none, will download>}"

# ---- count records → number of batches -----------------------------------

n_seqs=$(python -c "from Bio import SeqIO; print(sum(1 for _ in SeqIO.parse('$fasta','fasta')))")
n_batches=$(( (n_seqs + batch_size - 1) / batch_size ))
last_idx=$(( n_batches - 1 ))
if (( n_batches < 1 )); then
    echo "[mgr] ERROR: empty FASTA" >&2; exit 1
fi
echo "[mgr] $n_seqs sequences  /  batch_size=$batch_size  →  n_batches=$n_batches"
echo "[mgr] classifier=$classifier   n_jobs=$n_jobs"

# ---- decide whether we need an AF-DB download step -----------------------

needs_structures=0
case "$classifier" in plm_domains|both) needs_structures=1 ;; esac

dl_array_id=""
predict_extra=()
if (( needs_structures )) && [[ -z "$structures_dir" ]]; then
    mkdir -p "$output_root/structures" "$output_root/missing_csvs"
    dl_array_id=$(sbatch --parsable \
        --array=0-${last_idx} \
        --output="$logs_dir/download_%A_batch_%a.log" \
        "$scripts_dir/tps_screening_download_one_batch.sh" \
            "$fasta" "$batch_size" \
            "$output_root/structures" \
            "$output_root/missing_csvs")
    echo "[mgr] submitted download array job: $dl_array_id (logs: $logs_dir/download_${dl_array_id}_batch_*.log)"
    # Each predict task looks under <structures_root>/batch_<idx>/ and is
    # responsible for cleaning that dir up via its own exit-time trap.
    predict_extra+=(--structures-root "$output_root/structures")
elif (( needs_structures )); then
    # Shared, pre-populated structures dir; predict tasks never touch it.
    predict_extra+=(--structures-dir "$structures_dir")
    echo "[mgr] structures supplied externally: $structures_dir (no download)"
fi

# ---- predict array (1 GPU per task) --------------------------------------
# ``--cpus-per-task=$n_jobs`` overrides the SBATCH directive in the predict
# script so the SLURM allocation matches the multiprocessing pool size
# used by domain detection. Otherwise --n-jobs=30 on a 10-CPU allocation
# would saturate the host.

predict_deps=()
if [[ -n "$dl_array_id" ]]; then
    predict_deps+=(--dependency=aftercorr:"$dl_array_id")
fi

pred_array_id=$(sbatch --parsable \
    --array=0-${last_idx} \
    --cpus-per-task="$n_jobs" \
    --output="$logs_dir/predict_%A_batch_%a.log" \
    "${predict_deps[@]}" \
    "$scripts_dir/tps_screening_predict_one_batch.sh" \
        "$fasta" "$batch_size" \
        "$output_root/shards" \
        "$classifier" \
        "$n_jobs" \
        "${predict_extra[@]}")
echo "[mgr] submitted predict array job: $pred_array_id (logs: $logs_dir/predict_${pred_array_id}_batch_*.log)"

# ---- final gather (runs once after all predict tasks have completed) -----
# ``afterany`` so partial results are still aggregated when a small
# fraction of batches fails. The gather job also sweeps
# <output_root>/structures/ for any leftover batch dirs (SIGKILLed
# prediction tasks whose exit trap didn't fire).

gather_id=$(sbatch --parsable \
    --dependency=afterany:"$pred_array_id" \
    --output="$logs_dir/gather_%j.log" \
    "$scripts_dir/tps_screening_gather.sh" "$output_root")
echo "[mgr] submitted gather job: $gather_id (log: $logs_dir/gather_${gather_id}.log)"

echo "[mgr] all jobs submitted."
echo "[mgr] Final outputs:   $output_root/{plm,plm_domains,plm_domains_fallback,no_structure}.csv"
echo "[mgr] Logs:            $logs_dir/"
