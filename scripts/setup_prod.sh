#!/usr/bin/env bash
# EnzymeExplorer — production environment setup
# ==============================================
# Stands up a lean conda environment + downloads the deploy-side artifacts
# so the host can run the prediction & screening pipelines
# (predict_with_structures, predict_sequences_only, tps_predict_fasta).
#
# Unix only. Requires `conda` on PATH.
#
# Usage:
#   scripts/setup_prod.sh \
#       [--env-name enzyme_explorer_prod] \
#       [--cuda cu124 | --cpu] \
#       [--force]
#
# What it installs
#   * conda env <ENV_NAME> with python=3.10 + libstdcxx-ng (conda-forge,
#     supplies GLIBCXX_3.4.32+ that the bundled USalign binary needs) +
#     pymol-bundle + pymol-psico.
#   * pip deps + EnzymeExplorer itself (editable) so the console scripts
#     `predict_with_structures`, `predict_sequences_only`, and the
#     unified `enzyme_explorer_main` CLI land on PATH.
#   * Every Google-Drive artifact whose ``required_by`` in
#     ``drive/bundles.json`` includes ``prod``. Currently that means the
#     deploy model bundles + calibration table, the MARTS-DB reference
#     domains + foldseek DB cache, the pinned foldseek / USalign
#     binaries, and the AlphaFold-DB structures for the curated
#     showcase candidates.
#
# Download & extract flow is driven by ``scripts/drive_helper.py`` — the
# URLs live in ``drive/bundles.json`` (single source of truth).
# Cached zips land under ``.drive_cache/`` and are reused across
# re-runs; MANIFEST.txt sha256 verification is per-entry inside each zip.
#
# Idempotency
#   Re-running is safe: cached zips reuse, existing target files are
#   left in place unless --force is given. If <ENV_NAME> already exists,
#   we verify it (correct python, key deps importable, foldseek/USalign
#   on PATH) and warn+exit non-zero on failure — never silently skip.

set -euo pipefail

ENV_NAME="enzyme_explorer_prod"
CUDA_VER="cu124"           # blanked by --cpu
FORCE=0

usage() { sed -n '2,25p' "$0"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --cuda)     CUDA_VER="$2"; shift 2 ;;
        --cpu)      CUDA_VER=""; shift ;;
        --force)    FORCE=1; shift ;;
        -h|--help)  usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

log()  { printf '[setup_prod] %s\n' "$*"; }
warn() { printf '[setup_prod] WARN: %s\n' "$*" >&2; }
die()  { printf '[setup_prod] ERROR: %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v conda >/dev/null 2>&1 \
    || die "conda not found on PATH. Install Miniconda/Mambaforge and re-run."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1. env: build or verify
# ---------------------------------------------------------------------------

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

verify_env() {
    local env="$1"
    log "verifying env '$env' …"
    conda run -n "$env" --no-capture-output bash -c '
        set -e
        py_ver=$(python -c "import sys; print(sys.version_info[:2])")
        [[ "$py_ver" == "(3, 10)" ]] || { echo "wrong python: $py_ver"; exit 11; }
        python -c "import numpy, pandas, scipy, sklearn, torch, ankh, Bio, tqdm, yaml, configargparse, matplotlib, filelock, gdown" \
            || { echo "missing pip dep"; exit 12; }
        python -c "import sklearn; assert sklearn.__version__.startswith(\"1.5.\"), sklearn.__version__" \
            || { echo "wrong sklearn version"; exit 13; }
        python -c "from pymol import cmd" \
            || { echo "pymol import broken"; exit 14; }
        python -c "from psico.exporting import save_pdb_without_ter" \
            || { echo "pymol-psico import broken"; exit 15; }
        command -v foldseek >/dev/null || { echo "foldseek missing"; exit 16; }
        command -v USalign  >/dev/null || { echo "USalign missing";  exit 17; }
        python -c "from enzymeexplorer.src.prediction.pipeline import predict_sequences_only" \
            || { echo "enzymeexplorer not importable"; exit 18; }
        echo OK
    '
}

if env_exists "$ENV_NAME"; then
    if [[ $FORCE -eq 1 ]]; then
        log "--force given; removing existing env '$ENV_NAME'"
        conda env remove -n "$ENV_NAME" -y
    else
        if verify_env "$ENV_NAME"; then
            log "env '$ENV_NAME' verifies clean — skipping env build"
            SKIP_ENV_BUILD=1
        else
            warn "env '$ENV_NAME' exists but verification failed."
            warn "Re-run with --force to recreate, or repair manually."
            exit 3
        fi
    fi
fi

if [[ "${SKIP_ENV_BUILD:-0}" -ne 1 ]]; then
    log "creating conda env '$ENV_NAME' (python=3.10 + pymol + psico)"
    # pymol-bundle pinned to 3.1.6.1 (upstream PyMOL 3.1.6.1 — same
    # ABI as the pymol-bundle 3.1.6.1 used by the dev env, so pymol-psico
    # 3.4.19 links against both interchangeably).
    conda create -y -n "$ENV_NAME" \
        -c conda-forge -c bioconda -c speleo3 -c schrodinger \
        python=3.10.0 \
        pymol=3.1.6.1 \
        pymol-bundle=3.1.6.1 \
        pymol-psico=3.4.19

    log "installing pip deps into '$ENV_NAME'"
    PIP() { conda run -n "$ENV_NAME" --no-capture-output pip "$@"; }
    PIP install --upgrade pip

    if [[ -n "$CUDA_VER" ]]; then
        log "installing torch with $CUDA_VER wheels"
        PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url "https://download.pytorch.org/whl/$CUDA_VER"
    else
        log "installing CPU-only torch wheels"
        PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url "https://download.pytorch.org/whl/cpu"
    fi

    # sklearn 1.5.1 pin must match the version that pickled the fold bundles.
    PIP install \
        "scikit-learn==1.5.1" \
        "numpy==1.26.4" \
        "pandas==2.2.2" \
        "scipy==1.13.0" \
        "ankh==1.10.0" \
        "biopython==1.83" \
        "tqdm" \
        "pyyaml" \
        "dataclasses-json" \
        "configargparse" \
        "matplotlib" \
        "filelock" \
        "gdown" \
        "tables"

    PIP install -e "$REPO_ROOT"
fi

# ---------------------------------------------------------------------------
# 2. Drive artifacts — download + extract via drive_helper.py
# ---------------------------------------------------------------------------

log "fetching Drive artifacts (required_by=prod)"
DRIVE_ARGS=()
[[ $FORCE -eq 1 ]] && DRIVE_ARGS+=(--force)
conda run -n "$ENV_NAME" --no-capture-output \
    python -m scripts.drive_helper install-all --required-by prod "${DRIVE_ARGS[@]}"

# ---------------------------------------------------------------------------
# 3. Smoke test
# ---------------------------------------------------------------------------

log "running smoke test"
conda run -n "$ENV_NAME" --no-capture-output bash -c '
    set -e
    # PyMOL must load before NumPy so its libstdc++ wins the loader race;
    # otherwise import fails with GLIBCXX_3.4.30 not found.
    python -c "from pymol import cmd; import numpy, pandas, sklearn, torch, ankh; from enzymeexplorer.src.prediction.pipeline import predict_sequences_only, predict_with_structures; print(\"imports OK\")"
    foldseek version
    USalign 2>&1 | grep -m1 "US-align"
    test -f data/enzyme_explorer_checkpoints.pkl
    test -f data/enzyme_explorer_plm_checkpoints.pkl
    test -f data/calibration_fit_summary.csv
    test -f data/detected_domains/martsDB_detected_domains/martsDB_detected_domains.pkl
    test -f data/detected_domains/martsDB_detected_domains/secondary_structure_residues.pkl
    test -f data/detected_domains/martsDB_detected_domains/_ref_set_hash.txt
    test -d data/detected_domains/martsDB_detected_domains/domains
    n=$(find data/detected_domains/martsDB_detected_domains/domains -maxdepth 1 -name "*.pdb" | wc -l)
    [[ $n -gt 0 ]] || { echo "no domain PDBs extracted"; exit 1; }
    cache_hash=$(cat data/detected_domains/martsDB_detected_domains/_ref_set_hash.txt | tr -d "[:space:]")
    test -f "data/foldseek_cache/${cache_hash}/READY" \
        || { echo "foldseek-DB cache marker missing for ${cache_hash}"; exit 1; }
    echo "all data files present (n_domain_pdbs=$n, foldseek_cache=${cache_hash})"
'

cat <<EOF

[setup_prod] DONE.
  env:           $ENV_NAME
  cuda:          ${CUDA_VER:-cpu}
  data root:     $REPO_ROOT/data
  drive cache:   $REPO_ROOT/.drive_cache   (downloaded zips kept here for re-runs)

Activate with:
    conda activate $ENV_NAME

Run prediction:
    enzyme_explorer_main predict --sequences <fasta> --structures-dir <pdbs/> --output-dir <out/>
    enzyme_explorer_main predict --no-structures --sequences <fasta> --output-csv <out.csv>

Or the dedicated console scripts:
    predict_with_structures  --sequences <fasta> --structures-dir <pdbs/> --output-dir <out/>
    predict_sequences_only   --sequences <fasta> --output-csv <out.csv>

Detect domains:
    detect_domains --input-directory-with-structures data/dark_candidates/afdb --detections-output-path data/detected_domains/dark_candidates/dark_candidates_detected_domains.pkl --domains-output-path data/detected_domains/dark_candidates/domains/
EOF
