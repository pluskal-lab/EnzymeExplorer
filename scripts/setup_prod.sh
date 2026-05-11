#!/usr/bin/env bash
# EnzymeExplorer — production environment setup
# ==============================================
# Stands up a lean conda environment + downloads model checkpoints, reference
# domains, and pinned foldseek/USalign binaries so the host can run the
# prediction & screening pipelines (predict_sequences_only,
# predict_with_structures, and the screening cluster launcher).
#
# Unix only. Requires `conda` on PATH.
#
# Usage:
#   scripts/setup_prod.sh \
#       [--env-name enzyme_explorer_prod] \
#       [--cuda cu124 | --cpu] \
#       [--force]
#
# The three Google Drive URLs for the model-checkpoints, reference-domains,
# and binaries zips are HARDCODED below — see CHECKPOINTS_URL,
# REF_DOMAINS_URL, BINARIES_URL near the top of the script. Update those
# strings whenever the Drive uploads change.
#
# What it installs
#   * conda env <ENV_NAME> with python=3.10 + libstdcxx-ng (conda-forge,
#     supplies GLIBCXX_3.4.32+ that the bundled USalign binary needs) +
#     pymol-open-source + pymol-psico.
#   * pip deps (sklearn pinned to 1.5.1 — must match the version that pickled
#     the fold-checkpoint bundles; torch built against the requested CUDA wheel
#     channel or CPU-only; ankh for PLM inference; biopython, tqdm, pyyaml,
#     configargparse, matplotlib, filelock, gdown, scipy, pandas, numpy;
#     plus EnzymeExplorer itself via `pip install -e .` so the
#     `predict_sequences_only` / `predict_with_structures` console scripts
#     land on PATH).
#   * `data/enzyme_explorer_checkpoints.pkl`, `data/enzyme_explorer_plm_checkpoints.pkl`
#     from the model-checkpoints zip.
#   * `data/detected_domains/martsDB_detected_domains/{martsDB_detected_domains.pkl,
#     secondary_structure_residues.pkl,domains/*.pdb}` from the reference-domains zip.
#   * `$CONDA_PREFIX/bin/{foldseek,USalign}` from the binaries zip.
#
# Idempotency
#   Each zip is re-downloaded only when the local copy's sha256 doesn't
#   match the zip's MANIFEST.txt. Downloaded zips are kept under
#   .setup_prod_cache/ so re-runs (or future repairs) reuse them.
#   If <ENV_NAME> already exists, the script verifies the env (correct
#   python, key deps importable, foldseek/USalign on PATH) and warns +
#   lists what to fix — it does NOT silently skip. Pass --force to
#   recreate the env from scratch.

set -euo pipefail

# ---------------------------------------------------------------------------
# Hardcoded Google Drive URLs — fill these in after uploading the zips from
# dist/. 
# ("https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing") or a bare
# file id ("<FILE_ID>"); both work.
# ---------------------------------------------------------------------------
CHECKPOINTS_URL="https://drive.google.com/uc?id=1qJJU_pA5D6RIWKES-66c0708f2EFEyU1"
REF_DOMAINS_URL="https://drive.google.com/uc?id=1S1dNj1QDPY8ix7Tb3-zLJ45maWpHruf2"
BINARIES_URL="https://drive.google.com/uc?id=1T86gt8GFS0LJI6do2o-zWB6fbVwNZwcc"
# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

ENV_NAME="enzyme_explorer_prod"
CUDA_VER="cu124"           # set to empty by --cpu
FORCE=0

usage() {
    sed -n '2,30p' "$0"
    exit 1
}

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

# Refuse to run with placeholder URLs — better to fail fast than to try to
# gdown a literal "PLACEHOLDER_…" string and confuse the user.
for var in CHECKPOINTS_URL REF_DOMAINS_URL BINARIES_URL; do
    val="${!var}"
    case "$val" in
        PLACEHOLDER_*|"")
            echo "ERROR: $var is unset/placeholder. Edit $0 and paste the Drive URL." >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { printf '[setup_prod] %s\n' "$*"; }
warn() { printf '[setup_prod] WARN: %s\n' "$*" >&2; }
die()  { printf '[setup_prod] ERROR: %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Persistent cache for the three downloaded zips. Kept across runs so a
# re-invoke (e.g. to repair a broken env) reuses the bytes already on disk
# instead of re-pulling ~3.4 GB from Drive.
CACHE_DIR="$REPO_ROOT/.setup_prod_cache"
mkdir -p "$CACHE_DIR"

# ---------------------------------------------------------------------------
# 1. conda detection
# ---------------------------------------------------------------------------

command -v conda >/dev/null 2>&1 \
    || die "conda not found on PATH. Install Miniconda/Mambaforge and re-run."

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. env existence: verify or recreate
# ---------------------------------------------------------------------------

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

verify_env() {
    # Run a battery of import / which checks inside the env. Returns 0 if OK.
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
            log "env '$ENV_NAME' already exists and verifies clean — skipping env build"
        else
            warn "env '$ENV_NAME' exists but verification failed (see lines above)."
            warn "Re-run with --force to recreate, or repair the env manually."
            exit 3
        fi
        SKIP_ENV_BUILD=1
    fi
fi

# ---------------------------------------------------------------------------
# 3. env build
# ---------------------------------------------------------------------------

if [[ "${SKIP_ENV_BUILD:-0}" -ne 1 ]]; then
    log "creating conda env '$ENV_NAME' (python=3.10 + pymol + psico)"
    # libstdcxx-ng from conda-forge ensures GLIBCXX_3.4.32+ — required by the
    # bundled USalign binary. pymol-open-source is the FOSS PyMOL build.
    conda create -y -n "$ENV_NAME" \
        -c conda-forge -c bioconda -c speleo3 \
        python=3.10.0 \
        libstdcxx-ng \
        pymol-open-source \
        pymol-psico

    log "installing pip deps into '$ENV_NAME'"
    # Use `conda run` for every pip call so the env's pip is used, not the
    # base pip. --no-capture-output streams the install logs live.
    PIP() { conda run -n "$ENV_NAME" --no-capture-output pip "$@"; }

    PIP install --upgrade pip

    # PyTorch — channel depends on CUDA flag.
    if [[ -n "$CUDA_VER" ]]; then
        log "installing torch with $CUDA_VER wheels"
        PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url "https://download.pytorch.org/whl/$CUDA_VER"
    else
        log "installing CPU-only torch wheels"
        PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url "https://download.pytorch.org/whl/cpu"
    fi

    # Production runtime deps. sklearn==1.5.1 must match the version that
    # pickled the fold bundles; deviating risks unpickling errors or silent
    # behavioural drift on edge cases (tree traversal at exact-tied splits).
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

    # EnzymeExplorer itself, editable so updates from `git pull` take effect
    # immediately. This registers the `predict_sequences_only` /
    # `predict_with_structures` console scripts.
    PIP install -e "$REPO_ROOT"
fi

# ---------------------------------------------------------------------------
# 4. download & extract bundles
# ---------------------------------------------------------------------------

log "downloading + verifying bundles via gdown"
CHECKPOINTS_ZIP="$CACHE_DIR/model-checkpoints.zip"
REF_DOMAINS_ZIP="$CACHE_DIR/reference-domains.zip"
BINARIES_ZIP="$CACHE_DIR/binaries.zip"

# gdown comes from the just-installed env, so prefix every invocation.
download_and_verify_zip() {
    local url="$1" cache_path="$2" label="$3"
    if [[ ! -f "$cache_path" ]]; then
        log "$label: downloading from $url"
        conda run -n "$ENV_NAME" --no-capture-output gdown --output "$cache_path" "$url" \
            || die "$label: gdown failed; check the URL and your network"
    else
        log "$label: found cached $cache_path — re-verifying"
    fi

    local mtmp
    mtmp=$(mktemp -d)
    unzip -q -o "$cache_path" MANIFEST.txt -d "$mtmp" \
        || die "$label: MANIFEST.txt missing from zip"

    local lines_ok=1
    while IFS=$'\n' read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        local expected relpath
        expected=$(awk '{print $1}' <<<"$line")
        relpath=$(awk '{print $3}' <<<"$line")
        [[ "$relpath" == */ ]] && continue
        local entry_sha
        entry_sha=$(unzip -p "$cache_path" "$relpath" | sha256sum | awk '{print $1}')
        if [[ "$entry_sha" != "$expected" ]]; then
            warn "$label: $relpath sha mismatch (zip $entry_sha, MANIFEST $expected)"
            lines_ok=0
        fi
    done < "$mtmp/MANIFEST.txt"
    rm -rf "$mtmp"

    [[ $lines_ok -eq 1 ]] || die "$label: MANIFEST verification failed; delete $cache_path and re-run"
    log "$label: MANIFEST verified OK"
}

download_and_verify_zip "$CHECKPOINTS_URL"  "$CHECKPOINTS_ZIP"  "model-checkpoints"
download_and_verify_zip "$REF_DOMAINS_URL"  "$REF_DOMAINS_ZIP"  "reference-domains"
download_and_verify_zip "$BINARIES_URL"     "$BINARIES_ZIP"     "binaries"

# ---------------------------------------------------------------------------
# 5. place files at canonical paths
# ---------------------------------------------------------------------------

log "extracting model-checkpoints.zip -> data/"
mkdir -p "$REPO_ROOT/data"
unzip -q -o "$CHECKPOINTS_ZIP" \
    enzyme_explorer_checkpoints.pkl \
    enzyme_explorer_plm_checkpoints.pkl \
    -d "$REPO_ROOT/data"

log "extracting reference-domains.zip -> data/{detected_domains,foldseek_cache}/"
mkdir -p "$REPO_ROOT/data/detected_domains"
unzip -q -o "$REF_DOMAINS_ZIP" "martsDB_detected_domains/*" \
    -d "$REPO_ROOT/data/detected_domains"
# The zip also ships a prebuilt foldseek reference DB keyed by the
# content sha of the reference PDB set (see _ref_set_hash.txt sidecar).
# Drop it at data/foldseek_cache/<hash>/ so the very first prediction
# is a cache HIT instead of a multi-minute foldseek-createdb rebuild.
unzip -q -o "$REF_DOMAINS_ZIP" "foldseek_cache/*" \
    -d "$REPO_ROOT/data"

# Binaries: install into the conda env's bin/. CONDA_PREFIX is set by `conda activate`.
log "installing foldseek + USalign into env '$ENV_NAME'"
ENV_PREFIX=$(conda run -n "$ENV_NAME" --no-capture-output bash -c 'echo "$CONDA_PREFIX"')
[[ -d "$ENV_PREFIX/bin" ]] || die "could not resolve \$CONDA_PREFIX for '$ENV_NAME'"
unzip -q -o "$BINARIES_ZIP" foldseek USalign -d "$ENV_PREFIX/bin"
chmod +x "$ENV_PREFIX/bin/foldseek" "$ENV_PREFIX/bin/USalign"

# ---------------------------------------------------------------------------
# 6. smoke test
# ---------------------------------------------------------------------------

log "running smoke test"
conda run -n "$ENV_NAME" --no-capture-output bash -c '
    set -e
    # PyMOL must load before NumPy so its libstdc++ wins the loader race;
    # otherwise import fails with GLIBCXX_3.4.30 not found.
    python -c "from pymol import cmd; import numpy, pandas, sklearn, torch, ankh; from enzymeexplorer.src.prediction.pipeline import predict_sequences_only, predict_with_structures; print(\"imports OK\")"
    foldseek version
    USalign 2>&1 | grep -m1 "US-align"
    # Verify the data files landed.
    test -f data/enzyme_explorer_checkpoints.pkl
    test -f data/enzyme_explorer_plm_checkpoints.pkl
    test -f data/calibration_fit_summary.csv
    test -f data/detected_domains/martsDB_detected_domains/martsDB_detected_domains.pkl
    test -f data/detected_domains/martsDB_detected_domains/secondary_structure_residues.pkl
    test -f data/detected_domains/martsDB_detected_domains/_ref_set_hash.txt
    test -d data/detected_domains/martsDB_detected_domains/domains
    n=$(find data/detected_domains/martsDB_detected_domains/domains -maxdepth 1 -name "*.pdb" | wc -l)
    [[ $n -gt 0 ]] || { echo "no domain PDBs extracted"; exit 1; }
    # Foldseek cache: a single subdir keyed by the content sha; READY marker.
    cache_hash=$(cat data/detected_domains/martsDB_detected_domains/_ref_set_hash.txt | tr -d "[:space:]")
    test -f "data/foldseek_cache/${cache_hash}/READY" \
        || { echo "foldseek-DB cache marker missing for ${cache_hash}"; exit 1; }
    echo "all data files present (n_domain_pdbs=$n, foldseek_cache=${cache_hash})"
'

cat <<EOF

[setup_prod] DONE.
  env:           $ENV_NAME
  cuda:          ${CUDA_VER:-cpu}
  cache dir:     $CACHE_DIR   (downloaded zips kept here for re-runs)
  data dir:      $REPO_ROOT/data

Activate with:
    conda activate $ENV_NAME

Run prediction:
    predict_sequences_only --sequences <fasta> --output-csv <out.csv>
    predict_with_structures --sequences <fasta> --structures-dir <pdbs/> --output-csv <out.csv> --fallback-output-csv <fallback.csv>

EOF
