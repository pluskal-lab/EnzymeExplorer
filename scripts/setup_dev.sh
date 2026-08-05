#!/usr/bin/env bash
# EnzymeExplorer — development environment setup
# ==============================================
# Stands up the full developer conda environment and lays down every
# artifact needed to reproduce training, evaluation, calibration,
# screening and the rebuttal-only analyses.
#
# Unix only. Requires `conda` on PATH.
#
# Usage:
#   scripts/setup_dev.sh \
#       [--env-name enzyme_explorer_dev] \
#       [--cuda cu124 | --cpu] \
#       [--skip-drive]      (skip Drive downloads; env-only)
#       [--skip-pdbs]       (skip the ~48k AlphaFold-DB PDB download)
#       [--skip-gtdb]       (skip the GTDB ar53 archaeal tree download)
#       [--skip-clean]      (skip the CLEAN + facebook/esm install)
#       [--force]
#
# Pipeline
#   1. conda env <ENV_NAME> — python 3.10 + rdkit==2022.9.5 (pinned;
#      newer rdkit produces different canonical SMILES) + pymol +
#      foldseek + USalign + mmseqs2 + mafft + iqtree2 + dev pip deps.
#   2. `pip install -e .` — puts EnzymeExplorer + its CLIs on PATH.
#   3. Drive artifacts marked ``required_by: [dev]`` in
#      ``drive/bundles.json`` — model + reference-domains + PLM
#      checkpoints + gathered embeddings + structural features +
#      SwissProt export + MARTS-DB PDBs + domain-clustering cache +
#      detected-domains extras + dark-proteome-screening result +
#      trained fold checkpoints (outputs/models/*) + GO ontology +
#      candidate structures. See ``drive/bundles.json`` for the full
#      list.
#   4. Rebuild ``data/enzyme_explorer_pdbs/`` — copy the MARTS-DB
#      subset from ``data/martsDB_pdbs/`` (extracted in step 3) and
#      download the rest from AlphaFold-DB (v6 URL + REST-API fallback,
#      missing entries logged and skipped).
#   5. Provision the GTDB ar53 archaeal phylogeny — download the raw
#      Newick from data.gtdb.ecogenomic.org and post-process it into
#      an iTOL-loadable tree (see
#      ``scripts/archaeal_screening/download_gtdb_ar53_tree.sh``).
#   6. Install CLEAN + facebook/esm at ``<repo>/CLEAN/`` following the
#      upstream install docs (https://github.com/tttianhao/CLEAN#1-install).
#      Runs ``python build.py install`` inside the dev env so CLEAN's
#      Cython extension is built against the same Python. Appends
#      ``CLEAN/app/src`` to ``~/.bashrc``'s PATH so CLEAN's helper
#      scripts are resolvable in subsequent shells.
#
# Idempotency
#   Every stage is safe to re-run; downloaded zips are cached under
#   ``.drive_cache/``, already-present PDBs are not re-fetched, and
#   the tree provisioner short-circuits on cleaned-tree existence.
#   Pass ``--force`` to re-download Drive artifacts even when the
#   cached zip's sha256 matches.

set -euo pipefail

ENV_NAME="enzyme_explorer_dev"
CUDA_VER="cu124"
FORCE=0
SKIP_DRIVE=0
SKIP_PDBS=0
SKIP_GTDB=0
SKIP_CLEAN=0

usage() { sed -n '2,38p' "$0"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)   ENV_NAME="$2"; shift 2 ;;
        --cuda)       CUDA_VER="$2"; shift 2 ;;
        --cpu)        CUDA_VER=""; shift ;;
        --skip-drive) SKIP_DRIVE=1; shift ;;
        --skip-pdbs)  SKIP_PDBS=1; shift ;;
        --skip-clean) SKIP_CLEAN=1; shift ;;
        --skip-gtdb)  SKIP_GTDB=1; shift ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

log()  { printf '[setup_dev] %s\n' "$*"; }
warn() { printf '[setup_dev] WARN: %s\n' "$*" >&2; }
die()  { printf '[setup_dev] ERROR: %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v conda >/dev/null 2>&1 \
    || die "conda not found on PATH. Install Miniconda/Mambaforge and re-run."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1. conda env
# ---------------------------------------------------------------------------

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

if env_exists "$ENV_NAME"; then
    if [[ $FORCE -eq 1 ]]; then
        log "--force given; removing existing env '$ENV_NAME'"
        conda env remove -n "$ENV_NAME" -y
    else
        log "env '$ENV_NAME' already exists — skipping env build (pass --force to recreate)"
        SKIP_ENV_BUILD=1
    fi
fi

if [[ "${SKIP_ENV_BUILD:-0}" -ne 1 ]]; then
    log "creating conda env '$ENV_NAME' (python 3.10 + dev toolchain)"
    # rdkit pinned to 2022.9.5 — newer versions produce different SMILES canonical
    # forms and would silently invalidate the substrate-lookup CSVs.
    # foldseek + USalign + mmseqs2 + mafft + iqtree2 land inside the env's bin/
    # so no PATH juggling is needed at runtime.
    conda create -y -n "$ENV_NAME" \
        -c conda-forge -c bioconda -c speleo3 -c schrodinger \
        python=3.10.0 \
        pymol=3.1.6.1 \
        pymol-bundle=3.1.6.1 \
        pymol-psico=3.4.19 \
        foldseek==9.427df8a \
        bioconda::usalign \
        bioconda::mmseqs2 \
        mafft==7.525 \
        iqtree==2.3.0 \
        rdkit==2022.9.5 \
        biopython==1.83 \
        fastapi

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
    # goatools pinned to the version that indexes go-basic_2026_03_14.obo.
    PIP install \
        "scikit-learn==1.5.1" \
        "numpy==1.26.4" \
        "pandas==2.2.2" \
        "scipy==1.13.0" \
        "ankh==1.10.0" \
        "fair-esm==2.0.0" \
        "biopython==1.83" \
        "epam.indigo==1.45.0" \
        "tqdm" \
        "pyyaml" \
        "dataclasses-json" \
        "configargparse" \
        "matplotlib" \
        "seaborn" \
        "filelock" \
        "gdown" \
        "tables" \
        "openpyxl" \
        "hdbscan==0.8.33" \
        "scikit-learn-extra" \
        "dynamicTreeCut==0.1.1" \
        "goatools==1.6.5" \
        "plotly" \
        "umap-learn" \
        "py3Dmol" \
        "inquirer" \
        "jupyter"

    # ProFun (external — provides foldseek / blast / hmm baseline wrappers).
    PIP install "git+https://github.com/SamusRam/ProFun.git"

    # EnzymeExplorer itself, editable.
    PIP install -e "$REPO_ROOT"
fi

# ---------------------------------------------------------------------------
# 2. Drive artifacts
# ---------------------------------------------------------------------------

if [[ $SKIP_DRIVE -eq 0 ]]; then
    log "fetching Drive artifacts (required_by=dev)"
    DRIVE_ARGS=()
    [[ $FORCE -eq 1 ]] && DRIVE_ARGS+=(--force)
    conda run -n "$ENV_NAME" --no-capture-output \
        python -m scripts.drive_helper install-all --required-by dev "${DRIVE_ARGS[@]}"
else
    log "--skip-drive set; skipping Drive downloads"
fi

# ---------------------------------------------------------------------------
# 3. enzyme_explorer_pdbs — copy marts_ + download rest from AlphaFold-DB
# ---------------------------------------------------------------------------

if [[ $SKIP_PDBS -eq 0 ]]; then
    log "populating data/enzyme_explorer_pdbs/ (marts copies + AFDB downloads)"
    conda run -n "$ENV_NAME" --no-capture-output \
        python scripts/download_enzyme_explorer_pdbs.py --workers 16
else
    log "--skip-pdbs set; leaving data/enzyme_explorer_pdbs/ untouched"
fi

# ---------------------------------------------------------------------------
# 4. GTDB ar53 archaeal tree
# ---------------------------------------------------------------------------

if [[ $SKIP_GTDB -eq 0 ]]; then
    log "provisioning GTDB ar53 archaeal phylogeny"
    conda run -n "$ENV_NAME" --no-capture-output \
        bash scripts/archaeal_screening/download_gtdb_ar53_tree.sh
else
    log "--skip-gtdb set; leaving data/archaeal_screening/*.tree untouched"
fi

# ---------------------------------------------------------------------------
# 5. CLEAN — installed at <repo>/CLEAN/ per upstream README
#            https://github.com/tttianhao/CLEAN#1-install
# ---------------------------------------------------------------------------

if [[ $SKIP_CLEAN -eq 0 ]]; then
    log "installing CLEAN + facebook/esm at $REPO_ROOT/CLEAN/"
    CLEAN_ROOT="$REPO_ROOT/CLEAN"
    if [[ -d "$CLEAN_ROOT" && $FORCE -eq 1 ]]; then
        log "--force given; removing existing CLEAN/ tree"
        rm -rf "$CLEAN_ROOT"
    fi
    if [[ ! -d "$CLEAN_ROOT" ]]; then
        git clone https://github.com/tttianhao/CLEAN.git "$CLEAN_ROOT"
    else
        log "CLEAN/ already present — skipping git clone (--force to re-fetch)"
    fi

    # Build CLEAN's Cython extension with the dev env's python. The build
    # step must run from CLEAN/app because build.py resolves ``src/`` and
    # ``data/`` relative to its own directory.
    log "building CLEAN Cython extension (python build.py install)"
    conda run -n "$ENV_NAME" --no-capture-output bash -c "
        set -e
        cd '$CLEAN_ROOT/app'
        python build.py install
    "

    # facebook/esm — CLEAN uses ESM-1b embeddings; the app expects the
    # esm/ checkout to live at CLEAN/app/esm.
    if [[ ! -d "$CLEAN_ROOT/app/esm" ]]; then
        log "cloning facebook/esm into CLEAN/app/esm"
        git clone https://github.com/facebookresearch/esm.git "$CLEAN_ROOT/app/esm"
    else
        log "CLEAN/app/esm already present — skipping git clone"
    fi

    # Empty cache dir CLEAN writes per-sequence embeddings into.
    mkdir -p "$CLEAN_ROOT/app/data/esm_data"

    # Persist CLEAN's helper-scripts location on PATH so subsequent shells
    # can invoke them. Idempotent — the append only fires once.
    CLEAN_SRC="$CLEAN_ROOT/app/src"
    if ! grep -qsF "$CLEAN_SRC" "$HOME/.bashrc" 2>/dev/null; then
        printf '\n# EnzymeExplorer setup_dev.sh — CLEAN helper scripts\nexport PATH="%s:$PATH"\n' \
            "$CLEAN_SRC" >> "$HOME/.bashrc"
        warn "appended CLEAN src/ to ~/.bashrc — open a new shell or 'source ~/.bashrc' to pick it up"
    fi
else
    log "--skip-clean set; leaving CLEAN untouched"
fi

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------

log "running smoke test"
conda run -n "$ENV_NAME" --no-capture-output bash -c '
    set -e
    python -c "from pymol import cmd; import numpy, pandas, sklearn, torch, ankh; from enzymeexplorer.src.prediction.pipeline import predict_sequences_only; from enzymeexplorer.src.evaluation import cli as eval_cli; print(\"imports OK\")"
    foldseek version >/dev/null
    USalign 2>&1 | grep -m1 -q "US-align"
    command -v mafft   >/dev/null || { echo "mafft missing";   exit 1; }
    command -v iqtree2 >/dev/null || { echo "iqtree2 missing"; exit 1; }
    command -v mmseqs  >/dev/null || { echo "mmseqs missing";  exit 1; }
    enzyme_explorer_main --help >/dev/null
    echo "toolchain OK"
'

cat <<EOF

[setup_dev] DONE.
  env:           $ENV_NAME
  cuda:          ${CUDA_VER:-cpu}
  data root:     $REPO_ROOT/data
  drive cache:   $REPO_ROOT/.drive_cache   (downloaded zips kept here)

Activate with:
    conda activate $ENV_NAME

Full pipeline available:
    enzyme_explorer_main run       — train a model
    enzyme_explorer_main evaluate  — bootstrap evaluation over classifiers
    enzyme_explorer_main calibrate — fit per-(classifier, class) calibrators
    enzyme_explorer_main visualize — render plots from a saved evaluation
    enzyme_explorer_main predict   — inference (--no-structures for seq-only)
    detect_domains — TPS-family domain detection

EOF
