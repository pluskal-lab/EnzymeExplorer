#!/usr/bin/env bash
# Section 3 — Domain clustering & subtype identification driver.
# Chains the 8 canonical steps and produces all paper figures + iTOL
# annotations under outputs/domain_clustering/ and outputs/martsdb/phylogeny/.
#
# Usage:
#   scripts/run_domain_clustering.sh                    # canonical run
#   scripts/run_domain_clustering.sh --sanity <suffix>  # side-by-side outputs
#   scripts/run_domain_clustering.sh --skip-phylogeny   # skip MAFFT/IQ-TREE (slow)
#
# Sanity mode reroutes every writable output to outputs/domain_clustering/_rerun_<suffix>/
# so canonical artefacts stay untouched. The all-vs-all USalign cache
# (data/domain_clustering/all_vs_all/) is READ-only in either mode.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="normal"
SUFFIX=""
SKIP_PHYLO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --sanity) MODE="sanity"; SUFFIX="${2:-$(date +%Y%m%d_%H%M%S)}"; shift 2 ;;
        --skip-phylogeny) SKIP_PHYLO=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ "$MODE" = "sanity" ]; then
    OUT_DIR="outputs/domain_clustering/_rerun_${SUFFIX}"
    PHYLO_DIR="outputs/martsdb/phylogeny/_rerun_${SUFFIX}"
    SUBTYPE_PKL="$OUT_DIR/domain_module_id_2_domain_subtype.pkl"
    echo "[run_domain_clustering] Sanity re-run — outputs under $OUT_DIR"
else
    OUT_DIR="outputs/domain_clustering"
    PHYLO_DIR="outputs/martsdb/phylogeny"
    SUBTYPE_PKL="data/domain_module_id_2_domain_subtype.pkl"
    echo "[run_domain_clustering] Canonical run"
fi
mkdir -p "$OUT_DIR" "$PHYLO_DIR"

echo "[1] HAC (linkage + intermediate + overview plots)"
python scripts/run_hac_domain_clustering.py --output-dir "$OUT_DIR"

echo "[2] Dynamic tree cut sweep (deepSplit × minClusterSize grid)"
python scripts/run_dynamic_tree_cut_sweep.py --hac-dir "$OUT_DIR"

echo "[3] Canonical subtype labeling (d0_m3 + overrides + delta3)"
python scripts/run_domain_subtype_labeling.py \
    --hac-dir "$OUT_DIR" \
    --output-pickle "$SUBTYPE_PKL"

# The remaining "figure" steps have hardcoded paths (no --hac-dir/--subtype-pkl
# arg surface) and target the canonical outputs/domain_clustering/ tree.
# Skip them in sanity mode — reproducibility is verified by [1]-[3] alone.
if [ "$MODE" = "normal" ]; then
    echo "[4] Paper figures (dendrogram, metrics, palette)"
    python scripts/generate_paper_figures.py

    echo "[5] Scatter plots (PCA / tSNE / UMAP × subset × group)"
    python scripts/generate_scatter_plots.py

    echo "[6] Poster dendrogram"
    python scripts/plot_poster_dendrogram.py

    echo "[7] Supplementary domain-composition CSV"
    python scripts/build_supplementary_domain_table.py

    if [ "$SKIP_PHYLO" = "0" ]; then
        echo "[8] MARTS-DB phylogeny (MAFFT + IQ-TREE + iTOL annotations)"
        bash scripts/build_martsdb_tree.sh
        python scripts/build_martsdb_phylogeny_annotations.py \
            --output-dir "$PHYLO_DIR" \
            --subtype-pkl "$SUBTYPE_PKL"
    else
        echo "[8] SKIPPED phylogeny (per --skip-phylogeny)"
    fi
else
    echo "[sanity] Skipping figure + phylogeny steps (they read canonical outputs/)."
fi

echo "[run_domain_clustering] Done."
