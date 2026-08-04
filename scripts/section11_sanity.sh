#!/usr/bin/env bash
# Section 11 sanity re-run. Regenerates baselines + alpha-domain analysis
# + tree annotations against the post-refactor paths and verifies each
# stage produces the expected outputs. Does NOT rebuild the MAFFT+IQ-TREE
# phylogeny (~1h; the tree.nwk is treated as a fixed artefact here).
#
# Usage: bash scripts/section11_sanity.sh
# Assumes the ``enzyme_explorer`` conda env is active.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CANDS=(A0A0E3NXY0 A0A5E4I9B1 A0A537EJD0)

echo "=================================================="
echo "[section11_sanity] 1. Dark candidates re-prediction"
echo "=================================================="
scripts/run_dark_candidates.sh
test -s outputs/candidates/dark_candidates/predictions_plm_domains.csv
echo

echo "=================================================="
echo "[section11_sanity] 2. Baseline methods (BLAST/Foldseek/HMM/Pfam/Supfam/CLEAN)"
echo "=================================================="
bash scripts/rebuttal_only/dark_candidates_baselines/run_all.sh
for m in BLAST Foldseek HMM Pfam Supfam CLEAN; do
    test -s "outputs/rebuttal/dark_candidates_baselines/$m/mean.csv" \
        || { echo "MISSING: $m/mean.csv" >&2; exit 1; }
done
echo

echo "=================================================="
echo "[section11_sanity] 3. Alpha-domain per-candidate analysis"
echo "=================================================="
python -m scripts.rebuttal_only.analyze_hard_domain "${CANDS[@]}"
for c in "${CANDS[@]}"; do
    test -s "outputs/rebuttal/archaeal_alpha_domains/$c/tm_scores.csv" \
        || { echo "MISSING: $c/tm_scores.csv" >&2; exit 1; }
done
echo

echo "=================================================="
echo "[section11_sanity] 4. Three-candidate cross-similarity"
echo "=================================================="
python -m scripts.rebuttal_only.analyze_three_cross_similarity
test -d outputs/rebuttal/archaeal_alpha_domains/_cross
echo

echo "=================================================="
echo "[section11_sanity] 5. Novelty panels + PCA plots"
echo "=================================================="
python -m scripts.rebuttal_only.plot_novelty_panels
python -m scripts.rebuttal_only.plot_A0A5E4I9B1_pca
python -m scripts.rebuttal_only.plot_three_pca
echo

echo "=================================================="
echo "[section11_sanity] 6. Selection tree — patristic distances + iTOL"
echo "=================================================="
python -m scripts.dark_proteome_screening.compute_distances
python -m scripts.dark_proteome_screening.build_itol_annotations
test -s data/dark_proteome_screening/candidate_selection/phylo_tree/itol_kingdom_treecolors.txt
echo

echo "[section11_sanity] All stages OK."
