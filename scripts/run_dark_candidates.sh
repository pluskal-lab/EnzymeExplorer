#!/usr/bin/env bash
# Section 11 — Dark candidates driver.
#
# Re-runs EnzymeExplorer on the 11 dark candidates picked from the
# post-screening phylogenetic tree (uniprot-without-InterPro dark
# proteome → EnzymeExplorer screen → TPS_p > 0.95 filter → phylogeny
# → manual clade sampling). See
# data/dark_proteome_screening/candidate_selection/phylo_tree/ for the
# selection tree and
# outputs/candidates/dark_candidates/ for the final predictions.
#
# Inputs  : data/dark_candidates/{candidates.fasta, afdb/*.pdb}
# Outputs : outputs/candidates/dark_candidates/
#             predictions_plm_domains.csv
#             predictions_plm_only_fallback.csv
#
# Domain-detection defaults follow the Section-9 policy (no heuristic
# filters, one domain per iteration). Opt in via CLI flags if needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEQ="data/dark_candidates/candidates.fasta"
STRUCTURES="data/dark_candidates/afdb"
OUT="outputs/candidates/dark_candidates"

echo "[run_dark_candidates] predict sequences=$SEQ structures=$STRUCTURES output=$OUT"
enzyme_explorer_main predict \
    --sequences "$SEQ" \
    --structures-dir "$STRUCTURES" \
    --output-dir "$OUT" \
    "$@"
echo "[run_dark_candidates] Done. Outputs under $OUT/."
