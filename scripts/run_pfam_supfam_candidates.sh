#!/usr/bin/env bash
# Section 10 — Pfam+SUPFAM candidates driver.
#
# Regenerates predictions for the 9 hand-picked "high-priority" candidate
# proteins selected from the Pfam/SUPFAM screening of 5.1 billion sequences
# (BFD + UniParc + MGnify + 1KP + Phytozome + NCBI TSA). These are the
# nine sequences with the greatest phylogenetic distance from any
# characterised TPS — used in the paper as Table S1 to sanity-check the
# calibrated prediction pipeline on novel, functionally-uncharacterised
# TPS-like proteins.
#
# Inputs  : data/pfam_supfam_candidates/{candidates.fasta, afdb/*.pdb}
# Outputs : outputs/candidates/pfam_supfam_candidates/
#             predictions_plm_domains.csv
#             predictions_plm_only_fallback.csv
#
# Domain-detection defaults follow the Section-9 "no heuristic filters,
# one domain per iteration" policy — opt in via the CLI if you want the
# older behaviour.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEQ="data/pfam_supfam_candidates/candidates.fasta"
STRUCTURES="data/pfam_supfam_candidates/afdb"
OUT="outputs/candidates/pfam_supfam_candidates"

echo "[run_pfam_supfam_candidates] predict sequences=$SEQ structures=$STRUCTURES output=$OUT"
enzyme_explorer_main predict \
    --sequences "$SEQ" \
    --structures-dir "$STRUCTURES" \
    --output-dir "$OUT" \
    "$@"
echo "[run_pfam_supfam_candidates] Done. Outputs under $OUT/."
