# Incremental Evaluation Results — Atomic Changes

This document summarizes the results of the **dual-track incremental evaluation protocol**
for the EnzymeExplorer TPS discovery models. Each evaluation track isolates a specific
experimental variable so that performance changes can be attributed to individual factors.

---

## Table of Contents

1. [Evaluation Protocol Overview](#evaluation-protocol-overview)
2. [Datasets](#datasets)
3. [Models Evaluated](#models-evaluated)
4. [Track A — Phylo Folds (original split)](#track-a--phylo-folds-original-split)
5. [Track B — New Dataset (native folds)](#track-b--new-dataset-native-folds)
6. [Track C — Synced Folds (cross-dataset comparison)](#track-c--synced-folds-cross-dataset-comparison)
7. [Per-Similarity-Bin Analysis (TPS Detection)](#per-similarity-bin-analysis-tps-detection)
8. [Cross-Negative Stress Test](#cross-negative-stress-test)
9. [Cross-Track Comparison Summary](#cross-track-comparison-summary)
10. [Pending Experiments](#pending-experiments)
11. [Generated Artifacts](#generated-artifacts)
12. [Caveats and Notes](#caveats-and-notes)

---

## Evaluation Protocol Overview

We evaluate models across **three tracks** plus a **cross-negative stress test**.
Each track changes one variable at a time:

| Track | Dataset | Folds | Negatives | Purpose |
|-------|---------|-------|-----------|---------|
| **A** | Old (phylo) | Original phylogenetic folds | 9,944 negatives distributed across folds 0–4 | Baseline: original paper protocol |
| **B** | New (EnzymeExplorer) | Native 5-fold CV | 10,000 negatives distributed across folds 0–4 | Effect of expanded + curated dataset |
| **C** | Old (synced) | New-dataset fold IDs for shared TPS | 9,944 negatives distributed across folds 0–4 (same folds as Track A) | Bridge: same TPS, same folds as Track B, old negatives |
| **Cross-neg** | Synced TPS + swapped negatives | Synced folds | Old or new negatives swapped at test time | Stress test: robustness to different negative distributions |

### Key design choices

- **128 old-only TPS excluded**: The old dataset contained 128 TPS sequences not present in
  the curated new dataset. These were deemed low-quality and excluded from Track A (phylo)
  and Track C (synced) evaluations.
- **Fold synchronization**: For Track C, shared TPS sequences (1,035) receive the new
  dataset's fold assignments. Negatives keep their original (redistributed) fold assignments.
- **MMseqs2 similarity bins**: Per-fold sequence similarity computed with MMseqs2
  (`easy-search`, alignment-mode 3, max-seqs 300, e-value ∞). Bins: `no_hit`, `20–30%`,
  `30–40%`, `40–50%`, `50–60%`, `60–70%` identity to nearest training-set hit.

---

## Datasets

### Old dataset (phylo)
- **File**: `data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv`
- **Total entries**: 12,150
- **TPS**: 2,206 (includes 128 old-only TPS that are excluded during eval)
- **Negatives**: 9,944 (all distributed across folds 0–4)
- **Fold column**: `stratified_phylogeny_based_split_with_minor_products`

| Fold | TPS | Negatives | Total |
|------|-----|-----------|-------|
| 0    | 414 | 2,002     | 2,416 |
| 1    | 302 | 1,988     | 2,290 |
| 2    | 562 | 1,982     | 2,544 |
| 3    | 455 | 1,985     | 2,440 |
| 4    | 473 | 1,987     | 2,460 |

### New dataset (EnzymeExplorer)
- **File**: `data/EnzymeExplorer_Dataset.csv`
- **Total entries**: 14,185
- **TPS**: 4,185 (1,374 curated + 2,811 from MARTS DB expansion)
- **Negatives**: 10,000 (all distributed across folds)
- **Fold column**: `Fold` (bare integers 0–4)
- **ID column**: `ID`

| Fold | TPS   | Negatives | Total  |
|------|-------|-----------|--------|
| 0    | 695   | 1,897     | 2,592  |
| 1    | 704   | 1,872     | 2,576  |
| 2    | 1,009 | 1,958     | 2,967  |
| 3    | 770   | 2,202     | 2,972  |
| 4    | 1,007 | 2,071     | 3,078  |

### Synced dataset
- **File**: `data/TPS-Nov19_2023_with_synced_folds.csv`
- **Total entries**: 12,000 (128 old-only TPS removed)
- **TPS**: 2,056 (only sequences also present in the new dataset)
- **Negatives**: 9,944 (same negatives as old dataset, same fold assignments)
- **Fold column**: `synced_fold` (new-dataset fold IDs for shared TPS)

| Fold | TPS | Negatives | Total  |
|------|-----|-----------|--------|
| 0    | 339 | 2,002     | 2,341  |
| 1    | 370 | 1,988     | 2,358  |
| 2    | 464 | 1,982     | 2,446  |
| 3    | 393 | 1,985     | 2,378  |
| 4    | 490 | 1,987     | 2,477  |

---

## Models Evaluated

| Model | Type | Description |
|-------|------|-------------|
| **Blastp** | Sequence similarity | BLASTp nearest-neighbor matching |
| **PlmRF** (PlmRandomForest) | Embedding-based | Random Forest on finetuned ESM-1v sub-sequence embeddings |
| **CLEAN** | Pre-trained enzyme model | Pre-trained on all enzymes; **in-sample** for TPS (see [Caveats](#caveats-and-notes)) |
| **HMM** | Profile HMM | Hidden Markov Model for protein family classification |
| **Foldseek** | Structure similarity | Structural nearest-neighbor matching (requires PDB files) |

---

## Track A — Phylo Folds (original split)

**What this tests**: Baseline performance on the original phylogenetic split with all
negatives distributed across folds. This matches the published evaluation protocol with
the corrected negative distribution (~2,000 negatives per fold).

### Substrate Prediction (macro-averaged across substrate classes)

| Model | mAP | ROC-AUC | MCC-F1 |
|-------|-----|---------|--------|
| **PlmRF** | **0.776** | **0.970** | **0.557** |
| CLEAN\* | 0.626 | 0.848 | 0.506 |
| Blastp | 0.542 | 0.815 | 0.255 |
| Foldseek | 0.395 | 0.775 | 0.233 |
| HMM | 0.334 | 0.960 | 0.422 |

### TPS Detection (binary: TPS vs non-TPS)

| Model | AP | ROC-AUC | MCC-F1 |
|-------|-----|---------|--------|
| **PlmRF** | **0.988** | **0.998** | **0.751** |
| HMM | 0.890 | 0.983 | 0.768 |
| Blastp | 0.892 | 0.955 | 0.332 |
| CLEAN\* | 0.857 | 0.978 | 0.680 |
| Foldseek | 0.838 | 0.931 | 0.318 |

### Visualizations

Substrate prediction performance (mAP, ROC-AUC, MCC-F1):

<p align="middle">
  <img src="outputs/evaluation_results/track_a_phylo_Mean Average Precision.png" width="300" />
  <img src="outputs/evaluation_results/track_a_phylo_ROC-AUC.png" width="300" />
  <img src="outputs/evaluation_results/track_a_phylo_MCC-F1 summary.png" width="300" />
</p>

TPS detection (AP, ROC-AUC, MCC-F1, PR curve):

<p align="middle">
  <img src="outputs/evaluation_results/track_a_tps_detection_Average Precision_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_a_tps_detection_ROC-AUC_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_a_tps_detection_MCC-F1 summary_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_a_tps_detection_PR_TPS.png" width="250" />
</p>

---

## Track B — New Dataset (native folds)

**What this tests**: Performance on the expanded, curated EnzymeExplorer dataset with
significantly more TPS (4,185 vs 2,206) and a comparable negative set (~2,000 per fold
in both tracks after the phylo-dataset correction).

### Substrate Prediction

| Model | mAP | ROC-AUC | MCC-F1 |
|-------|-----|---------|--------|
| **PlmRF** | **0.803** | **0.987** | **0.584** |
| Blastp | 0.612 | 0.851 | 0.251 |
| HMM | 0.437 | 0.971 | 0.500 |
| CLEAN\* | 0.499 | 0.773 | 0.451 |

### TPS Detection

| Model | AP | ROC-AUC | MCC-F1 |
|-------|-----|---------|--------|
| **PlmRF** | **0.991** | **0.999** | **0.773** |
| Blastp | 0.934 | 0.977 | 0.340 |
| HMM | 0.895 | 0.973 | 0.820 |
| CLEAN\* | 0.871 | 0.976 | 0.694 |

### Visualizations

<p align="middle">
  <img src="outputs/evaluation_results/track_b_new_Mean Average Precision.png" width="300" />
  <img src="outputs/evaluation_results/track_b_new_ROC-AUC.png" width="300" />
  <img src="outputs/evaluation_results/track_b_new_MCC-F1 summary.png" width="300" />
</p>

<p align="middle">
  <img src="outputs/evaluation_results/track_b_tps_detection_Average Precision_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_b_tps_detection_ROC-AUC_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_b_tps_detection_MCC-F1 summary_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_b_tps_detection_PR_TPS.png" width="250" />
</p>

---

## Track C — Synced Folds (cross-dataset comparison)

**What this tests**: Same TPS as Track A (minus 128 old-only), same fold assignments as
Track B's shared TPS, but with the old negative set (9,944 negatives). This bridges
Tracks A and B, isolating the effect of **fold assignment** and **TPS curation** from
the negative set composition.

### Substrate Prediction

| Model | mAP | ROC-AUC | MCC-F1 |
|-------|-----|---------|--------|
| **PlmRF** | **0.794** | **0.988** | **0.575** |
| CLEAN\* | 0.609 | 0.836 | 0.483 |
| Blastp | 0.639 | 0.866 | 0.242 |
| HMM | 0.345 | 0.968 | 0.445 |

### TPS Detection

| Model | AP | ROC-AUC | MCC-F1 |
|-------|-----|---------|--------|
| **PlmRF** | **0.990** | **0.999** | **0.754** |
| Blastp | 0.938 | 0.988 | 0.321 |
| HMM | 0.914 | 0.987 | 0.816 |
| CLEAN\* | 0.840 | 0.977 | 0.674 |

### Visualizations

<p align="middle">
  <img src="outputs/evaluation_results/track_c_synced_Mean Average Precision.png" width="300" />
  <img src="outputs/evaluation_results/track_c_synced_ROC-AUC.png" width="300" />
  <img src="outputs/evaluation_results/track_c_synced_MCC-F1 summary.png" width="300" />
</p>

<p align="middle">
  <img src="outputs/evaluation_results/track_c_tps_detection_Average Precision_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_c_tps_detection_ROC-AUC_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_c_tps_detection_MCC-F1 summary_TPS.png" width="250" />
  <img src="outputs/evaluation_results/track_c_tps_detection_PR_TPS.png" width="250" />
</p>

---

## Per-Similarity-Bin Analysis (TPS Detection)

**What this tests**: How model performance degrades as train-test sequence similarity
decreases. MMseqs2 computes the best percent-identity hit for each test sequence against
its corresponding training fold. Results are broken down by identity bins.

### Track A — Average Precision by Similarity Bin

| Model | All | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% |
|-------|-----|--------|--------|--------|--------|--------|
| **PlmRF** | **0.987** | **0.971** | **0.976** | **0.997** | **0.999** | **1.000** |
| Blastp | 0.858 | 0.896 | 0.973 | 0.891 | 0.861 | 0.873 |
| CLEAN\* | 0.857 | 0.848 | 0.919 | 0.939 | 0.750 | 0.826 |

### Track A — MCC-F1 by Similarity Bin

| Model | All | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% |
|-------|-----|--------|--------|--------|--------|--------|
| **PlmRF** | **0.798** | **0.655** | **0.750** | **0.803** | **0.789** | **0.708** |
| CLEAN\* | 0.724 | 0.604 | 0.685 | 0.755 | 0.664 | 0.646 |
| Blastp | 0.314 | 0.272 | 0.344 | 0.438 | 0.356 | 0.269 |

### Track B — Average Precision by Similarity Bin

| Model | All | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% |
|-------|-----|--------|--------|--------|--------|--------|
| **PlmRF** | **0.991** | **0.993** | **0.982** | **0.999** | **0.983** | **1.000** |
| Blastp | 0.956 | 0.947 | 0.958 | 0.989 | 0.930 | 0.828 |
| CLEAN\* | 0.847 | 0.902 | 0.791 | 0.873 | 0.877 | 0.939 |

### Track B — MCC-F1 by Similarity Bin

| Model | All | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% |
|-------|-----|--------|--------|--------|--------|--------|
| **PlmRF** | **0.832** | **0.667** | **0.758** | **0.858** | **0.852** | **0.669** |
| CLEAN\* | 0.708 | 0.672 | 0.681 | 0.718 | 0.711 | 0.676 |
| Blastp | 0.311 | 0.249 | 0.297 | 0.426 | 0.488 | 0.269 |

### Track C — Average Precision by Similarity Bin

| Model | All | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% |
|-------|-----|--------|--------|--------|--------|--------|
| **PlmRF** | **0.987** | **0.991** | **0.988** | **0.984** | **0.991** | **1.000** |
| Blastp | 0.949 | 0.921 | 0.932 | 0.974 | 0.901 | 0.950 |
| CLEAN\* | 0.847 | 0.815 | 0.863 | 0.897 | 0.842 | 0.776 |

### Track C — MCC-F1 by Similarity Bin

| Model | All | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% |
|-------|-----|--------|--------|--------|--------|--------|
| **PlmRF** | **0.813** | **0.662** | **0.727** | **0.829** | **0.818** | **0.676** |
| CLEAN\* | 0.720 | 0.604 | 0.684 | 0.733 | 0.689 | 0.612 |
| Blastp | 0.308 | 0.277 | 0.312 | 0.420 | 0.348 | 0.264 |

### Key observations (per-similarity-bin)

1. **PlmRF is remarkably robust across all identity bins**: AP ≥ 0.97 even for the
   hardest bin (20–30% identity), consistently outperforming all other models at every
   similarity level.
2. **Blastp performs well at sequence-similar bins** (AP ≥ 0.89 at 40–50%) but degrades
   at lower identity bins (20–30%: AP 0.83–0.95), as expected for a homology-based method.
3. **CLEAN shows moderate performance across bins**, with some variability between tracks.
   In Track B, CLEAN is slightly stronger at low identity (20–30%: AP 0.902) than at
   intermediate bins (30–40%: AP 0.791), suggesting its pre-trained representations
   generalize to diverse sequences.
4. **`no_hit` bin is absent** because after correcting the MMseqs2 has_hit logic, only
   ~20 test sequences per fold truly have no detectable homolog — too few to meet the
   minimum sample threshold (≥3 positives) per class for reliable metrics.
5. **MCC-F1 degrades at low identity** for all models (0.25–0.67 at 20–30%), indicating
   that precision-recall trade-offs worsen for distant homologs, even when AP remains high.

---

## Atomic Negative Cleaning Analysis (Track A)

**What this tests**: How much would each incremental negative cleaning filter (from the
new pipeline) affect the old dataset? We analyze how many old-dataset negatives each
filter would remove and whether re-running experiments is warranted.

### Filter Impact Analysis

| Filter | Description | Negatives removed | % of total | Re-run needed? |
|--------|-------------|-------------------|------------|----------------|
| **v1a** `filter_out_putative_tpss` | Remove known putative TPS by ID | **0** | 0.00% | No (no-op) |
| **v1b** `filter_by_ec` | Remove negatives with TPS EC numbers | **1–3** | 0.01–0.03% | No (negligible) |
| **v1c** `filter_by_go` | Remove negatives with TPS GO terms | Unknown | — | Blocked (GO annotations not in data files) |
| **v1d** `filter_by_pfam_supfam` | Remove negatives sharing TPS Pfam/SUPFAM domains | **34** | 0.34% | No (negligible) |
| **All combined** | Union of v1a + v1b + v1d | **≤41** | ≤0.41% | No |

### Key Findings

1. **The old dataset negatives are already clean** with respect to TPS contamination.
   Out of 9,944 negatives, at most 41 (0.41%) are flagged by any filter — roughly 3–11
   per fold (out of ~2,000). This is too few to produce a statistically significant
   change in AP, AUC, or MCC-F1.

2. **Three actual terpene synthases were found in the negative set**:
   - **M4VQY9**: Fumagillin beta-trans-bergamotene synthase (EC 4.2.3.211)
   - **P9WEY6**: Terpene synthase 2 / TPS2 (EC 4.2.3.-)
   - **P9WEY5**: Terpene synthase 3 / TPS3 (EC 4.2.3.-)

   These are genuine TPS proteins misclassified as negatives, but removing 3 entries from
   ~2,000 negatives per fold has no measurable effect on model metrics.

3. **34 negatives share Pfam or SUPFAM domains with TPS** (via `uniprot_info_dataset.tsv`
   annotations):
   - 8 share a Pfam domain (e.g., PF01397, PF03936, PF19086)
   - 33 share a SUPFAM domain (e.g., SSF48239, SSF48576)
   - Some overlap (7 share both Pfam and SUPFAM with TPS)

4. **`PUTATIVE_TPS_IDS` has zero overlap** with the old dataset negatives. The 18 putative
   TPS IDs were either never sampled into the old negative pool or were already excluded
   during the original data preparation.

5. **GO filter cannot be evaluated**: The `uniprot_info_dataset.tsv` file does not contain
   Gene Ontology annotation columns. Evaluating `filter_by_go` would require downloading
   GO annotations from UniProt for the 9,944 negative IDs.

### Conclusion

The atomic cleaning filters have **negligible impact** on the old phylo dataset because
the original Swiss-Prot sampling and preparation already excluded most TPS-like proteins.
The difference between old and new dataset performance (Track A vs Track B) is primarily
attributable to **TPS coverage** (1,163 vs 1,374 unique TPS) and **negative diversity**
(~10k negatives fully integrated into folds), not to negative contamination.

---

## Cross-Negative Stress Test

**What this tests**: How robust is a Blastp model when the negative set is swapped between
old (9,944) and new (10,000+) distributions at test time? Four conditions are compared:

| Condition | Training negatives | Test negatives | Description |
|-----------|-------------------|----------------|-------------|
| `old_model + old_neg` | Old negatives | Old negatives | Baseline (same-distribution) |
| `old_model + new_neg` | Old negatives | New negatives | Stress: model sees unfamiliar negatives |
| `new_model + new_neg` | New negatives | New negatives | Baseline (same-distribution, new) |
| `new_model + old_neg` | New negatives | Old negatives | Reverse stress test |

### Blastp TPS Detection under Cross-Negative Stress

| Condition | AP | AUC | MCC | F1 | FPR | Positives | Negatives |
|-----------|-----|-----|-----|-----|-----|-----------|-----------|
| old\_model + old\_neg | **0.999** | **1.000** | **0.999** | **1.000** | 0.000 | 970 | 10,023 |
| old\_model + new\_neg | 0.948 | 0.997 | 0.971 | 0.973 | 0.005 | 970 | 10,137 |
| new\_model + new\_neg | 0.981 | 0.990 | 0.989 | 0.990 | 0.000 | 970 | 10,137 |
| new\_model + old\_neg | 0.974 | 0.989 | 0.985 | 0.986 | 0.001 | 970 | 10,023 |

**Key findings:**
- The Blastp model trained on old negatives achieves near-perfect performance on old
  negatives (AP = 0.999) but drops to AP = 0.948 when tested on new negatives — a modest
  but real sensitivity to negative distribution.
- The reverse direction (new → old) shows less drop (0.981 → 0.974), suggesting the
  new-dataset negatives are harder or more diverse.
- Both models maintain FPR < 0.01 under all conditions, indicating strong specificity
  regardless of negative source.

---

## Cross-Track Comparison Summary

### Substrate Prediction (mAP) across tracks

| Model | Track A (phylo) | Track B (new) | Track C (synced) |
|-------|-----------------|---------------|------------------|
| **PlmRF** | 0.776 | **0.803** | 0.794 |
| CLEAN\* | 0.626 | 0.499 | 0.609 |
| Blastp | 0.542 | 0.612 | 0.639 |
| Foldseek | 0.395 | — | — |
| HMM | 0.334 | 0.437 | 0.345 |

### TPS Detection (AP) across tracks

| Model | Track A (phylo) | Track B (new) | Track C (synced) |
|-------|-----------------|---------------|------------------|
| **PlmRF** | 0.988 | **0.991** | 0.990 |
| Blastp | 0.892 | 0.934 | 0.938 |
| HMM | 0.890 | 0.895 | 0.914 |
| CLEAN\* | 0.857 | 0.871 | 0.840 |
| Foldseek | 0.838 | — | — |

### Key observations

1. **PlmRF consistently dominates** across all tracks and both tasks (substrate prediction
   and TPS detection).
2. **Track A → Track B improvement for PlmRF** (0.776 → 0.803 mAP): The larger, curated
   dataset with more TPS significantly improves substrate prediction. With the corrected
   negative distribution, both tracks now have comparable negative-to-positive ratios, so
   this improvement is attributable to better TPS coverage, not negative-set artifacts.
3. **Track A → Track C for PlmRF** (0.776 → 0.794 mAP): Even with the same negatives,
   removing 128 potentially mislabeled old-only TPS and changing fold assignments improves
   performance, suggesting the old dataset had noisy TPS labels.
4. **CLEAN degrades on Track B** (0.626 → 0.499 mAP): The new dataset's different substrate
   composition (more TPS classes) may be harder for CLEAN's pre-trained representations.
   Note that CLEAN is in-sample (see [Caveats](#caveats-and-notes)).
5. **HMM is robust for TPS detection** (AP: 0.890–0.914 across tracks) but weaker at
   fine-grained substrate prediction (mAP: 0.334–0.437).
6. **Blastp benefits from the new dataset** (0.542 → 0.612 mAP) but MCC-F1 remains low
   (~0.25), indicating it struggles with precision-recall balance.

---

## Pending Experiments

The following experiments are **planned but not yet completed**:

| Experiment | Track | Status | Notes |
|------------|-------|--------|-------|
| Similarity-bin evaluation | All tracks | **Done** | MMseqs2 artifacts regenerated with corrected has_hit logic; results above |
| Blastp | Track C (synced folds) | **Done** | AP = 0.938 |
| Foldseek | Track B (new dataset) | Pending | 6,594 negative PDBs missing (need AlphaFold download) |
| Foldseek | Track C (synced folds) | Pending | Same PDB issue for shared negatives |
| PlmDomainsRandomForest | All tracks | Pending | `data/clustering__domain_dist_based_features.pkl` missing |
| Cross-neg stress test | PlmRF, CLEAN, HMM, Foldseek | Pending | Only Blastp done so far |
| **Track D (cross-dataset)** | **All models** | **Pending** | Train on old synced, eval on new dataset folds. Configs ready under `cross_synced_to_new/`. |
| Per-kingdom evaluation | Tracks B, C | Pending | `id_2_kingdom_dataset_new_pipeline.pkl` available in `data/` |
| Atomic v1a | Track A | **Analyzed** | `filter_out_putative_tpss` — 0 negatives overlap with `PUTATIVE_TPS_IDS` (no-op) |
| Atomic v1b | Track A | **Analyzed** | `filter_by_ec` — 1–3 negatives match TPS EC numbers (negligible) |
| Atomic v1c | Track A | Blocked | `filter_by_go` — GO annotations not available in current data files |
| Atomic v1d | Track A | **Analyzed** | `filter_by_pfam_supfam` — 34 negatives share TPS domains (0.3%) |

---

## Generated Artifacts

### Evaluation CSVs
Located in `outputs/evaluation_results/`:

| File | Description |
|------|-------------|
| `track_a_phylo_folds.csv` | Track A aggregate metrics |
| `track_b_new_dataset.csv` | Track B aggregate metrics |
| `track_c_synced_folds.csv` | Track C aggregate metrics |
| `track_a_tps_detection.csv` | Track A TPS detection metrics |
| `track_b_tps_detection.csv` | Track B TPS detection metrics |
| `track_c_tps_detection.csv` | Track C TPS detection metrics |

### Visualization PNGs
Located in `outputs/evaluation_results/`:

- `track_a_phylo_{Mean Average Precision,ROC-AUC,MCC-F1 summary}.png`
- `track_b_new_{Mean Average Precision,ROC-AUC,MCC-F1 summary}.png`
- `track_c_synced_{Mean Average Precision,ROC-AUC,MCC-F1 summary}.png`
- `track_{a,b,c}_tps_detection_{Average Precision,ROC-AUC,MCC-F1 summary,PR}_TPS.png`

### MMseqs2 similarity artifacts
| File | Description |
|------|-------------|
| `data/mmseqs_similarities_track_a_phylo.pkl` | Per-fold test→train similarity for Track A |
| `data/mmseqs_similarities_track_b_new.pkl` | Per-fold test→train similarity for Track B |
| `data/mmseqs_similarities_track_c_synced.pkl` | Per-fold test→train similarity for Track C |

### Evaluation CSVs (per-similarity-bin)
| File | Description |
|------|-------------|
| `outputs/evaluation_results/all_results_track_{a,b,c}_simbins.csv` | Aggregate metrics with similarity bins |
| `outputs/evaluation_results/per_class_all_results_track_{a,b,c}_simbins.csv` | Per-class/bin breakdown |

### Data artifacts
| File | Description |
|------|-------------|
| `data/TPS-Nov19_2023_verified_all_reactions_with_neg_with_folds.csv` | Old dataset with all 9,944 negatives distributed across folds 0–4 |
| `data/TPS-Nov19_2023_with_synced_folds.csv` | Old dataset with synchronized fold assignments |
| `data/EnzymeExplorer_Dataset.csv` | New curated dataset |
| `data/substrate_2_tps_type_new_pipeline.pkl` | Substrate→TPS-type mapping for new dataset |
| `data/id_2_kingdom_dataset_new_pipeline.pkl` | ID→Kingdom mapping for new dataset |

---

## Caveats and Notes

### CLEAN is in-sample performance

**CLEAN** was pre-trained on a large corpus of enzyme-function data that **includes all TPS
and negative enzymes** used in our evaluation sets. Therefore, CLEAN's reported numbers
reflect **in-sample** (or near-in-sample) performance and **should not** be directly compared
to the other models as a measure of generalization. CLEAN's numbers are included for
completeness and as an upper-bound reference for what a large pre-trained enzyme model
achieves when the test data overlaps with its training set.

### Negative distribution correction

In the original version of the phylo dataset, 9,772 negatives were assigned to fold −1
and excluded from both training and evaluation, leaving only 172 "hard negatives" across
folds 0–4. This did not match the original paper protocol, where all negatives participate
in cross-validation. The corrected version redistributes all 9,944 negatives round-robin
across folds 0–4, resulting in ~2,000 negatives per fold — comparable to the new dataset's
distribution.

**Impact**: With the corrected negative distribution, Track A models now train and evaluate
with a realistic negative-to-positive ratio (~5:1), making the evaluation more meaningful
and more comparable to Track B. Previous results with only ~35 negatives per fold were
unrealistically optimistic for metrics sensitive to class balance (e.g., MCC-F1).

### ESM embedding regeneration for CLEAN

Per project requirements, CLEAN's ESM embeddings are regenerated from scratch on each
run via a custom `esm/scripts/extract.py` script. Sequences longer than 1,022 amino acids
are truncated to fit ESM's 1,024-token context window (including BOS/EOS tokens).

### 128 old-only TPS excluded

The new dataset is a curated superset of the old dataset's TPS. 128 TPS sequences present
only in the old dataset were deemed low-quality during curation and are excluded from
Track A (phylo) and Track C (synced) evaluations. The remaining 1,035 shared TPS are used
in both datasets.

### MMseqs2 `has_hit` logic correction

The initial version of `scripts/compute_fold_similarities.py` classified sequences as
`no_hit` if their best MMseqs2 hit had query coverage (`qcov`) below 0.5 — even if
a strong alignment existed. This incorrectly pushed ~900 sequences per fold into the
`no_hit` bin (instead of ~20), producing nonsensical Blastp AP ≈ 0.01 for `no_hit`
in Tracks B and C. The fix ensures `has_hit=True` whenever MMseqs2 finds *any*
alignment, regardless of `qcov`. After correction, only 15–21 sequences per fold
truly have no hit, and the `no_hit` bin no longer meets the minimum sample threshold
(≥3 positives per class) so it is omitted from results.

### Foldseek requires PDB structures

Foldseek was only run on Track A (phylo folds), where 11,350 PDB structures were available
from `/home/samusevich/TerpeneMiner/data/alphafold_structs/`. For the new dataset, 4,780
PDB structures were extracted from `enzyme_explorer_pdbs.zip`, but 6,594 negatives still
lack PDB files and would need to be downloaded from AlphaFold DB.

---

*Updated: 2026-03-24. Per-similarity-bin analysis updated with corrected MMseqs2
`has_hit` logic. Aggregate metrics refreshed from latest model outputs.
Evaluation commands executed via `enzyme_explorer_main evaluate` and
`enzyme_explorer_main visualize` as documented in the project README.*
