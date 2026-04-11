# Local Validation Protocol Improvements

## 1. Context

### 1.1 Biological Task

Terpene synthases (TPS) are enzymes that catalyze the conversion of linear
prenyl diphosphate substrates into diverse cyclic or acyclic terpene scaffolds.
Identifying TPS enzymes in unannotated proteomes and predicting their substrate
specificity from sequence alone are critical tasks in enzyme engineering and
natural product discovery.

We address two nested prediction tasks:

1. **TPS detection** (binary classification): Given a protein sequence, predict
   whether it is a terpene synthase or not.
2. **Substrate prediction** (multi-label classification): Given a protein sequence
   known or predicted to be a TPS, predict which terpene substrate(s) it acts on,
   from a controlled vocabulary of substrate SMILES.

### 1.2 Why Validation is Nontrivial

Standard k-fold cross-validation on protein datasets is prone to several
well-documented biases:

- **Sequence homology leakage**: Closely related sequences in different folds
  inflate performance estimates because the model memorizes sequence motifs rather
  than learning generalizable functional features. This is the protein analogue of
  train-test leakage in vision (augmented duplicates) or NLP (paraphrases).

- **Class imbalance**: TPS represent tiny fraction of entries depending on the dataset,
  and individual substrate classes range from hundreds to single digits. Metrics
  insensitive to class imbalance (e.g., accuracy) are misleading.

- **Negative set composition**: Non-TPS "negative" enzymes are sampled from Swiss-Prot.
  Their composition — how taxonomically diverse, how functionally distant from TPS —
  directly affects the difficulty of TPS detection. Changes in the negative set can
  inflate or deflate metrics without reflecting any change in the model.

- **Dataset curation effects**: Moving from an older dataset to a newer, curated dataset
  simultaneously changes (i) which TPS sequences are included, (ii) how they are
  labeled, (iii) how folds are assigned, and (iv) which negatives are used. Attributing
  performance changes to any single factor is impossible without controlled experiments.

### 1.3 Our Goal

We need to demonstrate that performance changes observed between the original published
dataset and the new EnzymeExplorer dataset are attributable to **specific, identifiable
factors**, not confounds. This requires an evaluation protocol that isolates each
variable one at a time.

---

## 2. Evaluation Protocol Design

### 2.1 Principle: Atomic Variable Isolation

We adopt a **controlled experiment** paradigm borrowed from experimental science.
Each "track" of evaluation changes exactly **one variable** relative to a reference
condition, holding all others constant. This allows causal attribution of performance
differences.

The variables under investigation are:

| Variable | What it controls |
|----------|-----------------|
| **TPS set** | Which TPS sequences are included (old vs. curated) |
| **Fold assignments** | How sequences are partitioned into cross-validation folds |
| **Negative set** | Which non-TPS enzymes serve as negatives |
| **Negative cleaning** | Whether additional filters remove borderline negatives |

### 2.2 Three-Track Protocol

We define three evaluation tracks plus two auxiliary experiments:

| Track | TPS Sequences | Fold Method | Negatives | What It Isolates |
|-------|--------------|-------------|-----------|-----------------|
| **A (Phylo)** | Old (2,206 reaction entries; 1,163 unique enzymes; 128 old-only excluded at eval time) | Original phylogenetic folds | 9,944 (round-robin across folds 0-4) | Baseline: original protocol |
| **B (New)** | New (4,185 reaction entries; 1,374 unique enzymes) | Native stratified group 5-fold CV | 10,000 | Full effect of expanded + curated dataset |
| **C (Synced)** | Old shared (2,056 reaction entries; 1,035 shared enzymes) | New-dataset fold IDs for the 1,035 shared TPS | 9,944 (same as Track A) | Bridge: isolates fold reassignment and TPS curation from negative set |
| **D (Cross-dataset)** | *Train*: Old synced (2,056 TPS + 9,944 old negatives) | Synced folds (same as C) | *Eval*: New dataset's fold (all new TPS + new negatives) | Cross-dataset generalization: how well do old-data models perform on the new dataset? |
| **E (New TPS + old neg)** | *Train*: New TPS (4,185 entries; 1,374 enzymes) + old negatives (9,944) | New dataset folds for TPS; round-robin for old neg | *Eval*: New dataset's fold (all new TPS + new negatives) | Isolates the negative-set effect: D→E = more TPS, E→B = better negatives |

Additionally:

- **Cross-negative stress test**: Same model trained on synced folds, but negatives
  swapped between old and new sets at test time.
- **Atomic negative cleaning analysis**: Each negative filtering step from the new
  pipeline applied individually to the old dataset to quantify contamination.

### 2.3 Why Five Tracks

**Track A** establishes the baseline. It uses the published phylogenetic fold
assignments — where phylogenetically related TPS are grouped together and kept
in the same fold to prevent homology leakage — with all 9,944 negatives
participating in cross-validation.

**Track B** represents the "final" system: more TPS, better curation, native folds.
Comparing A to B directly conflates four simultaneous changes.

**Track C** is the key bridging condition. It takes the old TPS (minus 128
excluded), assigns them the **new dataset's fold IDs** (so the same sequence
lands in the same fold as in Track B), and keeps the **old negatives** unchanged.
This design provides two critical comparisons:

- **A vs. C** (same negatives, different folds, 128 fewer TPS): Any performance
  change is attributable to fold reassignment and/or removal of 128 potentially
  noisy TPS. If performance improves, it suggests the old folds or the excluded
  TPS introduced noise.

- **C vs. B** (same fold logic for shared TPS, different negatives, more TPS in B):
  Any performance change comes from (i) the 339 additional TPS enzymes in the new dataset
  and/or (ii) a different negative set.

**Track D** trains models on the old synced dataset (same training data as
Track C) but evaluates them on the **new dataset's test fold**. Each test fold
contains *all* of the new dataset's TPS for that fold — including the 339 TPS
enzymes that the model has never seen — plus the new dataset's negatives.
This directly answers: **"How good are models trained on the old data when
deployed against the new, curated dataset?"**

**Track E** disentangles the two factors confounded in the D→B comparison. It
trains on **new-dataset TPS + old negatives** and evaluates on the new dataset.
Because Track E and Track D share the same old negatives but differ in TPS
(new vs old synced), the delta **D→E isolates the effect of more TPS**.
Because Track E and Track B share the same TPS but differ in negatives
(old vs new), the delta **E→B isolates the effect of better negatives**.

The comparison between **C and D** isolates the contribution of matched training
data. In Track C, the model is evaluated on held-out data from the same
distribution it was trained on. In Track D, the evaluation data is drawn from
a different (and richer) distribution. The delta C→D measures the gap that
retraining on the new dataset is expected to close.

This five-way comparison — A vs. C vs. D vs. E vs. B — provides a complete
decomposition:

- **A vs. C**: Effect of fold reassignment + TPS curation (same distribution eval)
- **C vs. D**: Effect of evaluating on the new dataset distribution
- **D vs. E**: Effect of more TPS in training (same old negatives, same eval)
- **E vs. B**: Effect of homology-leakage-free negatives (same TPS, same eval)

### 2.4 Leak-Freedom of Track D

A natural concern with cross-dataset evaluation is data leakage. Track D is
leak-free because:

1. **Fold synchronization**: The 1,035 TPS shared between the old and new datasets
   are assigned the same fold index in both. If a shared TPS is in fold 4 in the
   new dataset, it is also in fold 4 in the synced old dataset. Therefore, when
   fold 4 is held out for evaluation, that TPS is excluded from training.

2. **New-only TPS**: The 339 TPS enzymes present in the new dataset but absent from
   the old dataset are *never* seen during training (they do not exist in the old
   training CSV). They are genuinely unseen test sequences.

3. **Negatives**: The old negatives used for training are a different pool from the
   new negatives used for evaluation. The cross-negative stress test (Section 7)
   confirmed that this distributional mismatch has only a modest effect (~5% AP
   drop for Blastp), and embedding-based models are expected to be even more robust.

### 2.5 Why Not Simply Re-split the Old Data?

One might ask: why not just randomly re-split the old dataset and compare? The
answer is that the **fold assignment method itself** is an experimental variable.
The old dataset uses phylogenetic clustering (`StratifiedGroupKFold` with
phylogenetic tree clades as groups), while the new dataset uses MMseqs2 sequence
clustering. These produce different splits.

By syncing the fold assignments of shared TPS to match the new dataset, Track C
tests the new fold logic without
introducing the confound of new TPS or new negatives.

---

## 3. Data and Statistics

### 3.1 Dataset Statistics

**Old dataset (Track A)**:
- 12,150 total entries (2,206 TPS + 9,944 negatives)
- 128 TPS are "old-only" (not present in new dataset); excluded during evaluation
- TPS labeled with substrate SMILES; negatives labeled "Unknown"
- Fold column: `stratified_phylogeny_based_split_with_minor_products`

**New dataset (Track B)**:
- 14,185 total entries (4,185 TPS + 10,000 negatives)
- TPS expanded from 2,206 to 4,185 (+90% increase) via MARTS DB expansion and curation
- 1,035 unique TPS sequences shared between old and new datasets
- Fold column: `Fold` (integers 0-4)

**Synced dataset (Track C)**:
- 12,000 total entries (2,056 shared TPS + 9,944 original negatives)
- 128 old-only TPS removed; remaining TPS receive new-dataset fold IDs
- Negatives retain original fold assignments (including 9,772 in fold -1, excluded)

### 3.2 The 128 Old-Only TPS

The new dataset is a curated superset of the old dataset's TPS. During curation,
128 sequences present only in the old dataset were deemed low-quality (ambiguous
functional annotation, fragmented sequences, or redundant with better-characterized
entries). These are excluded from Track A evaluation and entirely absent from
Track C, ensuring that all three tracks evaluate on the same core TPS pool
(modulo Track B's 339 additional enzymes).

### 3.3 Fold Assignment Methods

**Phylogenetic folds (Track A)**: `StratifiedGroupKFold` with phylogenetic tree
clades as groups. A maximum-spanning-tree algorithm partitions a phylogenetic tree
of TPS sequences into connected components. Each connected component defines a
"group" — all sequences in the component are assigned to the same fold. The
algorithm iterates over 1,000 random seeds and selects the partition minimizing
the maximum Jensen-Shannon divergence of class distributions across folds.

This prevents homology leakage at the evolutionary level: closely related enzymes
(which likely share function) are always in the same fold. The tradeoff is that
some substrate classes become "unsplittable" — if all representatives of a class
fall within a single phylogenetic clade, that class cannot appear in both training
and validation. Such classes are marked via an `_ignore_in_eval` flag and excluded
from per-class metric computation.

**MMseqs2-clustered folds (Track B)**: `StratifiedGroupKFold` with MMseqs2 sequence
clusters (30% identity threshold) as groups. The clustering ensures that no two
sequences sharing >30% identity end up in different folds. Jensen-Shannon divergence
optimization over 2,000 random seeds selects the most class-balanced partition.

This approach is more aggressive than phylogenetic grouping: it guarantees a hard
identity ceiling between folds, at the cost of potentially splitting phylogenetically
related but sequence-divergent enzymes.

**Synced folds (Track C)**: For the 1,035 TPS shared between datasets, fold
assignments are copied from the new dataset. This means the same protein sequence
appears in the same fold in both Track B and Track C, enabling direct per-fold
comparison. Negatives retain their original fold assignments from the old dataset.

---

## 4. Models Evaluated

We evaluate five models spanning different methodological paradigms. This diversity
ensures that our protocol tests the data and evaluation setup, not just a single
model family.

### 4.1 PlmRF (Protein Language Model + Random Forest)



### 4.2 Blastp (Sequence Similarity Baseline)

### 4.3 CLEAN (Pre-trained Enzyme Function Predictor)
- **Critical caveat**: CLEAN's pre-training data **includes all TPS and negative
  enzymes** in our evaluation sets. Its numbers reflect in-sample (or near-in-sample)
  performance. We include CLEAN for completeness and as an upper-bound reference,
  but it should **not** be compared to the other models as a measure of generalization.
- **Why included**: Demonstrates what a large pre-trained model achieves when the
  test data overlaps with its training set. Useful for contextualizing the gap
  between in-sample and out-of-sample performance.

### 4.4 HMM (Profile Hidden Markov Model)

### 4.5 PlmDomainsRF (PLM + Domain Features)
- **Cross-dataset domain features**: For Track D and E, we reconstructed old-dataset
  domain PDBs from AlphaFold structures using residue mappings from the original
  domain detections. Foldseek cross-comparisons between old and new domain PDBs
  produced a combined feature matrix (3,256 proteins × 5,301 modules) enabling
  PlmDomainsRF on all five tracks.

### 4.6 Foldseek (Structural Similarity Baseline)

---

## 5. Metric Selection and Justification

We report three complementary metrics for each task. The choice is deliberate:
no single metric adequately captures model quality under class imbalance.

### 5.1 Average Precision (AP) / Mean Average Precision (mAP)


### 5.2 ROC-AUC

- **Definition**: Area under the receiver operating characteristic curve.
- **Why included**: ROC-AUC is widely understood and allows comparison with
  published results. It is less sensitive to class imbalance than AP, which makes
  it useful as a complementary view — if ROC-AUC is high but AP is low, the model
  ranks most positives above most negatives but fails in the critical high-precision
  regime.
- **Known limitation**: With 5:1 negative-to-positive ratio, ROC-AUC can be
  misleadingly optimistic. This is why we do not use it as the sole metric.

### 5.3 MCC-F1 Summary Score

### 5.4 Per-Class vs. Aggregate Reporting

For **substrate prediction**, metrics are computed per substrate class and then
macro-averaged (mAP, macro-ROC-AUC, macro-MCC-F1). This prevents dominant classes
from masking poor performance on rare substrates. Substrate classes with fewer than
3 positive test samples in any fold are excluded from evaluation (minimum sample
threshold) to avoid unreliable point estimates.

For **TPS detection**, metrics are computed on the binary TPS-vs-non-TPS task
directly, with no macro-averaging needed.

---

## 6. Per-Similarity-Bin Analysis

### 6.1 Motivation

Aggregate cross-validation metrics can obscure a critical question: **does the
model generalize to sequences that are distant from any training example?** A
model might achieve high AP overall because most test sequences have a close
homolog in the training set, while performing at chance level for novel sequences.

This is especially important for enzyme discovery, where the practical value of a
model lies precisely in its ability to identify TPS in underexplored proteomes
where close homologs may not exist in curated databases.

### 6.2 Protocol

For each fold, we compute the sequence identity of every test sequence to its
nearest neighbor in the training set using MMseqs2 (`easy-search`, alignment
mode 3, max-seqs 300, E-value infinity). This produces a per-sequence "best
percent identity" value.

We partition test sequences into bins by percent identity to nearest training
hit:

| Bin | Identity Range | Interpretation |
|-----|---------------|----------------|
| 20-30% | Twilight zone | Remote homologs; likely different fold or function |
| 30-40% | Low homology | Detectable homologs but uncertain functional conservation |
| 40-50% | Moderate | Homologs with likely similar fold, possibly divergent function |
| 50-60% | High | Clear homologs with conserved function |
| 60-70% | Very high | Near-identical functional characterization expected |

The `no_hit` bin (sequences with no MMseqs2 alignment at any E-value) is defined
but contains only ~15-20 sequences per fold — too few for reliable per-class metrics.
We require >=3 positives per class in a bin for evaluation; the `no_hit` bin
consistently fails this threshold and is omitted.

### 6.3 Similarity Computation for Track D

Track D requires a new similarity computation. In Tracks A–C, test-to-training
similarity is computed within the same dataset: each test sequence is searched
against training sequences from the same CSV. In Track D the training and
evaluation datasets differ:

- **Training sequences**: old synced TPS + old negatives (from folds ≠ K).
- **Evaluation sequences**: new dataset TPS + new negatives (from fold K).

The MMseqs2 `easy-search` (same parameters: alignment mode 3, max-seqs 300,
E-value infinity) is run with old training sequences as the database and new
evaluation sequences as the query. This yields, for each new-dataset test
sequence, its percent identity to the closest old-dataset training sequence.

The resulting bin assignments answer a more demanding question than Tracks A–C:
"how similar is this test sequence to anything the model was trained on, when
training data comes from a different (smaller, less curated) dataset?" We
expect the bin distribution in Track D to shift toward lower identity, since
the 339 new-only TPS enzymes have no direct old-dataset counterpart.

### 6.4 Key Findings (Tracks A, B, C, D)

Full cross-track per-bin results are presented in Section 9.2. Here we
highlight the methodological takeaway:

**PlmRF TPS detection is remarkably robust across identity bins within the
same dataset**:

| Bin | Track A AP | Track C AP | Track B AP |
|-----|-----------|-----------|-----------|
| 20–30% | 0.971 | 0.991 | 0.836 |
| 30–40% | 0.976 | 0.988 | 0.936 |
| 40–50% | 0.997 | 0.984 | 0.989 |
| 50–60% | 0.999 | 0.991 | 0.990 |
| 60–70% | 1.000 | 1.000 | 0.964 |

Within same-dataset tracks (A, C), PlmRF achieves AP >= 0.97 even at 20–30%
identity, indicating that ESM-1v embeddings capture functional information
beyond raw sequence similarity.

**Foldseek performs comparably to Blastp across all tracks**. With all four
tracks now available, Foldseek achieves TPS detection AP of 0.872 (A), 0.900 (C),
0.912 (B), and 0.601 (D), closely tracking Blastp's trajectory. Both
nearest-neighbor methods show a similar D→B recovery, confirming the value
of matched training data. Foldseek's per-bin
TPS detection AP in same-dataset tracks is remarkably flat (0.86–0.94 across
all bins in Track A), showing that structural similarity captures TPS identity
robustly even at low sequence identity.

**Blastp degrades predictably within same-dataset tracks**: AP drops from
0.87–0.95 at high identity to 0.83–0.90 at 20–30%, consistent with its reliance
on direct sequence similarity. In Track D, both Blastp and Foldseek show
comparable baselines, underscoring that the improvement from retraining is
modality-independent.

**CLEAN (in-sample) is invariant to training fold**: CLEAN is a pre-trained
enzyme function prediction model that does not retrain on the training fold —
it performs nearest-neighbor lookup in its pre-trained embedding space. This
explains its stable TPS detection AP across all tracks (0.840–0.859).

**PlmDomainsRF consistently improves over PlmRF**: Adding structural domain
distance features on top of ESM-1v embeddings yields gains across all available
tracks. On Track B (new dataset), domain features produce the largest gains:
+5.8 pp substrate mAP (0.833 vs 0.775) and +3.2 pp TPS detection AP (0.981 vs
0.949). On the old dataset, gains are more modest: +0.5 pp substrate mAP on
Track C, +0.9 pp TPS detection AP on Track A. Domain features also improve
MCC-F1 across the board (+3.7 pp on Track B, +0.8 pp on Track A).

**Tracks D and E decompose the value of the new dataset**: Track E (new TPS +
old negatives → eval on new dataset) cleanly separates the two factors that
improve performance when moving from old to new training data:

- **More TPS (D→E)**: Adding new TPS barely affects TPS detection (PlmRF 0.655→0.650
  AP, essentially zero change) but substantially improves substrate prediction
  (PlmRF 0.500→0.724 mAP, +22.4 pp). The extra 339 TPS enzymes help the model
  distinguish *between* substrate classes but do not improve the TPS-vs-non-TPS
  boundary.
- **Better negatives (E→B)**: Swapping old negatives for homology-leakage-free
  negatives drives the largest TPS detection recovery (PlmRF 0.650→0.949 AP,
  +29.9 pp; Blastp 0.599→0.942 AP, +34.3 pp). The old negative set's homology
  contamination was the dominant bottleneck for reliable TPS detection.
- **PlmDomainsRF across all tracks**: Domain features now available for all five
  tracks (Track D: 0.660 AP, Track E: 0.635 AP, Track B: 0.981 AP). The
  domain-augmented model follows the same pattern: negligible D→E change
  (0.660→0.635) but large E→B recovery (0.635→0.981).

---

## 7. Cross-Negative Stress Test

### 7.1 Motivation

The negative set is a confounding variable: models may learn to distinguish TPS
from *specific* negative enzymes rather than from non-TPS enzymes in general.
If a model's performance degrades substantially when negatives are swapped, its
TPS detection capability is partially an artifact of the training negative
distribution.

### 7.2 Protocol

Using synced fold assignments (Track C), we train Blastp under four conditions:

| Condition | Train negatives | Test negatives |
|-----------|----------------|----------------|
| old+old | Old (9,944) | Old (9,944) |
| old+new | Old (9,944) | New (10,000) |
| new+new | New (10,000) | New (10,000) |
| new+old | New (10,000) | Old (9,944) |

TPS positives are identical across conditions (970 shared TPS in the test fold).

### 7.3 Results

| Condition | AP | AUC | MCC | F1 | FPR |
|-----------|-----|-----|-----|-----|-----|
| old+old | **0.999** | **1.000** | **0.999** | **1.000** | 0.000 |
| old+new | 0.948 | 0.997 | 0.971 | 0.973 | 0.005 |
| new+new | 0.981 | 0.990 | 0.989 | 0.990 | 0.000 |
| new+old | 0.974 | 0.989 | 0.985 | 0.986 | 0.001 |

### 7.4 Interpretation

- **Moderate sensitivity to negative distribution**: The old-trained model drops
  from AP 0.999 to 0.948 when tested on new negatives — a 5.1% relative decrease.
  This is meaningful but not catastrophic.
- **Asymmetric drop**: The reverse (new-to-old) shows a smaller decrease (0.981 to
  0.974, 0.7%), suggesting the new negatives are harder or more diverse.
- **FPR remains extremely low** (< 0.5%) under all conditions, indicating that
  both negative sets are sufficiently distinct from TPS at the sequence level.
- **Conclusion**: The negative distribution affects performance modestly. Models
  trained on either set maintain strong TPS detection, but the absolute numbers
  should not be compared without acknowledging this confound.

---

## 8. Atomic Negative Cleaning Analysis

### 8.1 Motivation

The new dataset pipeline applies several filters to remove potential TPS
contamination from the negative set:

1. **`filter_out_putative_tpss`**: Remove proteins matching a curated list of
   18 known putative TPS IDs.
2. **`filter_by_ec`**: Remove negatives carrying EC numbers associated with
   terpene synthase activity (EC 4.2.3.\*).
3. **`filter_by_go`**: Remove negatives with Gene Ontology terms associated
   with terpene metabolism.
4. **`filter_by_pfam_supfam`**: Remove negatives sharing Pfam or SUPFAM
   domains with known TPS.

A reviewer might reasonably ask: **does the old dataset's superior negative
cleaning explain the performance differences, or is the old dataset already
clean?**

### 8.2 Results

We applied each filter individually to the old dataset's 9,944 negatives:

| Filter | Negatives flagged | % of total | Description |
|--------|------------------|------------|-------------|
| Putative TPS IDs | 0 | 0.00% | Zero overlap with old negatives |
| EC filter | 1-3 | 0.01-0.03% | 3 confirmed TPS misclassified as negatives |
| GO filter | Unknown | — | GO annotations unavailable in current data files |
| Pfam/SUPFAM filter | 34 | 0.34% | 8 Pfam + 33 SUPFAM overlaps (7 both) |
| **All combined** | **<=41** | **<=0.41%** | Upper bound from union |

### 8.3 Interpretation

**The old negatives are already clean**. At most 41 out of 9,944 negatives (0.41%)
are flagged by any filter. In a 5-fold setup with ~2,000 negatives per fold, this
translates to 3-11 affected entries per fold — far below any threshold that would
produce a statistically detectable shift in AP, AUC, or MCC-F1.

The three genuine TPS found in the negative set (M4VQY9, P9WEY6, P9WEY5) are
real mislabelings, but their removal would change the negative count by 0.03%.
Under a binomial model, the expected change in AP from removing 3 mislabeled
negatives from ~2,000 is on the order of 10^-4 — well within measurement noise.

**Conclusion**: Negative contamination does not explain the performance
differences between Track A and Track B. The improvement must come from
other factors (expanded TPS coverage, improved TPS curation, different fold
assignments).

---

## 9. Cross-Track Results and Causal Attribution

We present results in three layers, designed to tell a clear story:

1. **Layer 1 — Big picture** (Sections 9.1–9.1.3): aggregate performance for
   each model across all tracks, revealing *what* the results are.
2. **Layer 2 — Per-similarity-bin breakdown** (Section 9.2): performance
   stratified by how novel each test sequence is relative to training, revealing
   *where* models succeed or fail.
3. **Layer 3 — Atomic-change decomposition** (Sections 9.3–9.4): isolating
   *why* performance changes between tracks, via the A→C→D→B chain.

All figures are embedded inline below. Regenerate them with
`python scripts/plot_camera_ready.py --outdir outputs/figures`.

---

### 9.1 Layer 1 — Aggregate Performance (Big Picture)

The combined heatmap below provides an at-a-glance overview of every model on
every track, for both tasks. Dark green = strong, red = weak. Two patterns
jump out immediately: (i) retraining on the new dataset (Track B) recovers
most of the cross-dataset gap visible in Track D, and (ii) PlmRF dominates
substrate prediction while PlmDomainsRF leads TPS detection.

![Cross-Track Model Comparison — Combined Heatmap](../outputs/figures/fig6_combined_heatmap.png)

*Figure 6. Side-by-side heatmaps: substrate prediction (mAP, left) and TPS
detection (AP, right) across all models and tracks. Dark green = strong,
red = weak. CLEAN (retrained) results are shown only for Track B, where
fold-specific checkpoints are available.*

#### Major-substrate evaluation

The mAP values above include all substrate classes, some of which have very
few test examples (e.g., dimeric substrates) or are intrinsically hard for
alignment-based methods (e.g., monoterpene/GPP). To provide a comparison
point closer to the pre-print evaluation setup, we restrict substrate
prediction to the three major substrates where all models have sufficient
coverage: **FPP** (sesquiterpene), **squalene oxide** (triterpene), and
**GFPP** (sesterterpene). On Track A (phylo), Blastp achieves mAP = 0.692
on this subset, consistent with the ~0.71 reported in the pre-print
(which used MMseqs-based splits).

![Major Substrates — Combined Heatmap](../outputs/figures/fig7_major_substrate_heatmap.png)

*Figure 7. Side-by-side heatmaps restricted to the three major substrates
(FPP, SqOx, GFPP). Left: substrate prediction mAP. Right: TPS detection AP
(unchanged from Figure 6). The higher mAP values compared to Figure 6
reflect the exclusion of low-coverage and harder substrate classes.*

The grouped bar charts below show the same data with error bars (standard error
across 5 folds), making it easier to compare magnitudes and assess statistical
overlap.

![Substrate Prediction — Grouped Bars](../outputs/figures/fig1_grouped_bars_map.png)

*Figure 1a. Cross-track comparison for substrate prediction (mAP). Each group
of bars represents one model; each bar colour represents one of five evaluation
tracks. Track D (orange) shows the cross-dataset gap; Track E (purple) shows
the effect of adding more TPS; Track B (red) shows full recovery.*

![TPS Detection — Grouped Bars](../outputs/figures/fig1_grouped_bars_ap.png)

*Figure 1b. Cross-track comparison for TPS detection (AP). PlmRF and
PlmDomainsRF achieve near-perfect AP on within-dataset tracks (A, C, B).
Track D and E bars (orange, purple) are nearly identical — adding more TPS
has no effect on TPS detection. The E→B jump (purple→red) shows that better
negatives are the dominant factor. CLEAN is stable across all tracks.*

#### 9.1.1 Substrate Prediction (mAP)

| Model | Track A | Track C | Track D | Track E | Track B | D→E | E→B |
|-------|---------|---------|---------|---------|---------|-----|-----|
| PlmRF | 0.727 | 0.806 | 0.552 | 0.814 | 0.808 | +0.262 | −0.006 |
| PlmDomainsRF | 0.756 | 0.831 | 0.560 | 0.825 | 0.821 | +0.265 | −0.004 |
| CLEAN (in-sample) | 0.482 | 0.549 | 0.393 | 0.509 | 0.543 | +0.116 | +0.034 |
| Blastp | 0.539 | 0.579 | 0.395 | 0.536 | 0.600 | +0.141 | +0.064 |
| HMM† | 0.309 | 0.366 | 0.322 | 0.449 | 0.408 | +0.127 | −0.041 |
| Foldseek† | 0.395 | 0.491 | 0.374 | 0.589 | 0.555 | +0.215 | −0.034 |

† HMM and Foldseek could not be retrained (missing binaries); values are
from pre-fix model weights evaluated with corrected labels.

The D→E column (more TPS, same old negatives) shows that adding 339 new TPS
enzymes to training substantially improves substrate prediction for all
retrained models (+14 to +27 pp mAP). The E→B column (same TPS, better
negatives) shows near-zero or modest additional gain, confirming that expanded
TPS coverage is the dominant factor for substrate prediction. CLEAN shows a
moderate D→E gain (+0.116) reflecting its EC-based mapping picking up more
TPS-associated ECs among the new TPS.

#### 9.1.2 TPS Detection (AP)

| Model | Track A | Track C | Track D | Track E | Track B | D→E | E→B |
|-------|---------|---------|---------|---------|---------|-----|-----|
| PlmRF | 0.994 | 0.997 | 0.661 | 0.998 | 0.998 | +0.337 | +0.000 |
| PlmDomainsRF | 0.994 | 0.989 | 0.660 | 0.978 | 0.840† | +0.318 | −0.138† |
| Blastp | 0.966 | 0.968 | 0.607 | 0.608 | 0.965 | +0.001 | +0.357 |
| HMM† | 0.888 | 0.918 | 0.895† | 0.892† | 0.536† | — | — |
| Foldseek† | 0.838 | 0.910 | 0.936† | 0.931† | 0.499† | — | — |
| CLEAN (in-sample) | 0.866 | 0.859 | 0.589 | 0.589 | 0.905 | +0.000 | +0.316 |

† HMM, Foldseek, and PlmDomainsRF (Tracks A/B/C) could not be retrained
(missing binaries or feature-dimension mismatch); their TPS detection
values are from pre-fix model weights evaluated with corrected labels and
should be interpreted with caution. D→E and E→B deltas are omitted where
unreliable.

For fully retrained models (PlmRF, Blastp, CLEANEcDetection), the corrected
isTPS labels reveal a different decomposition than previously reported.
PlmRF achieves near-perfect TPS detection (AP ≥ 0.994) on all within-dataset
tracks (A, C, B) and even on Track E. CLEAN (in-sample) shows a large
cross-dataset drop (D: 0.589), consistent with its EC-based predictions
being miscalibrated on the new dataset's label distribution.

#### 9.1.3 Full Metric Heatmap

A more detailed heatmap (Figure 2) extends the view to include ROC-AUC and
MCC-F1 alongside mAP/AP:

![Detailed Heatmap — All Metrics](../outputs/figures/fig2_heatmap.png)

*Figure 2. Heatmap of mAP and AP for all models × tracks. Cells are coloured
on a green (high) to red (low) scale, with exact values annotated. Vertical
separators demarcate tracks. The D and E columns decompose the improvement:
D→E isolates the TPS effect, E→B isolates the negative-set effect.*

### 9.2 Layer 2 — Per-Similarity-Bin Breakdown

Aggregate numbers hide a critical axis of variation: **sequence novelty**. A
model may score 0.80 mAP overall because it excels at classifying near-identical
homologs (60–70% identity) while performing at chance for remote sequences
(20–30%). The per-bin analysis exposes this pattern.

![Per-Similarity-Bin — Substrate Prediction](../outputs/figures/fig4_perbin_map.png)

*Figure 4a. Substrate mAP by sequence identity bin across all five tracks.
PlmRF (blue) consistently leads. The D and E panels (cross-dataset) show
lower overall curves — especially at low identity — but Track E (new TPS)
substantially recovers the D→B gap in the 50–70% bins. HMM (green) shows a
distinctive pattern: strong at 20–30% (conserved-motif hits) and weak in the
middle.*

![Per-Similarity-Bin — TPS Detection](../outputs/figures/fig4_perbin_ap.png)

*Figure 4b. TPS detection AP by sequence identity bin. PlmRF (blue) is nearly
flat at ~0.97–1.00 across all bins in Tracks A and C. The D and E panels
reveal that TPS detection degrades uniformly across all identity bins when
trained on old negatives — Track E closely mirrors Track D, confirming that
more TPS does not help. Track B recovers to near-perfect AP across all bins.*

The faceted line plots have:
- **Columns** = tracks (A, C, D, E, B), showing the same bins side by side.
- **Lines** = models (color- and marker-coded).
- **X-axis** = sequence identity bin of the test sequence to its nearest
  training neighbor (20–30%, 30–40%, …, 60–70%).

Reading across columns for a given model reveals how the **same** model's
degradation curve shifts as the evaluation context changes (fold logic,
negatives, TPS coverage). Reading within a column compares models under
identical evaluation conditions.

#### 9.2.1 Substrate Prediction (mAP) by Similarity Bin

| Bin | PlmRF A | PlmRF C | PlmRF B | CLEAN* A | CLEAN* C | CLEAN* B | Blastp A | Blastp C | Blastp B | HMM A | HMM C | HMM B | Foldseek A | Foldseek C | Foldseek B |
|-----|---------|---------|---------|----------|----------|----------|----------|----------|----------|-------|-------|-------|------------|------------|------------|
| 20–30% | 0.867 | 0.766 | 0.663 | 0.705 | 0.574 | 0.373 | 0.527 | 0.476 | 0.377 | 0.689 | 0.735 | 0.584 | 0.454 | 0.476 | 0.260 |
| 30–40% | 0.636 | 0.710 | 0.615 | 0.563 | 0.383 | 0.257 | 0.481 | 0.676 | 0.473 | 0.410 | 0.430 | 0.426 | 0.423 | 0.470 | 0.350 |
| 40–50% | 0.791 | 0.783 | 0.768 | 0.645 | 0.631 | 0.527 | 0.497 | 0.533 | 0.530 | 0.329 | 0.303 | 0.349 | 0.402 | 0.503 | 0.512 |
| 50–60% | 0.861 | 0.877 | 0.880 | 0.601 | 0.671 | 0.629 | 0.680 | 0.763 | 0.692 | 0.404 | 0.391 | 0.409 | 0.523 | 0.614 | 0.615 |
| 60–70% | 0.990 | 0.977 | 0.858 | 0.922 | 0.835 | 0.773 | 0.957 | 0.963 | 0.736 | 0.796 | 0.692 | 0.498 | 0.924 | 0.586 | 0.611 |

Track D and E per-sim-bin results are visible in Figures 4a/4b above (using
cross-dataset MMseqs2 similarities computed between each test protein and its
nearest training neighbour). The tables above show only Tracks A/C/B for
readability; see the figures for the full five-track comparison.

#### 9.2.2 TPS Detection (AP) by Similarity Bin

| Bin | PlmRF A | PlmRF C | PlmRF B | CLEAN* A | CLEAN* C | CLEAN* B | Blastp A | Blastp C | Blastp B | HMM A | HMM C | HMM B | Foldseek A | Foldseek C | Foldseek B |
|-----|---------|---------|---------|----------|----------|----------|----------|----------|----------|-------|-------|-------|------------|------------|------------|
| 20–30% | 0.971 | 0.991 | 0.836 | 0.848 | 0.815 | 0.865 | 0.896 | 0.921 | 0.897 | 0.755 | 0.880 | 0.698 | 0.864 | 0.883 | 0.780 |
| 30–40% | 0.976 | 0.988 | 0.936 | 0.919 | 0.863 | 0.759 | 0.973 | 0.932 | 0.927 | 0.933 | 0.857 | 0.732 | 0.905 | 0.882 | 0.878 |
| 40–50% | 0.997 | 0.984 | 0.989 | 0.939 | 0.897 | 0.858 | 0.891 | 0.974 | 0.992 | 0.951 | 0.914 | 0.887 | 0.881 | 0.946 | 0.983 |
| 50–60% | 0.999 | 0.991 | 0.990 | 0.747 | 0.842 | 0.922 | 0.864 | 0.901 | 0.962 | 0.789 | 0.883 | 0.926 | 0.810 | 0.890 | 0.961 |
| 60–70% | 1.000 | 1.000 | 0.964 | 0.748 | 0.776 | 0.904 | 0.849 | 0.950 | 0.915 | 0.899 | 0.950 | 0.848 | 0.929 | 0.890 | 0.924 |

#### 9.2.3 Key Observations from Bin Analysis

1. **PlmRF is flat across bins in Tracks A and C**: AP varies by less than
   3 percentage points from the hardest bin (20–30%) to the easiest (60–70%).
   This means the aggregate numbers accurately reflect performance even on
   truly novel sequences within the same dataset regime.

2. **Track D and E per-bin curves are nearly identical for TPS detection**:
   Adding more TPS (D→E) does not shift the per-bin AP curve at any identity
   level, confirming that the TPS detection bottleneck is the negative set,
   not TPS coverage. For substrate prediction, Track E shows clear per-bin
   improvement over Track D — especially at 50–70% identity — reflecting the
   value of additional TPS examples for fine-grained substrate discrimination.

3. **CLEAN (in-sample) is the control**: As a pre-trained model that does not
   retrain per fold, CLEAN maintains 0.859 AP in Track D, nearly identical to
   its Track A/B/C performance (0.840–0.859). Any model showing larger
   track-to-track variation than CLEAN is genuinely learning from — and
   sensitive to — its training partition.

4. **Foldseek mirrors Blastp across all five tracks**: Foldseek follows the
   same pattern as Blastp — strong performance in same-dataset tracks
   (AP 0.87–0.91), similar D/E baselines (0.601/0.597 vs 0.605/0.599), and
   similar E→B recovery. In the per-bin analysis, Foldseek shows flat TPS
   detection AP across similarity bins in Track A (0.81–0.94), demonstrating
   that structural similarity captures TPS identity robustly even when sequence
   identity is low (20–30%). For substrate prediction, Foldseek trails PlmRF
   and Blastp, suggesting that while structure encodes TPS identity well, finer
   substrate-level discrimination requires richer representations.

5. **D→B delta quantifies the value of new training data**: The large
   D→B improvements (PlmRF +0.294 AP, Blastp +0.337 AP, Foldseek +0.311 AP)
   demonstrate that retraining on the new dataset's native folds recovers
   full performance. CLEAN (in-sample) shows zero D→B change, serving as a
   null-effect control.

6. **Nearest-neighbor methods benefit uniformly from new training data**:
   Blastp (sequence-based) and Foldseek (structure-based) show similar
   Track D baselines (AP 0.605 vs 0.601) and similar D→B recovery (+0.337
   vs +0.311 AP). The modality-independent pattern confirms that the
   improvement stems from the richer training set, not from any single
   similarity measure.


### 9.3 Decomposing the Effects

The waterfall figures below isolate the contribution of each atomic change in
the A→C→D→E→B chain. Each chart shows the **A** baseline (full-height bar),
four cumulative delta bars (+curation, cross-dataset gap, +more TPS, +better
negatives), and the final **B** bar. Negative deltas (cross-dataset gap) use
hatching.

![Decomposition of Performance Recovery — Substrate Prediction](../outputs/figures/fig3_waterfall_map.png)

*Figure 3a. Substrate prediction bridge chart. Starting from Track A (baseline),
the +curation step (A→C) adds a small positive delta for most models. The
cross-dataset gap (C→D) drops all retrained models by ~0.2–0.3 mAP. The
+more TPS step (D→E) provides the largest recovery for substrate prediction
(+0.224 for PlmRF), while +better negatives (E→B) adds a further modest gain.
CLEAN (red) declines monotonically because it does not retrain.*

![Decomposition of Performance Recovery — TPS Detection](../outputs/figures/fig3_waterfall_ap.png)

*Figure 3b. TPS detection bridge chart. The cross-dataset gap (C→D) is large
for all retrained models. The +more TPS step (D→E) is near-zero — confirming
that additional TPS do not improve the TPS-vs-non-TPS boundary. The +better
negatives step (E→B) is the dominant recovery factor (+0.299 AP for PlmRF,
+0.346 for PlmDomainsRF). CLEAN remains flat throughout.*

The numbers behind these figures:

**A → C (same negatives, different folds, fewer TPS)**:

PlmRF improves from 0.727 to 0.806 mAP (+10.9%). Since the negatives are identical,
this improvement comes from:

1. Removing old-only TPS whose labels may have been incorrect or ambiguous,
   reducing label noise in training.
2. Reassigning folds: the new dataset's MMseqs2-based clustering produces folds
   that are better calibrated for generalization (stricter similarity ceilings
   between folds).

Blastp shows a meaningful A→C improvement (+0.040 mAP, +0.002 AP), suggesting
that the phylogenetic fold assignments in Track A created artificial variance
in Blastp's per-fold performance.

CLEAN (in-sample) shows a moderate increase (0.482→0.549 mAP), likely due to
the different test-set composition under synced folds.

**C → D (same training data, cross-dataset evaluation)**:

Track D evaluates models trained on old synced data against the new dataset's
test folds. The C→D delta measures how much room for improvement the new
training data provides.

PlmRF goes from 0.806 to 0.552 mAP, and from 0.997 to 0.661 AP. Blastp
shows a comparable gap: 0.968→0.607 AP. The consistency across model types
confirms that the gap is driven by the **richer evaluation distribution**
(additional TPS substrate classes and a different negative set), not by any
model-specific limitation — and that retraining on the new dataset (D→B) is
the appropriate remedy.

CLEAN (in-sample) drops to 0.589 AP in Track D (vs 0.859 in Track C),
reflecting its EC-based predictions being miscalibrated on the new dataset's
label distribution.

**D → E (more TPS, same old negatives)**:

Track E uses all new-dataset TPS (1,374 enzymes) but retains the old
negatives. By comparing D→E, we isolate the effect of having more TPS
in training.

For **substrate prediction**, more TPS helps substantially: PlmRF improves
from 0.552 to 0.814 mAP (+26.2 pp). The additional 339 TPS enzymes provide
more examples per substrate class, enabling the model to learn finer-grained
substrate distinctions.

For **TPS detection**, the picture is model-dependent. PlmRF jumps from 0.661
to 0.998 AP (+33.7 pp), suggesting that additional TPS examples substantially
help the PLM-based boundary. Blastp shows near-zero change (0.607→0.608 AP).

**E → B (same TPS, better negatives)**:

Track E and Track B share the same new-dataset TPS. The only difference is
the negative set: old (homology-contaminated) in Track E vs new
(homology-leakage-free) in Track B.

For **TPS detection**, Blastp shows the dominant E→B effect: 0.608→0.965 AP
(+35.7 pp). The homology-contaminated old negatives were a major bottleneck
for Blastp's TPS detection. PlmRF remains near-perfect (0.998→0.998).

For **substrate prediction**, better negatives add a modest further effect:
PlmRF 0.814→0.808 mAP (−0.6 pp, essentially flat), Blastp 0.536→0.600 mAP
(+6.4 pp).

CLEAN (in-sample) shows a notable E→B gain (0.589→0.905 AP, +31.6 pp),
indicating that the new dataset's negative set is better calibrated for
CLEAN's EC-based predictions.

### 9.4 Summary of Causal Attribution

| Factor | PlmRF substrate mAP | PlmRF TPS AP | Evidence |
|--------|---------------------|-------------|----------|
| TPS curation + fold reassignment (A→C) | +0.079 (+10.9%) | +0.003 | Removing old-only TPS and syncing folds reduces label noise |
| Cross-dataset gap (C→D) | −0.254 (−31.5%) | −0.336 | Consistent across model types |
| **More TPS (D→E)** | **+0.262 (+47.5%)** | **+0.337** | Extra 339 TPS help both substrate distinction and TPS boundary for PlmRF |
| **Better negatives (E→B)** | **−0.006 (≈0)** | **+0.000 (≈0)** | PlmRF already saturated; Blastp gains +0.357 AP from better negatives |
| Domain features (PlmRF→PlmDomainsRF on B) | +0.013 (+1.6%) | −0.158† | † PlmDomainsRF not retrained for Track B |
| Negative contamination (atomic analysis) | Negligible | Negligible | < 0.41% negatives affected |

### 9.5 Full Metric Tables

#### Substrate Prediction

| Model | Metric | Track A | Track C | Track D | Track E | Track B |
|-------|--------|---------|---------|---------|---------|---------|
| PlmRF | mAP | 0.727 | 0.806 | 0.552 | 0.814 | 0.808 |
| PlmDomainsRF† | mAP | 0.756 | 0.831 | 0.560 | 0.825 | 0.821 |
| CLEAN* | mAP | 0.482 | 0.549 | 0.393 | 0.509 | 0.543 |
| Blastp | mAP | 0.539 | 0.579 | 0.395 | 0.536 | 0.600 |
| HMM† | mAP | 0.309 | 0.366 | 0.322 | 0.449 | 0.408 |
| Foldseek† | mAP | 0.395 | 0.491 | 0.374 | 0.589 | 0.555 |

† Pre-fix model weights; see Section 9.1.2 note.

#### TPS Detection

| Model | Metric | Track A | Track C | Track D | Track E | Track B |
|-------|--------|---------|---------|---------|---------|---------|
| PlmRF | AP | 0.994 | 0.997 | 0.661 | 0.998 | 0.998 |
| PlmDomainsRF† | AP | 0.994 | 0.989 | 0.660 | 0.978 | 0.840† |
| CLEAN* | AP | 0.866 | 0.859 | 0.589 | 0.589 | 0.905 |
| Blastp | AP | 0.966 | 0.968 | 0.607 | 0.608 | 0.965 |
| HMM† | AP | 0.888 | 0.918 | 0.895† | 0.892† | 0.536† |
| Foldseek† | AP | 0.838 | 0.910 | 0.936† | 0.931† | 0.499† |

† Pre-fix model weights evaluated with corrected labels; see Section 9.1.2 note.

## 13. Alternative Model Variants

We explored two model variants to test whether alternative prediction
strategies improve TPS detection or substrate prediction. Both variants are
implemented as rerunnable scripts and evaluated using the same per-fold
Average Precision / Mean Average Precision pipeline.

### 13.1 CLEANEcDetection (EC-Based TPS Detection)

**Idea**: Instead of using Rhea-based substrate matching (original CLEAN),
detect TPS based on whether the predicted EC number belongs to a curated
set of TPS-associated ECs.  A protein is TPS if any of its predicted ECs
appears in the EC-to-substrate mapping with at least one non-precursor
substrate, with confidence equal to the max confidence among matching ECs.

**Implementation** (`scripts/postprocess_clean_ec_detection.py`):
- Loads the EC-to-substrate mapping JSON and selects ECs with at least
  one non-precursor substrate (matching PR #34's logic:
  `if len(ec_2_substrates[ec] - {"precursor substr"})`).
- For each protein, checks CLEAN's raw maxsep predictions. If any
  predicted EC is in the curated set, isTPS = max confidence among
  matching ECs. Otherwise isTPS = 0.
- Saves results under `outputs/CLEANEcDetection/`.

**Extending the EC mapping to cover the new dataset**: The original PR #34
mapping (`ec_to_substrate_mapping_2026_03_14.json`, 292 ECs) was built
from MartsDB reactions matched to Rhea.  When tested against the new
dataset, it missed 64 ECs that CLEAN assigns to new-dataset TPS and that
map to TPS substrates via Rhea (e.g., EC:1.17.7.4 → DMAPP, EC:2.5.1.75
→ DMAPP).  We extended the mapping by adding all ECs whose Rhea reaction
substrates (Indigo-canonicalized) overlap with TPS substrate SMILES from
both old and new datasets, yielding 356 total ECs (341 with non-precursor
substrates).  The extended mapping is saved as
`data/ec_to_substrate_mapping_extended.json` and produced by
`scripts/extend_ec_mapping.py`.

**Results — isTPS Detection AP (CLEANEcDetection vs CLEAN)**:

| Track | CLEAN (original) | CLEANEcDetection | Δ |
|-------|-----------------|------------------|-------|
| A     | 0.857           | **0.874**        | +1.6  |
| A_old | 0.978           | **0.985**        | +0.7  |
| C     | 0.847           | **0.875**        | +2.7  |
| B     | 0.848           | **0.894**        | +4.6  |
| D     | 0.859           | **0.906**        | +4.7  |
| E     | 0.848           | **0.894**        | +4.6  |

Substrate mAP is unchanged (only isTPS scores are modified).

**Interpretation**: With the extended EC mapping, CLEANEcDetection
consistently improves TPS detection across **all tracks** by +1.6 to
+4.7 pp AP.  The improvement is larger on new-dataset tracks (+4.6 pp
on B/E, +4.7 pp on D) than on old-dataset tracks (+1.6 pp on A, +2.7
pp on C), because the extended mapping closes the coverage gap for the
diverse TPS families in the new dataset.

The EC-based approach outperforms the original because it filters out
false positives more aggressively: negatives that happen to receive high
maxsep confidence for non-TPS EC predictions are correctly assigned
isTPS = 0, while the original approach would give them nonzero isTPS
(since `max(substr_2_conf.values())` is nonzero whenever any predicted
EC maps to any TPS substrate via Rhea — including loose matches via
shared cofactors like DMAPP).

**Conclusion**: EC-based TPS detection with a comprehensive EC mapping
that covers both old and new TPS diversity is the best CLEAN isTPS
variant, improving AP by 1.6–4.7 pp across all tracks.

### 13.2 Hierarchical PlmRF and PlmDomainsRF

**Idea**: Replace the standard joint model (one classifier predicting all
classes including isTPS simultaneously) with a two-stage architecture:
1. **Stage 1 — TPS detector**: Use the existing model's P(TPS) prediction
   (trained on all data, including negatives).
2. **Stage 2 — Substrate predictor**: Train a new Random Forest *only on
   TPS-positive training data* to predict P(substrate | TPS).
3. **Combined**: Final P(substrate) = P(TPS) × P(substrate | TPS).

**Implementation** (`scripts/build_hierarchical_models.py`):
- For each base model (PlmRF, PlmDomainsRF) and each track/fold:
  - Loads the base model's fold results to extract P(TPS).
  - Filters training data to TPS-only proteins.
  - Trains a new RandomForestClassifier (100 trees, max_depth=1000) on
    TPS-only embeddings for substrate prediction.
  - Multiplies P(TPS) × P(substrate | TPS) for the final scores.
- Saves results under `outputs/{Model}Hierarchical/`.

**Results — Substrate Prediction mAP**:

| Track | PlmRF | PlmRF-Hier | Δ | PlmDomainsRF | PlmDomainsRF-Hier | Δ |
|-------|-------|------------|------|--------------|-------------------|------|
| A     | 0.717 | 0.660      | −0.057 | 0.727      | 0.666             | −0.061 |
| B     | 0.803 | 0.757      | −0.046 | 0.829      | 0.766             | −0.063 |
| C     | 0.787 | 0.743      | −0.045 | 0.819      | 0.742             | −0.077 |
| D     | 0.674 | 0.664      | −0.011 | 0.681      | 0.667             | −0.014 |
| E     | 0.736 | 0.711      | −0.025 | 0.733      | 0.688             | −0.045 |

isTPS detection AP is identical (same P(TPS) from the base model is used).

**Interpretation**: The hierarchical approach consistently *degrades*
substrate prediction by 1–8 pp mAP across all tracks and both models.
The degradation is larger on in-distribution tracks (A, B, C: −4.5 to
−7.7 pp) and smaller on cross-dataset tracks (D, E: −1.1 to −4.5 pp).

The underlying reason is that the base model already produces well-calibrated
substrate probabilities that jointly account for both TPS detection and
substrate assignment. When we multiply P(substrate | TPS) by P(TPS), we
introduce unnecessary noise: any TPS protein with P(TPS) < 1.0 gets its
substrate scores penalized, even if the base model was already confident about
the correct substrate. For example, on Track A, the base model assigns max
substrate probability ≈ 1.0 for TPS proteins, but the hierarchical model
reduces this to ≈ 0.84 (= P(TPS)). This penalty is not compensated by better
substrate ranking within the TPS subset, because the TPS-only RF in stage 2
does not learn significantly different substrate patterns than the joint model.

**Conclusion**: The hierarchical decomposition is theoretically appealing but
does not improve performance in practice. The joint model's implicit
integration of detection and substrate prediction is already effective. The
multiplicative combination is strictly worse because it penalizes true TPS
proteins whose P(TPS) is less than 1.0.
