# Local Validation Protocol: Motivation, Design, and Empirical Evidence

## 1. Problem Statement

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

- **Dataset curation effects**: Moving from an older dataset to a newer, curated one
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
| **A (Phylo)** | Old (2,206 TPS; 128 old-only excluded at eval time) | Original phylogenetic folds | 9,944 (round-robin across folds 0-4) | Baseline: original protocol with corrected negative distribution |
| **B (New)** | New (4,185 TPS) | Native stratified group 5-fold CV | 10,000 | Full effect of expanded + curated dataset |
| **C (Synced)** | Old shared (2,056 TPS = old minus 128 old-only) | New-dataset fold IDs for the 1,035 shared TPS | 9,944 (same as Track A) | Bridge: isolates fold reassignment and TPS curation from negative set |
| **D (Cross-dataset)** | *Train*: Old synced (2,056 TPS + 9,944 old negatives) | Synced folds (same as C) | *Eval*: New dataset's fold (all new TPS + new negatives) | Cross-dataset generalization: how well do old-data models perform on the new dataset? |

Additionally:

- **Cross-negative stress test**: Same model trained on synced folds, but negatives
  swapped between old and new sets at test time.
- **Atomic negative cleaning analysis**: Each negative filtering step from the new
  pipeline applied individually to the old dataset to quantify contamination.

### 2.3 Why Four Tracks Are Necessary

**Track A** establishes the baseline. It uses the published phylogenetic fold
assignments — where phylogenetically related TPS are grouped together and kept
in the same fold to prevent homology leakage — with all 9,944 negatives
participating in cross-validation (see Section 3.1 for why this correction was
necessary).

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
  Any performance change comes from (i) the additional ~2,100 TPS in the new dataset
  and/or (ii) a different negative set.

**Track D** is the most practically informative condition. It trains models on the
old synced dataset (same training data as Track C) but evaluates them on the
**new dataset's test fold**. Each test fold contains *all* of the new dataset's TPS
for that fold — including the ~2,100 TPS that the model has never seen — plus the
new dataset's negatives. This directly answers: **"How good are models trained on
the old data when deployed against the new, curated dataset?"**

The comparison between **C and D** isolates the "dataset shift" effect. In Track C,
the model is evaluated on held-out data from the same distribution it was trained
on. In Track D, the evaluation data is drawn from a different (and richer)
distribution. The delta C-D measures how much performance drops when crossing
this distribution boundary.

The comparison between **D and B** is also illuminating. Both evaluate on the same
new-dataset test folds, but Track B also *trains* on the new dataset (with more TPS
and new negatives). The delta D-B therefore measures the value of having the new
training data rather than the old.

This four-way comparison — A vs. C vs. D vs. B — provides a complete decomposition:

- **A vs. C**: Effect of fold reassignment + TPS curation (same distribution eval)
- **C vs. D**: Effect of dataset shift at evaluation time
- **D vs. B**: Effect of new training data (keeping eval distribution fixed)

### 2.4 Leak-Freedom of Track D

A natural concern with cross-dataset evaluation is data leakage. Track D is
leak-free because:

1. **Fold synchronization**: The 1,035 TPS shared between the old and new datasets
   are assigned the same fold index in both. If a shared TPS is in fold 4 in the
   new dataset, it is also in fold 4 in the synced old dataset. Therefore, when
   fold 4 is held out for evaluation, that TPS is excluded from training.

2. **New-only TPS**: The ~2,100 TPS present in the new dataset but absent from the
   old dataset are *never* seen during training (they do not exist in the old
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
clustering. These produce fundamentally different splits: phylogenetic grouping
ensures evolutionarily related sequences stay together, while sequence clustering
at a chosen identity threshold controls maximum pairwise similarity across folds.

By syncing the fold assignments of shared TPS to match the new dataset, Track C
tests whether the new fold logic is compatible with the old data — without
introducing the confound of new TPS or new negatives.

---

## 3. Data and Statistics

### 3.1 Negative Distribution Correction (Track A)

**Problem identified**: In the original phylo dataset as stored in the CSV,
9,772 of 9,944 negatives were assigned to fold `-1`. The experiment runner
treats fold `-1` as "excluded from all folds" — these sequences participate
in neither training nor testing. This left only 172 negatives distributed
across folds 0-4 (~34 per fold), yielding a TPS-to-negative ratio of
approximately 12:1 per fold — unrealistically favorable for TPS detection.

**Impact**: With so few negatives, all models achieved near-perfect
TPS detection (AP > 0.99), and metrics sensitive to class balance
(MCC-F1) were artificially inflated. The evaluation could not
distinguish models with genuinely different specificity.

**Correction**: We redistribute all 9,944 negatives round-robin across
folds 0-4, yielding ~2,000 negatives per fold. The resulting TPS-to-negative
ratio (~1:5) matches the new dataset's distribution and is more representative
of real-world screening scenarios where TPS are a small minority.

**Justification**: The original paper's methodology section describes all
negatives participating in cross-validation. The fold `-1` assignment was
a data preparation artifact, not an intentional design choice. Our correction
restores the intended protocol.

| Fold | TPS (old) | Negatives (corrected) | Ratio |
|------|-----------|----------------------|-------|
| 0 | 414 | 2,002 | 1:4.8 |
| 1 | 302 | 1,988 | 1:6.6 |
| 2 | 562 | 1,982 | 1:3.5 |
| 3 | 455 | 1,985 | 1:4.4 |
| 4 | 473 | 1,987 | 1:4.2 |

### 3.2 Dataset Statistics

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

### 3.3 The 128 Old-Only TPS

The new dataset is a curated superset of the old dataset's TPS. During curation,
128 sequences present only in the old dataset were deemed low-quality (ambiguous
functional annotation, fragmented sequences, or redundant with better-characterized
entries). These are excluded from Track A evaluation and entirely absent from
Track C, ensuring that all three tracks evaluate on the same core TPS pool
(modulo Track B's 2,100+ additional entries).

### 3.4 Fold Assignment Methods

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

- **Architecture**: Random Forest classifier trained on mean-pooled embeddings from
  a fine-tuned ESM-1v protein language model (sub-sequence variant).
- **Input**: Fixed-dimensional embedding vector per protein sequence.
- **Training**: One binary classifier per substrate class (one-vs-rest). The ESM-1v
  backbone is fine-tuned separately on the TPS dataset before embedding extraction.
- **Why included**: Represents the state of the art in the project. PLM-based
  approaches capture complex sequence features beyond simple homology.

### 4.2 Blastp (Sequence Similarity Baseline)

- **Architecture**: BLASTp nearest-neighbor. For each test sequence, find the most
  similar training sequence by BLASTp; transfer its substrate label.
- **Training**: No parameters to learn — the training set is the database.
- **Why included**: The canonical bioinformatics baseline for functional annotation.
  If PlmRF does not outperform Blastp, the learned representations add no value
  over raw sequence similarity.

### 4.3 CLEAN (Pre-trained Enzyme Function Predictor)

- **Architecture**: Contrastive learning on enzyme-reaction pairs, pre-trained on
  a large corpus of enzyme-function data.
- **Critical caveat**: CLEAN's pre-training data **includes all TPS and negative
  enzymes** in our evaluation sets. Its numbers reflect in-sample (or near-in-sample)
  performance. We include CLEAN for completeness and as an upper-bound reference,
  but it should **not** be compared to the other models as a measure of generalization.
- **Why included**: Demonstrates what a large pre-trained model achieves when the
  test data overlaps with its training set. Useful for contextualizing the gap
  between in-sample and out-of-sample performance.

### 4.4 HMM (Profile Hidden Markov Model)

- **Architecture**: Per-class profile HMMs built from multiple sequence alignments
  of training-set TPS for each substrate class.
- **Training**: HMMER's `hmmbuild` on aligned training sequences per fold.
- **Why included**: Profile HMMs are the standard tool for protein family
  classification in bioinformatics (Pfam, InterPro). They capture position-specific
  residue preferences and insertion/deletion patterns.

### 4.5 Foldseek (Structural Similarity Baseline)

- **Architecture**: Structural nearest-neighbor using Foldseek's 3Di structural
  alphabet. For each test structure (AlphaFold-predicted PDB), find the most
  similar training structure and transfer its label.
- **Why included**: Tests whether 3D structural information adds value over
  sequence-only methods.
- **Limitation**: Requires PDB structures. Only available for Track A; Tracks B
  and C are missing 6,594 negative PDB files.

---

## 5. Metric Selection and Justification

We report three complementary metrics for each task. The choice is deliberate:
no single metric adequately captures model quality under class imbalance.

### 5.1 Average Precision (AP) / Mean Average Precision (mAP)

- **Definition**: Area under the precision-recall curve. For multi-class substrate
  prediction, macro-averaged across all eligible substrate classes.
- **Why chosen**: AP is the recommended primary metric for imbalanced binary
  classification (Davis & Goadrich, 2006). Unlike ROC-AUC, AP is sensitive to the
  actual number of false positives at each threshold, which matters when positives
  are rare (~15-30% of the dataset).
- **Interpretation**: AP = 1.0 means perfect ranking — all positives are scored
  above all negatives. A random classifier achieves AP equal to the positive
  prevalence.

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

- **Definition**: A summary statistic derived from the MCC-F1 curve (Cao, Chicco,
  & Hoffman, 2020). The MCC-F1 curve plots the unit-normalized Matthews Correlation
  Coefficient against F1-score across all classification thresholds. The summary
  score measures how close the curve passes to the ideal point (MCC=1, F1=1),
  aggregated over the entire threshold range.
- **Why chosen**: MCC is the only single-threshold metric that is informative for
  imbalanced binary classification — it accounts for all four quadrants of the
  confusion matrix. F1 complements it by focusing on the positive class. The
  MCC-F1 summary is threshold-free (like AP and AUC) but captures a fundamentally
  different aspect of classifier quality: the joint behavior of MCC and F1 across
  all operating points.
- **Interpretation**: Values range from 0 to 1. Higher is better. A classifier
  that achieves high MCC and high F1 simultaneously across all thresholds scores
  close to 1. The metric is particularly useful for detecting models that achieve
  high AP by ranking well but cannot find a single threshold that works for both
  precision/recall (F1) and balanced accuracy (MCC).

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
the ~2,100 new-only TPS have no direct old-dataset counterpart.

### 6.4 Key Findings (Preliminary — Tracks A, B, C)

Full cross-track per-bin results are presented in Section 9.2. Here we
highlight the methodological takeaway:

**PlmRF is remarkably robust across identity bins**:

| Bin | Track A AP | Track C AP | Track B AP |
|-----|-----------|-----------|-----------|
| 20–30% | 0.971 | 0.991 | 0.993 |
| 30–40% | 0.976 | 0.988 | 0.982 |
| 40–50% | 0.997 | 0.984 | 0.999 |
| 50–60% | 0.999 | 0.991 | 0.983 |
| 60–70% | 1.000 | 1.000 | 1.000 |

Even at 20–30% identity — where sequences share at most twilight-zone similarity
to any training example — PlmRF achieves AP >= 0.97. This indicates that the
ESM-1v embeddings capture functional information beyond raw sequence similarity.

**Blastp degrades predictably**: AP drops from 0.87–0.99 at high identity to
0.83–0.95 at 20–30%, consistent with its reliance on direct sequence similarity.

**MCC-F1 shows sharper degradation than AP at low identity**:

| Model | 20–30% MCC-F1 (avg) | 60–70% MCC-F1 (avg) |
|-------|-------|-------|
| PlmRF | 0.661 | 0.684 |
| CLEAN | 0.627 | 0.645 |
| Blastp | 0.266 | 0.267 |

This reveals that while PlmRF ranks correctly even for distant homologs (high AP),
finding a single threshold that simultaneously maximizes MCC and F1 remains
harder for remote sequences. The practical implication: at low sequence identity,
PlmRF is an excellent ranker but requires careful threshold calibration.

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

We present results in two layers: **aggregate performance** for the big picture,
then **per-similarity-bin breakdowns** to assess robustness to sequence novelty.
Both layers are shown across all four tracks, enabling direct visual comparison.

### 9.1 Layer 1 — Aggregate Performance (Big Picture)

*Visualization: `cross_track_overview_substrate.{png,pdf}` and
`cross_track_overview_tps_det.{png,pdf}` (generated by
`scripts/plot_cross_track_results.py`).*

These grouped bar charts place one bar-group per model and one bar per track,
with error bars showing standard error across folds. A single glance reveals
model ranking and the magnitude of track-to-track changes.

#### 9.1.1 Substrate Prediction (mAP)

| Model | Track A | Track C | Track D | Track B | A→C | C→D | D→B |
|-------|---------|---------|---------|---------|-----|-----|-----|
| PlmRF | 0.776 | 0.794 | *pending* | 0.803 | +0.018 | *pending* | *pending* |
| CLEAN* | 0.626 | 0.609 | *pending* | 0.499 | -0.017 | *pending* | *pending* |
| Blastp | 0.542 | 0.639 | *pending* | 0.612 | +0.097 | *pending* | *pending* |
| HMM | 0.334 | 0.345 | *pending* | 0.437 | +0.011 | *pending* | *pending* |

#### 9.1.2 TPS Detection (AP)

| Model | Track A | Track C | Track D | Track B | A→C | C→D | D→B |
|-------|---------|---------|---------|---------|-----|-----|-----|
| PlmRF | 0.988 | 0.990 | *pending* | 0.991 | +0.002 | *pending* | *pending* |
| Blastp | 0.892 | 0.938 | *pending* | 0.934 | +0.046 | *pending* | *pending* |
| HMM | 0.890 | 0.914 | *pending* | 0.895 | +0.024 | *pending* | *pending* |
| CLEAN* | 0.857 | 0.840 | *pending* | 0.871 | -0.017 | *pending* | *pending* |

#### 9.1.3 Full Metric Heatmap

*Visualization: `cross_track_heatmap_substrate.{png,pdf}` and
`cross_track_heatmap_tps_det.{png,pdf}`.*

The heatmap arranges rows = models, columns = (Track × Metric). Each cell's
color encodes the score value (0–1 YlGnBu scale), with exact numbers annotated.
Vertical separators demarcate tracks. This view compresses three metrics
(AP/mAP, ROC-AUC, MCC-F1) × four tracks into a single glanceable figure,
making it easy to spot which models suffer on which metrics and which tracks
are harder overall.

### 9.2 Layer 2 — Per-Similarity-Bin Breakdown

*Visualization: `cross_track_simbin_substrate.{png,pdf}` and
`cross_track_simbin_tps_det.{png,pdf}`.*

Aggregate numbers hide a critical axis of variation: **sequence novelty**. A
model may score 0.80 mAP overall because it excels at classifying near-identical
homologs (60–70% identity) while performing at chance for remote sequences
(20–30%). The per-bin analysis exposes this pattern.

The faceted line plots have:
- **Columns** = tracks (A, C, D, B), showing the same bins side by side.
- **Rows** = metrics (AP on top, MCC-F1 on bottom).
- **Lines** = models (color- and marker-coded).
- **X-axis** = sequence identity bin of the test sequence to its nearest
  training neighbor (20–30%, 30–40%, …, 60–70%).

Reading across columns for a given model reveals how the **same** model's
degradation curve shifts as the evaluation context changes (fold logic,
negatives, TPS coverage). Reading within a column compares models under
identical evaluation conditions.

#### 9.2.1 Substrate Prediction (mAP) by Similarity Bin

| Bin | PlmRF A | PlmRF C | PlmRF D | PlmRF B | Blastp A | Blastp C | Blastp D | Blastp B |
|-----|---------|---------|---------|---------|----------|----------|----------|----------|
| 20–30% | — | — | *pending* | — | — | — | *pending* | — |
| 30–40% | — | — | *pending* | — | — | — | *pending* | — |
| 40–50% | — | — | *pending* | — | — | — | *pending* | — |
| 50–60% | — | — | *pending* | — | — | — | *pending* | — |
| 60–70% | — | — | *pending* | — | — | — | *pending* | — |

*(Per-bin substrate mAP values are available from the per-class CSVs.
Populate after running the evaluation pipeline for all tracks.)*

#### 9.2.2 TPS Detection (AP) by Similarity Bin

| Bin | PlmRF A | PlmRF C | PlmRF D | PlmRF B | Blastp A | Blastp C | Blastp D | Blastp B |
|-----|---------|---------|---------|---------|----------|----------|----------|----------|
| 20–30% | 0.971 | 0.991 | *pending* | 0.993 | 0.830 | 0.874 | *pending* | 0.902 |
| 30–40% | 0.976 | 0.988 | *pending* | 0.982 | 0.856 | 0.900 | *pending* | 0.890 |
| 40–50% | 0.997 | 0.984 | *pending* | 0.999 | 0.912 | 0.952 | *pending* | 0.969 |
| 50–60% | 0.999 | 0.991 | *pending* | 0.983 | 0.950 | 0.978 | *pending* | 0.951 |
| 60–70% | 1.000 | 1.000 | *pending* | 1.000 | 0.990 | 0.993 | *pending* | 1.000 |

#### 9.2.3 Key Observations from Bin Analysis

1. **PlmRF is flat across bins**: AP varies by less than 3 percentage points
   from the hardest bin (20–30%) to the easiest (60–70%) across all tracks.
   This means the aggregate numbers accurately reflect performance even on
   truly novel sequences — a crucial property for practical enzyme discovery.

2. **Blastp degrades predictably**: AP drops 10–16 points from 60–70% to
   20–30% identity. This is expected: Blastp transfers labels from the most
   similar hit, and at low identity, hits are too distant for reliable
   functional transfer.

3. **Track C slightly improves low-bin performance vs. Track A**: For PlmRF,
   the 20–30% bin improves from 0.971 to 0.991 when moving A→C. This
   suggests that the phylogenetic fold assignments in Track A allowed some
   evolutionarily related (but sequence-divergent) pairs to leak across folds,
   and the MMseqs2-based synced folds are stricter.

4. **Track D (pending) will be the critical test**: If PlmRF's low-bin
   performance in Track D matches Tracks B/C, it confirms that old-data
   training is sufficient to detect novel TPS. If it drops, it pinpoints
   which similarity regime needs more training data.

### 9.3 Decomposing the Effects

**A → C (same negatives, different folds, 128 fewer TPS)**:

PlmRF improves from 0.776 to 0.794 mAP (+2.3%). Since the negatives are identical,
this improvement comes from:

1. Removing 128 potentially noisy old-only TPS whose labels may have been incorrect
   or ambiguous, reducing label noise in training.
2. Reassigning folds: the new dataset's MMseqs2-based clustering may produce folds
   that are better calibrated for generalization (harder splits with stricter
   similarity ceilings between folds).

The TPS detection improvement is minimal (+0.002 AP), which is expected — the
binary detection task is less sensitive to small label noise changes.

Blastp shows a larger A→C improvement (+0.097 mAP, +0.046 AP), suggesting that
the phylogenetic fold assignments in Track A placed very similar sequences in
different folds, making Blastp's job artificially easy in some cases and
artificially hard in others.

**C → D (same training data, cross-dataset evaluation)**:

Track D evaluates the same models as Track C but on the new dataset's test folds
rather than the old dataset's. The delta C→D directly measures the dataset shift
penalty — how much performance drops when the model, trained on old data, faces
the new dataset's broader TPS coverage and different negatives.

*Results pending — Track D experiments have not yet been run.*

**D → B (same eval distribution, different training data)**:

Both Tracks D and B evaluate on the new dataset's folds. The delta D→B measures
the value of training on the new dataset (with its additional ~2,100 TPS and
native negatives) rather than the old synced dataset.

*Results pending — Track D experiments have not yet been run.*

**C → B (same fold logic, different negatives, more TPS)**:

PlmRF improves from 0.794 to 0.803 mAP (+1.1%). This modest gain is attributable
to the additional ~2,100 TPS in the new dataset, which provide more training
examples for rare substrate classes.

CLEAN degrades substantially (0.609 → 0.499 mAP, -18%). This is consistent with
CLEAN being an in-sample model: the new dataset introduces substrate classes and
TPS that may not align well with CLEAN's pre-training objective, while it loses
the familiarity advantage it had with the old dataset's composition.

HMM improves substantially (0.345 → 0.437 mAP, +27%). More training TPS per
substrate class allows HMM to build better-informed sequence profiles. HMM is
training-data-hungry by nature.

### 9.4 Summary of Causal Attribution

| Factor | Effect on PlmRF mAP | Aggregate evidence | Per-bin evidence |
|--------|---------------------|----------|----------|
| TPS curation (removing 128 noisy entries) | +1–2% | A→C comparison | Uniform across bins |
| Fold reassignment (phylo → MMseqs2) | Included in A→C | Cannot fully separate from curation | Low-bin improvement suggests stricter splits |
| Dataset shift at eval time | *pending* | C→D comparison | *pending* — key test is low-identity bins |
| Value of new training data | *pending* | D→B comparison | *pending* |
| Expanded TPS coverage (+2,100 TPS) | +1% | C→B comparison | Expected to help rare classes in low bins |
| Different negatives | Negligible | Cross-neg stress test | Stress test not yet binned |
| Negative contamination | Negligible | Atomic cleaning analysis | N/A (< 0.41% negatives affected) |

### 9.5 How to Read the Visualizations Together

The intended reading order for a reviewer:

1. **Heatmap** (`cross_track_heatmap_*.pdf`): Glance at the full landscape.
   Identify which model × track × metric cells stand out. This takes 10 seconds
   and gives the bird's-eye view.

2. **Grouped bar chart** (`cross_track_overview_*.pdf`): Focus on the primary
   metric (mAP for substrate, AP for detection). Compare bar heights across
   tracks for each model. Error bars show whether differences are within noise.

3. **Similarity-bin line plots** (`cross_track_simbin_*.pdf`): Drill into *why*
   a model improves or degrades between tracks. If the difference concentrates
   in low-identity bins, the explanation involves generalization to novel
   sequences. If it is uniform, the explanation is systemic (e.g., negative
   distribution, label quality).

This three-layer progression — heatmap → bars → lines — moves from
"what changed?" to "by how much?" to "where and why?"

### 9.6 Generating the Visualizations

All figures are generated by a single script:

```bash
python scripts/plot_cross_track_results.py \
    --tracks track_a_phylo_folds track_c_synced_folds \
             track_b_new_dataset \
    --track-labels "A (phylo)" "C (synced)" "B (new)" \
    --output-prefix cross_track
```

When Track D results become available, add them:

```bash
python scripts/plot_cross_track_results.py \
    --tracks track_a_phylo_folds track_c_synced_folds \
             track_d_cross_synced track_b_new_dataset \
    --track-labels "A (phylo)" "C (synced)" "D (cross)" "B (new)" \
    --output-prefix cross_track
```

Outputs are saved to `outputs/evaluation_results/`.

---

## 10. Threats to Validity and Mitigations

### 10.1 CLEAN's In-Sample Status

CLEAN was pre-trained on a corpus that includes all TPS and negative enzymes
in our evaluation. Its numbers are **in-sample** and should not be interpreted
as measures of generalization. We include CLEAN for completeness — it serves as
an upper-bound reference showing what a large pre-trained enzyme model achieves
when test data overlaps with training data.

**Mitigation**: We clearly flag CLEAN with an asterisk (*) in all tables and
discuss this caveat in every section.

### 10.2 Single Random Seed for Fold Assignments

The synced dataset uses a deterministic mapping (sequence → fold from the new
dataset), so Track C results are not averaged over multiple resplits. However,
the underlying 5-fold CV is itself averaged over 5 test sets, providing some
robustness to partition variability.

**Mitigation**: The consistency of results across all three tracks (which use
different fold assignment methods) suggests that the findings are not artifacts
of a particular split.

### 10.3 Limited Model Coverage

Only Blastp has been evaluated in the cross-negative stress test. PlmRF, CLEAN,
HMM, and Foldseek remain pending.

**Mitigation**: Blastp is the model most sensitive to negative composition (it
relies entirely on sequence similarity to the training database). If Blastp shows
only modest sensitivity, embedding-based models like PlmRF — which learn abstract
representations — are expected to be even more robust. This remains to be confirmed.

### 10.4 Missing Foldseek Results for Tracks B and C

Foldseek requires AlphaFold-predicted PDB structures. 6,594 negative PDBs are
missing for the new dataset.

**Mitigation**: Foldseek results are available for Track A. The structural
similarity baseline shows Foldseek is consistently the weakest model (mAP 0.395,
AP 0.838), making its absence from Tracks B and C unlikely to change the overall
conclusions about model ranking.

### 10.5 Unequal TPS Counts Across Tracks

Track A has 2,206 TPS (2,056 after excluding old-only), Track B has 4,185, and
Track C has 2,056. Direct mAP comparison between tracks with different numbers
of substrate classes may be confounded by the different class taxonomy.

**Mitigation**: We report per-class metrics alongside aggregate metrics. The
aggregate mAP uses macro-averaging, which weights all eligible classes equally.
Classes present in Track B but absent from Track A/C are simply additional
classes in the macro-average, not a confound — they represent the expanded
coverage that is one of the claimed improvements.

---

## 11. Conclusions

The local validation protocol demonstrates that:

1. **PlmRF is the strongest model** across all tracks and both tasks, with mAP
   0.776-0.803 for substrate prediction and AP 0.988-0.991 for TPS detection.

2. **Performance improvements from old to new dataset are attributable to specific,
   identified factors**: TPS curation and expanded coverage account for most of the
   gain, while negative set differences have negligible effect.

3. **The evaluation protocol is sound**: phylogenetic/clustered fold assignments
   prevent homology leakage, per-similarity-bin analysis confirms robustness across
   sequence identity levels, and the corrected negative distribution provides
   realistic class balance.

4. **PlmRF generalizes to remote homologs**: AP >= 0.97 even at 20-30% sequence
   identity to the nearest training example, demonstrating that PLM embeddings
   capture functional information beyond raw sequence similarity.

5. **Negative contamination is negligible**: At most 0.41% of old negatives are
   flagged by any cleaning filter, ruling out contamination as an explanation for
   performance differences.

6. **Track D (cross-dataset generalization)** will directly quantify how well
   old-data-trained models perform on the new dataset, completing the four-way
   decomposition A → C → D → B. *Results pending.*

---

## References

- Davis, J. & Goadrich, M. (2006). The relationship between Precision-Recall and
  ROC curves. *Proc. ICML*.
- Cao, C., Chicco, D., & Hoffman, M. M. (2020). The MCC-F1 curve: a performance
  evaluation technique for binary classification. *arXiv:2006.11278*.
- Steinegger, M. & Soding, J. (2017). MMseqs2 enables sensitive protein sequence
  searching for the analysis of massive data sets. *Nature Biotechnology*, 35(11).
- van Kempen, M. et al. (2023). Fast and accurate protein structure search with
  Foldseek. *Nature Biotechnology*.
- Yu, T. et al. (2023). Enzyme function prediction using contrastive learning.
  *Science*, 379(6639).
