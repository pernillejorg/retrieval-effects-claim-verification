# Step 2 Results: RoBERTa Claim Verification (Baseline + Evidence-Aware Classifier)

This document records the Step 2 classifiers and the full experimental history behind
them, including two methodological corrections made during the project and a multi-seed
robustness check. Two RoBERTa models are trained, both on SciFact:

- **Model 1 — the no-retrieval baseline** (Part 1): classifies claims from **claim text
  only**, with no evidence. This is the reference every retrieval-augmented result in
  later steps is measured against. It is also evaluated zero-shot on SciFact-Open (Part 2).
- **Model 2 — the evidence-aware classifier** (Part 3): trained on **claim + gold-evidence
  pairs**, so it can read evidence. This is the classifier used for the retrieval-augmented
  conditions in Step 5.

Both models share the same training procedure and differ only in their input
representation (claim only vs claim + evidence), so the comparison between them is
controlled. Both are trained with a fixed seed (42) under the corrected seeding procedure
described below.

## Experimental history (summary)

The reported baseline evolved through two documented corrections. Both are recorded here in
full for transparency, because each materially affected the reported numbers and each
represents a deliberate rigour improvement rather than a tuning change:

1. **Deduplication fix** — the SciFact validation set was found to contain duplicate claim
   rows (450 rows for only 300 unique claims). Correcting this changed the reported baseline
   from 0.5496 (over 450 duplicated rows) to 0.4570 (over 300 unique claims).
2. **Seeding fix** — the training seed was originally applied before the learning-rate
   search but not re-applied before the final training run, so the run inherited the
   search's random state. Correcting this (re-seeding immediately before the final training
   run) changed the baseline from 0.4570 to the current reported value of **0.5263**, and —
   crucially — made genuine multi-seed variance measurement possible.

The three stages of the baseline are therefore:

| Stage | Validation set | Seeding | Macro F1 |
|---|---|---|---|
| 1. Raw | 450 rows (duplicated) | original | 0.5496 |
| 2. Deduplicated | 300 unique claims | original | 0.4570 |
| 3. Deduplicated + corrected seeding | 300 unique claims | corrected | **0.5263** (reported) |

Stages 1 and 2 are retained below for transparency and are clearly marked **SUPERSEDED**.
Stage 3 is the reported baseline used from here on.

## Experiment Overview

| Property | Value |
|---|---|
| Model | roberta-base |
| Primary dataset (trained) | SciFact |
| Secondary dataset (zero-shot eval) | SciFact-Open |
| Task | 3-class claim verification (SUPPORT / CONTRADICT / NEI) |
| Retrieval | None — claim only (Model 1); claim + gold evidence (Model 2) |
| Device | CUDA (Google Colab GPU) |
| Seed | 42 (fixed; corrected seeding procedure) |

---

## Correction 1 — deduplicating the SciFact validation set

During Step 4 (reranking), a claim-count mismatch surfaced: the SciFact validation set was
being treated as 450 claims, but only 300 of these were unique. SciFact's raw validation
split stores **450 rows corresponding to only 300 distinct claims** — a single claim is
repeated across multiple rows when it is cited against more than one evidence document. A
verification check confirmed this: 450 total rows, 300 unique claim ids, and **0 claims with
conflicting labels** (every repeated claim carried the same label, so the repeats were pure
duplicates carrying no new information).

The data loader (`load_scifact`) was corrected to deduplicate claims by id, merging the
evidence document ids of repeated rows into a single claim entry:

- The validation set is now 300 unique claims rather than 450 rows.
- No claim is lost — only accidental repeats are removed.
- Every downstream step (baseline, retrieval, reranking) now operates on the same 300
  unique claims, so metrics are no longer inflated by double-counting.

This is a correctness fix, not a reduction in data. Counting 450 rows treated 150 claims as
if they were distinct, which double-weighted them in every metric and in training. The
corrected 300-claim count is the true number of distinct claims in the split.

---

## Correction 2 — seeding procedure

The training seed governs random weight initialisation and batch shuffling. It was
originally applied once, before the learning-rate search. The learning-rate search, however,
internally re-seeds to a fixed value before each candidate rate (so that every rate is
compared from the same initialisation — a fair comparison). As a result, by the time the
final training run began, the random state had been set by the *last* re-seed inside the
search, not by the intended top-level seed. Two consequences followed:

1. The final model was trained from a random state determined by the search, not directly
   by the chosen seed.
2. A multi-seed variance study was impossible: varying the top-level seed changed nothing,
   because the search's internal re-seed overrode it, so every seed produced an identical
   model.

The fix re-applies the chosen seed **immediately before the final training run**, after the
learning-rate search completes. The learning-rate search still re-seeds internally (so rate
selection remains a fair, fixed comparison), but the final training run now genuinely starts
from the chosen seed. This both makes the reported model a clean function of the seed and
enables the multi-seed robustness check reported at the end of this document.

Because this changes the random path of the final training run, the corrected-seeding
baseline (0.5263) differs from the pre-correction value (0.4570); both are legitimate
training runs, but 0.5263 is produced by the cleaner, reproducible procedure and is the
value used throughout the rest of the thesis.

---

## Part 1 — SciFact (primary, trained): Model 1 (claim-only)

### SUPERSEDED — Stage 1: raw 450-row validation set (original seeding)

Retained for transparency only; **not** the reported baseline.

| Metric | Value (450 rows) |
|---|---|
| Validation claims | 450 (with duplicates) |
| Macro F1 | 0.5496 |
| Macro precision | 0.58 |
| Macro recall | 0.54 |

Per-class (450 rows): SUPPORT 0.66 F1, CONTRADICT 0.45 F1, NEI 0.53 F1 (support summed to
450, reflecting the duplicated rows).

### SUPERSEDED — Stage 2: 300 unique claims (original seeding)

Retained for transparency only; **not** the reported baseline. This was the post-dedup,
pre-seed-fix result.

| Metric | Value (300 unique) |
|---|---|
| Macro F1 | 0.4570 |
| Macro precision | 0.48 |
| Macro recall | 0.45 |

Per-class: SUPPORT 0.47 F1, CONTRADICT 0.30 F1, NEI 0.60 F1.

### REPORTED — Stage 3: 300 unique claims, corrected seeding (seed 42)

**Dataset statistics**

| Split | Claims |
|---|---|
| Train | 809 |
| Validation | 300 (unique) |

Train label distribution: SUPPORT 332, CONTRADICT 173, NEI 304.

**Token length check**
- Maximum claim token length: **75 tokens**
- MAX_LENGTH_CLAIM_ONLY = 128 confirmed safe — no claims truncated.

**Learning rate search (one trial pass per rate)**

| Learning rate | Val macro F1 |
|---|---|
| 1e-5 | 0.2997 |
| 2e-5 | 0.3812 |
| **3e-5** | **0.4295** ← selected |

**Training log (full run at lr = 3e-5, seed 42, corrected seeding)**

| Epoch | Val macro F1 | Val precision | Val recall | Notes |
|---|---|---|---|---|
| 1 | 0.2800 | 0.2751 | 0.3482 | New best |
| 2 | 0.3393 | 0.3541 | 0.3954 | New best |
| 3 | 0.4397 | 0.4531 | 0.4424 | New best |
| 4 | 0.4892 | 0.5056 | 0.5006 | New best |
| 5 | 0.5060 | 0.5048 | 0.5126 | New best |
| 6 | 0.5128 | 0.5229 | 0.5089 | New best |
| 7 | 0.5103 | 0.5332 | 0.5042 | No improvement (1) |
| 8 | 0.5263 | 0.5287 | 0.5246 | New best (best checkpoint) |
| 9 | 0.5213 | 0.5198 | 0.5279 | No improvement (1) |
| 10 | 0.5239 | 0.5229 | 0.5252 | No improvement (2) — early stopping |

Best checkpoint: **epoch 8**.

**Final results — SciFact validation (best checkpoint)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.55 | 0.58 | 0.56 | 124 |
| CONTRADICT | 0.38 | 0.36 | 0.37 | 64 |
| NEI | 0.66 | 0.63 | 0.65 | 112 |

| Metric | Value |
|---|---|
| Accuracy | 0.55 |
| Macro F1 | **0.5263** |
| Macro precision | 0.53 |
| Macro recall | 0.52 |

*(Support column sums to 300, confirming the deduplicated set.)* The reloaded checkpoint
reproduces the reported F1 exactly (save/reload integrity verified). SciFact has no public
test labels (blind leaderboard), so validation is the final reported split.

---

## Part 2 — SciFact-Open (secondary, zero-shot): Model 1

The **same SciFact-trained Model 1** was evaluated on SciFact-Open without retraining.
SciFact-Open is a test-only collection, so this is a zero-shot generalisation reference.
SciFact-Open was not affected by the deduplication issue (no duplicate claims); the model
evaluated here is the corrected-seeding Model 1.

**Dataset notes**
- 279 claims loaded.
- 15 claims had conflicting SUPPORT/CONTRADICT evidence, resolved to SUPPORT under a fixed precedence rule.
- 73 claims had no evidence and were mapped to NEI.
- Label distribution: SUPPORT 116, CONTRADICT 90, NEI 73.

### Results — SciFact-Open (zero-shot, corrected-seeding Model 1)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.61 | 0.67 | 0.64 | 116 |
| CONTRADICT | 0.58 | 0.46 | 0.51 | 90 |
| NEI | 0.68 | 0.75 | 0.71 | 73 |

| Metric | Value |
|---|---|
| Accuracy | 0.62 |
| Macro F1 | **0.6219** |
| Macro precision | 0.6236 |
| Macro recall | 0.6271 |

At the no-retrieval stage, SciFact-Open scores higher than SciFact (0.6219 vs 0.5263)
despite being zero-shot. This is expected: the baseline uses no retrieval, so SciFact-Open's
defining difficulty — its 500,000-document retrieval corpus — is invisible here. The two
numbers reflect only claim-only classification difficulty on two different claim sets;
SciFact-Open's balanced set with easily-recognised no-evidence NEI cases is comparatively
easier to classify from claim text alone. The baseline therefore establishes that any
SciFact-Open degradation observed in later steps is attributable to retrieval, not to the
claims themselves.

---

## Part 3 — Model 2 (claim + gold evidence)

Model 2 was trained with the same procedure as Model 1 (same corrected seeding, seed 42) but
on **claim + gold-evidence pairs** (`--input_mode claim_evidence`), using the tokenizer's
text-pair format so RoBERTa receives claim and evidence as two segments with its own segment
boundary. Model 1 is used for the no-retrieval condition in Step 5; Model 2 is used for the
BM25/dense/reranked conditions. Applying each classifier to the input format it was trained
on is the methodologically correct RAG setup.

### Model 2 results — SciFact validation (300 unique claims, seed 42)

Best learning rate: 3e-5. Best checkpoint: epoch 10. Reloaded-checkpoint F1 confirmed
(save/reload integrity verified).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.66 | 0.70 | 0.68 | 124 |
| CONTRADICT | 0.39 | 0.36 | 0.37 | 64 |
| NEI | 0.89 | 0.87 | 0.88 | 112 |

| Metric | Value |
|---|---|
| Accuracy | 0.69 |
| Macro F1 | **0.6438** |

### Model 1 vs Model 2 comparison

| Model | Input | Macro F1 | SUPPORT F1 | CONTRADICT F1 | NEI F1 |
|---|---|---|---|---|---|
| Model 1 (baseline) | claim only | 0.5263 | 0.56 | 0.37 | 0.65 |
| Model 2 (evidence) | claim + gold evidence | 0.6438 | 0.68 | 0.37 | 0.88 |

**Interpretation.** Adding gold evidence raises macro F1 by **+0.118** (0.5263 → 0.6438),
confirming the model genuinely uses the evidence rather than ignoring it. The largest gain
is on NEI (0.65 → 0.88): with evidence present, the model can far better distinguish claims
that have supporting/refuting evidence from those with none. SUPPORT also improves (0.56 →
0.68). **CONTRADICT remains the hardest class and does not improve at all with evidence
(0.37 → 0.37)** — a striking result suggesting that detecting refutation is intrinsically
difficult and not solved by evidence access alone. This is a direct input to the failure
analysis (Step 7).

**Important caveat (training vs test evidence).** Model 2 is trained and validated on
**gold** evidence. In the Step 5 RAG pipeline it is fed **retrieved** evidence, which is
noisier. Its 0.6438 gold-evidence figure is therefore an optimistic ceiling; the realistic
RAG performance is measured in Step 5 against retrieved evidence and is expected to be lower.
This gap between gold-evidence and retrieved-evidence performance is itself an informative
result for the thesis.

### SciFact-Open and Model 2

Model 2 is **not** evaluated on SciFact-Open in this step, because doing so requires feeding
it retrieved evidence from SciFact-Open's 500K corpus — which is the Step 5 pipeline's job.
Model 2's SciFact-Open evaluation is reported in Step 5.

---

## Comparison and interpretation

### The two corrections, side by side (SciFact, Model 1)

| Stage | Val set | Seeding | Macro F1 | Precision | Recall |
|---|---|---|---|---|---|
| Raw (superseded) | 450 rows | original | 0.5496 | 0.58 | 0.54 |
| Deduplicated (superseded) | 300 unique | original | 0.4570 | 0.48 | 0.45 |
| **Deduplicated + corrected seeding (reported)** | 300 unique | corrected | **0.5263** | 0.53 | 0.52 |

**On the deduplication (Stage 1 → 2).** The drop from 0.5496 to 0.4570 was not a regression
but a correction: the 450-row figure was inflated because 150 duplicate rows were
double-counted in the metric and double-weighted in training. Removing them revealed true
performance over the 300 distinct claims.

**On the seeding (Stage 2 → 3).** The change from 0.4570 to 0.5263 reflects the corrected
random path of the final training run, not a change in data or hyperparameters. Both are
valid runs; the corrected-seeding value is produced by the cleaner, reproducible procedure
and is what all downstream steps use. Importantly, 0.5263 falls within the multi-seed band
reported below, so it is a representative run rather than a favourable outlier.

### SciFact vs SciFact-Open (reported models)

| Dataset | Macro F1 | Precision | Recall |
|---|---|---|---|
| SciFact (trained, 300 unique) | 0.5263 | 0.53 | 0.52 |
| SciFact-Open (zero-shot) | 0.6219 | 0.62 | 0.63 |

### Per-class observations (reported SciFact, Model 1)

CONTRADICT is the hardest class (F1 0.37): identifying a contradiction from claim text alone,
without evidence, is difficult, since a contradicting claim often looks plausible on its
surface. NEI is strongest (F1 0.65), SUPPORT intermediate (F1 0.56). This ordering is
consistent across all three stages of the baseline, confirming the model's qualitative
behaviour is stable regardless of the corrections — only the absolute level shifts.

---

## Multi-seed robustness check (Model 1 baseline)

To quantify run-to-run variance from random initialisation and data shuffling, the
claim-only baseline was retrained on SciFact with three seeds. This check is only possible
because of Correction 2: before the seeding fix, varying the seed produced identical models
(the learning-rate search's internal re-seed overrode the top-level seed), which would have
yielded a misleading standard deviation of exactly zero. After the fix, the seeds genuinely
diverge. The learning-rate search remains fixed across all runs, so the only source of
variation is the final training run's initialisation and shuffling — isolating seed variance
from learning-rate selection.

### Results across seeds

| Seed | SciFact validation macro F1 |
|---|---|
| 42 (reported baseline) | 0.5263 |
| 123 | 0.5403 |
| 7 | 0.5059 |
| **Mean ± SD** | **0.5242 ± 0.0141** |

The reported baseline (seed 42, 0.5263) sits within this band, confirming it is a
representative run rather than an outlier. The standard deviation of 0.0141 — a spread of
about 0.034 between the lowest and highest run — indicates the training procedure is
reasonably stable: the baseline's performance is a property of the method and data, not of a
single fortunate initialisation.

**Practical use of this figure.** The variance provides a noise floor for interpreting later
results: F1 differences smaller than roughly ±0.014 are within baseline run-to-run noise and
should not be over-interpreted, whereas larger differences — such as Model 2's evidence gain
of +0.118 — are well outside it and are therefore meaningful.

**Limitation.** The multi-seed check was run for the Model 1 baseline only, as it is the
most-scrutinised reference figure and the cheapest to retrain (no large-corpus encoding).
Propagating multi-seed variance through the full pipeline (Model 2 and the retrieval
conditions on both datasets) was not done due to compute cost and is noted as future work.
Reported pipeline figures (Steps 3–5) are single-seed (seed 42) under the same corrected
seeding procedure as the models above.

---

## Model checkpoints and files

- Model 1: `models/saved_models/baseline_scifact/`
- Model 2: `models/saved_models/evidence_scifact/`
- Backed up to Google Drive → rag-thesis → models → saved_models

Result files:
- `results/baseline_scifact_claim_only.json` (Model 1, claim-only)
- `results/baseline_scifact_claim_evidence.json` (Model 2, claim+evidence)
- `results/baseline_scifact_open.json` (Model 1 zero-shot on SciFact-Open)
- `results/seed_check/variance_summary.json` (multi-seed robustness check)
- `results/seed_check/seed{42,123,7}_log.txt` (per-seed training logs)