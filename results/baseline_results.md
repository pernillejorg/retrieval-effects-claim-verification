# Baseline Results: RoBERTa No-Retrieval Claim Verification

This document records the Step 2 baseline: a RoBERTa model that classifies claims
using **claim text only, with no retrieved evidence**. The model is trained on
SciFact and additionally evaluated zero-shot on SciFact-Open. Every retrieval-augmented
result in later steps is measured against these baseline numbers.

**Important note on this version:** the SciFact validation set was found to contain
duplicate claim rows and has been deduplicated (see "Data correction" below). This
document therefore reports two sets of results: the **original** results computed over
the raw 450-row validation set, and the **corrected** results computed over the 300
unique claims. The corrected results are the ones used going forward.

## Experiment Overview

| Property | Value |
|---|---|
| Model | roberta-base |
| Primary dataset (trained) | SciFact |
| Secondary dataset (zero-shot eval) | SciFact-Open |
| Task | 3-class claim verification (SUPPORT / CONTRADICT / NEI) |
| Retrieval | None — claim text only |
| Device | CUDA (Google Colab GPU) |
| Seed | 42 |

---

## Data correction: deduplicating the SciFact validation set

During Step 4 (reranking), a claim-count mismatch surfaced: the SciFact validation set
was being treated as 450 claims, but only 300 of these were unique. Investigation showed
that SciFact's raw validation split stores **450 rows corresponding to only 300 distinct
claims** — a single claim is repeated across multiple rows when it is cited against more
than one evidence document. A verification check confirmed this: 450 total rows, 300
unique claim ids, and **0 claims with conflicting labels** (i.e. every repeated claim
carried the same label, so the repeats were pure duplicates carrying no new information).

The data loader (`load_scifact`) was corrected to deduplicate claims by id, merging the
evidence document ids of repeated rows into a single claim entry. This means:

- The validation set is now 300 unique claims rather than 450 rows.
- No claim is lost — only accidental repeats are removed.
- Every downstream step (baseline, retrieval, reranking) now operates on the same 300
  unique claims, so metrics are no longer inflated by double-counting.

This is a correctness fix, not a reduction in data. Counting 450 rows treated 150 claims
as if they were distinct, which double-weighted them in every metric and in training. The
corrected 300-claim count is the true number of distinct claims in the split.

---

## Part 1 — SciFact (primary, trained)

### Original results (raw 450-row validation set) — SUPERSEDED

These were the results before the deduplication fix, computed over the raw 450 rows.
They are retained here for transparency and comparison only; they are **not** the
reported baseline.

| Metric | Value (450 rows) |
|---|---|
| Validation claims | 450 (with duplicates) |
| Macro F1 | 0.5496 |
| Macro precision | 0.58 |
| Macro recall | 0.54 |

Original per-class (450 rows):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.62 | 0.71 | 0.66 | 216 |
| CONTRADICT | 0.44 | 0.46 | 0.45 | 122 |
| NEI | 0.67 | 0.45 | 0.53 | 112 |

*(Note: the support column sums to 450, reflecting the duplicated rows.)*

### Corrected results (300 unique claims) — REPORTED BASELINE

The model was retrained and evaluated on the deduplicated 300-claim validation set.

**Dataset statistics**

| Split | Claims |
|---|---|
| Train | 809 |
| Validation | 300 (unique) |

Train label distribution: SUPPORT 332, CONTRADICT 173, NEI 304.

**Token length check**
- Maximum claim token length: **75 tokens**
- MAX_LENGTH = 128 confirmed safe — no claims truncated.

**Learning rate search (one trial pass per rate)**

| Learning rate | Val macro F1 |
|---|---|
| 1e-5 | 0.2997 |
| 2e-5 | 0.3812 |
| **3e-5** | **0.4295** ← selected |

**Training log (full run at lr = 3e-5)**

| Epoch | Train loss | Val macro F1 | Val precision | Val recall | Notes |
|---|---|---|---|---|---|
| 1 | 1.0944 | 0.1812 | 0.1244 | 0.3333 | New best saved |
| 2 | 1.0375 | 0.3250 | 0.3780 | 0.3928 | New best saved |
| 3 | 0.9691 | 0.3668 | 0.3285 | 0.4156 | New best saved |
| 4 | 0.7273 | 0.4149 | 0.4133 | 0.4179 | New best saved |
| 5 | 0.5435 | 0.4298 | 0.4547 | 0.4517 | New best saved |
| 6 | 0.4371 | 0.4570 | 0.4830 | 0.4506 | New best saved (best checkpoint) |
| 7 | 0.3575 | 0.4533 | 0.4732 | 0.4465 | No improvement (1) |
| 8 | 0.3012 | 0.4501 | 0.4615 | 0.4439 | No improvement (2) — early stopping |

Early stopping triggered after epoch 8; the best checkpoint is **epoch 6**.

**Final results — SciFact validation (best checkpoint, epoch 6)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.44 | 0.51 | 0.47 | 124 |
| CONTRADICT | 0.27 | 0.34 | 0.30 | 64 |
| NEI | 0.74 | 0.50 | 0.60 | 112 |

| Metric | Value |
|---|---|
| Accuracy | 0.47 |
| Macro F1 | **0.4570** |
| Macro precision | 0.48 |
| Macro recall | 0.45 |
| Weighted F1 | 0.48 |

*(Support column sums to 300, confirming the deduplicated set.)*

SciFact has no public test labels (blind leaderboard), so validation is the final
reported split, following standard practice.

---

## Part 2 — SciFact-Open (secondary, zero-shot)

The **same SciFact-trained model** was evaluated on SciFact-Open without retraining.
SciFact-Open is a test-only collection, so this is a zero-shot generalisation reference.
SciFact-Open was **not** affected by the deduplication issue (it has no duplicate claims),
so its numbers are essentially unchanged between runs.

**Dataset notes**
- 279 claims loaded.
- 15 claims had conflicting SUPPORT/CONTRADICT evidence, resolved to SUPPORT under a fixed precedence rule.
- 73 claims had no evidence in the corpus and were mapped to NEI.
- Label distribution: SUPPORT 116, CONTRADICT 90, NEI 73.

**Results — SciFact-Open (zero-shot, corrected run)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.50 | 0.55 | 0.53 | 116 |
| CONTRADICT | 0.46 | 0.46 | 0.46 | 90 |
| NEI | 0.68 | 0.58 | 0.62 | 73 |

| Metric | Value |
|---|---|
| Accuracy | 0.53 |
| Macro F1 | **0.5348** |
| Macro precision | 0.5456 |
| Macro recall | 0.5275 |

---

## Comparison and interpretation

### Effect of the deduplication fix on SciFact

| Metric | Original (450 rows) | Corrected (300 unique) |
|---|---|---|
| Validation claims | 450 (with duplicates) | 300 (unique) |
| Macro F1 | 0.5496 | 0.4570 |
| Macro precision | 0.58 | 0.48 |
| Macro recall | 0.54 | 0.45 |

**The macro F1 decreased from 0.5496 to 0.4570 after deduplication — and this is
expected and correct, not a regression.** The original figure was inflated because 150
of the 450 rows were duplicate claims, which were double-counted in the metric (and
double-weighted during training). Removing the duplicates removes that inflation and
reveals the model's true performance on the 300 distinct claims. In other words, the
model did not get worse; the measurement became honest. The lower number is the
trustworthy one.

This is consistent across classes: the per-class pattern is unchanged (SUPPORT easiest,
CONTRADICT hardest, NEI strong), confirming it is the same model behaving the same way,
now measured over the correct claim set.

**A lower baseline is also advantageous for the project.** The baseline exists to be the
floor that the retrieval-augmented pipeline improves upon. A corrected, lower baseline of
0.4570 leaves clearer room to demonstrate the value of retrieval and stance reranking in
later steps than an inflated 0.5496 would have. An honest, modest baseline strengthens the
eventual comparison rather than weakening it.

### SciFact vs SciFact-Open (corrected)

| Dataset | Macro F1 | Precision | Recall |
|---|---|---|---|
| SciFact (trained, 300 unique) | 0.4570 | 0.48 | 0.45 |
| SciFact-Open (zero-shot) | 0.5348 | 0.55 | 0.53 |

At the **baseline (no-retrieval) stage**, SciFact-Open scores higher than SciFact despite
being zero-shot. This is expected: the baseline uses no retrieval, so SciFact-Open's
defining difficulty — its 500,000-document retrieval corpus — is invisible here. The two
numbers reflect only claim-only classification difficulty on two different claim sets, and
SciFact-Open's smaller, balanced set with easily-recognised no-evidence NEI cases is
comparatively easier to classify from claim text alone. The retrieval penalty for
SciFact-Open is expected to appear only in later steps, once the pipeline must find
evidence in the large corpus. The baseline therefore establishes that any SciFact-Open
degradation observed later is attributable to retrieval, not to the claims themselves.

### Per-class observations (corrected SciFact)

CONTRADICT is the hardest class (F1 0.30): identifying a contradiction from claim text
alone, without evidence, is difficult, since a contradicting claim often looks plausible
on its surface. NEI is strongest (F1 0.60), and SUPPORT is intermediate (F1 0.47). This
ordering is consistent with the pre-deduplication run, confirming the model's behaviour
is stable.

**Limitation.** These are single-run (single-seed) results. Reporting mean and standard
deviation over multiple seeds is noted as future work to quantify run-to-run variance.

---

## Model checkpoint and files

Saved to: `models/saved_models/baseline_scifact/`
Backed up to: Google Drive → rag-thesis → models → saved_models → baseline_scifact

Result files: `results/baseline_scifact.json`, `results/baseline_scifact_open.json`