# Baseline Results: RoBERTa No-Retrieval Claim Verification

This document records the Step 2 baseline: a RoBERTa model that classifies claims
using **claim text only, with no retrieved evidence**. The model is trained on
SciFact and additionally evaluated zero-shot on SciFact-Open. Every retrieval-augmented
result in later steps is measured against these baseline numbers.

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

## Part 1 — SciFact (primary, trained)

### Dataset statistics

| Split | Claims |
|---|---|
| Train | 1261 |
| Validation | 450 |

Train label distribution: SUPPORT 616, CONTRADICT 341, NEI 304.

### Token length check
- Maximum claim token length in train split: **75 tokens**
- MAX_LENGTH = 128 confirmed safe — no claims truncated.

### Learning rate search (one trial pass per rate)

| Learning rate | Val macro F1 |
|---|---|
| 1e-5 | 0.3163 |
| **2e-5** | **0.4881** ← selected |
| 3e-5 | 0.4715 |

The search selected **2e-5**. Note this is a quick single-pass search used only to
pick the rate; the full training run below is what produces the reported result.

### Training log (full run at lr = 2e-5)

| Epoch | Train loss | Val macro F1 | Val precision | Val recall | Notes |
|---|---|---|---|---|---|
| 1 | 1.0653 | 0.2162 | 0.1600 | 0.3333 | New best saved |
| 2 | 0.9145 | 0.4492 | 0.4610 | 0.4538 | New best saved |
| 3 | 0.6543 | 0.5001 | 0.5231 | 0.4937 | New best saved |
| 4 | 0.4378 | 0.4960 | 0.4949 | 0.4973 | No improvement (1) |
| 5 | 0.2661 | 0.5123 | 0.5374 | 0.5038 | New best saved |
| 6 | 0.1800 | 0.5496 | 0.5765 | 0.5395 | New best saved (best checkpoint) |
| 7 | 0.1195 | 0.5246 | 0.5262 | 0.5242 | No improvement (1) |
| 8 | 0.0879 | 0.5282 | 0.5359 | 0.5236 | No improvement (2) — early stopping |

Early stopping triggered after epoch 8; the best checkpoint is **epoch 6**.

### Final results — SciFact validation (best checkpoint, epoch 6)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.62 | 0.71 | 0.66 | 216 |
| CONTRADICT | 0.44 | 0.46 | 0.45 | 122 |
| NEI | 0.67 | 0.45 | 0.53 | 112 |

| Metric | Value |
|---|---|
| Accuracy | 0.58 |
| Macro F1 | **0.5496** |
| Macro precision | 0.58 |
| Macro recall | 0.54 |
| Weighted F1 | 0.57 |

SciFact has no public test labels (blind leaderboard), so validation is the final
reported split, following standard practice.

---

## Part 2 — SciFact-Open (secondary, zero-shot)

The **same SciFact-trained model** was evaluated on SciFact-Open without any retraining.
SciFact-Open is a test-only collection, so this is a zero-shot generalisation reference.

### Dataset notes
- 279 claims loaded.
- 15 claims had conflicting SUPPORT/CONTRADICT evidence, resolved to SUPPORT under a fixed precedence rule.
- 73 claims had no evidence in the corpus and were mapped to NEI.
- Label distribution: SUPPORT 116, CONTRADICT 90, NEI 73.

### Results — SciFact-Open (zero-shot)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.66 | 0.71 | 0.68 | 116 |
| CONTRADICT | 0.58 | 0.60 | 0.59 | 90 |
| NEI | 0.79 | 0.67 | 0.73 | 73 |

| Metric | Value |
|---|---|
| Accuracy | 0.66 |
| Macro F1 | **0.6665** |
| Macro precision | 0.6774 |
| Macro recall | 0.6594 |

---

## Comparison and interpretation

| Dataset | Macro F1 | Precision | Recall |
|---|---|---|---|
| SciFact (trained) | 0.5496 | 0.58 | 0.54 |
| SciFact-Open (zero-shot) | 0.6665 | 0.68 | 0.66 |

**Why is the zero-shot SciFact-Open score higher than the trained SciFact score?**
This looks counterintuitive but is expected at the baseline stage, and it is important
to state explicitly rather than treat as an anomaly. Two reasons:

1. **Retrieval is not involved yet.** This is the claim-only baseline — neither run
   uses the evidence corpus. SciFact-Open's defining difficulty is its 500,000-document
   retrieval corpus, but that difficulty is invisible to a model that never retrieves.
   The retrieval penalty for SciFact-Open is expected to appear only in later steps,
   once the pipeline must find evidence in the large corpus. The baseline therefore
   establishes that any SciFact-Open degradation observed later is attributable to
   *retrieval*, not to the claims themselves.

2. **Different claim sets and balance.** SciFact-Open has fewer claims (279 vs 450)
   and a cleaner label balance, and its NEI cases are the no-evidence claims, which
   can be comparatively easy to classify from claim text alone (reflected in the high
   NEI F1 of 0.73).

**Per-class observations.** On SciFact, CONTRADICT is the hardest class (F1 0.45):
identifying contradiction without evidence is difficult, since a contradicting claim
often looks superficially plausible. SUPPORT performs best on both datasets. NEI is
notably stronger on SciFact-Open (0.73 vs 0.53), consistent with its no-evidence NEI
construction being easier to recognise.

**Role of these numbers.** These are the no-retrieval reference points. The research
questions of this project — how retrieval affects performance, and how it fails —
are answered by comparing the retrieval-augmented pipelines in later steps against
these baselines, separately on each corpus.

**Limitation.** These are single-run (single-seed) results. Reporting mean and standard
deviation over multiple seeds is noted as future work to quantify run-to-run variance.

---

## Model checkpoint

Saved to: `models/saved_models/baseline_scifact/`
Backed up to: Google Drive → rag-thesis → models → saved_models → baseline_scifact

Result files: `results/baseline_scifact.json`, `results/baseline_scifact_open.json`