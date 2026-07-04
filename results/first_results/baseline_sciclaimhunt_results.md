# Step 2: RoBERTa No-Retrieval Baseline for SciClaimHunt Results

## Overview

This file records the results of fine-tuning a RoBERTa-base classifier on SciClaimHunt claim text alone,
with no retrieved evidence. This is the no-retrieval baseline for the SciClaimHunt dataset.
All downstream RAG pipeline results will be compared against this.

## Dataset Split

| Split      | Claims  |
|------------|---------|
| Train      | 87,135  |
| Validation | 10,872  |

**Train label distribution:**
- CONTRADICT: 49,503
- SUPPORT: 37,632

**Note:** SciClaimHunt is a binary classification task (SUPPORT / CONTRADICT only, no NEI class).

## Token Length Check

- Maximum claim token length in train split: **178 tokens**
- MAX_LENGTH set to: **128 tokens**
- **Warning:** Some claims exceed MAX_LENGTH=128 and will be truncated.
  This is a cross-dataset difference from SciFact (max 39 tokens) and is documented as a design limitation.

## Learning Rate Search

Using 2,000 random examples for LR search (large dataset cap).

| Learning Rate | Val Macro F1 |
|---------------|--------------|
| 1e-05         | 0.9391       |
| 2e-05         | 0.9385       |
| 3e-05         | 0.9387       |

**Best learning rate:** 1e-05 (val F1 = 0.9391)

## Full Training Run (lr = 1e-05)

| Epoch | Train Loss | Val Macro F1 | Val Precision | Val Recall | Notes                  |
|-------|------------|--------------|---------------|------------|------------------------|
| 1     | 0.1939     | 0.9746       | 0.9737        | 0.9756     | New best saved         |
| 2     | 0.0779     | 0.9819       | 0.9818        | 0.9819     | New best saved         |
| 3     | 0.0498     | 0.9846       | 0.9845        | 0.9847     | New best saved         |
| 4     | 0.0305     | 0.9840       | 0.9845        | 0.9835     | No improvement (1)     |
| 5     | 0.0190     | 0.9870       | 0.9870        | 0.9871     | New best saved         |
| 6     | 0.0131     | 0.9870       | 0.9870        | 0.9871     | No improvement (1)     |
| 7     | 0.0131     | 0.9870       | 0.9870        | 0.9871     | No improvement (2) — Early stop |

**Early stopping triggered after epoch 7 (patience = 2).**

## Final Results (Best Checkpoint — Epoch 5)

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| SUPPORT    | 0.98      | 0.99   | 0.99     | 4,664   |
| CONTRADICT | 0.99      | 0.99   | 0.99     | 6,208   |
| **macro avg** | **0.99** | **0.99** | **0.99** | **10,872** |
| weighted avg | 0.99   | 0.99   | 0.99     | 10,872  |

**Accuracy: 0.99**

## Summary

| Metric              | Value  |
|---------------------|--------|
| Best val macro F1   | 0.9870 |
| Best learning rate  | 1e-05  |
| Best epoch          | 5      |
| Early stopping      | Yes (epoch 7) |
| Model saved to      | `models/saved_models/baseline_sciclaimhunt` |

## Notes

- SciClaimHunt is a binary task with no NEI class, making it inherently easier than SciFact's 3-class problem.
- The high baseline F1 (0.987) suggests strong linguistic signal in the claim text itself, independent of retrieved evidence.
- This finding is relevant for the RAG analysis: if the no-retrieval baseline is already very strong,
  the added value of retrieval must be measured carefully in subsequent steps.
- Some claims exceed the 128-token MAX_LENGTH and are truncated; this is standard practice and does not
  affect model validity.