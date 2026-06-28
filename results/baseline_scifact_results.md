# Baseline Results: RoBERTa No-Retrieval — SciFact

## Experiment Overview

| Property | Value |
|---|---|
| Model | roberta-base |
| Dataset | SciFact (primary) |
| Task | 3-class claim verification (SUPPORT / CONTRADICT / NEI) |
| Retrieval | None — claim text only |
| Device | CUDA (Google Colab A100) |

---

## Dataset Statistics

| Split | Claims |
|---|---|
| Train | 1261 |
| Validation | 450 |

### Train label distribution

| Label | Count |
|---|---|
| SUPPORT | 616 |
| CONTRADICT | 341 |
| NEI | 304 |

---

## Hyperparameter Selection

### Token length check
- Maximum claim token length in train split: **75 tokens**
- MAX_LENGTH = 128 confirmed safe — no claims truncated

### Learning rate search (3 trial epochs each, on full train split)

| Learning rate | Val macro F1 |
|---|---|
| 1e-5 | 0.3219 |
| 2e-5 | 0.4481 |
| **3e-5** | **0.4838** ← selected |

---

## Training Log

| Epoch | Train loss | Val macro F1 | Val precision | Val recall | Notes |
|---|---|---|---|---|---|
| 1 | 1.0346 | 0.3010 | 0.4005 | 0.3257 | New best saved |
| 2 | 0.8772 | 0.4549 | 0.4535 | 0.4570 | New best saved |
| 3 | 0.5923 | 0.5119 | 0.5165 | 0.5167 | New best saved |
| 4 | 0.3952 | 0.4973 | 0.4976 | 0.4971 | No improvement (1) |
| 5 | 0.2587 | 0.5596 | 0.5710 | 0.5628 | New best saved |
| 6 | 0.2162 | 0.5596 | 0.5710 | 0.5628 | No improvement (1) |
| 7 | 0.2118 | 0.5596 | 0.5710 | 0.5628 | No improvement (2) — early stopping triggered |

---

## Final Results (best checkpoint — epoch 5)

### Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORT | 0.62 | 0.65 | 0.64 | 216 |
| CONTRADICT | 0.58 | 0.40 | 0.47 | 122 |
| NEI | 0.52 | 0.63 | 0.57 | 112 |

### Summary metrics (validation set, 450 claims)

| Metric | Value |
|---|---|
| Accuracy | 0.58 |
| Macro F1 | **0.5596** |
| Macro precision | 0.5710 |
| Macro recall | 0.5628 |
| Weighted F1 | 0.58 |

---

## Configuration Summary

| Hyperparameter | Value | Justification |
|---|---|---|
| Model | roberta-base | Standard for NLP classification; Liu et al. (2019) |
| MAX_LENGTH | 128 | Safe — max claim length is 75 tokens |
| BATCH_SIZE | 16 | Standard for MacBook/Colab training |
| MAX_EPOCHS | 10 | Hard ceiling; early stopping used in practice |
| EARLY_STOPPING_PATIENCE | 2 | Avoids overfitting on small dataset |
| Best learning rate | 3e-5 | Selected via grid search over {1e-5, 2e-5, 3e-5} |
| SCHEDULER_EPOCH_ESTIMATE | 5 | Conservative estimate; Wadden et al. (2020) |
| Optimiser | AdamW, weight_decay=0.01 | Standard for transformer fine-tuning |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Warmup steps | 10% of total | Well-established heuristic |

---

## Interpretation

- **Macro F1 = 0.56** is the official no-retrieval baseline. All RAG pipeline results in Steps 3–6 are compared against this.
- **CONTRADICT is the hardest class** (F1 = 0.47) — the model struggles most to identify contradicting evidence without retrieval. Expected, as contradictions require precise evidence matching.
- **SUPPORT performs best** (F1 = 0.64) — likely because SUPPORT claims align most closely with patterns in RoBERTa's pre-training data.
- **NEI performance** (F1 = 0.57) reflects the model's ability to recognise when claims make statements not well-evidenced in pre-training knowledge.
- Early stopping triggered at epoch 7 (best at epoch 5), confirming that 3 fixed epochs would have been insufficient and the early stopping design was necessary.
- The learning rate search confirmed 3e-5 as optimal, consistent with standard RoBERTa fine-tuning recommendations.

---

## Model checkpoint

Saved to: `models/saved_models/baseline_scifact/`  
Also backed up to: `Google Drive > rag-thesis > models > saved_models > baseline_scifact`