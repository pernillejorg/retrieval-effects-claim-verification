# Data

This folder contains scripts and documentation for the two datasets used in this project. Raw data files are not committed to the repository — they are downloaded locally using the scripts provided.

---

## Dataset 1: SciFact (Primary)

**Source:** [allenai/scifact](https://huggingface.co/datasets/allenai/scifact) on Hugging Face  
**Paper:** Wadden et al. (2020) — *Fact or Fiction: Verifying Scientific Claims*  
**Task:** Classify a scientific claim as `SUPPORT`, `CONTRADICT`, or `NOT ENOUGH INFO` given a corpus of paper abstracts as evidence.

### Why SciFact?
SciFact is the primary dataset for this project because retrieval is genuinely hard — the evidence corpus contains ~5,000 abstracts and retrieval quality varies substantially, which is exactly the condition we want to study. Claims are precise and domain-specific, meaning topically similar but non-committal documents are a real problem, which is the core motivation for stance-aware reranking.

### Structure
- `train`: 809 claims with labelled evidence
- `validation`: 300 claims
- `corpus`: 5,183 paper abstracts (the retrieval corpus)

### Labels
| Label | Meaning |
|---|---|
| `SUPPORT` | The evidence supports the claim |
| `CONTRADICT` | The evidence contradicts the claim |
| `NOT_ENOUGH_INFO` | No evidence found that directly addresses the claim |

### Loading
```python
from datasets import load_dataset
dataset = load_dataset("allenai/scifact")
corpus = load_dataset("allenai/scifact", "corpus")
```

---

## Dataset 2: SciClaimHunt (Secondary)

**Source:** To be confirmed — check Hugging Face and recent ACL/EMNLP proceedings  
**Task:** Scientific claim verification, similar label structure to SciFact

### Why SciClaimHunt?
SciClaimHunt is used as the secondary dataset to establish that findings generalise beyond SciFact. Running the core pipeline variants and experimental matrix on both datasets allows us to distinguish between results that are SciFact-specific and results that reflect broader properties of retrieval-augmented claim verification. This is what separates a case study from a research contribution.

### Key differences from SciFact
- More recent dataset with a different source corpus
- Allows cross-dataset comparison of failure patterns and stance reranking effectiveness
- Full manual failure annotation is done on SciFact only; SciClaimHunt uses quantitative failure rate comparison

---

## What is NOT stored here

Raw data files (JSON, JSONL, etc.) are excluded from version control via `.gitignore`. Download and cache them locally using the loading scripts. This keeps the repository lightweight and avoids licensing issues with redistributing dataset files.

---

## Folder structure (once data is downloaded locally)

```
data/
├── README.md               # This file
├── scifact/
│   ├── download.py         # Script to download and cache SciFact
│   └── (cached files)      # gitignored
└── sciclaimhunt/
    ├── download.py         # Script to download and cache SciClaimHunt
        └── (cached files)      # gitignored
        ```
