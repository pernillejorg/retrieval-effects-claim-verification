# Data

This folder holds the dataset loaders and cached data for the project. Two datasets are
used: **SciFact** (primary, used for training) and **SciFact-Open** (secondary, used
test-only / zero-shot as a large-corpus stress test). All loading and label normalisation
is handled by `utils.py`, which both datasets share so that claims and labels are
represented consistently downstream.

```
data/
├── utils.py                 # shared loaders and label normalisation for both datasets
├── scifact/
│   └── cache/               # HuggingFace cache for SciFact (auto-populated on first load)
└── scifact_open/
    └── cache/               # SciFact-Open files (claims.jsonl, corpus.jsonl) - see below
```

## Requirements

The SciFact loader requires a specific `datasets` version:

```
pip install "datasets==2.21.0"
```

This pin is required because the `allenai/scifact` loading script is not compatible with
newer `datasets` releases. In a notebook, install this **first** and restart the runtime
before importing anything, or the loader will fail.

## SciFact (primary, trained)

SciFact is loaded directly from the HuggingFace Hub (`allenai/scifact`) via `load_scifact`,
which pulls both the claims and the corpus and caches them under `scifact/cache/`. Nothing
needs to be downloaded manually; the first call populates the cache.

```python
from data.utils import load_scifact

train_claims, corpus = load_scifact(split="train")
val_claims, corpus   = load_scifact(split="validation")
```

Splits available: `train`, `validation`, `test`. SciFact has no public test labels (it uses
a blind leaderboard), so the **validation** split is used as the final reported evaluation
set throughout the project.

### Validation deduplication

The raw SciFact validation split stores **450 rows for only 300 unique claims**: a claim is
repeated across rows when it is cited against more than one evidence document. `load_scifact`
**deduplicates claims by id**, merging the evidence document ids of repeated rows into a
single claim entry, so the loader returns 300 unique validation claims. This is a
correctness fix (repeated rows carry the same label and add no information); without it,
metrics would be inflated by double-counting. The deduplication is described in more detail,
as a methodological finding, in `results/step2_results.md`.

Labels are normalised to three classes: **SUPPORT**, **CONTRADICT**, **NEI** (not enough
information). Claims with no evidence are mapped to NEI.

## SciFact-Open (secondary, zero-shot)

SciFact-Open (Wadden et al., 2022) is **not on the HuggingFace Hub**, so it is read from
local cached `.jsonl` files rather than downloaded automatically. The files must be placed
under `scifact_open/cache/` before use:

```
data/scifact_open/cache/
├── claims.jsonl             # 279 test claims
├── corpus.jsonl            # full 500,000-document corpus (used for reported numbers)
└── corpus_candidates.jsonl  # optional smaller candidate corpus (faster, for debugging)
```

`corpus.jsonl` (the full 500K corpus) is large and is **gitignored** — it is not stored in
the repository and must be obtained from the SciFact-Open authors' release and placed in the
cache folder manually. In Colab, this folder is staged from Google Drive before running.

```python
from data.utils import load_scifact_open

# full 500K corpus (reported thesis numbers)
claims, corpus = load_scifact_open(corpus_file="full")

# smaller candidate corpus (faster, for debugging only)
claims, corpus = load_scifact_open(corpus_file="candidates")
```

SciFact-Open is used **test-only (zero-shot)**: models trained on SciFact are evaluated on
it without any further training. It provides the large, hard retrieval setting that
contrasts with SciFact's small, tractable corpus.

### Label handling

SciFact-Open loads **279 claims** (SUPPORT 116, CONTRADICT 90, NEI 73). A small number of
claims (15) carry conflicting SUPPORT/CONTRADICT evidence; these are resolved to SUPPORT
under a fixed precedence rule, and the loader prints how many were resolved. Claims with no
evidence are mapped to NEI.

## Note on the earlier dataset (SciClaimHunt)

The project originally planned to use SciClaimHunt as the second dataset, and the early
steps were run with it. It was replaced by SciFact-Open because it does not provide the kind
of large open retrieval corpus this analysis requires (the small-vs-large corpus contrast is
central to the project). The earlier SciClaimHunt work is preserved on the
`main-sciclaimhunt-archive` branch. The rationale for the switch is documented in
`results/step1_results.md`.
