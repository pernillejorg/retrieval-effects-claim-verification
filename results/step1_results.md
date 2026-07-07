# Step 1 Results: Datasets

This document records Step 1, the datasets used in the project, the rationale for choosing them, and the preprocessing decisions applied to each. It is the analysis-and-rationale counterpart to `data/README.md`, which covers the operational setup (how to obtain and load the data). Two datasets are used: **SciFact** as the primary, trained dataset, and **SciFact-Open** as a secondary, test-only dataset that provides a much larger and harder retrieval corpus in the same domain.

## Primary dataset: SciFact

SciFact contains scientific claims verified against paper abstracts, with retrieval over a corpus of roughly 5,000 abstracts. Retrieval on SciFact is genuinely non-trivial (the correct evidence must be found among thousands of scientifically similar abstracts), which is what makes it a meaningful testbed for studying retrieval effects rather than a solved lookup task.

SciFact is the main experimental dataset: the baseline classifiers are trained on it, the full experimental matrix is run on it, and the manual failure annotation (Step 7) is done on it. It is loaded directly from the HuggingFace Hub (`allenai/scifact`).

### Splits and reported evaluation

SciFact provides `train` and `validation` splits with public labels; the official `test` set uses a blind leaderboard with no public labels. The **validation split is therefore used as the final reported evaluation set** throughout the project, which is standard practice for SciFact when the leaderboard is not being targeted.

### Label scheme

Labels are normalised to three classes: **SUPPORT**, **CONTRADICT**, and **NEI** (not enough information). Claims with no cited evidence are treated as NEI. This 3-class scheme is used consistently across both datasets so that results are directly comparable.

### Validation deduplication (a preprocessing correction)

The raw SciFact validation split contains **450 rows corresponding to only 300 unique
claims**. A single claim appears in multiple rows when it is cited against more than one evidence document. Because each of those rows carries the same claim and the same label, treating them as separate evaluation instances would **double-count** some claims and inflate the metrics.

The loader (`load_scifact`) therefore **deduplicates by claim id**, merging the evidence document ids of the repeated rows into a single claim entry, and returns **300 unique validation claims**. This is a correctness fix rather than a modelling choice: it ensures each claim contributes exactly once to evaluation. The deduplication and its effect on the reported numbers are documented in more detail in `step2_results.md`.

## Secondary dataset: SciFact-Open

SciFact-Open (Wadden et al., 2022) is a more recent collection in the same scientific
(biomedical) domain, providing a much larger open retrieval corpus of roughly **500,000 documents**. It is the key element that lets the project study retrieval behaviour as a function of corpus scale.

### Test-only, zero-shot use

SciFact-Open is a **test-only collection with no training split**. It is therefore **not** used to retrain any model. Instead, the SciFact-trained pipeline is evaluated on SciFact-Open directly (zero-shot), testing whether the behaviour observed on SciFact holds when the retrieval problem becomes substantially harder. Running the core pipeline variants and the key retrieval depths on both corpora is what makes the conclusions about retrieval behaviour a property of the method under different retrieval difficulty, rather than an artefact of one corpus size.

### Claims and label handling

The loader (`load_scifact_open`) loads **279 claims** with the distribution SUPPORT 116, CONTRADICT 90, NEI 73. The raw SciFact-Open data provides evidence with SUPPORT or CONTRADICT labels but has **no explicit NEI label**; claims with no evidence are mapped to NEI to align with SciFact's 3-class scheme.

A small number of claims (**15**) carry **conflicting** evidence, with some documents
supporting and others contradicting the same claim. These are resolved by a fixed precedence rule (**any SUPPORT wins, otherwise CONTRADICT**), and the loader reports how many were resolved so the decision can be cited transparently. This is a deliberate, documented normalisation rather than a hidden choice.

### Corpus options

The loader supports two corpus sizes via `corpus_file`: the full **500,000 document** corpus (`"full"`, used for the reported thesis numbers) and a smaller **candidate** corpus (`"candidates"`, roughly 12,000 documents) used only for fast development and debugging. All reported SciFact-Open results use the full corpus.

## Why these two datasets (design rationale)

The pairing is deliberate and is central to the project's design. SciFact and SciFact-Open are in the **same scientific domain** but differ by roughly **100× in corpus size** (about 5,000 versus 500,000 documents). This isolates the axis the project is actually about — **retrieval difficulty**, while holding the domain and task fixed.

This lets the project ask a question a same-size second dataset could not answer as cleanly: does the behaviour of retrieval, reranking, and retrieval depth hold when the candidate pool becomes 100× larger and noisier? Comparing the two corpora directly is what turns single-corpus observations into claims about how retrieval behaviour scales with corpus difficulty.

### Honest scope of the generalisation

Because both datasets are biomedical-scientific, the generalisation tested here is across **retrieval difficulty (corpus scale)**, not across **subject domain**. Domain transfer is a separate question, and it is addressed separately by the real-world seafood and sustainability case study in Step 10, which applies the pipeline to out-of-domain social-media claims. Keeping these two kinds of generalisation distinct scale here, domain in Step 10 is deliberate, so that neither is overclaimed.

## Note on the earlier dataset choice (SciClaimHunt)

The project originally planned to use SciClaimHunt as the second dataset, and the early steps were run with it. It was replaced by SciFact-Open because it does not provide the kind of large open retrieval corpus this analysis requires: the small-versus-large corpus contrast is central to the project, and SciClaimHunt could not supply the large, hard retrieval setting that SciFact-Open does. The earlier SciClaimHunt work is preserved on the `main-sciclaimhunt-archive` branch rather than deleted, so the full history of the project's development remains available.

## Relevance to later steps

- **Step 2** trains the classifiers on SciFact and evaluates them zero-shot on SciFact-Open; the deduplication described here is what makes the SciFact validation numbers correct.
- **Steps 5 and 6** run the pipeline and the retrieval-depth matrix on both corpora, using the scale contrast established here.
- **Step 9** compares the two corpora directly, which is only possible because both are loaded into the same 3-class scheme.
- **Step 10** addresses domain transfer separately, complementing the scale-transfer studied across these two datasets.

## Files

- Loaders and normalisation: `data/utils.py` (`load_scifact`, `load_scifact_open`)
- Operational setup and download instructions: `data/README.md`
- SciFact cache: `data/scifact/cache/` (auto-populated from HuggingFace)
- SciFact-Open cache: `data/scifact_open/cache/` (claims and corpus `.jsonl` files; the full 500K corpus is gitignored and staged separately)
