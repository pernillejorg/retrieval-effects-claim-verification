# Empirical Analysis of Retrieval Effects and Failure Behaviour in Retrieval-Augmented Scientific Claim Verification

*MSc Artificial Intelligence dissertation, Queen Mary University of London*

 ---

**Author:** Pernille Bergesen · **Supervisor:** Dr. Arkaitz Zubiaga

 ---

## Overview

Most fact-checking research assumes oracle evidence, meaning clean, perfectly retrieved documents handed to the model. This project looks at what actually happens under realistic retrieval conditions: when the system has to find its own evidence and sometimes gets it wrong.

The project is centred on the effect and failure behaviour side of retrieval augmentation. The work is focused on characterising where and why it fails for scientific claim verification, and on the conditions under which retrieval helps versus the failure modes it can introduce (such as evidence overload, degradation at scale, and unstable predictions), rather than re-confirming that retrieval helps, which is already well established.

A key component under investigation is a stance-aware reranking step: after standard retrieval, a zero-shot NLI model scores whether each retrieved document actually takes a *stance* on the claim, with the intention of demoting documents that are topically related but say nothing specific about the claim. The project studies retrieval, reranking, retrieval depth, and confidence behaviour across two scientific claim datasets of very different corpus sizes, and examines whether the observed behaviour transfers to an out-of-domain real-world setting.

**The thesis question:**
*This project asks not whether retrieval improves automated fact-checking, which prior work largely assumes, but under what conditions it fails: it systematically characterises the failure behaviour of retrieval-augmented scientific claim verification across retrieval method, retrieval depth, and corpus scale, links each failure to a defined failure category, and tests whether the behaviour holds on out-of-domain real-world claims, so that the reliability of retrieval augmentation for automated fact-checking can be assessed rather than assumed.*

**Scope:** 
Ten pipeline stages evaluated across two corpora differing 100-fold in size; a 32-cell retrieval-depth matrix, four conditions by four depths on each of two corpora, run under three training seeds; both classifiers and the full pipeline retrained and re-run at each seed; 70 hand-annotated errors with a blind second annotation pass (Cohen's kappa 0.914); and a 30-claim out-of-domain case study with five pre-registered expectations. Three hypotheses were fixed before any experiment. Two were refuted, and both refutations are reported as findings rather than revised away.
 
  ---
 
## Final Project Structure

```
retrieval-effects-claim-verification/
├── data/                           #dataset loading and preprocessing
│   ├── README.md                   #dataset setup and loading instructions
│   ├── utils.py                    #load_scifact / load_scifact_open loaders
│   ├── scifact/                    #SciFact (primary dataset, trained)
│   └── scifact_open/               #SciFact-Open (secondary dataset, zero-shot)
├── models/
│   ├── baseline.py                 #RoBERTa classifiers (claim-only + claim+evidence)
│   ├── retrieval.py                #BM25 + dense (mpnet) retrieval
│   ├── reranker.py                 #stance-aware reranking via NLI
│   └── pipeline.py                 #full pipeline (also used for the Step 6 k-sweep)
├── analysis/
│   ├── annotation_guide.md         #failure category annotation rules
│   ├── tools/                      #annotation and analysis helper scripts
│   ├── failure_taxonomy.py         #failure category annotation and automatic signals
│   ├── confidence_analysis.py      #retrieval-aware confidence scoring
│   └── cross_corpus.py             #cross-dataset (SciFact vs SciFact-Open) comparison
├── notebooks/                      #Colab notebooks, one per step
│   ├── README.md                   #maps each Step 2 notebook to its role
│   ├── step2/
│   │   ├── Step2_reported_baseline.ipynb
│   │   ├── Step2_variance_study.ipynb
│   │   ├── additional_checks/
│   │   │   └── Step2_variance_evidence_model.ipynb
│   │   └── earlier_versions/
│   │       ├── Step2_stage1_raw_450rows.ipynb
│   │       ├── Step2_stage2_deduplicated_model1.ipynb
│   │       └── Step2_stage2_deduplicated_with_evidence.ipynb
│   ├── Step3_retrieval_Mini.ipynb
│   ├── Step3_retrieval_mpnet.ipynb
│   ├── Step4_reranker.ipynb
│   ├── Step5_pipeline_multipleseeds.ipynb
│   ├── Step5_pipeline.ipynb
│   ├── Step6_experiments.ipynb
│   ├── Step6_experiments_multiseed.ipynb
│   ├── Step7_FailureTaxonomy.ipynb
│   ├── Step8_ConfidenceScoring.ipynb
│   ├── Step9_CrossCorpus.ipynb
│   └── Step10_RealWorld.ipynb
├── realworld/                      #Step 10 real-world case study
│   ├── seafood_claims.py           #case study runner
│   ├── seafood_claims.csv          #the 30 collected claims
│   ├── collection_guide.md         #collection protocol, fixed before collection
│   └── results_annotation_guide.md #annotation scheme for the case study
├── results/                        #result write-ups and output tables
│   ├── step2_baseline/             #Step 2 output JSONs
│   ├── step3_retrieval/            #Step 3 recall JSONs
│   ├── step4_reranker/             #Step 4 output tables
│   ├── step5_pipeline/             #Step 5 pipeline records
│   ├── step6_matrix/               #Step 6 k-sweep matrix and records
│   │   └── step6_multiseed_matrix/ #seed 123 and 7 runs, plus assembled summary
│   ├── step7_failure/              #annotations, rates, analysis JSONs
│   ├── step8_confidence/           #confidence JSONs and CSV tables
│   ├── step9_comparison/           #cross-corpus comparison JSON and CSVs
│   ├── step10_realworld/           #case study records and summary
│   ├── step1_results.md
│   ├── step2_results.md
│   ├── step3_results.md
│   ├── step4_results.md
│   ├── step5_results.md
│   ├── step6_results.md
│   ├── step7_results.md
│   ├── step8_results.md
│   ├── step9_results.md
│   └── step10_results.md
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Running the Project

Install dependencies, then run the stages in order. Each script writes to
`results/`, and the later analyses read those files rather than re-invoking
models.

```bash
pip install -r requirements.txt
```

| Stage | Command |
|---|---|
| 2. Baseline | `python models/baseline.py --dataset scifact` |
| 3. Retrieval | `python models/retrieval.py --dataset scifact` |
| 4. Reranking | `python models/reranker.py --dataset scifact --mode soft` |
| 5. Pipeline | `python models/pipeline.py --dataset scifact --top_k 3` |
| 7. Failure taxonomy | `python analysis/failure_taxonomy.py` |
| 8. Confidence | `python analysis/confidence_analysis.py` |
| 9. Cross-corpus | `python analysis/cross_corpus.py` |
| 10. Case study | `python realworld/seafood_claims.py --claims_csv realworld/seafood_claims.csv` |

All stages were run on a Google Colab GPU. Encoding the 500,000-document
SciFact-Open corpus takes roughly 40 minutes, and the full depth matrix
under three seeds takes several hours. Full arguments and seed settings for
each stage are in the corresponding `results/stepN_results.md`.

---
 
## Methodology
 
### Step 1: Datasets

The primary dataset is SciFact, which contains scientific claims verified against paper abstracts, with genuinely hard retrieval over a roughly 5,000-abstract corpus. This is the main experimental dataset: baseline training, the full matrix, and failure annotation all happen here.

SciFact-Open is used as the secondary dataset. It provides a much larger open retrieval corpus (roughly 500,000 documents) in the same scientific domain, and because it is a test-only collection with no training split, the SciFact-trained pipeline is evaluated on it zero-shot. This contrasts retrieval on a small, tractable corpus (SciFact) with a large, hard one (SciFact-Open), testing whether the findings hold as retrieval difficulty scales up.

Core experiments run on both datasets. Manual failure annotation is done on SciFact only.

(An earlier version of the project used SciClaimHunt as the second dataset, but it was replaced by SciFact-Open because it does not provide the kind of large open retrieval corpus this analysis requires. The SciClaimHunt work is preserved on the `main-sciclaimhunt-archive` branch.)

### Step 2: Baseline Model

A RoBERTa model trained to verify claims without any retrieved evidence, and a second RoBERTa model trained on claim plus evidence pairs. Trained and evaluated on SciFact (F1, precision, recall), and additionally evaluated zero-shot on SciFact-Open. Both classifiers were retrained under three seeds (42, 123, 7), so the baseline figures carry a measured standard deviation rather than resting on a single run. This is the reference point that everything else is measured against.

### Step 3: Evidence Retrieval

Two retrieval methods are implemented:

- BM25: keyword-based sparse retrieval
- Dense retrieval: semantic similarity via sentence-transformers (all-mpnet-base-v2)

For each claim, a candidate pool of documents is retrieved from the evidence corpus using each method. These are the standard baselines the stance reranker is compared against.

### Step 4: Stance-Aware Reranking

After standard retrieval, a filtering step using `cross-encoder/nli-deberta-v3-small` (Hugging Face) scores each retrieved document for entailment, contradiction, or neutral. Neutral documents sink in the ordering, or in hard mode are removed outright, so that documents which actually take a stance on the claim are prioritised before verification.

The motivation is that topical similarity may not be enough. A document about omega-3 and cardiovascular health might be retrieved for a related claim but say nothing specific about it. Stance scoring is intended to catch this. Whether it actually improves evidence selection is one of the questions the project investigates.

| Retrieval condition | What it does |
|---|---|
| BM25 | keyword baseline, no filtering |
| Dense | semantic similarity baseline, no filtering |
| Dense + stance reranking | semantic retrieval reordered by NLI stance scores |

Two modes are tested: soft, which reorders the candidate pool without removing anything, and hard, which discards documents whose neutral probability exceeds a threshold set either loosely at 0.5 or strictly at 0.8. Hard filtering proved unusable at either threshold, leaving 0.6 to 1.2 documents per claim and collapsing R@10 from 0.803 to 0.077. Soft reranking also harms top-rank recall but at least preserves the full pool, so it is the mode carried into the pipeline as the lesser damage rather than as an improvement.

### Step 5: Full Pipeline

The retrieved evidence is integrated into the verification model. Four pipeline variants are compared: no retrieval, BM25 + RoBERTa, Dense + RoBERTa, and Dense + stance reranking + RoBERTa. This is where the effect of retrieved evidence on final verification is measured end to end.

An initial version used naive concatenation, which revealed that retrieval depth stopped mattering above roughly two documents because the context window saturated. That saturation was itself a finding, and it was addressed with per-document token budgeting so that k genuinely varies the evidence the model sees. The pipeline was also run across three seeds (42, 123, 7) to give the pipeline numbers a variance band rather than a single run.

The variance study revealed that every retrieval condition has a seed standard deviation two to four times the claim-only baseline's, so retrieval-augmentation introduces instability the baseline does not have.

### Step 6: Controlled Experimental Matrix

The pipeline variants are run under systematically varied retrieval depth:

| Variable | Values tested |
|---|---|
| Retrieval condition | no retrieval, BM25, dense, dense + stance |
| k (number of docs retrieved) | 1, 3, 5, 10 |
| Training seed | 42, 123, 7 |

Metric: macro F1. Run on both datasets. The matrix spans four conditions by four depths on each of the two corpora, and was run in full under all three seeds, so every cell carries a mean and standard deviation rather than a single value. The stance threshold is not swept here: soft reranking retains all documents, so the threshold does not filter and the matrix runs at the loose setting only.

Two readings from the original seed-42 run did not survive averaging: retrieval's apparent advantage at k = 1 on SciFact, and the corpus-dependent optimum on SciFact-Open. Both are reported in `results/step6_results.md` alongside the seed-42 tables.

The multi-seed re-run was added late in the project specifically to test whether the depth findings were seed artefacts. It confirmed the overload trend and the SciFact-Open result, and overturned two readings that had been taken from seed 42 alone. The run is recorded in `notebooks/Step6_experiments_multiseed.ipynb`.

### Step 7: Failure Taxonomy

Four failure categories, defined before running experiments:

1. Irrelevant retrieval: retrieved documents are not really about the claim
2. Contradictory retrieval: retrieved documents argue against the correct label
3. Evidence overload: too many documents confuse or dilute the model
4. Confident wrong prediction: model is wrong despite having reasonably relevant evidence

70 errors from SciFact (35 each from the dense and dense-plus-rerank conditions) are manually labelled into these categories as the primary failure analysis, and all 70 were then relabelled from scratch in a blind second pass, giving an intra-annotator agreement of Cohen's kappa 0.914 over the 65 categorised rows. For SciFact-Open, quantitative failure rates are compared across conditions without full manual annotation, which is an honest scoping decision for a solo project. The reranker was designed to reduce the first two categories. Both instead roughly double as a share of each condition's errors, while the two downstream categories fall, so it moves failures upstream rather than removing them. The samples are small (around 33 per condition) and the Wilson intervals are wide, so the shift is directional rather than precisely quantified.

### Step 8: Retrieval-Aware Confidence Scoring

RoBERTa outputs a softmax probability distribution, and the highest probability is used as a confidence score. The analysis looks at whether low-confidence predictions are more likely to be wrong, whether stance reranking changes the confidence–correctness relationship, and whether a simple flagging rule catches more errors. The central finding is an inversion: on refutation claims the classifier is on average more confident when wrong than when right, in all eight condition-by-corpus cells, and reranking extends the reversal to a second class on SciFact.

### Step 9: Cross-Dataset Comparison

Three questions are asked of the two corpora side by side: does reranking help consistently as retrieval difficulty scales, do the failure indicators shift as the corpus grows roughly 100 times larger, and does the confidence–correctness pattern hold under harder retrieval. The main finding is that on the large corpus no retrieval configuration beats the no-retrieval baseline, so retrieval becomes counterproductive at scale rather than merely harder, and the confidence inversion on refutation claims holds on both corpora. Framing is retrieval-difficulty generalisation; domain generalisation is handled by Step 10.

### Step 10: Real-World Application

30 real seafood and sustainability claims collected from public social media, run through the same pipeline and corpus under three conditions and four depths. Five expectations were fixed in advance, each carried in from an earlier step, and the claims were selected to exercise different failure modes rather than sampled at random, so the accuracy figures illustrate transferability rather than estimating deployment performance. The clearest result is that the confidence inversion on refutation claims reproduces in a domain the pipeline was never built for: the model scored zero of seven on the true-CONTRADICT claims in all nine condition-by-depth runs, at mean confidence rising to 0.96.
 
  ---

## Models and Libraries

**Models**

- `roberta-base`: claim verification classifier, fine-tuned in two variants
- `sentence-transformers/all-mpnet-base-v2`: dense retrieval (primary)
- `sentence-transformers/all-MiniLM-L6-v2`: dense retrieval (comparison)
- `cross-encoder/nli-deberta-v3-small`: zero-shot stance scoring

**Libraries**

- `torch`, `transformers`: model training and inference
- `sentence-transformers`: dense embedding and retrieval
- `rank-bm25`: BM25 sparse retrieval
- `datasets`: dataset loading from Hugging Face
- `scikit-learn`: evaluation metrics and Cohen's kappa
- `numpy`, `pandas`: analysis and result tables

Full pinned versions are in `requirements.txt`.

---
 
## What Goes Beyond Prior Work
 
| | MAPLE | Stammbach and Neumann | This project |
|---|---|---|---|
| Datasets | FEVER, cFEVER, SciFact (oracle and retrieved) | FEVER (Wikipedia) | SciFact + SciFact-Open |
| Retrieval analysis | Fixed depth k = 3, no sweep | Single comparison of five sentences against a thresholded variant | Controlled matrix: method, depth k, stance threshold, three seeds |
| Stance-aware retrieval | Not done | Not done (supervised sentence ranker; entailment used only as verifier) | NLI filtering for scientific claims + full evaluation |
| Failure taxonomy | Not done | Not done | Pre-defined 4-category taxonomy |
| Confidence scoring | Not done | Not done | Retrieval-aware confidence signal |
| Cross-dataset | Not possible | Not possible | Core findings compared across both |
| Real-world domain | Not done | Not done | Seafood/sustainability social media claims |
| Retrieval value at scale | Quality studied, not depth or scale | Not studied | Shown counterproductive on the large corpus |
 

- **MAPLE** (Zeng and Zubiaga, 2024): a few-shot claim verification method using seq2seq training dynamics as classifier features
- **Stammbach and Neumann** (2019), the DOMLIN system: a FEVER shared-task entry built on a supervised sentence ranker followed by an entailment classifier

Both are systems proposed to verify claims better. This project holds the task fixed and varies retrieval instead, so the comparison is one of scope: neither prior system set out to characterise how retrieval fails, which is the question asked here.

  ---
 
## Completed Stages
 
- [x] Step 1: Dataset loading and preprocessing
- [x] Step 2: RoBERTa baseline
- [x] Step 3: BM25 + dense retrieval
- [x] Step 4: Stance-aware reranker
- [x] Step 5: Full pipeline
- [x] Step 6: Experimental matrix
- [x] Step 7: Failure taxonomy and annotation
- [x] Step 8: Confidence scoring analysis
- [x] Step 9: Cross-dataset comparison
- [x] Step 10: Real-world case study
 