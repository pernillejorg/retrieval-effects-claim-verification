# RAG Claim Verification — MSc Thesis

**Empirical Analysis of Retrieval Effects and Failure Behaviour in RAG Models for Scientific Claim Verification**

 *MSc AI Thesis Project: supervised research*
 ---
  Author: Pernille Bergesen (MSc AI Student)
 
  Supervisor: Arkaitz Zubiaga (Senior Lecturer in NLP)
 ---

## Overview

Most fact-checking research assumes oracle evidence, meaning clean, perfectly retrieved documents handed to the model. This project looks at what actually happens under realistic retrieval conditions: when the system has to find its own evidence and sometimes gets it wrong.

The project is centred on the effect and failure behaviour side of RAG. The work is focused on characterising where and why RAG fails for scientific claim verification, and on the conditions under which retrieval helps versus the failure modes it can introduce (such as evidence overload, degradation at scale, and unstable predictions), rather than re-confirming that retrieval helps, which is already well established. Part of the motivation is practical: several companies interested in adopting RAG for fact-checking appear largely unaware of its failure modes and cost/benefit trade-offs. There is real value in evidence about when these components actually hurt, so that adopting RAG becomes an informed decision rather than an assumption.

A key component under investigation is a stance-aware reranking step: after standard retrieval, a zero-shot NLI model scores whether each retrieved document actually takes a *stance* on the claim, with the intention of filtering out documents that are topically related but say nothing specific about the claim. The project studies retrieval, reranking, retrieval depth, and confidence behaviour across two scientific claim datasets of very different corpus sizes, and examines whether the observed behaviour transfers to an out-of-domain real-world setting.

**The thesis question:**
*This project asks not whether retrieval improves automated fact-checking, which prior work largely assumes, but under what conditions it fails: it systematically characterises the failure behaviour of retrieval-augmented scientific claim verification across retrieval method, retrieval depth, and corpus scale, links each failure to a defined failure category, and tests whether the behaviour holds on out-of-domain real-world claims, so that the reliability of RAG for automated fact-checking can be assessed rather than assumed.*

  ---
 
  ## Project Timeline
 
  ```mermaid
  gantt
      title MSc Thesis Project Timeline
      dateFormat  YYYY-MM-DD
      axisFormat  %d %b
 
      section 1. Literature Review
      Background Research           :lit1, 2026-05-25, 2026-06-08
      Related Work (Fact-checking/RAG) :lit2, 2026-05-29, 2026-06-15
 
      section 2. Data Preparation
      Dataset Download & Exploration of SciFact :data1, 2026-05-28, 2026-06-10
      Data Cleaning & Formatting    :data2, 2026-06-04, 2026-06-19
 
      section 3. Baseline Model
      Model Setup (RoBERTa)         :base1, 2026-06-10, 2026-06-22
      Training & Evaluation         :base2, 2026-06-17, 2026-06-30
 
      section 4. Retrieval Module
      BM25 Implementation           :ret1, 2026-06-15, 2026-06-26
      Dense Retrieval Implementation :ret2, 2026-06-19, 2026-07-06
 
      section 5. RAG System
      Integration Claim             :rag1, 2026-06-29, 2026-07-08
      Integration Evidence          :rag2, 2026-06-29, 2026-07-10
 
      section 6. Evaluation & Analysis
      Initial Evaluation (for draft) :eval1, 2026-07-02, 2026-07-17
      Extended Experiments          :eval2, 2026-07-10, 2026-07-24
      Error & Failure Analysis      :eval3, 2026-07-17, 2026-07-29
 
      section 7. Real-world Application
      Model behaviour on real-world scenarios :rw1, 2026-07-20, 2026-07-30
 
      section 8. Writing
      Draft Writing                 :write1, 2026-06-25, 2026-07-17
      Final Writing: Refinement     :write2, 2026-07-08, 2026-08-11
      Editing & Proofreading        :write3, 2026-08-05, 2026-08-12
 
      section 9. Presentation
      Presentation Prep & Finishing :pres1, 2026-08-05, 2026-08-12
  ```
 
  ---
 
## Final Project Structure

```
rag-claim-verification/
├── data/                           #dataset loading and preprocessing
│   ├── README.md                   #dataset setup and loading instructions
│   ├── utils.py                    #load_scifact / load_scifact_open loaders
│   ├── scifact/                    #SciFact (primary dataset, trained)
│   └── scifact_open/               #SciFact-Open (secondary dataset, zero-shot)
├── models/
│   ├── baseline.py                 #RoBERTa classifiers (claim-only + claim+evidence)
│   ├── retrieval.py                #BM25 + dense (mpnet) retrieval
│   ├── reranker.py                 #stance-aware reranking via NLI
│   ├── pipeline.py                 #full RAG pipeline (also used for the Step 6 k-sweep)
│   └── pipeline1.py                #earlier pipeline variant (retained)
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
│   ├── Step7_FailureTaxonomy.ipynb
│   ├── Step8_ConfidenceScoring.ipynb
│   └── Step9_CrossCorpus.ipynb
├── realworld/                      #Step 10 real-world case study (in progress)
│   └── seafood_claims.py           #seafood/sustainability social-media claims
├── results/                        #result write-ups and output tables
│   ├── step2_baseline/             #Step 2 output JSONs
│   ├── step3_retrieval/            #Step 3 recall JSONs
│   ├── step4_reranker/             #Step 4 output tables
│   ├── step5_pipeline/             #Step 5 pipeline records
│   ├── step6_matrix/               #Step 6 k-sweep matrix and records
│   ├── step7_failure/              #annotations, rates, analysis JSONs
│   ├── step8_confidence/           #confidence JSONs and CSV tables
│   ├── step9_comparison/           #cross-corpus comparison JSON and CSVs
│   ├── step1_results.md
│   ├── step2_results.md
│   ├── step3_results.md
│   ├── step4_results.md
│   ├── step5_results.md
│   ├── step6_results.md
│   ├── step7_results.md
│   ├── step8_results.md
│   └── step9_results.md
├── .gitignore
├── README.md
└── requirements.txt
```
  ---
 
  ## Methodology
 
### Step 1: Datasets

The primary dataset is SciFact, which contains scientific claims verified against paper abstracts, with genuinely hard retrieval over a roughly 5,000-abstract corpus. This is the main experimental dataset: baseline training, the full matrix, and failure annotation all happen here.

SciFact-Open is used as the secondary dataset. It provides a much larger open retrieval corpus (roughly 500,000 documents) in the same scientific domain, and because it is a test-only collection with no training split, the SciFact-trained pipeline is evaluated on it zero-shot. This contrasts retrieval on a small, tractable corpus (SciFact) with a large, hard one (SciFact-Open), testing whether the findings hold as retrieval difficulty scales up.

Core experiments run on both datasets. Manual failure annotation is done on SciFact only.

(An earlier version of the project used SciClaimHunt as the second dataset, but it was replaced by SciFact-Open because it does not provide the kind of large open retrieval corpus this analysis requires. The SciClaimHunt work is preserved on the `main-sciclaimhunt-archive` branch.)

### Step 2: Baseline Model

A RoBERTa model trained to verify claims without any retrieved evidence, and a second RoBERTa model trained on claim plus evidence pairs. Trained and evaluated on SciFact (F1, precision, recall), and additionally evaluated zero-shot on SciFact-Open. This is the reference point that everything else is measured against.

### Step 3: Evidence Retrieval

Two retrieval methods are implemented:

- BM25: keyword-based sparse retrieval
- Dense retrieval: semantic similarity via sentence-transformers (all-mpnet-base-v2)

For each claim, a candidate pool of documents is retrieved from the evidence corpus using each method. These are the standard baselines the stance reranker is compared against.

### Step 4: Stance-Aware Reranking

After standard retrieval, a filtering step using `cross-encoder/nli-deberta-v3-small` (Hugging Face) scores each retrieved document for entailment, contradiction, or neutral. Neutral documents are filtered out or downranked, so that documents which actually take a stance on the claim are prioritised before verification.

The motivation is that topical similarity may not be enough. A document about omega-3 and cardiovascular health might be retrieved for a related claim but say nothing specific about it. The stance filter is intended to catch this. Whether it actually improves evidence selection is one of the questions the project investigates.

| Retrieval condition | What it does |
|---|---|
| BM25 | keyword baseline, no filtering |
| Dense | semantic similarity baseline, no filtering |
| Dense + stance reranking | semantic retrieval filtered by NLI stance scores |

Two filter thresholds are tested: loose and strict.

### Step 5: RAG Pipeline

The retrieved evidence is integrated into the verification model. Four pipeline variants are compared: no retrieval, BM25 + RoBERTa, Dense + RoBERTa, and Dense + stance reranking + RoBERTa. This is where the effect of retrieved evidence on final verification is measured end to end.

An initial version used naive concatenation, which revealed that retrieval depth stopped mattering above roughly two documents because the context window saturated. That saturation was itself a finding, and it was addressed with per-document token budgeting so that k genuinely varies the evidence the model sees. The pipeline was also run across three seeds (42, 123, 7) to give the pipeline numbers a variance band rather than a single run.

### Step 6: Controlled Experimental Matrix

The pipeline variants are run under systematically varied retrieval depth:

| Variable | Values tested |
|---|---|
| Retrieval condition | BM25, dense, dense + stance |
| k (number of docs retrieved) | 1, 3, 5, 10 |
| Stance filter threshold | loose, strict |

Metric: F1, precision, recall. Run on both datasets for the core conditions. This isolates the effect of how many documents are supplied to the verifier.

### Step 7: Failure Taxonomy

Four failure categories, defined before running experiments:

1. Irrelevant retrieval: retrieved documents are not really about the claim
2. Contradictory retrieval: retrieved documents argue against the correct label
3. Evidence overload: too many documents confuse or dilute the model
4. Confident wrong prediction: model is wrong despite having reasonably relevant evidence

70 errors from SciFact (35 each from the dense and dense-plus-rerank conditions) are manually labelled into these categories as the primary failure analysis, with a second blind annotation pass giving an intra-annotator agreement of Cohen's kappa 0.914. For SciFact-Open, quantitative failure rates are compared across conditions without full manual annotation, which is an honest scoping decision for a solo project. The analysis examines which retrieval conditions produce which failure types, and whether stance reranking affects the first two categories in particular.

### Step 8: Retrieval-Aware Confidence Scoring

RoBERTa outputs a softmax probability distribution, and the highest probability is used as a confidence score. The analysis looks at whether low-confidence predictions are more likely to be wrong, whether stance reranking changes the confidence–correctness relationship, and whether a simple flagging rule (mark predictions below a threshold as unreliable) catches more errors.

### Step 9: Cross-Dataset Comparison

Once results are in for both datasets, the key questions are: does reranking help consistently as retrieval difficulty scales, do the failure indicators shift as the corpus grows roughly 100 times larger, and does the confidence–correctness pattern hold under harder retrieval. The main finding is that on the large corpus no retrieval configuration beats the no-retrieval baseline, so retrieval becomes counterproductive at scale rather than merely harder, and the confidence inversion on refutation claims holds on both corpora. Framing is retrieval-difficulty generalisation; domain generalisation is handled by Step 10.

### Step 10: Real-World Application

20 to 30 real seafood and sustainability claims from social media, run through the best pipeline. The analysis is qualitative: which failure types come up, whether the stance filter catches irrelevant evidence on out-of-domain claims, and whether the confidence score flags uncertain predictions appropriately. This tests whether the behaviour observed on scientific benchmarks transfers to a real-world domain.
 
  ---
 
  ## Models and Libraries
 
  - `transformers` + `roberta-base`: claim verification model
  - - `rank_bm25`: BM25 retrieval
    - - `sentence-transformers`: dense retrieval
      - - `cross-encoder/nli-deberta-v3-small`: stance-aware reranking
        - - `datasets`: dataset loading from Hugging Face
          - - `scikit-learn`: evaluation metrics
           
            - ---
 
  ## What Goes Beyond Prior Work
 
  | | MAPLE | Stammbach & Neumann (2019) | This project |
  |---|---|---|---|
  | Datasets | SciFact only | FEVER (Wikipedia) | SciFact + SciFact-Open |
  | Retrieval analysis | Performance drop noted | Single noisy vs filtered evidence comparison | Controlled matrix: k, method, threshold |
  | Stance-aware retrieval | Not done | Not done (supervised sentence ranker; entailment used only as verifier) | NLI filtering for scientific claims + full evaluation |
  | Failure taxonomy | Not done | Not done | Pre-defined 4-category taxonomy |
  | Confidence scoring | Not done | Not done | Retrieval-aware confidence signal |
  | Cross-dataset | Not possible | Not possible | Core findings compared across both |
  | Real-world domain | Not done | Not done | Seafood/sustainability social media claims |
  | Retrieval value at scale | Not studied | Not studied | Shown counterproductive on the large corpus |
 
  ---
 
  ## Status
 
  - [ x] Step 1: Dataset loading and preprocessing
  - [ x] Step 2: RoBERTa baseline
  - [ x] Step 3: BM25 + dense retrieval
  - [ x] Step 4: Stance-aware reranker
  - [ x] Step 5: Full RAG pipeline
  - [ x] Step 6: Experimental matrix
  - [ x] Step 7: Failure taxonomy and annotation
  - [ x] Step 8: Confidence scoring analysis
  - [ x] Step 9: Cross-dataset comparison
  - [ x] Step 10: Real-world case study
 