# RAG Claim Verification — MSc Thesis

**Empirical Analysis of Retrieval Effects and Failure Behaviour in RAG Models for Scientific Claim Verification**

> *MSc Thesis Project — supervised research*
> ---
>>  Author: Pernille Bergesen (MSc AI Student)
> 
>>  Supervisor: Arkaitz Zubiaga (Senior Lecturer in NLP)
> ---
>
> ## Overview
>
> Most fact-checking research assumes oracle evidence, meaning clean, perfectly retrieved documents handed to the model. This project looks at what actually happens under realistic retrieval conditions: when the system has to find its own evidence and sometimes gets it wrong.
>
> The goal is to systematically study how retrieval quality affects claim verification performance, understand why it fails, and check whether those findings hold across two datasets and a real-world domain. The main technical contribution is a stance-aware reranking step: after standard retrieval, a zero-shot NLI model scores whether each retrieved document actually takes a *stance* on the claim. Documents that are topically related but say nothing specific about the claim get filtered out before verification.
>
> **The one-sentence version of the thesis:**
> > *Topical similarity is not enough for fact-checking evidence selection. Stance-aware reranking using zero-shot NLI reduces two of the four failure categories we identify, produces better-calibrated model confidence, and this pattern holds across two scientific claim datasets and out-of-domain real-world claims.*
> >
> > ---
> >
> > ## Project Timeline
> >
> > ```mermaid
> > gantt
> >     title MSc Thesis Project Timeline
> >     dateFormat  YYYY-MM-DD
> >     axisFormat  %d %b
> >
> >     section 1. Literature Review
> >     Background Research           :lit1, 2026-05-25, 2026-06-08
> >     Related Work (Fact-checking/RAG) :lit2, 2026-05-29, 2026-06-15
> >
> >     section 2. Data Preparation
> >     Dataset Download & Exploration of SciFact :data1, 2026-05-28, 2026-06-10
> >     Data Cleaning & Formatting    :data2, 2026-06-04, 2026-06-19
> >
> >     section 3. Baseline Model
> >     Model Setup (RoBERTa)         :base1, 2026-06-10, 2026-06-22
> >     Training & Evaluation         :base2, 2026-06-17, 2026-06-30
> >
> >     section 4. Retrieval Module
> >     BM25 Implementation           :ret1, 2026-06-15, 2026-06-26
> >     Dense Retrieval Implementation :ret2, 2026-06-19, 2026-07-06
> >
> >     section 5. RAG System
> >     Integration Claim             :rag1, 2026-06-29, 2026-07-08
> >     Integration Evidence          :rag2, 2026-06-29, 2026-07-10
> >
> >     section 6. Evaluation & Analysis
> >     Initial Evaluation (for draft) :eval1, 2026-07-02, 2026-07-17
> >     Extended Experiments          :eval2, 2026-07-10, 2026-07-24
> >     Error & Failure Analysis      :eval3, 2026-07-17, 2026-07-29
> >
> >     section 7. Real-world Application
> >     Model behaviour on real-world scenarios :rw1, 2026-07-20, 2026-07-30
> >
> >     section 8. Writing
> >     Draft Writing                 :write1, 2026-06-25, 2026-07-17
> >     Final Writing: Refinement     :write2, 2026-07-08, 2026-08-11
> >     Editing & Proofreading        :write3, 2026-08-05, 2026-08-12
> >
> >     section 9. Presentation
> >     Presentation Prep & Finishing :pres1, 2026-08-05, 2026-08-12
> > ```
> >
> > ---
> >
> > ## Project Structure
> >
> > ```
> > rag-claim-verification/
> > ├── data/                    #dataset loading and preprocessing
> > │   ├── scifact/             #SciFact (primary dataset)
> > │   └── sciclaimhunt/        #SciClaimHunt (secondary dataset)
> > ├── models/
> > │   ├── baseline.py          #RoBERTa no-retrieval baseline
> > │   ├── retrieval.py         #BM25 + dense retrieval
> > │   ├── reranker.py          #stance-aware reranking via NLI
> > │   └── pipeline.py          #full RAG pipeline
> > ├── experiments/
> > │   ├── run_experiments.py   #controlled experimental matrix
> > │   └── configs/             #experiment config files
> > ├── analysis/
> > │   ├── failure_taxonomy.py  #failure category annotation and analysis
> > │   ├── confidence.py        #retrieval-aware confidence scoring
> > │   └── cross_dataset.py     #cross-dataset comparison
> > ├── realworld/
> > │   └── seafood_claims.py    #real-world seafood/sustainability case study
> > ├── results/                 #output tables and figures
> > ├── notebooks/               #exploratory analysis notebooks
> > ├── requirements.txt
> > └── README.md
> > ```
> >
> > ---
> >
> > ## Methodology
> >
> > ### Step 1: Datasets
> > The primary dataset is SciFact, which contains scientific claims verified against paper abstracts. Retrieval is genuinely hard here, which is exactly what makes it interesting. SciClaimHunt is used as a secondary dataset with a similar structure, so that results can be compared across corpora rather than just reported for one.
> >
> > Core experiments run on both datasets. Manual failure annotation is done on SciFact only.
> >
> > ### Step 2: Baseline Model
> > A RoBERTa model trained to verify claims without any retrieved evidence. Evaluated on both datasets, reporting F1, precision, and recall. This is the reference point that everything else is measured against.
> >
> > ### Step 3: Evidence Retrieval
> > Two retrieval methods are implemented:
> > - BM25: keyword-based sparse retrieval
> > - - Dense retrieval: semantic similarity via sentence-transformers
> >  
> >   - ### Step 4: Stance-Aware Reranking *(the novel bit)*
> >   - After standard retrieval, a filtering step using `cross-encoder/nli-deberta-v3-small` (Hugging Face) scores each retrieved document for entailment, contradiction, or neutral. Neutral documents are filtered out or downranked — only documents that actually take a stance on the claim get passed to the verifier.
> >  
> >   - The idea is that topical similarity is not enough. A document about omega-3 and cardiovascular health might be retrieved for a related claim but say nothing specific about it. The stance filter catches this.
> >  
> >   - | Retrieval condition | What it does |
> >   - |---|---|
> >   - | BM25 | keyword baseline, no filtering |
> >   - | Dense | semantic similarity baseline, no filtering |
> >   - | Dense + stance reranking | semantic retrieval filtered by NLI stance scores |
> >
> >   - Two filter thresholds are tested: loose and strict.
> >
> >   - ### Step 5: RAG Pipeline
> >   - Four pipeline variants: no retrieval / BM25 + RoBERTa / Dense + RoBERTa / Dense + stance reranking + RoBERTa
> >
> >   - ### Step 6: Controlled Experimental Matrix
> >
> >   - | Variable | Values tested |
> >   - |---|---|
> >   - | Retrieval condition | BM25, dense, dense + stance |
> >   - | k (number of docs retrieved) | 1, 5, 10 |
> >   - | Stance filter threshold | loose, strict |
> >
> >   - Metric: F1, precision, recall. Run on both datasets for the core conditions.
> >
> >   - ### Step 7: Failure Taxonomy
> >   - Four failure categories, defined before running any experiments:
> >
> >   - 1. Irrelevant retrieval: retrieved documents are not really about the claim
> >     2. 2. Contradictory retrieval: retrieved documents argue against the correct label
> >        3. 3. Evidence overload: too many documents confuse the model
> >           4. 4. Confident wrong prediction: model is wrong despite having reasonably relevant evidence
> >             
> >              5. 50–75 errors from SciFact are manually labelled into these categories. For SciClaimHunt, quantitative failure rates are compared across conditions without full annotation.
> >             
> >              6. ### Step 8: Retrieval-Aware Confidence Scoring
> >              7. RoBERTa outputs a softmax probability distribution. The highest probability is used as a confidence score. The analysis looks at whether low-confidence predictions are more likely to be wrong, whether stance reranking produces better-calibrated confidence, and whether a simple flagging rule (mark predictions below a threshold as unreliable) actually catches more errors.
> >             
> >              8. ### Step 9: Cross-Dataset Comparison
> > Once results are in for both datasets, the key questions are: does stance reranking help on both? Are the dominant failure categories similar? Does the confidence-correctness pattern hold? Either consistent results or dataset-specific divergence is a valid and interesting outcome.
> >
> > ### Step 10: Real-World Application
> > 20–30 real seafood and sustainability claims from social media, run through the best pipeline. The analysis is qualitative: which failure types come up, does the stance filter catch irrelevant evidence on out-of-domain claims, and does the confidence score flag uncertain predictions appropriately.
> >
> > ---
> >
> > ## Models and Libraries
> >
> > - `transformers` + `roberta-base`: claim verification model
> > - - `rank_bm25`: BM25 retrieval
> >   - - `sentence-transformers`: dense retrieval
> >     - - `cross-encoder/nli-deberta-v3-small`: stance-aware reranking
> >       - - `datasets`: dataset loading from Hugging Face
> >         - - `scikit-learn`: evaluation metrics
> >          
> >           - ---
> >
> > ## What Goes Beyond Prior Work
> >
> > | | MAPLE | Stammbach & Neumann (2019) | This project |
> > |---|---|---|---|
> > | Datasets | SciFact only | Health claims only | SciFact + SciClaimHunt |
> > | Retrieval analysis | Performance drop noted | Not systematic | Controlled matrix: k, method, threshold |
> > | Stance-aware retrieval | Not done | NLI filtering for health claims | NLI filtering for scientific claims + full evaluation |
> > | Failure taxonomy | Not done | Not done | Pre-defined 4-category taxonomy |
> > | Confidence scoring | Not done | Not done | Retrieval-aware confidence signal |
> > | Cross-dataset | Not possible | Not possible | Core findings compared across both |
> > | Real-world domain | Not done | Not done | Seafood/sustainability social media claims |
> >
> > ---
> >
> > ## Status
> >
> > - [ x] Step 1: Dataset loading and preprocessing
> > - [ x] Step 2: RoBERTa baseline
> > - [ x] Step 3: BM25 + dense retrieval
> > - [ x] Step 4: Stance-aware reranker
> > - [ ] Step 5: Full RAG pipeline
> > - [ ] Step 6: Experimental matrix
> > - [ ] Step 7: Failure taxonomy and annotation
> > - [ ] Step 8: Confidence scoring analysis
> > - [ ] Step 9: Cross-dataset comparison
> > - [ ] Step 10: Real-world case study
> >
> >
> >
