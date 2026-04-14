# RAG Claim Verification — MSc Thesis

**Empirical Analysis of Retrieval Effects and Failure Behaviour in RAG Models for Scientific Claim Verification**

> *MSc Thesis Project — supervised research*
>
> ---
>
> ## Overview
>
> Most fact-checking research assumes oracle evidence — clean, perfectly retrieved documents handed to the model. This project studies what happens under **realistic retrieval conditions**: when the system must find its own evidence, and sometimes gets it wrong.
>
> The project systematically analyses how retrieval quality affects claim verification performance, categorises why it fails, and shows these findings generalise across two datasets and into a real-world domain. It introduces **stance-aware reranking** as a novel retrieval filtering step: after standard retrieval, a zero-shot NLI model scores whether each retrieved document actually takes a *stance* on the claim. Neutral (topically similar but non-committal) documents are filtered out before verification.
>
> **Core thesis claim:**
> > *Topical similarity is insufficient for fact-checking evidence selection. Stance-aware reranking using zero-shot NLI reduces two of our four failure categories systematically, produces better-calibrated model confidence, and the pattern generalises across two scientific claim datasets and into out-of-domain real-world claims.*
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
> > ├── data/                    # Dataset loading and preprocessing
> > │   ├── scifact/             # SciFact dataset (primary)
> > │   └── sciclaimhunt/        # SciClaimHunt dataset (secondary)
> > ├── models/
> > │   ├── baseline.py          # RoBERTa no-retrieval baseline
> > │   ├── retrieval.py         # BM25 + dense retrieval
> > │   ├── reranker.py          # Stance-aware reranking (NLI)
> > │   └── pipeline.py          # Full RAG pipeline
> > ├── experiments/
> > │   ├── run_experiments.py   # Controlled experimental matrix
> > │   └── configs/             # Experiment configuration files
> > ├── analysis/
> > │   ├── failure_taxonomy.py  # Failure category annotation & analysis
> > │   ├── confidence.py        # Retrieval-aware confidence scoring
> > │   └── cross_dataset.py     # Cross-dataset comparison
> > ├── realworld/
> > │   └── seafood_claims.py    # Real-world seafood/sustainability case study
> > ├── results/                 # Output tables and figures
> > ├── notebooks/               # Exploratory analysis notebooks
> > ├── requirements.txt
> > └── README.md
> > ```
> >
> > ---
> >
> > ## Methodology
> >
> > ### Step 1 — Datasets
> > - **Primary:** SciFact — scientific claims verified against paper abstracts
> > - - **Secondary:** SciClaimHunt — a more recent dataset with similar structure
> >   - - Core experiments run on both; failure annotation done primarily on SciFact
> >    
> >     - ### Step 2 — Baseline Model
> >     - RoBERTa-based claim verification without any retrieved evidence. Trained and evaluated on both datasets. Records F1, precision, recall. Establishes what the model knows before retrieval is introduced.
> >    
> >     - ### Step 3 — Evidence Retrieval
> >     - Two standard retrieval methods:
> >     - - **BM25** — classic keyword-based sparse retrieval
> > - **Dense retrieval** — sentence-transformers semantic similarity
> >
> > - ### Step 4 — Stance-Aware Reranking *(novel contribution)*
> > - After standard retrieval, a zero-shot NLI step filters the candidate pool. Uses `cross-encoder/nli-deberta-v3-small` (Hugging Face) to score entailment/contradiction/neutral for each retrieved document. Neutral documents are downranked or removed — only stance-bearing evidence passes to the verifier.
> >
> > - | Retrieval Condition | Description |
> > - |---|---|
> > - | BM25 | Keyword baseline, no filtering |
> > - | Dense | Semantic similarity baseline, no filtering |
> > - | Dense + Stance Reranking | Semantic retrieval filtered by NLI stance scores |
> >
> > - Two filter thresholds tested: **loose** and **strict**.
> >
> > - ### Step 5 — RAG Pipeline
> > - Four pipeline variants: No retrieval / BM25+RoBERTa / Dense+RoBERTa / Dense+Stance+RoBERTa
> >
> > - ### Step 6 — Controlled Experimental Matrix
> >
> > - | Variable | Values |
> > - |---|---|
> > - | Retrieval condition | BM25, dense, dense+stance |
> > - | k (docs retrieved) | 1, 5, 10 |
> > - | Stance filter threshold | Loose, strict |
> >
> > - Metric: F1, precision, recall. Run on both datasets for core conditions.
> >
> > - ### Step 7 — Failure Taxonomy
> > - Four pre-defined categories (defined *before* experiments):
> > - 1. **Irrelevant retrieval** — retrieved docs topically unrelated to the claim
> >   2. 2. **Contradictory retrieval** — retrieved docs argue against the correct label
> >      3. 3. **Evidence overload** — too many docs confuse or dilute the model
> >         4. 4. **Confident wrong prediction** — model wrong despite reasonably relevant retrieval
> >           
> >            5. Manual annotation of 50–75 errors from SciFact. Quantitative failure rate analysis on SciClaimHunt.
> >           
> >            6. ### Step 8 — Retrieval-Aware Confidence Scoring
> >            7. - Record softmax confidence scores across all pipeline variants
> >               - - Analyse whether low-confidence predictions correlate with errors
> > - Test whether stance reranking produces better-calibrated confidence
> > - - Simple flagging rule: predictions below threshold marked unreliable
> >  
> >   - ### Step 9 — Cross-Dataset Comparison
> >   - Compare key findings across SciFact and SciClaimHunt:
> >   - - Does stance reranking improve F1 consistently on both?
> >     - - Are dominant failure categories similar?
> >       - - Does confidence-correctness correlation hold on both?
> >        
> >         - ### Step 10 — Real-World Application
> >         - 20–30 real seafood and sustainability claims from social media, run through the best pipeline. Qualitative analysis of failure categories, stance filtering, and confidence behaviour on out-of-domain claims.
> >        
> >         - ---
> >
> > ## Models & Libraries
> >
> > - `transformers` + `roberta-base` — claim verification
> > - - `rank_bm25` — BM25 retrieval
> >   - - `sentence-transformers` — dense retrieval
> >     - - `cross-encoder/nli-deberta-v3-small` — stance-aware reranking
> >       - - `datasets` — dataset loading
> >         - - `scikit-learn` — evaluation metrics
> >          
> >           - ---
> >
> > ## What Goes Beyond Prior Work
> >
> > | | MAPLE | Stammbach & Neumann (2019) | This project |
> > |---|---|---|---|
> > | Datasets | SciFact only | Health claims only | SciFact + SciClaimHunt |
> > | Retrieval analysis | Performance drop noted | Not systematic | Controlled matrix: k, method, threshold |
> > | Stance-aware retrieval | Not done | NLI filtering (health) | NLI filtering (scientific) + full evaluation |
> > | Failure taxonomy | Not done | Not done | Pre-defined 4-category taxonomy |
> > | Confidence scoring | Not done | Not done | Retrieval-aware confidence signal |
> > | Cross-dataset | Not possible | Not possible | Core findings compared across both |
> > | Real-world domain | Not done | Not done | Seafood/sustainability social media claims |
> >
> > ---
> >
> > ## Status
> >
> > - [ ] Step 1 — Dataset loading & preprocessing
> > - [ ] - [ ] Step 2 — RoBERTa baseline
> > - [ ] - [ ] Step 3 — BM25 + dense retrieval
> > - [ ] - [ ] Step 4 — Stance-aware reranker
> > - [ ] - [ ] Step 5 — Full RAG pipeline
> > - [ ] - [ ] Step 6 — Experimental matrix
> > - [ ] - [ ] Step 7 — Failure taxonomy & annotation
> > - [ ] - [ ] Step 8 — Confidence scoring analysis
> > - [ ] - [ ] Step 9 — Cross-dataset comparison
> > - [ ] - [ ] Step 10 — Real-world case study
> >
> > - [ ] ---
> >
> > - [ ] ## License
> >
> > - [ ] Private repository — MSc thesis work in progress.
