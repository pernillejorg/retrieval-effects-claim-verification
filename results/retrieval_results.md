# Step 3 Results: Evidence Retrieval (BM25 & Dense)

This document records the Step 3 retrieval evaluation. Two standard retrieval methods —
BM25 (sparse keyword matching) and dense (semantic similarity) — are run over each
dataset's evidence corpus, and Recall@k measures how often the gold evidence document
appears among the top-k retrieved documents. These retrievers are the baselines that
the stance reranker in Step 4 builds upon: the reranker cannot recover a document that
retrieval never surfaced, so retrieval quality sets a ceiling on the whole pipeline.

Two dense models were evaluated to check whether the findings depend on the specific
embedding model: all-MiniLM-L6-v2 (a fast, lightweight 384-dimensional model) and
all-mpnet-base-v2 (a stronger, heavier 768-dimensional model).

## Setup

| Property | Value |
|---|---|
| Sparse retriever | BM25 (rank-bm25) |
| Dense retrievers | all-MiniLM-L6-v2 (384-dim) and all-mpnet-base-v2 (768-dim) |
| Candidates retrieved per claim | top 10 |
| Metric | Recall@k (k = 1, 5, 10) |
| NEI claims | excluded from Recall@k (no gold evidence document to retrieve) |
| Device | CUDA (Google Colab GPU) |

Recall@k is computed only over claims that have annotated evidence documents. A claim
labelled NEI (Not Enough Information) has no gold evidence document in the corpus, so it
cannot contribute to a retrieval recall measurement and is excluded from the denominator.

## SciFact (primary dataset)

- Corpus: 5,183 abstracts
- Validation claims: 450 (all with evidence; 0 NEI in this split)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 | 0.487 | 0.676 | 0.758 |
| Dense (MiniLM) | 0.562 | 0.809 | 0.851 |
| Dense (mpnet) | 0.609 | 0.813 | 0.867 |

On SciFact, both dense models clearly outperform BM25 at every k, consistent with the
general finding that semantic matching beats keyword overlap for claim–evidence
retrieval. mpnet improves on MiniLM at every k, most noticeably at R@1 (0.609 vs 0.562).

## SciFact-Open (secondary dataset, large-corpus)

- Corpus: 500,000 abstracts
- Claims: 279 (206 with evidence, 73 NEI excluded from Recall@k)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 | 0.403 | 0.578 | 0.650 |
| Dense (MiniLM) | 0.369 | 0.641 | 0.762 |
| Dense (mpnet) | 0.408 | 0.699 | 0.752 |

## Interpretation

**Retrieval degrades as the corpus scales.** For every method, Recall@k falls when
moving from SciFact's ~5,000-abstract corpus to SciFact-Open's 500,000-abstract corpus.
For example, mpnet dense R@5 drops from 0.813 to 0.699, and BM25 R@10 drops from 0.758
to 0.650. Both sparse and dense retrievers lose ground as the search space grows roughly
100×, which provides direct quantitative evidence for the project's central premise:
retrieval becomes harder at scale. Importantly, the degradation is meaningful but
moderate — dense retrieval still recovers the gold document within the top 10 for roughly
75% of evidenced claims even against half a million documents. The honest characterisation
is therefore a robust-but-degrading retriever, not a collapse. This nuance matters, because
overstating the effect as a "collapse" would misrepresent the data.

**A stronger dense model recovers top-rank precision at scale.** The most interesting
finding concerns R@1 on the large corpus. With MiniLM, dense retrieval unexpectedly
*underperformed* BM25 at R@1 on SciFact-Open (0.369 versus 0.403) — suggesting that dense
retrieval's precision at the single top rank suffers disproportionately when the corpus is
large. The mpnet results show this was model-specific rather than a general limitation of
dense retrieval: mpnet dense recovers R@1 to 0.408, edging past BM25, and improves R@5
substantially (0.641 to 0.699). This indicates that embedding quality — not the dense
retrieval paradigm itself — drives top-rank performance at scale. Running both models was
what allowed this distinction to be drawn; a single model would have left the R@1 result
ambiguous.

**One counter-observation, reported honestly.** mpnet is not uniformly superior. On
SciFact-Open at R@10, mpnet (0.752) is marginally *below* MiniLM (0.762). So the stronger
model improves precision at the top ranks (R@1 and R@5) but not at R@10 on the large
corpus. This is a small but genuine nuance rather than a clean across-the-board win, and
it is reported rather than hidden because selectively presenting only the favourable
metrics would misrepresent the comparison.

## Model selection

Both dense models are valid, standard general-purpose retrievers, and neither is a
domain-specialised scientific model — this is stated plainly so the dense retriever is not
oversold as domain-specific. **all-mpnet-base-v2 is selected as the primary dense retriever
for the downstream pipeline** (Steps 4 onward), for two reasons. First, it improves Recall@1
and Recall@5 on both datasets — and these top ranks are the ones most relevant to the stance
reranker, which operates only on the top candidates, so improvements at R@1/R@5 matter more
to the pipeline than the marginal R@10 difference. Second, it resolves the anomalous R@1
underperformance seen with MiniLM, giving a more coherent and defensible retrieval story.

The cost of this choice is encoding time: mpnet's 768-dimensional embeddings took roughly
39 minutes to encode the 500,000-document corpus, versus about 7 minutes for MiniLM's
384-dimensional embeddings. This is a one-time cost per corpus (the encoded vectors are
reused for all subsequent retrievals), so it is an acceptable trade for the retrieval-quality
gain. all-MiniLM-L6-v2 is retained and reported as a lighter-weight comparison point. Its
value is scientific rather than merely practical: because the project's central findings —
retrieval degradation at scale, and dense retrieval generally exceeding BM25 — hold under
*both* dense models, those findings are shown to be robust properties of the task rather than
artifacts of a single embedding model.

*(Consistency note: because mpnet is the primary retriever, Step 4 stance reranking is run on
the mpnet candidate files, i.e. `retrieval_candidates_mpnet_*.json`. The reranker and all
downstream steps use the same retriever as this selection, so the pipeline is internally
consistent.)*

## Files

- **MiniLM:** `results/retrieval_recall_scifact.json`,
  `results/retrieval_recall_scifact_open.json`, and matching
  `results/retrieval_candidates_*.json`
- **mpnet:** `results/retrieval_recall_mpnet_scifact.json`,
  `results/retrieval_recall_mpnet_scifact_open.json`, and matching
  `results/retrieval_candidates_mpnet_*.json`