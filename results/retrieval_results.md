# Step 3 Results: Evidence Retrieval (BM25 & Dense)

This document records the Step 3 retrieval evaluation. Two standard retrieval
methods — BM25 (sparse keyword) and dense (semantic) — are run over each dataset's
evidence corpus, and Recall@k measures how often the gold evidence document appears
in the top-k retrieved. These retrievers are the baselines the stance reranker
(Step 4) builds upon.

## Setup

| Property | Value |
|---|---|
| Sparse retriever | BM25 (rank-bm25) |
| Dense retriever | sentence-transformers/all-MiniLM-L6-v2 (general-purpose) |
| Candidates retrieved per claim | top 10 |
| Metric | Recall@k (k = 1, 5, 10) |
| NEI claims | excluded from Recall@k (no gold evidence document to retrieve) |
| Device | CUDA (Google Colab GPU) |

Recall@k is computed only over claims that have annotated evidence documents, since
a claim with no gold document cannot contribute to a retrieval recall measurement.

## SciFact (primary)

- Corpus: 5,183 abstracts
- Validation claims: 450 (all with evidence; 0 NEI in this split)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 | 0.487 | 0.676 | 0.758 |
| Dense | 0.562 | 0.809 | 0.851 |

Dense retrieval clearly outperforms BM25 at every k, consistent with the general
finding that semantic matching beats keyword overlap for claim–evidence retrieval.

## SciFact-Open (secondary, large-corpus)

- Corpus: 500,000 abstracts
- Claims: 279 (206 with evidence, 73 NEI excluded from Recall@k)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 | 0.403 | 0.578 | 0.650 |
| Dense | 0.369 | 0.641 | 0.762 |

## Interpretation

**Retrieval degrades as the corpus scales.** Moving from SciFact's ~5K corpus to
SciFact-Open's 500K corpus, dense R@10 falls from 0.851 to 0.762 and BM25 R@10 from
0.758 to 0.650. Both retrievers lose ground as the search space grows roughly 100×,
providing direct quantitative evidence for the project's central premise that
retrieval becomes harder at scale. The degradation is meaningful but moderate —
dense retrieval still recovers the gold document within the top 10 for 76% of
evidenced claims even against 500K documents — so the honest characterisation is a
robust-but-degrading retriever, not a collapse.

**Dense's top-rank advantage does not hold at scale.** On SciFact, dense beats BM25
at every k, including R@1 (0.562 vs 0.487). On SciFact-Open, this reverses at R@1:
BM25 (0.403) outperforms dense (0.369) for the single top-ranked document, while
dense retains its advantage at R@5 and R@10. This suggests that dense retrieval's
precision at the very top rank suffers disproportionately when the corpus is large,
an observation worth carrying into the failure analysis (Step 7).

**Choice of retriever for downstream steps.** The dense retriever's candidates are
used as the input pool for stance reranking (Step 4), as dense achieves the higher
Recall@k overall and provides the semantically-related-but-not-necessarily-stance-
bearing documents the stance filter is designed to refine.

## Retriever robustness check (all-mpnet-base-v2)

To test whether the findings above depend on the specific dense model, retrieval was
additionally run with a stronger general-purpose dense model, all-mpnet-base-v2
(768-dim embeddings), saved separately to avoid overwriting the MiniLM results.

*(Results pending — to be filled once the mpnet run completes. Key question: does
mpnet close the R@1 gap on SciFact-Open where MiniLM dense underperformed BM25?)*

## Files

- `results/retrieval_recall_scifact.json`, `results/retrieval_recall_scifact_open.json`
  — Recall@k summaries
- `results/retrieval_candidates_scifact.json`, `results/retrieval_candidates_scifact_open.json`
  — retrieved candidates (input to Step 4)
- mpnet variants saved with `_mpnet` suffix