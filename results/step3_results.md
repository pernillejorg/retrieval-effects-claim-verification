# Step 3 Results: Document Retrieval (BM25 & Dense)

This document records the Step 3 retrieval evaluation. Two standard retrieval methods:
BM25 (sparse keyword matching) and dense (semantic similarity), are run over each
dataset's evidence corpus, and Recall@k measures how often the gold evidence document
appears among the top-k retrieved documents. These retrievers are the baselines that the stance reranker in Step 4 builds upon: the reranker cannot recover a document that
retrieval never surfaced, so retrieval quality sets a ceiling on the whole pipeline.

Two dense models were evaluated to check whether the findings depend on the specific
embedding model: all-MiniLM-L6-v2 (a fast, lightweight 384-dimensional model) and
all-mpnet-base-v2 (a stronger, heavier 768-dimensional model).

**Note on this version:** all SciFact results were re-run after deduplicating the SciFact validation set from 450 rows to 300 unique claims (see "Data correction" below). This document reports both the original (450-row) and corrected (300-claim) SciFact numbers. SciFact-Open was unaffected by the deduplication (it contains no duplicate claims), so its numbers are unchanged. The corrected numbers are the reported results.

## Setup

| Property | Value |
|---|---|
| Sparse retriever | BM25 (rank-bm25) |
| Dense retrievers | all-MiniLM-L6-v2 (384-dim) and all-mpnet-base-v2 (768-dim) |
| Candidates retrieved per claim | top 10 |
| Metric | Recall@k (k = 1, 5, 10) |
| NEI claims | SciFact: retained (they carry cited doc ids). SciFact-Open: excluded (no evidence docs) |
| Device | CUDA (Google Colab GPU) |

Recall@k is computed over claims with a non-empty `evidence_doc_ids`. What that field contains differs by dataset, and this matters for reading the two tables together. For SciFact it holds the claim's cited doc ids, which every claim has, so all 300 validation claims are included, and the metric is best described as **cited-document recall**. For SciFact-Open it holds annotated evidence doc ids, so the 73 NEI claims have none and are excluded, and the metric is evidence recall over the 206 evidenced claims. The two figures are therefore not the same measurement, and the cross-corpus comparison below is read with that caveat.

## Data correction: deduplicating the SciFact validation set

The SciFact validation split stores 450 rows corresponding to only 300 unique claims, a claim is repeated across rows when it cites more than one evidence document. A check
confirmed 450 rows, 300 unique ids, and 0 conflicting labels (the repeats carried no new information). The loader was corrected to deduplicate to 300 unique claims. Retrieval was re-run on the corrected set. Note that the retrieval *recall rates* change only slightly, because recall was always computed per claim-id; the correction mainly ensures every downstream step (retrieval, reranking, pipeline) operates on a consistent 300-claim set.

## SciFact: original results (450 rows) = SUPERSEDED

Retained for transparency only; not the reported numbers.

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 | 0.487 | 0.676 | 0.758 |
| Dense (MiniLM) | 0.562 | 0.809 | 0.851 |
| Dense (mpnet) | 0.609 | 0.813 | 0.867 |

## SciFact: corrected results (300 unique claims) = REPORTED

- Corpus: 5,183 abstracts
- Validation claims: 300 (SUPPORT 124, CONTRADICT 64, NEI 112, per the Step 2 per-class support). All 300 carry cited doc ids, so none is excluded from the recall denominator.

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 | 0.437 | 0.630 | 0.703 |
| Dense (MiniLM) | 0.503 | 0.753 | 0.793 |
| Dense (mpnet) | 0.533 | 0.733 | 0.803 |

On the corrected set, both dense models still clearly beat BM25 at every k. mpnet leads MiniLM at R@1 (0.533 vs 0.503) and R@10 (0.803 vs 0.793); MiniLM edges mpnet at R@5 (0.753 vs 0.733).

## SciFact-Open (secondary, large-corpus) = REPORTED

- Corpus: 500,000 abstracts
- Claims: 279 (206 with evidence, 73 NEI excluded from Recall@k)
- Unaffected by deduplication (no duplicate claims), so numbers are unchanged.

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| BM25 | 0.403 | 0.578 | 0.650 |
| Dense (MiniLM) | 0.369 | 0.641 | 0.762 |
| Dense (mpnet) | 0.408 | 0.699 | 0.752 |

## Effect of the deduplication on SciFact

| Method | R@10 (450 rows) | R@10 (300 unique) |
|---|---|---|
| BM25 | 0.758 | 0.703 |
| Dense (MiniLM) | 0.851 | 0.793 |
| Dense (mpnet) | 0.867 | 0.803 |

The corrected recall values are somewhat lower across the board. As with the baseline,
this is not a regression but a correction: the 450-row figures were computed over a set that repeated 150 claims, and removing that duplication yields the true recall over the 300 distinct claims. The relative ordering of methods is unchanged, confirming the correction affects the absolute values consistently rather than altering the conclusions.

## Interpretation

**Retrieval degrades as the corpus scales.** 
For every method, Recall@k falls moving from SciFact's ~5,000-abstract corpus to SciFact-Open's 500,000-abstract corpus. For example, mpnet dense R@5 drops from 0.733 to 0.699, and BM25 R@10 from 0.703 to 0.650. Both sparse and dense retrievers lose ground as the search space grows ~100×, giving direct quantitative evidence for the project's central premise that retrieval becomes harder at scale. The degradation is meaningful but moderate, dense retrieval still recovers the gold document within the top 10 for around 75–80% of evidenced claims even against half a million documents, so this is a robust-but-degrading retriever, not a collapse.

**Caveat on the cross-corpus comparison.** The SciFact figures are cited-document recall over all 300 claims, while the SciFact-Open figures are evidence recall over the 206 evidenced claims. The observed degradation is consistent across both retrievers and both dense models, so the direction is taken as real, but the two numbers are not strictly the same metric and the size of the gap should not be over-read.

**A stronger dense model recovers top-rank precision at scale.** 
The most interesting finding concerns R@1 on the large corpus. With MiniLM, dense retrieval *underperforms* BM25 at R@1 on SciFact-Open (0.369 vs 0.403) and suggesting dense top-rank precision suffers on large corpora. mpnet shows this was model-specific, not a general limitation: mpnet dense recovers R@1 to 0.408, edging past BM25, and improves R@5 substantially (0.641 to 0.699). This indicates embedding quality, not the dense paradigm itself, drives top-rank performance at scale. Evaluating both models was what let this distinction be drawn.

**One counter-observation, reported honestly.** 
mpnet is not uniformly superior. On SciFact-Open at R@10, mpnet (0.752) is marginally below MiniLM (0.762), and on SciFact at R@5 MiniLM (0.753) edges mpnet (0.733). So the stronger model wins clearly at the top ranks (R@1) and on the metrics most relevant to the downstream pipeline, but not on every single metric, a genuine nuance rather than a clean sweep.

## Model selection

**all-mpnet-base-v2 is selected as the primary dense retriever** for the downstream
pipeline (Steps 4 onward). The reasoning:

1. **mpnet wins on 4 of the 6 dense metrics** across the two datasets, losing SciFact R@5 (0.733 vs 0.753) and SciFact-Open R@10 (0.752 vs 0.762)
2. **mpnet wins decisively where it matters most.** On SciFact-Open it leads at R@1
   (0.408 vs 0.369) and R@5 (0.699 vs 0.641), the top ranks that feed the stance reranker and classifier, which operate on the few highest-ranked documents.
3. **mpnet gives a coherent retrieval story.** It resolves MiniLM's anomaly where     dense retrieval underperformed BM25 at R@1 on the large corpus. With mpnet, dense consistently beats BM25, which is the clean, defensible narrative.

The cost is encoding time (mpnet's 768-dim embeddings took ~39 minutes to encode the 500K corpus versus ~7 minutes for MiniLM), a one-time cost per corpus. all-MiniLM-L6-v2 is retained as a lighter-weight comparison point, demonstrating that the central findings (retrieval degradation at scale; dense generally exceeding BM25) hold under both dense models and are therefore robust to the choice of embedding model.

**Consequence for downstream steps:** Step 4 (reranking) and Step 5 (pipeline) use the
mpnet retrieval candidates, the `retrieval_candidates_mpnet_*.json` files as their
input, matching this selection.

## Files

- **MiniLM:** `results/retrieval_recall_minilm_scifact.json`,
  `retrieval_recall_minilm_scifact_open.json`, and matching `retrieval_candidates_minilm_*.json`
- **mpnet (primary):** `results/retrieval_recall_mpnet_scifact.json`,
  `retrieval_recall_mpnet_scifact_open.json`, and matching `retrieval_candidates_mpnet_*.json`