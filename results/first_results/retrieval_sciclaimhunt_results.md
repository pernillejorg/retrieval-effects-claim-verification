# Step 3: Retrieval Evaluation for SciClaimHunt Results

## Overview

This file records BM25 and dense retrieval Recall@k results on the SciClaimHunt validation split.
Retrieval is evaluated by checking whether the correct evidence document appears in the top-k
retrieved documents for each claim.

## Dataset

| Split      | Claims | Corpus Size |
|------------|--------|-------------|
| Validation | 10,872 | 10,872 docs |

- All 10,872 validation claims have annotated evidence (no NEI class in SciClaimHunt)
- Each claim maps to exactly one synthetic evidence passage
- Corpus is constructed from the Evidence field of each row, keyed by row index

## Retrieval Models

| Method | Description |
|--------|-------------|
| BM25   | Sparse retrieval using BM25Okapi (rank-bm25), whitespace tokenisation |
| Dense  | sentence-transformers/all-MiniLM-L6-v2, cosine similarity, normalised embeddings |

## Recall@k Results

| Method | R@1   | R@5   | R@10  |
|--------|-------|-------|-------|
| BM25   | 0.312 | 0.474 | 0.507 |
| Dense  | 0.344 | 0.527 | 0.567 |

## Comparison with SciFact

| Dataset      | Method | R@1   | R@5   | R@10  |
|--------------|--------|-------|-------|-------|
| SciFact      | BM25   | 0.595 | 0.730 | 0.758 |
| SciFact      | Dense  | 0.703 | 0.838 | 0.851 |
| SciClaimHunt | BM25   | 0.312 | 0.474 | 0.507 |
| SciClaimHunt | Dense  | 0.344 | 0.527 | 0.567 |

## Analysis

- Dense retrieval outperforms BM25 on both datasets, consistent with the literature.
- SciClaimHunt recall is substantially lower than SciFact across all k values and both methods.
- This reflects the harder retrieval setting in SciClaimHunt: the corpus and claims are synthetically
  generated with more abstract relationships, reducing lexical overlap and making retrieval harder.
- In SciFact, claims are written directly about specific abstracts, creating stronger keyword overlap.
- The lower recall on SciClaimHunt indicates that retrieval is a meaningful bottleneck on this dataset,
  making it a more challenging and informative testbed for RAG analysis.

## Notes

- No NEI claims exist in SciClaimHunt; all 10,872 validation claims are included in Recall@k computation.
- Results saved to Drive: `retrieval_recall_sciclaimhunt.json`