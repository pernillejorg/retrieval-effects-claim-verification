# Retrieval Results: BM25 and Dense Retrieval for SciFact

## Experiment Overview

| Property | Value |
|---|---|
| Dataset | SciFact (primary) |
| Split evaluated | Validation |
| Corpus size | 5,183 abstracts |
| Validation claims | 450 |
| Claims with annotated evidence | 450 (all) |
| NEI claims excluded from Recall@k | 0 |
| Device | CUDA (Google Colab A100) |

---

## Retrieval Methods

| Method | Description |
|---|---|
| BM25 | Sparse keyword-based retrieval using rank-bm25 (BM25Okapi). Documents tokenised by whitespace. Robertson & Zaragoza (2009). |
| Dense | Semantic similarity retrieval using sentence-transformers/all-MiniLM-L6-v2. Corpus encoded upfront; cosine similarity at retrieval time. Reimers & Gurevych (2019). |

---

## Recall@k Results

Recall@k measures the fraction of claims for which at least one gold evidence document appears in the top-k retrieved documents. Higher is better.

| Method | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| BM25 | 0.487 | 0.676 | 0.758 |
| **Dense** | **0.562** | **0.809** | **0.851** |

Dense retrieval outperforms BM25 at every k value.

---

## Exact Values (from saved JSON)

```json
{
  "bm25": {
    "1": 0.4867,
    "5": 0.6756,
    "10": 0.7578
  },
  "dense": {
    "1": 0.5622,
    "5": 0.8089,
    "10": 0.8511
  }
}
```

---

## Interpretation

- **Dense retrieval is consistently better than BM25** across all k values. At k=10, dense retrieval finds at least one correct evidence document for 85.1% of claims, compared to 75.8% for BM25.
- **The gap widens with k** — at k=1 the difference is 7.5 percentage points; at k=10 it is 9.3 points. This suggests dense retrieval ranks correct documents higher on average, not just that it finds more of them.
- **Why dense beats BM25 on SciFact:** Scientific claims use precise vocabulary that does not always match the exact words in paper abstracts. Semantic similarity captures meaning beyond keyword overlap, which is critical for this domain.
- **Recall@10 = 0.851 for dense retrieval** means the RAG pipeline (Step 5) has access to at least one correct evidence document for 85% of claims. The remaining 15% represent cases where retrieval fails entirely, these will appear prominently in the failure taxonomy (Step 7).
- **These numbers are the retrieval upper bound** for your pipeline. Your RAG model cannot verify a claim correctly if retrieval fails to include any relevant evidence. The gap between retrieval recall and final F1 tells you how much the verification model loses even when evidence is available.

---

## Configuration

| Parameter | Value |
|---|---|
| Dense model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| Encoding batch size | 64 |
| k values evaluated | 1, 5, 10 |
| BM25 tokenisation | Whitespace, lowercased |
| Similarity metric | Cosine similarity (normalised dot product) |

---

## Next step

These retrieval results feed directly into Step 4 (stance-aware reranking). The candidate pool at k=10 from dense retrieval is passed to the NLI-based stance filter, which removes neutral documents before passing to the verification model.