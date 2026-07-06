# Step 4 Results: Stance-Aware Reranking

This document records Step 4, the stance-aware reranking stage, the project's novel
technical component. After dense retrieval (Step 3) produces a candidate pool of
documents for each claim, a stance reranker uses a zero-shot NLI model
(cross-encoder/nli-deberta-v3-small) to score whether each document takes a *stance*
on the claim (entailment or contradiction) rather than merely being topically related.
Documents are then either reordered (soft) or filtered (hard) by that stance score.

The central hypothesis was that topical similarity alone is insufficient for evidence
selection that filtering for stance-bearing documents would improve the quality of the candidate pool. **This step tests that hypothesis directly, and the result is a clear and well-mechanised negative finding: on scientific claims, stance reranking degrades retrieval rather than improving it.** This is a substantive result for a project on retrieval *failure behaviour*: the mechanism is diagnosed below with score-level evidence, and the failure mode it reveals is carried forward for formal categorisation in the Step 7 failure taxonomy.

## Setup

| Property | Value |
|---|---|
| Reranker model | cross-encoder/nli-deberta-v3-small (zero-shot NLI) |
| NLI label order | contradiction, entailment, neutral |
| Stance score | max(entailment probability, contradiction probability) |
| Input candidates | top-10 dense (mpnet) candidates from Step 3 |
| Thresholds | loose (neutral > 0.5), strict (neutral > 0.8) |
| Modes | soft (reorder, keep all), hard (remove docs above neutral threshold) |
| Metric | Recall@k (k = 1, 5, 10), vs dense-before-reranking |

Two modes are evaluated so the choice between them is justified empirically:
- **Soft reranking** reorders all documents by stance score but keeps them all.
- **Hard filtering** removes documents whose neutral probability exceeds the threshold.

## SciFact: Soft reranking

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.533 | 0.733 | 0.803 |
| Dense + stance rerank soft (loose) | 0.120 | 0.520 | 0.803 |
| Dense + stance rerank soft (strict) | 0.120 | 0.520 | 0.803 |

Avg docs surviving: 10.0 (both thresholds). Avg docs the threshold *would* filter: 9.4
(loose), 9.0 (strict).

Soft reranking keeps all 10 documents (R@10 is therefore unchanged at 0.803, the gold
document is still somewhere in the pool), but reordering by stance **collapses R@1 from 0.533 to 0.120**. In other words, stance reranking pushes the gold evidence document *down* the ranking. R@5 also drops (0.733 to 0.520) because reranking displaces gold documents from the top 5 into the 6–10 range, they remain in the pool (so R@10 is preserved) but no longer appear in the top 5. The loose and strict thresholds give identical results in soft mode, because soft mode never removes documents; the threshold only affects the (unused) filter flag, so it cannot change the ordering.

## SciFact: Hard filtering

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.533 | 0.733 | 0.803 |
| Dense + stance rerank hard (loose) | 0.060 | 0.077 | 0.077 |
| Dense + stance rerank hard (strict) | 0.080 | 0.133 | 0.133 |

Avg docs surviving: **0.6 (loose), 1.0 (strict)**.

Hard filtering is catastrophic. Because the NLI model flags ~9 of 10 documents as neutral, filtering removes almost the entire candidate pool: on average only **0.6 documents survive** the loose filter and **1.0 the strict filter**. Recall collapses accordingly, R@10 falls from 0.803 to 0.077 (loose). Note the counter-intuitive detail that the *strict* threshold retains slightly more documents and gives higher recall than loose here: the strict threshold (neutral > 0.8) removes only very-confidently-neutral documents, whereas loose (neutral > 0.5) removes anything leaning neutral, which on this near-all-neutral pool is almost everything.

## SciFact-Open: Soft reranking

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.408 | 0.699 | 0.752 |
| Dense + stance rerank soft (loose) | 0.126 | 0.490 | 0.752 |
| Dense + stance rerank soft (strict) | 0.126 | 0.490 | 0.752 |

Avg docs surviving: 10.0. Would-filter: 9.1 (loose), 8.8 (strict). The same pattern holds on the 500K-corpus dataset: R@10 preserved (0.752), but R@1 collapses from 0.408 to 0.126.

## SciFact-Open: Hard filtering

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.408 | 0.699 | 0.752 |
| Dense + stance rerank hard (loose) | 0.083 | 0.112 | 0.112 |
| Dense + stance rerank hard (strict) | 0.102 | 0.160 | 0.160 |

Avg docs surviving: 0.9 (loose), 1.2 (strict). Again the filter removes ~9 of 10 documents and recall collapses.

## Soft vs Hard: comparison

| Aspect | Soft reranking | Hard filtering |
|---|---|---|
| Documents kept | all 10 | ~0.6–1.2 |
| SciFact R@10 | 0.803 (preserved) | 0.077–0.133 (destroyed) |
| SciFact-Open R@10 | 0.752 (preserved) | 0.112–0.160 (destroyed) |
| Effect on R@1 | severe drop | severe drop |
| Usable downstream? | yes (preserves pool) | no (removes evidence) |

Both modes harm top-rank recall, but hard filtering additionally destroys the candidate pool by removing almost every document. Soft reranking at least preserves R@10 (the gold document remains retrievable within the pool), whereas hard filtering discards it entirely for most claims.

## Why this happens: diagnosed mechanism (with evidence)

Inspecting the raw NLI stance scores for a representative claim's retrieved documents
reveals the cause directly:

```
stance=0.977  ent=0.001  con=0.977  neu=0.023
stance=0.165  ent=0.165  con=0.001  neu=0.834
stance=0.016  ent=0.002  con=0.016  neu=0.982
stance=0.010  ent=0.001  con=0.010  neu=0.989
stance=0.009  ent=0.009  con=0.004  neu=0.987
```

The pattern is stark: **the NLI model assigns very high neutral probability (0.83–0.99) to almost all scientific abstracts**, with only a single document receiving a confident stance (0.977). The stance reranker therefore promotes that one confidently-stanced document to the top and demotes everything else as "neutral."

This is not an isolated example. Aggregated across all 3,000 retrieved documents in the SciFact set (300 claims × 10 candidates), **93.6% received a neutral score above 0.5, and the mean neutral score was 0.924**. The NLI model therefore rates the overwhelming majority of scientific abstracts as taking no stance, confirming that the pattern seen in the single-claim example above is systematic, not anecdotal. Because only ~6% of documents are scored as stance-bearing, reranking is driven by a small, unrepresentative minority of documents, and hard filtering removes roughly nine in ten candidates.

The problem is that scientific evidence is written in hedged, cautious, technical language, where it rarely resembles the assertive entailment/contradiction sentence pairs the general-domain NLI model was trained on. As a result, the true gold-evidence abstract is typically among the documents the model rates as "neutral," and it gets pushed down the ranking, while whichever document happens to trigger a confident (and usually spurious) stance reading is promoted to rank 1, which is why R@1 collapses. Hard filtering then removes the ~9 "neutral" documents per claim, discarding the gold evidence along with them.

This is a **domain-mismatch failure**: a general-purpose NLI stance model applied to
scientific text systematically mis-scores hedged evidence as stance-free.

## Decision: soft reranking carried forward

**Soft reranking is selected over hard filtering for the downstream pipeline (Step 5).**
Hard filtering is demonstrably unusable where it removes almost the entire candidate pool (0.6–1.2 documents surviving) and destroys recall on both datasets. Soft reranking, while it harms top-rank recall, at least preserves the full candidate pool (R@10 unchanged), so it remains a viable condition to test in the full pipeline. Evaluating both modes was what allowed this to be an evidence-based decision rather than an assumption, where the empirical collapse of hard filtering is itself a documented result.

## Relevance to the project hypotheses and later steps

**Answering the project's hypothesis.** The project hypothesised that stance-based
selection would improve evidence quality over pure topical similarity. At the retrieval level, this hypothesis is **refuted** on scientific claims: stance reranking degrades recall, because a general-domain NLI model cannot reliably identify stance in hedged scientific text. This is a genuine, mechanised finding rather than an inconclusive one.

**Relevance to Step 5 (RAG pipeline).** Step 4 measures the effect of reranking on
*retrieval recall*. The distinct and still-open question is whether this recall damage
translates into *classification* damage, like does feeding the reordered (soft) candidates to the verifier hurt final F1, or can the classifier compensate? Step 5 answers this directly by comparing the dense and dense+rerank conditions on downstream verification F1. Given the severity of the recall collapse, reranking is expected to harm Step 5 as well, but this is measured, not assumed.

**Relevance to Step 7 (failure taxonomy).** 
This step contributes a concrete, named failure mode: *confidently-stanced-but-irrelevant document promoted over hedged true evidence*, with
a quantified frequency (≈9 of 10 documents mis-scored as neutral) and a clear cause (NLI domain mismatch). This is a direct input to the failure taxonomy.

**Why this is interesting for the thesis.** 
A project on "retrieval effects and failure
behaviour" is strengthened, not weakened, by a well-diagnosed negative result. Showing that an intuitively reasonable technique fails, and explaining precisely *why*, with score-level evidence, is a more substantive contribution than confirming an expected improvement. It also points to a concrete future direction: a domain-adapted (scientific) NLI reranker may succeed where the general-domain model fails, which is a natural extension of this work.

## Files

- Results: `results/reranking_mpnet_{soft,hard}_{scifact,scifact_open}.json`
- Reranked candidates (input to Step 5): `results/reranked_candidates_mpnet_{soft,hard}_{scifact,scifact_open}.json`