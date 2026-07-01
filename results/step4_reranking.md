# Step 4: Stance-Aware Reranking Results

## What This Step Does

After standard retrieval gives a candidate pool of documents, this step adds a stance-aware reranking layer on top. The idea is that topical similarity is not enough for scientific claim verification. A document can be about the same topic as a claim without actually saying anything specific about it. The retriever has no way to tell these apart because it only looks at semantic overlap. The reranker fixes this by using a zero-shot NLI model to score whether each document takes a stance on the claim, either supporting it or contradicting it, rather than just being topically nearby.

The model used is `cross-encoder/nli-deberta-v3-small` from HuggingFace. For each claim, all retrieved documents are encoded together with the claim in one batched forward pass through the NLI model. The model outputs three probability scores per document: entailment, contradiction, and neutral. The stance score for each document is defined as the maximum of its entailment and contradiction probabilities, which captures whether the document takes any position on the claim at all.

Two threshold values are tested to measure threshold sensitivity. The loose threshold is set to 0.5 and the strict threshold is set to 0.8. These refer to the neutral probability score above which a document would be considered non-stance-bearing.

## Datasets Used

Both datasets were evaluated on their validation splits.

SciFact: 450 validation claims, corpus of 5183 abstracts. All 450 validation claims have annotated evidence so NEI claims are 0 in this split.

SciClaimHunt: 10872 validation claims, corpus of 10872 documents. All claims have annotated evidence.

## First Approach: Hard Filtering (Abandoned)

The first implementation used hard filtering. This means documents whose neutral probability score exceeded the threshold were completely removed from the candidate pool before the reranker returned results. The two thresholds defined what counted as too neutral. If a document scored above 0.5 neutral probability in the loose condition, it was discarded. If it scored above 0.8 in the strict condition, it was discarded.

The results from the hard filtering run were as follows.

**SciFact (hard filter):**

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.562 | 0.809 | 0.851 |
| Dense + loose hard filter | 0.062 | 0.096 | 0.096 |
| Dense + strict hard filter | 0.116 | 0.169 | 0.169 |

Average documents remaining after loose filter: 0.4 out of 10
Average documents remaining after strict filter: 0.7 out of 10

**SciClaimHunt (hard filter):**

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.344 | 0.527 | 0.567 |
| Dense + loose hard filter | 0.047 | 0.086 | 0.086 |
| Dense + strict hard filter | 0.054 | 0.116 | 0.117 |

Average documents remaining after loose filter: 1.3 out of 10
Average documents remaining after strict filter: 1.9 out of 10

The recall numbers collapsed completely. R@10 dropped from 0.851 to 0.096 on SciFact under the loose threshold, and from 0.567 to 0.086 on SciClaimHunt. The reason is that the NLI model was classifying almost every retrieved document as neutral, leaving on average fewer than 1 document per claim in the candidate pool after the loose filter on SciFact. Once there are almost no documents left, recall is guaranteed to be near zero regardless of document quality.

This is not a model failure in the ordinary sense. It is a domain mismatch problem. The NLI model was trained on general language datasets such as SNLI and MultiNLI where entailment and contradiction relationships are expressed directly and explicitly. Scientific abstracts express their stances in a much more implicit, hedged, and domain-specific way. A sentence like "our results are consistent with the hypothesis that omega-3 supplementation affects cardiovascular markers" is technically entailing a claim about omega-3, but the NLI model trained on general text is unlikely to recognise this as a strong entailment. It will assign a high neutral probability because the relationship is not expressed in the direct way the model learned to look for. This causes genuine evidence documents to be discarded along with irrelevant ones.

This failure mode has been documented in recent work on retrieval for scientific claim verification. A 2026 paper studying retrieval for the TREC BioGen track using SciFact as a development benchmark explicitly identified what they called Retrieval Asymmetry: applying NLI filtering on top of dense retrieval improves contradiction detection but degrades support recall because the filtering is too aggressive on scientific text. Our hard filtering results independently reproduce this finding.

## Revised Approach: Soft Reranking (Final Version)

Because hard filtering was destroying the candidate pool, the approach was changed to soft reranking. In soft reranking, no documents are removed. Instead, all retrieved documents are kept but reordered so that stance-bearing documents appear at the top of the ranked list. The stance score, which is the maximum of entailment and contradiction probability, is used as the sorting key. A document that the NLI model believes clearly supports or refutes the claim will rank above a document the model considers topically related but neutral.

The threshold parameters are kept in the code but now serve a different purpose. Rather than deciding which documents to remove, they are used to record how many documents would have been removed under hard filtering. This is reported as a diagnostic statistic so the over-filtering problem can be quantified and discussed in the thesis.

The model label order confirmed by the runtime output was `{0: contradiction, 1: entailment, 2: neutral}`, which matches the index constants defined in the code. All inference was batched per claim using a single forward pass through the NLI model, which is important for efficiency especially on the SciClaimHunt validation set of 10872 claims.

## Final Results: Soft Reranking

### SciFact (validation, 450 claims, 5183 corpus documents)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.562 | 0.809 | 0.851 |
| Dense + loose soft reranking | 0.153 | 0.580 | 0.851 |
| Dense + strict soft reranking | 0.153 | 0.580 | 0.851 |

Average documents retrieved: 10.0
Average documents that would be filtered by loose threshold (0.5): 6.2 out of 10
Average documents that would be filtered by strict threshold (0.8): 6.0 out of 10

### SciClaimHunt (validation, 10872 claims, 10872 corpus documents)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Dense (before reranking) | 0.344 | 0.527 | 0.567 |
| Dense + loose soft reranking | 0.064 | 0.298 | 0.567 |
| Dense + strict soft reranking | 0.064 | 0.298 | 0.567 |

Average documents retrieved: 10.0
Average documents that would be filtered by loose threshold (0.5): 8.7 out of 10
Average documents that would be filtered by strict threshold (0.8): 8.1 out of 10

## Analysis and Discussion

**R@10 is fully preserved on both datasets.** Because no documents are removed in soft reranking, the maximum recall ceiling is unchanged. The evidence document that was in the top 10 before reranking is still in the top 10 after reranking. This confirms the soft reranking approach does not hurt retrieval coverage.

**R@5 drops after reranking on SciFact** (from 0.809 to 0.580) and **on SciClaimHunt** (from 0.527 to 0.298). This means the evidence document is being pushed lower in the ranking in some cases. The likely cause is that documents with strong but incorrect stance signals are being promoted above the true evidence. A document might score high on entailment or contradiction simply because it uses assertive scientific language, even if it is not the annotated evidence document for that specific claim. This is a known limitation of zero-shot NLI reranking and is directly relevant to the failure taxonomy developed in this thesis.

**R@1 drops noticeably on both datasets** (from 0.562 to 0.153 on SciFact, from 0.344 to 0.064 on SciClaimHunt). The same explanation applies. The top-ranked document after reranking is not always the correct one because stance score does not perfectly correlate with being the annotated evidence document.

**The would-be-filter statistics are the most important finding of this step.** On SciFact, 6.2 out of 10 retrieved documents would have been removed by hard filtering at the loose threshold. On SciClaimHunt, 8.7 out of 10 would have been removed. This confirms that zero-shot NLI on scientific text classifies the overwhelming majority of retrieved documents as neutral, which is why hard filtering is not viable for this domain. The difference between the two datasets is also informative. SciClaimHunt has more aggressive filtering (8.7 vs 6.2) likely because its claims are more domain-specific and the NLI model is even less calibrated to recognise stance in that kind of text.

**The loose and strict thresholds produce identical recall results** on both datasets. This is because the threshold does not affect the ranking order in soft reranking, only the would-be-filter count changes between thresholds (6.2 vs 6.0 on SciFact, 8.7 vs 8.1 on SciClaimHunt). The fact that the two thresholds produce almost the same would-be-filter counts suggests most documents fall clearly above both thresholds rather than between them, meaning the NLI model is not giving borderline neutral scores but consistently high neutral scores for most scientific documents.

## Setup and Environment

Using notebook and running it in Google Colab:

Runtime: Google Colab A100 GPU
NLI model: cross-encoder/nli-deberta-v3-small
Retrieval model: sentence-transformers/all-MiniLM-L6-v2
Candidate pool size: k=10 per claim
Loose neutral threshold: 0.5
Strict neutral threshold: 0.8
Inference: batched per claim, all 10 documents in one forward pass
Results saved to: Google Drive rag-thesis/results/