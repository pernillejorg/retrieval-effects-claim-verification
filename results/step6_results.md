# Step 6 Results: Controlled Experimental Matrix (Retrieval Depth k)

This document records Step 6, the controlled experimental matrix that varies the retrieval depth **k**, the number of retrieved documents supplied to the classifier, across all pipeline conditions and both datasets. Where Step 5 fixed the pipeline at a single depth (k = 3, following MAPLE), Step 6 sweeps **k ∈ {1, 3, 5, 10}** to isolate the effect of retrieval quantity on final verification performance. This is the step that answers whether "more evidence is better," and it produces the project's main quantitative depth result.

## Design

| Property | Value |
|---|---|
| Swept variable | retrieval depth k ∈ {1, 3, 5, 10} |
| Conditions | no retrieval, BM25, dense, dense + soft rerank |
| Datasets | SciFact (5,183 docs), SciFact-Open (500,000 docs) |
| Classifiers | seed-42 Model 1 and Model 2 (from Step 2) |
| Truncation | Option B (per-document token budget), so k genuinely varies |
| Metric | macro F1 |
| Seed | 42 (fixed; the seed axis is studied separately in the Step 5 variance study) |

### Why k ∈ {1, 3, 5, 10}

The retrieval depth k = 3 is taken from MAPLE (Zeng and Zubiaga, 2024), which retrieves the top-3 BM25 abstracts as evidence; it is retained as one point in the sweep so the matrix is anchored to prior work. The sweep extends this to test sensitivity: k = 1 is minimal retrieval (a single top document), k = 10 approaches the practical ceiling set by RoBERTa's 512-token input, and k = 3 and k = 5 sample the low-to-moderate range. A single fixed value, as used in prior work, cannot reveal whether performance is stable across depth, the sweep can.

### Why this sweep is only meaningful under Option B

Step 5 established that with naive whole-document concatenation (Option A), the 512-token input saturates after roughly two abstracts, so k above ~2 had no effect. Step 6 therefore uses the Option B per-document budget, which divides the token budget equally across the k documents so every document contributes and k genuinely changes the input. The clear variation with k in the results below confirms Option B works as intended, it is the direct validation of the Step 5 design fix. The no-retrieval condition, which uses no documents, is by construction identical across all k and serves as a flat reference line.

## Results: SciFact (macro F1 by k)

| Condition | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| No retrieval | 0.5263 | 0.5263 | 0.5263 | 0.5263 |
| BM25 | **0.5727** | 0.5127 | 0.4778 | 0.4667 |
| Dense | **0.5947** | 0.5583 | 0.5179 | 0.4828 |
| Dense + soft rerank | 0.3852 | 0.4879 | 0.5389 | 0.4824 |

## Results: SciFact-Open (macro F1 by k)

| Condition | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| No retrieval | 0.6219 | 0.6219 | 0.6219 | 0.6219 |
| BM25 | **0.5378** | 0.5229 | 0.5176 | 0.5127 |
| Dense | 0.5429 | **0.5560** | 0.5454 | 0.5200 |
| Dense + soft rerank | 0.4102 | 0.5075 | 0.5238 | **0.5406** |

## Key findings

### 1. For plain retrieval, fewer documents are better (performance declines with k)

The clearest result of the sweep: for the non-reranked retrieval conditions (BM25 and dense), macro F1 **peaks at k = 1 and declines as k grows**. On SciFact, dense falls from 0.5947 at k = 1 to 0.4828 at k = 10, and BM25 from 0.5727 to 0.4667 both roughly monotone decreases of about 0.11 across the range. On SciFact-Open the same downward trend holds for BM25 (0.5378 to 0.5127) and for dense beyond its k = 3 peak (0.5560 to 0.5200). Supplying more retrieved documents dilutes rather than enriches the evidence: additional documents introduce more noise and, under the per-document budget, shorter fragments of each, and the classifier's performance suffers. This is a direct, empirical demonstration of the **"evidence overload"** failure mode, one of the four failure categories defined for Step 7, and it is a counterintuitive, decision-relevant finding: for scientific claim verification, one well-chosen document beats many.

### 2. At the optimal depth (k = 1), retrieval clearly helps on SciFact

At k = 1, both retrieval conditions clearly beat the no-retrieval baseline on SciFact: dense 0.5947 and BM25 0.5727 versus 0.5263, gains of +0.068 and +0.046. This is the cleanest "retrieval helps" signal in the whole project. It qualifies the earlier Step 5 reading (at k = 3, where retrieval's benefit was within noise): retrieval *does* help on the tractable corpus, but only at shallow depth, by k = 3 the dilution has already eroded most of the benefit, and MAPLE's k = 3 is in fact already past the optimum. The reported Step 5 depth (k = 3) is therefore a conservative choice, not an optimal one, which the sweep makes explicit.

### 3. Reranking behaves oppositely where it improves with k, but never leads

The dense + soft rerank condition shows the **opposite** trend: it is worst at k = 1
(catastrophic on SciFact, 0.3852) and improves with more documents (SciFact 0.3852 to 0.5389 at k = 5; SciFact-Open 0.4102 to 0.5406 at k = 10). The interpretation is consistent with the reranking failure diagnosed in Steps 4 and 5: with only one document, the reranker's tendency to promote a confidently-but-wrongly-stanced document has nowhere to hide, so a single mis-ranked document is disastrous; with more documents, the true evidence is more likely to be present somewhere in the set, partially masking the damage. Crucially, **even at its best k the reranked condition never beats plain dense at that dataset's best k** (like SciFact rerank peaks
at 0.5389 versus dense's 0.5947 at k = 1). Reranking's "improvement" with k is recovery from self-inflicted harm, not a genuine benefit.

### 4. No retrieval is unaffected by k, as expected

The no-retrieval condition is identical at every k (SciFact 0.5263, SciFact-Open 0.6219), because it uses no retrieved documents. This is the correct control behaviour and confirms the matrix is wired properly: only the retrieval-dependent conditions respond to k.

### 5. On SciFact-Open, no retrieval still wins at every k

Even at the retrieval conditions' best depths, no retrieval (0.6219) remains above every retrieval condition at every k on SciFact-Open (best retrieval value is dense 0.5560 at k = 3). The Step 5 finding that retrieval does not help at scale is therefore robust across the entire depth sweep, not an artefact of one k. Depth tuning does not rescue retrieval on the large, hard corpus.

## Interpretation for the thesis

### The overload finding is a core contribution

Step 6 turns an intuition into a measured result: **for scientific claim verification, adding more retrieved evidence degrades performance.** This runs against the common assumption where held widely, including in industry deployments of RAG, that more retrieved context is better. For plain retrieval the best depth is the shallowest tested (k = 1), and performance falls off steadily with more documents. This is precisely the kind of failure behaviour the project set out to characterise, and it is directly decision-relevant: a practitioner building a RAG fact-checker over scientific text should not assume that retrieving more context helps, and this work provides empirical evidence and a mechanism (evidence dilution / overload) for why.

### Relationship to prior work (MAPLE)

MAPLE fixed k = 3. Step 6 shows that, for the retrieval conditions studied here, k = 3 is already past the optimum (k = 1), and performance would have been higher at shallower depth. This is offered not as a criticism of MAPLE, a fixed reasonable depth is a legitimate choice for their scope, but as exactly the kind of insight a controlled sweep yields that a single fixed value cannot. (As noted in Step 5, MAPLE is a methodologically different system and no direct F1 comparison is drawn; the connection is limited to retrieval depth.)

### On which k to use going forward

The sweep identifies **k = 1 as the empirically optimal depth** for the plain retrieval conditions. Nonetheless the reported pipeline (Steps 5, 7, 8) retains **k = 3**, for two reasons: it is the MAPLE-anchored depth on which the rest of the pipeline and the per-claim records were built, and retaining it keeps the whole project internally consistent. The optimal-depth finding (k = 1) is reported here as a result in its own right rather than used to re-baseline the pipeline. Where a single "best" configuration is wanted, for example the real-world case study in Step 10, dense retrieval at k = 1 is the empirically supported choice and can be adopted there with this justification.

### Robustness caveat

These matrix figures are single-seed (seed 42), as the seed axis is studied separately in the Step 5 variance study, which found the retrieval conditions to be seed-sensitive (standard deviations of 0.03–0.06). The **shape** of the k-curves, plain retrieval declining with k, reranking rising with k, no retrieval flat is clear and consistent across both datasets and is the main result; the precise F1 at any single cell should be read with the seed-variancem noise floor in mind. The direction of the overload effect (roughly a 0.11 decline from k = 1 to k = 10 on SciFact) is larger than that floor and is treated as a real effect.

## Relevance to later steps

**Step 7 (failure taxonomy).** 
Step 6 provides direct quantitative evidence for the "evidence overload" failure category: performance falls as k grows for plain retrieval. The per-claim
records saved for each k (`records_*_k{1,3,5,10}_thr0_5.json`) allow the manual error analysis to examine, for specific claims, how predictions change as more documents are added.

**Step 8 (confidence analysis).** 
The per-claim records include confidence scores at each k, enabling analysis of whether confidence tracks the accuracy decline as k grows.

**Step 10 (real-world case study).** 
The overload finding motivates using a shallow retrieval depth (k = 1, the empirical optimum) for the best-configuration case study, rather than a larger k.

## Files

- Aggregate metrics: `results/step6_matrix/matrix_{scifact,scifact_open}_k{1,3,5,10}_thr0_5.json`
- Per-claim records: `results/step6_matrix/records_{scifact,scifact_open}_k{1,3,5,10}_thr0_5.json`
- Run logs: `results/step6_matrix/log_*_k{1,3,5,10}.txt`

## Note on run logs

The tokenizer length warning ("Token indices sequence length is longer than 512") appears in the logs as in earlier steps. Truncation to 512 tokens is applied, so the model always receives a valid input; the warning is informational. Under Option B the per-document budget means each document is truncated to its share, so at larger k each document is shorter.