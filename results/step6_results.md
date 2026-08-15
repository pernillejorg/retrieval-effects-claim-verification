# Step 6 Results: Controlled Experimental Matrix (Retrieval Depth k)

This document records Step 6, the controlled experimental matrix that varies the retrieval depth **k**, the number of retrieved documents supplied to the classifier, across all pipeline conditions and both datasets. Where Step 5 fixed the pipeline at a single depth (k = 3, following MAPLE), Step 6 sweeps **k ∈ {1, 3, 5, 10}** to isolate the effect of retrieval quantity on final verification performance. This is the step that answers whether "more evidence is better," and it produces the project's main quantitative depth result.

## Design

| Property | Value |
|---|---|
| Swept variable | retrieval depth k ∈ {1, 3, 5, 10} |
| Conditions | no retrieval, BM25, dense, dense + soft rerank |
| Datasets | SciFact (5,183 docs), SciFact-Open (500,000 docs) |
| Classifiers | Model 1 and Model 2 (from Step 2), at each of seeds 42, 123 and 7 |
| Truncation | Option B (per-document token budget), so k genuinely varies |
| Metric | macro F1 |
| Seeds | 42, 123, 7 (full matrix re-run under each; results reported as mean ± SD) |

### Why k ∈ {1, 3, 5, 10}

The retrieval depth k = 3 is taken from MAPLE (Zeng and Zubiaga, 2024), which retrieves the top-3 BM25 abstracts as evidence; it is retained as one point in the sweep so the matrix is anchored to prior work. The sweep extends this to test sensitivity: k = 1 is minimal retrieval (a single top document), k = 10 approaches the practical ceiling set by RoBERTa's 512-token input, and k = 3 and k = 5 sample the low-to-moderate range. A single fixed value, as used in prior work, cannot reveal whether performance is stable across depth, the sweep can.

### Why this sweep is only meaningful under Option B

Step 5 established that with naive whole-document concatenation (Option A), the 512-token input saturates after roughly two abstracts, so k above ~2 had no effect. Step 6 therefore uses the Option B per-document budget, which divides the token budget equally across the k documents so every document contributes and k genuinely changes the input. The clear variation with k in the results below confirms Option B works as intended, it is the direct validation of the Step 5 design fix. The no-retrieval condition, which uses no documents, is by construction identical across all k and serves as a flat reference line.

## Results: SciFact (macro F1 by k, mean ± SD over seeds 42, 123, 7)

| Condition | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| No retrieval | 0.5242 ± 0.0141 | 0.5242 ± 0.0141 | 0.5242 ± 0.0141 | 0.5242 ± 0.0141 |
| BM25 | **0.5375 ± 0.0250** | 0.4601 ± 0.0516 | 0.4565 ± 0.0209 | 0.4449 ± 0.0244 |
| Dense | **0.5478 ± 0.0394** | 0.5183 ± 0.0339 | 0.4888 ± 0.0525 | 0.4565 ± 0.0341 |
| Dense + soft rerank | 0.3482 ± 0.0292 | 0.4434 ± 0.0423 | **0.5000 ± 0.0494** | 0.4583 ± 0.0395 |

## Results: SciFact-Open (macro F1 by k, mean ± SD over seeds 42, 123, 7)

| Condition | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| No retrieval | **0.5938 ± 0.0199** | **0.5938 ± 0.0199** | **0.5938 ± 0.0199** | **0.5938 ± 0.0199** |
| BM25 | 0.4972 ± 0.0421 | 0.4875 ± 0.0502 | 0.4725 ± 0.0548 | 0.4734 ± 0.0649 |
| Dense | 0.5045 ± 0.0359 | 0.4980 ± 0.0641 | 0.5033 ± 0.0631 | 0.4997 ± 0.0628 |
| Dense + soft rerank | 0.3743 ± 0.0299 | 0.4637 ± 0.0608 | 0.4823 ± 0.0732 | 0.5113 ± 0.0554 |

### Seed-42 run (retained for reference)

The matrix was originally run at seed 42 alone; those figures are kept here because
several observations in this document were first made from them, and because the
comparison between the two readings is itself informative.

SciFact:

| Condition | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| No retrieval | 0.5263 | 0.5263 | 0.5263 | 0.5263 |
| BM25 | **0.5727** | 0.5127 | 0.4778 | 0.4667 |
| Dense | **0.5947** | 0.5583 | 0.5179 | 0.4828 |
| Dense + soft rerank | 0.3852 | 0.4879 | 0.5389 | 0.4824 |

SciFact-Open:

| Condition | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| No retrieval | 0.6219 | 0.6219 | 0.6219 | 0.6219 |
| BM25 | **0.5378** | 0.5229 | 0.5176 | 0.5127 |
| Dense | 0.5429 | **0.5560** | 0.5454 | 0.5200 |
| Dense + soft rerank | 0.4102 | 0.5075 | 0.5238 | **0.5406** |

## Key findings

### 1. For plain retrieval, fewer documents are better (performance declines with k)

The clearest result of the sweep: for the non-reranked retrieval conditions (BM25 and dense), macro F1 **peaks at the shallowest depth and declines as k grows on SciFact**, while on SciFact-Open dense is flat within noise. On SciFact, dense falls from 0.5478 at k = 1 to 0.4565 at k = 10, and BM25 from 0.5375 to 0.4449, both roughly monotone decreases of about 0.09 across the range and larger than any condition's standard deviation. On SciFact-Open the same downward trend holds for BM25 (0.4972 to 0.4734). Supplying more retrieved documents dilutes rather than enriches the evidence: additional documents introduce more noise and, under the per-document budget, shorter fragments of each, and the classifier's performance suffers. This is a direct, empirical demonstration of the **"evidence overload"** failure mode, one of the four failure categories defined for Step 7, and it is a counterintuitive, decision-relevant finding: for scientific claim verification, one well-chosen document beats many.

### 2. Retrieval's apparent advantage at k = 1 does not survive averaging

At seed 42, both retrieval conditions clearly beat the no-retrieval baseline on SciFact at
k = 1: dense 0.5947 and BM25 0.5727 against 0.5263, gains of +0.068 and +0.046. Read alone
this was the cleanest "retrieval helps" signal in the project.

Averaged over three seeds it disappears. Dense at k = 1 is 0.5478 ± 0.0394 against a
baseline of 0.5242 ± 0.0141, a gain of +0.024 that sits well inside its own standard
deviation; BM25 at k = 1 is 0.5375 ± 0.0250, a gain of +0.013. Neither margin is
distinguishable from run-to-run noise. The honest reading is that no retrieval configuration
in this matrix improves on the claim-only baseline by more than its own variance, on either
corpus, at any depth.

This sharpens rather than weakens the thesis. It also demonstrates the specific way a
single-seed evaluation can mislead: the seed-42 run showed a retrieval advantage that three
seeds do not support, and it showed a corpus-dependent optimum (k = 1 on SciFact against
k = 3 on SciFact-Open) that also dissolves, since dense on SciFact-Open reads 0.5045, 0.4980,
0.5033 and 0.4997 across the sweep, flat within its own deviation.

### 3. Reranking behaves oppositely where it improves with k, but never leads

The dense + soft rerank condition shows the **opposite** trend: it is worst at k = 1 (0.3482 on SciFact, 0.3743 on SciFact-Open, in both cases far outside noise) and improves with more documents (SciFact to 0.5000 at k = 5; SciFact-Open to 0.5113 at k = 10). The interpretation is consistent with the reranking failure diagnosed in Steps 4 and 5: with only one document, the reranker's tendency to promote a confidently-but-wrongly-stanced document has nowhere to hide, so a single mis-ranked document is disastrous; with more documents, the true evidence is more likely to be present somewhere in the set, partially masking the damage. Crucially, **even at its best k the reranked condition never beats plain dense at that dataset's best k** (like SciFact rerank peaks at 0.5000 versus dense's 0.5478 at k = 1). Reranking's "improvement" with k is recovery from self-inflicted harm, not a genuine benefit.

### 4. No retrieval is unaffected by k, as expected

The no-retrieval condition is identical at every k (SciFact 0.5242 ± 0.0141, SciFact-Open 0.5938 ± 0.0199), because it uses no retrieved documents. This is the correct control behaviour and confirms the matrix is wired properly: only the retrieval-dependent conditions respond to k.

### 5. On SciFact-Open, no retrieval still wins at every k

Even at the retrieval conditions' best depths, no retrieval (mean 0.5938) remains above every retrieval condition at every k on SciFact-Open (the best retrieval mean is 0.5113 for reranking at k = 10, still 0.0824 below the baseline). The Step 5 finding that retrieval does not help at scale is therefore robust across the entire depth sweep, not an artefact of one k. Depth tuning does not rescue retrieval on the large, hard corpus.

## Interpretation for the thesis

### The overload finding is a core contribution

Step 6 turns an intuition into a measured result: **for scientific claim verification, adding more retrieved evidence degrades performance.** This runs against the common assumption where held widely, including in industry deployments of RAG, that more retrieved context is better. For plain retrieval the shallowest depth tested is the least harmful, and performance falls off steadily with more documents. This is precisely the kind of failure behaviour the project set out to characterise, and it is directly decision-relevant: a practitioner building a RAG fact-checker over scientific text should not assume that retrieving more context helps, and this work provides empirical evidence and a mechanism (evidence dilution / overload) for why.

### Relationship to prior work (MAPLE)

MAPLE fixed k = 3. Step 6 shows that, for the retrieval conditions studied here, performance at k = 3 is already below what a shallower depth gives, though on seed means no depth clears the claim-only baseline. This is offered not as a criticism of MAPLE, a fixed reasonable depth is a legitimate choice for their scope, but as exactly the kind of insight a controlled sweep yields that a single fixed value cannot. (As noted in Step 5, MAPLE is a methodologically different system and no direct F1 comparison is drawn; the connection is limited to retrieval depth.)

### On which k to use going forward

On seed means no depth gives a retrieval condition a clear advantage over the claim-only
baseline, so the sweep does not identify an optimal depth so much as show that shallow
retrieval is least harmful for plain retrieval. The reported pipeline (Steps 5, 7, 8, 10)
retains k = 3: it is the MAPLE-anchored depth on which the per-claim records were built, and
the evidence-overload failure category is defined as a correct-to-wrong transition as k
grows, so it cannot be assigned at k = 1 at all.

### Robustness caveat

The matrix was re-run under seeds 123 and 7, so every cell carries a mean and standard
deviation over three seeds. The shape of the k-curves is unchanged: plain retrieval declines
with k, reranking rises with k, no retrieval is flat. The decline from k = 1 to k = 10 on
SciFact is about 0.09 for both retrievers, larger than any condition's deviation, so the
overload effect is treated as real. Two single-seed readings did not survive averaging, and
are corrected above: retrieval's advantage at k = 1 on SciFact, and the corpus-dependent
optimum on SciFact-Open. The retrieval conditions remain markedly less stable than the
baseline (deviations of 0.02 to 0.07 against 0.014 and 0.020), which is itself one of the
project's findings.

## Relevance to later steps

**Step 7 (failure taxonomy).** 
Step 6 provides direct quantitative evidence for the "evidence overload" failure category: performance falls as k grows for plain retrieval. The per-claim
records saved for each k (`records_*_k{1,3,5,10}_thr0_5.json`), enriched with full document text and the exact classifier input (see the record-enrichment section below), allow the manual error analysis to examine, for specific claims, how predictions change as more documents are added.

**Step 8 (confidence analysis).** 
The per-claim records include confidence scores at each k, enabling analysis of whether confidence tracks the accuracy decline as k grows.

**Step 10 (real-world case study).** 
The case study sweeps all four depths and anchors its primary test at k = 3, matching the rest of the pipeline.

## Record enrichment for Step 7 (matrix re-run, results unchanged)

The Step 6 matrix was re-run after the pipeline (`pipeline.py`) was extended with additional
per-claim record fields required by the Step 7 failure taxonomy. The motivation is the same as
described in the Step 5 results: the original records saved only a 300-character snippet of
each retrieved document (too short for a human annotator to judge relevance reliably) and did
not store the exact text the classifier saw after concatenation and 512-token truncation
(needed to tell a genuine confident wrong prediction from an input-construction/truncation
failure). The Step 6 records matter here specifically because Step 7's evidence-overload
analysis reads the records across all four depths (k = 1, 3, 5, 10) to trace how a claim's
prediction changes as documents are added, so every depth's records need the richer fields.

The added fields are **purely additive** and non-behavioural: full retrieved document text
(the 300-character cap removed), `classifier_input_text` with its
`input_token_count_before_truncation`, `input_token_count_after_truncation`, and
`was_truncated` flag, and, for the reranked condition, each document's original dense
retrieval `score` alongside its stance and neutral scores. Retrieval, concatenation, the
tokenisation used for inference, and the predictions are unchanged.

The re-run reproduced the entire matrix **exactly**: every macro-F1 cell of the then-current 
seed-42 matrix was identical to the original run (for example SciFact dense 0.5947 at k = 1 
falling to 0.4828 at k = 10; SciFact-Open no-retrieval 0.6219 flat across k), and the assembled 
matrix printed in the notebook matched cell for cell. This confirms the enrichment changed only 
what is recorded, not what is computed. A verification cell in the Step 6 notebook checks the 
regenerated k = 3 records for both datasets and confirms the new fields are present and the saved 
document text is now the full abstract (938 characters for the first SciFact dense record, 
1448 for the first SciFact-Open dense record, versus the previous 300-character cap). All findings, 
the overload result, and the k-curves above are therefore unaffected; the records simply now carry 
the information Step 7 needs.

A note on filenames: the pipeline now embeds the retrieval depth and threshold in the output
filenames automatically (for example `records_scifact_k3_thr0_5.json`), so the per-k matrix
files retain the naming used throughout this document, and runs at different depths cannot
overwrite one another.

## Files

- Aggregate metrics: `results/step6_matrix/matrix_{scifact,scifact_open}_k{1,3,5,10}_thr0_5.json`
- Per-claim records: `results/step6_matrix/records_{scifact,scifact_open}_k{1,3,5,10}_thr0_5.json`
- Run logs: `results/step6_matrix/log_*_k{1,3,5,10}.txt`
- Multi-seed matrix: `results/step6_matrix/step6_multiseed_matrix/k{1,3,5,10}/matrix_{scifact,scifact_open}_seed{123,7}_k{1,3,5,10}_thr0_5.json`
- Assembled summary: `results/step6_matrix/step6_multiseed_matrix/matrix_multiseed_summary.json`

- Figure coordinates: `analysis/tools/emit_figure_coords.py`, which prints the
  pgfplots coordinate lines used in the dissertation directly from the saved
  results, so the figures are regenerated rather than retyped.

The multi-seed runs applied a small runtime patch to `models/retrieval.py`,
shown in cell 7 of `Step6_experiments_multiseed.ipynb`, which caches the dense
embedding matrix so the 500,000-document corpus is encoded once rather than once
per run. The cache is keyed on embedding model and corpus size and is only used
when the stored document ids match exactly, so it changes runtime and not the
embeddings or any reported result. The patch is not applied in the committed
`retrieval.py`, which re-encodes on every run as before.

## Note on run logs

The tokenizer length warning ("Token indices sequence length is longer than 512") appears in the logs as in earlier steps. Truncation to 512 tokens is applied, so the model always receives a valid input; the warning is informational. Under Option B the per-document budget means each document is truncated to its share, so at larger k each document is shorter.