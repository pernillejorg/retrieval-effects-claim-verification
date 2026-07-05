# Step 5 Results: The RAG Pipeline

This document records Step 5, the full retrieval-augmented generation (RAG) pipeline for
scientific claim verification. It integrates every component built in the previous steps —
retrieval (Step 3), stance reranking (Step 4), and the two fine-tuned RoBERTa classifiers
(Step 2) — and evaluates four conditions on each dataset. This is the step where the
project's central questions are answered at the level of final classification performance:
does retrieved evidence help a verifier, and does stance reranking help or hurt?

**Note on this version.** These results use the **corrected-seeding classifiers** from
Step 2 (Model 1 macro F1 0.5263, Model 2 0.6438). An earlier version of this pipeline used
the pre-seed-correction models (Model 1 0.4570); those earlier numbers are retained in the
comparison tables below, clearly marked, so the effect of the classifier correction on the
pipeline is transparent. Retrieval (Step 3) and reranking (Step 4) were **not** affected by
the seeding correction — they do not use the classifiers — so their results are unchanged;
only the classification stage re-ran.

## Pipeline conditions

1. **No retrieval** — Model 1 (claim-only classifier), no evidence context.
2. **BM25 + RoBERTa** — sparse retrieval, then classification with Model 2 (evidence classifier).
3. **Dense + RoBERTa** — dense (mpnet) retrieval, then classification with Model 2.
4. **Dense + soft rerank + RoBERTa** — dense retrieval, soft stance reranking, then Model 2.

The no-retrieval condition uses Model 1 (trained on claim text alone); the three retrieval
conditions use Model 2 (trained on claim + gold evidence pairs). Each classifier is
therefore applied to the input format it was trained on, which is the methodologically
correct RAG setup.

## Setup

| Property | Value |
|---|---|
| Classifiers | Model 1 (claim-only, F1 0.5263), Model 2 (claim+evidence, F1 0.6438), from Step 2 |
| Retrievers | BM25 (sparse), all-mpnet-base-v2 (dense) |
| Reranker | cross-encoder/nli-deberta-v3-small, soft mode (from Step 4) |
| Documents to classifier | top_k = 5 |
| Rerank pool size | 10 |
| Neutral threshold | 0.5 (loose) |
| Metric | macro F1 (precision, recall also reported), present-label scoped |
| Device | CUDA (Google Colab GPU) |

Model 2 was trained on gold evidence but is fed **retrieved** evidence here, which is the
standard "train on gold, test on retrieved" RAG evaluation. Retrieval and reranking are run
live within the pipeline so the exact document text seen by the classifier is consistent
end to end.

## Results — SciFact (5,183-document corpus, 300 claims)

| Condition | Macro F1 | Precision | Recall |
|---|---|---|---|
| No retrieval (Model 1) | 0.5263 | 0.5287 | 0.5246 |
| BM25 + RoBERTa | 0.5388 | 0.5572 | 0.5357 |
| Dense + RoBERTa | **0.5666** | 0.5905 | 0.5611 |
| Dense + soft rerank + RoBERTa | 0.4939 | 0.5405 | 0.5043 |

Per-class F1 (SciFact):

| Condition | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|
| No retrieval | 0.56 | 0.37 | 0.65 |
| BM25 | 0.63 | 0.35 | 0.64 |
| Dense | 0.64 | 0.37 | 0.69 |
| Dense + rerank | 0.57 | 0.29 | 0.63 |

## Results — SciFact-Open (500,000-document corpus, 279 claims)

| Condition | Macro F1 | Precision | Recall |
|---|---|---|---|
| No retrieval (Model 1) | **0.6219** | 0.6236 | 0.6271 |
| BM25 + RoBERTa | 0.5815 | 0.5976 | 0.5761 |
| Dense + RoBERTa | 0.5881 | 0.6076 | 0.5804 |
| Dense + soft rerank + RoBERTa | 0.4727 | 0.4829 | 0.4890 |

Per-class F1 (SciFact-Open):

| Condition | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|
| No retrieval | 0.64 | 0.51 | 0.71 |
| BM25 | 0.61 | 0.50 | 0.64 |
| Dense | 0.63 | 0.54 | 0.60 |
| Dense + rerank | 0.54 | 0.38 | 0.50 |

## Effect of the classifier seeding correction on the pipeline

For transparency, the table below compares the pipeline before and after the Step 2 seeding
correction (see step2_results.md, Correction 2). The earlier pipeline used the
pre-correction Model 1 (0.4570); this version uses the corrected-seeding models. Retrieval
and reranking are identical between the two — only the classifiers changed.

**SciFact:**

| Condition | Earlier (pre-seed-fix) | Reported (corrected) |
|---|---|---|
| No retrieval | 0.4570 | 0.5263 |
| BM25 | 0.4875 | 0.5388 |
| Dense | 0.5424 | 0.5666 |
| Dense + rerank | 0.4115 | 0.4939 |

**SciFact-Open:**

| Condition | Earlier (pre-seed-fix) | Reported (corrected) |
|---|---|---|
| No retrieval | 0.5348 | 0.6219 |
| BM25 | 0.5035 | 0.5815 |
| Dense | 0.4962 | 0.5881 |
| Dense + rerank | 0.4225 | 0.4727 |

All conditions rose under the corrected seeding, as expected — the corrected classifiers are
uniformly stronger. Crucially, the **relationships between conditions — which are what the
findings are about — are preserved**: retrieval helps on SciFact, retrieval does not help on
SciFact-Open, and reranking is worst on both. The corrected numbers are the reported results;
the earlier ones are shown only to demonstrate that the findings are robust to the classifier
correction rather than an artefact of a particular training run.

## Consistency check

In both datasets, the no-retrieval condition reproduces the Step 2 baseline exactly (SciFact
0.5263; SciFact-Open zero-shot 0.6219). Because the no-retrieval condition uses the same
Model 1 on the same claim-only inputs as Step 2, this exact match confirms the pipeline is
wired correctly to the earlier steps and that the four conditions are directly comparable.

## Key findings

### 1. On the small corpus (SciFact), retrieval helps

Both retrieval conditions beat the no-retrieval baseline on SciFact: dense (0.5666) and BM25
(0.5388) both exceed 0.5263, and dense improves on the baseline by +0.040. Dense also beats
BM25, consistent with dense retrieval's higher recall in Step 3. So when the corpus is small
and retrieval is reliable, feeding retrieved evidence to an evidence-trained classifier
improves claim verification, as a RAG system is intended to. (The improvement margin is
smaller than in the pre-seed-fix run, +0.040 vs +0.085, because the corrected baseline is
itself stronger; the direction and ordering are unchanged.)

### 2. On the large corpus (SciFact-Open), retrieval does NOT help

On the 500,000-document corpus, the no-retrieval baseline (0.6219) is *higher* than both
BM25 (0.5815) and dense (0.5881). Adding retrieved evidence made classification **worse**,
not better — by roughly 0.03–0.04. The interpretation is direct and central to the project:
when the corpus is roughly 100× larger, retrieval is substantially harder (as quantified in
Step 3, where recall dropped at scale), so the evidence retrieved is noisier and less
reliable. That noisy evidence misleads the classifier more than the absence of evidence
would, so retrieval becomes a net negative. Retrieval's benefit is therefore **not universal
— it is conditional on the retrieval task being tractable**, and it inverts as corpus
difficulty grows. This finding is unchanged by the seeding correction: no-retrieval remains
the top condition on SciFact-Open in both runs.

### 3. Stance reranking hurts on both datasets

The dense + soft rerank condition is the **worst** condition on both datasets (SciFact
0.4939, below the no-retrieval baseline; SciFact-Open 0.4727, well below every other
condition). This confirms, at the classification level, the failure diagnosed at the
retrieval level in Step 4. Stance reranking degraded retrieval recall in Step 4 because the
general-domain NLI model rates hedged scientific abstracts as overwhelmingly neutral (93.6%
of documents scored neutral > 0.5, mean neutral score 0.924) and promotes a rare
confidently-but-wrongly-stanced document over the true evidence. Step 5 shows this recall
damage propagates all the way to final F1: the classifier, fed reranking's misordered
evidence, performs worse than with no reranking, and worse than with no retrieval at all.
The CONTRADICT class is hit hard (F1 drops to 0.29 on SciFact) and NEI recall inflates (the
model defaults toward "not enough information"), exactly the behaviour expected when the
evidence supplied is unreliable. This failure is consistent across both the pre- and
post-seed-fix runs, confirming it is a property of the reranker, not of a particular
classifier.

## Cross-dataset comparison (reported, corrected models)

| Condition | SciFact F1 | SciFact-Open F1 | Change at scale |
|---|---|---|---|
| No retrieval | 0.5263 | 0.6219 | +0.096 |
| BM25 | 0.5388 | 0.5815 | +0.043 |
| Dense | 0.5666 | 0.5881 | +0.022 |
| Dense + rerank | 0.4939 | 0.4727 | −0.021 |

The most informative row is **dense**: it is the best condition on SciFact but on
SciFact-Open it drops *below* the no-retrieval baseline (0.5881 vs 0.6219). The strong
retriever that wins on a small corpus loses its advantage — and becomes a net negative — on
a large one. This crossover is the empirical heart of the thesis: the effect of retrieval is
not a fixed property of the method but depends on how hard the retrieval problem is.

## Relevance to the project hypotheses

**Hypothesis: retrieved evidence improves claim verification.** Partly supported, and
conditionally. Retrieval helps on SciFact but not on SciFact-Open. The honest, evidenced
conclusion is that retrieval's benefit is conditional on corpus difficulty rather than
universal — a more nuanced and defensible claim than "retrieval always helps". This holds
under both the pre- and post-seed-correction classifiers, so it is robust to the training
run.

**Hypothesis: stance-based reranking improves evidence quality and therefore verification.**
Refuted, consistently and with a diagnosed mechanism. Reranking is the worst condition on
both datasets, at both the retrieval level (Step 4) and the classification level (Step 5). A
general-domain NLI reranker applied to scientific text systematically demotes true evidence,
and this harm carries through the pipeline to final performance.

## Relevance to later steps

**Step 6 (controlled experimental matrix).** Step 5 fixes top_k = 5. Step 6 varies k ∈
{1, 5, 10} across the retrieval conditions to isolate the effect of the number of documents
supplied to the classifier — testing, for example, whether fewer documents reduce the noise
that harmed retrieval on SciFact-Open, or whether more documents help or dilute performance.
The pipeline already supports this via its top_k argument, so no code change is required.

**Step 7 (failure taxonomy).** The per-claim records saved by this pipeline (claim, true and
predicted label, confidence, retrieved document ids and text snippets, and — for the
reranked condition — the pre-rerank order) are the direct input to the failure taxonomy. The
SciFact-Open retrieval-does-not-help result and the reranking collapse are prime cases for
the "irrelevant retrieval" and "contradictory retrieval" categories.

**Step 8 (confidence analysis).** Each record includes the classifier's confidence (max
softmax probability), enabling the Step 8 analysis of whether low-confidence predictions
correlate with errors, without re-running the pipeline.

## Note on the interpretation of magnitudes

The Step 2 multi-seed robustness check found a baseline standard deviation of about 0.014.
Several of the differences here — particularly on SciFact-Open, where BM25 (0.5815) and
dense (0.5881) sit close together and about 0.03–0.04 below no-retrieval — are of a
comparable scale to run-to-run noise. The *direction* of the findings (retrieval helps on
SciFact, does not help on SciFact-Open, reranking hurts on both) is consistent and holds
across two independently-seeded classifier runs, but the precise magnitude of the smaller
gaps should be read with this noise floor in mind. The reranking penalty (0.05–0.11 below
the best condition) and the evidence gain of Model 2 in Step 2 (+0.118) are well outside the
noise floor and are therefore firm.

## Result for the thesis

A project titled "Retrieval Effects and Failure Behaviour" is well served by these results.
Rather than a single expected outcome, the pipeline produces a **conditional,
mechanistically explained picture**: retrieval helps when tractable and does not help when
the corpus makes retrieval hard; stance reranking fails consistently for a diagnosed reason.
Both the retrieval-at-scale crossover and the reranking failure are the kind of negative and
conditional findings that a failure-focused empirical study is designed to surface, both are
supported by evidence across multiple steps, and both are shown to be robust to the
classifier seeding correction.

## Files

- Aggregate metrics: `results/step5_pipeline_scifact_thr0_5.json`,
  `results/step5_pipeline_scifact_open_thr0_5.json`
- Per-claim records (input to Steps 7 and 8): `results/step5_records_scifact_thr0_5.json`,
  `results/step5_records_scifact_open_thr0_5.json`

## Note on run logs

A tokenizer warning ("Token indices sequence length is longer than 512") appears in the logs
when a claim+document pair exceeds 512 tokens before truncation. Truncation
(`max_length=512`) is applied, so the model always receives a valid 512-token input; the
warning is informational and does not affect results. It reflects that some scientific
abstracts are long and are truncated to fit the model's input limit, as expected in any RAG
system with a fixed context window.

