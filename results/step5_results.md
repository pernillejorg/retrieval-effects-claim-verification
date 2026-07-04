# Step 5 Results: The RAG Pipeline

This document records Step 5, the full retrieval-augmented generation (RAG) pipeline for scientific claim verification. It integrates every component built in the previous steps, where retrieval (Step 3), stance reranking (Step 4), and the two fine-tuned RoBERTa classifiers (Step 2), and evaluates four conditions on each dataset. This is the step where the project's central questions are answered at the level of final classification performance: does retrieved evidence help a verifier, and does stance reranking help or hurt?

## Pipeline conditions

1. **No retrieval** — Model 1 (claim-only classifier), no evidence context.
2. **BM25 + RoBERTa** — sparse retrieval, then classification with Model 2 (evidence classifier).
3. **Dense + RoBERTa** — dense (mpnet) retrieval, then classification with Model 2.
4. **Dense + soft rerank + RoBERTa** — dense retrieval, soft stance reranking, then Model 2.

The no-retrieval condition uses Model 1 (trained on claim text alone); the three retrieval conditions use Model 2 (trained on claim + gold evidence pairs). Each classifier is therefore applied to the input format it was trained on, which is the methodologically correct RAG setup.

## Setup

| Property | Value |
|---|---|
| Classifiers | Model 1 (claim-only), Model 2 (claim+evidence), both from Step 2 |
| Retrievers | BM25 (sparse), all-mpnet-base-v2 (dense) |
| Reranker | cross-encoder/nli-deberta-v3-small, soft mode (from Step 4) |
| Documents to classifier | top_k = 5 |
| Rerank pool size | 10 |
| Neutral threshold | 0.5 (loose) |
| Metric | macro F1 (precision, recall also reported), present-label scoped |
| Device | CUDA (Google Colab GPU) |

Model 2 was trained on gold evidence but is fed **retrieved** evidence here, which is the standard "train on gold, test on retrieved" RAG evaluation. Retrieval and reranking are run live within the pipeline so the exact document text seen by the classifier is consistent end to end.

## Results for SciFact (5,183-document corpus, 300 claims)

| Condition | Macro F1 | Precision | Recall |
|---|---|---|---|
| No retrieval (Model 1) | 0.4570 | 0.4830 | 0.4506 |
| BM25 + RoBERTa | 0.4875 | 0.5021 | 0.4835 |
| Dense + RoBERTa | **0.5424** | 0.5577 | 0.5360 |
| Dense + soft rerank + RoBERTa | 0.4115 | 0.4253 | 0.4453 |

Per-class F1 (SciFact):

| Condition | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|
| No retrieval | 0.47 | 0.30 | 0.60 |
| BM25 | 0.56 | 0.26 | 0.64 |
| Dense | 0.59 | 0.33 | 0.72 |
| Dense + rerank | 0.49 | 0.11 | 0.63 |

## Results fo SciFact-Open (500,000-document corpus, 279 claims)

| Condition | Macro F1 | Precision | Recall |
|---|---|---|---|
| No retrieval (Model 1) | **0.5348** | 0.5456 | 0.5275 |
| BM25 + RoBERTa | 0.5035 | 0.5043 | 0.5216 |
| Dense + RoBERTa | 0.4962 | 0.5204 | 0.4953 |
| Dense + soft rerank + RoBERTa | 0.4225 | 0.4296 | 0.4450 |

Per-class F1 (SciFact-Open):

| Condition | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|
| No retrieval | 0.53 | 0.46 | 0.62 |
| BM25 | 0.59 | 0.29 | 0.63 |
| Dense | 0.55 | 0.34 | 0.60 |
| Dense + rerank | 0.47 | 0.29 | 0.51 |

## Consistency check

In both datasets, the no-retrieval condition reproduces the Step 2 baseline exactly
(SciFact 0.4570; SciFact-Open zero-shot 0.5348). Because the no-retrieval condition uses the same Model 1 on the same claim-only inputs as Step 2, this exact match confirms the pipeline is wired correctly to the earlier steps and that the four conditions are directly comparable.

## Key findings

### 1. On the small corpus (SciFact), retrieval helps

Both retrieval conditions beat the no-retrieval baseline on SciFact: dense (0.5424) and BM25 (0.4875) both exceed 0.4570, and dense improves on the baseline by +0.085. Dense also beats BM25, consistent with dense retrieval's higher recall in Step 3. So when the corpus is small and retrieval is reliable, feeding retrieved evidence to an evidence-trained classifier improves claim verification, as a RAG system is intended to.

### 2. On the large corpus (SciFact-Open), retrieval does NOT help, it slightly hurts

This is the more striking and important finding. On the 500,000-document corpus, the
no-retrieval baseline (0.5348) is *higher* than both BM25 (0.5035) and dense (0.4962).
Adding retrieved evidence made classification **worse**, not better. The interpretation is direct and central to the project: when the corpus is roughly 100× larger, retrieval is substantially harder (as quantified in Step 3, where recall dropped at scale), so the evidence retrieved is noisier and less reliable. That noisy evidence misleads the classifier more than the absence of evidence would, so retrieval becomes a net negative. Retrieval's benefit is therefore **not universal — it is conditional on the retrieval task being tractable**, and it inverts as corpus difficulty grows.

### 3. Stance reranking hurts on both datasets

The dense + soft rerank condition is the **worst** condition on both datasets (SciFact
0.4115, below even the no-retrieval baseline; SciFact-Open 0.4225). This confirms, at the classification level, the failure diagnosed at the retrieval level in Step 4. Stance reranking degraded retrieval recall in Step 4 because the general-domain NLI model rates hedged scientific abstracts as overwhelmingly neutral (93.6% of documents scored neutral > 0.5, mean neutral score 0.924) and promotes a rare confidently-but-wrongly-stanced document over the true evidence. Step 5 shows this recall damage propagates all the way to final F1: the classifier, fed reranking's misordered evidence, performs worse than with no reranking, and worse than with no retrieval at all. The CONTRADICT class is hit hardest (F1 collapses to 0.11 on SciFact), and NEI recall inflates (the model defaults toward "not enough information"), exactly the behaviour expected when the evidence supplied is
unreliable.

## Cross-dataset comparison

| Condition | SciFact F1 | SciFact-Open F1 | Change at scale |
|---|---|---|---|
| No retrieval | 0.4570 | 0.5348 | +0.078 |
| BM25 | 0.4875 | 0.5035 | +0.016 |
| Dense | 0.5424 | 0.4962 | −0.046 |
| Dense + rerank | 0.4115 | 0.4225 | +0.011 |

The most informative row is **dense**: it is the best condition on SciFact but *falls* on SciFact-Open, dropping below its own no-retrieval baseline. The strong retriever that wins on a small corpus loses its advantage — and becomes harmful — on a large one. This crossover is the empirical heart of the thesis: the effect of retrieval is not a fixed property of the method but depends on how hard the retrieval problem is.

## Relevance to the project hypotheses

**Hypothesis: retrieved evidence improves claim verification.** Partly supported, and
conditionally. Retrieval helps on SciFact but hurts on SciFact-Open. The honest, evidenced conclusion is that retrieval's benefit is conditional on corpus difficulty rather than universal, a more nuanced and defensible claim than "retrieval always helps."

**Hypothesis: stance-based reranking improves evidence quality and therefore verification.**
Refuted, consistently and with a diagnosed mechanism. Reranking is the worst condition on both datasets, at both the retrieval level (Step 4) and the classification level (Step 5). A general-domain NLI reranker applied to scientific text systematically demotes true evidence, and this harm carries through the pipeline to final performance.

## Relevance to later steps

**Step 6 (controlled experimental matrix).** Step 5 fixes top_k = 5. Step 6 varies k ∈
{1, 5, 10} across the retrieval conditions to isolate the effect of the number of documents supplied to the classifier — testing, for example, whether fewer documents reduce the noise that harmed retrieval on SciFact-Open, or whether more documents help or dilute performance. The pipeline already supports this via its top_k argument.

**Step 7 (failure taxonomy).** The per-claim records saved by this pipeline (claim, true and predicted label, confidence, retrieved document ids and snippets, and — for the reranked condition — the pre-rerank order) are the direct input to the failure taxonomy. The SciFact-Open retrieval-hurts result and the reranking collapse are prime cases for the "irrelevant retrieval" and "contradictory retrieval" categories.

**Step 8 (confidence analysis).** Each record includes the classifier's confidence (max softmax probability), enabling the Step 8 analysis of whether low-confidence predictions correlate with errors, without re-running the pipeline.

## Result for the thesis

A project titled "Retrieval Effects and Failure Behaviour" is well served by these results. Rather than a single expected outcome, the pipeline produces a **conditional, mechanistically explained picture**: retrieval helps when tractable and hurts when the corpus makes it hard; stance reranking fails consistently for a diagnosed reason. Both the retrieval-at-scale crossover and the reranking failure are the kind of negative and conditional findings that a failure-focused empirical study is designed to surface, and both are supported by evidence across multiple steps rather than asserted.

## Files

- Aggregate metrics: `results/step5_pipeline_scifact_thr0_5.json`,
  `results/step5_pipeline_scifact_open_thr0_5.json`
- Per-claim records (input to Steps 7 and 8): `results/step5_records_scifact_thr0_5.json`,
  `results/step5_records_scifact_open_thr0_5.json`

## Note on run logs

A tokenizer warning ("Token indices sequence length is longer than 512") appears in the logs when a claim+document pair exceeds 512 tokens before truncation. Truncation
(`max_length=512`) is applied, so the model always receives a valid 512-token input; the warning is informational and does not affect results. It reflects that some scientific abstracts are long and are truncated to fit the model's input limit, as expected in any RAG system with a fixed context window.
