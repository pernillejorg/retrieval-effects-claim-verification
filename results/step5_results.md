# Step 5 Results: The RAG Pipeline

This document records Step 5, the full retrieval-augmented generation (RAG) pipeline for
scientific claim verification. It integrates every component built in the previous steps —
retrieval (Step 3), stance reranking (Step 4), and the two fine-tuned RoBERTa classifiers
(Step 2) — and evaluates four conditions on each dataset. This is the step where the
project's central questions are answered at the level of final classification performance:
does retrieved evidence help a verifier, and does stance reranking help or hurt?

Step 5 was carried out in **two parts**, reflecting a design issue discovered during the
work:

- **Part A — naive concatenation.** Retrieved documents are concatenated whole and fed to
  the classifier until the 512-token limit is reached. This is the natural first
  implementation and mirrors how prior work (MAPLE) feeds retrieved abstracts.
- **Part B — per-document token budget.** After discovering that Part A saturates the
  context window (below), the evidence budget is instead divided equally across the
  retrieved documents, so that all documents contribute and the retrieval depth k genuinely
  affects the input.

Both parts use retrieval depth **k = 3**, following the depth used by MAPLE (Zeng and
Zubiaga, 2024), which retrieves the top-3 BM25 abstracts as evidence. Part B is the
reported pipeline; Part A is retained as a documented finding that motivates it. The
systematic sweep over k is carried out in Step 6, using the Part B design.

## Pipeline conditions

1. **No retrieval** — Model 1 (claim-only classifier), no evidence context.
2. **BM25 + RoBERTa** — sparse retrieval, then classification with Model 2 (evidence classifier).
3. **Dense + RoBERTa** — dense (mpnet) retrieval, then classification with Model 2.
4. **Dense + soft rerank + RoBERTa** — dense retrieval, soft stance reranking, then Model 2.

The no-retrieval condition uses Model 1 (trained on claim text alone); the three retrieval
conditions use Model 2 (trained on claim + gold evidence pairs). Each classifier is applied
to the input format it was trained on, which is the methodologically correct RAG setup.

## Setup

| Property | Value |
|---|---|
| Classifiers | Model 1 (claim-only, F1 0.5263), Model 2 (claim+evidence, F1 0.6438), from Step 2 |
| Retrievers | BM25 (sparse), all-mpnet-base-v2 (dense) |
| Reranker | cross-encoder/nli-deberta-v3-small, soft mode (from Step 4) |
| Retrieval depth | k = 3 (following MAPLE, Zeng and Zubiaga 2024) |
| Rerank pool size | 10 |
| Neutral threshold | 0.5 (loose) |
| Classifier max length | 512 tokens |
| Metric | macro F1 (precision, recall also reported), present-label scoped |
| Device | CUDA (Google Colab GPU) |

Model 2 was trained on gold evidence but is fed **retrieved** evidence here, which is the
standard "train on gold, test on retrieved" RAG evaluation. Retrieval and reranking are run
live within the pipeline so the exact document text seen by the classifier is consistent
end to end.

---

## Part A — naive concatenation, and the context-window saturation finding

The first implementation concatenated retrieved documents whole, adding each document in
rank order until the 512-token input limit was reached, then stopping. Running the pipeline
at k = 3, k = 5 and k = 10 under this scheme produced **identical results** at every k above
a small value.

Investigation showed why. Scientific abstracts are long (typically 200–300 tokens each).
With a 512-token input and the claim consuming part of it, only about **two whole abstracts
fit** before the budget is exhausted; any further retrieved documents are truncated away and
never reach the classifier. Increasing k beyond ~2 therefore changes nothing, because the
additional documents are discarded. **The effective retrieval depth is bounded by the
context window, not by k.**

This is a genuine finding rather than a mere implementation detail, and it is directly
relevant to prior work. MAPLE (Zeng and Zubiaga, 2024) retrieves the top-3 BM25 abstracts
and feeds them to a 512-token model, and the MAPLE paper itself notes that its
retrieved-evidence instances are lengthy and "may exceed the maximum context length". Our
analysis makes the consequence explicit: with whole-document concatenation, the nominal
retrieval depth (k = 3) overstates the evidence the model actually reads, because the
context window saturates after roughly two abstracts. In other words, the effective depth in
such setups is set by the model's context length, not by the retrieval parameter.

*Note on scope.* We cite MAPLE only for its retrieval depth (k = 3) and its shared
context-length limitation. MAPLE is a methodologically different system — a few-shot
T5-small model measuring semantic-similarity evolution, feeding a logistic classifier —
whereas this project uses a fully fine-tuned RoBERTa classifier reading claim + evidence.
The two are therefore not directly comparable in absolute performance, and no such
comparison is drawn; the connection is limited to retrieval depth and the context-length
observation.

### Part A results (naive concatenation, k = 3)

| Condition | SciFact F1 | SciFact-Open F1 |
|---|---|---|
| No retrieval | 0.5263 | 0.6219 |
| BM25 | 0.5388 | 0.5815 |
| Dense | 0.5666 | 0.5881 |
| Dense + rerank | 0.4939 | 0.4727 |

These are valid results for the naive scheme, but because they saturate above k ≈ 2 they
cannot support a study of retrieval depth. This motivated Part B.

---

## Part B — per-document token budget (reported pipeline)

To make the retrieval depth k a meaningful variable, the evidence token budget is divided
**equally across the k retrieved documents**, rather than filling it with whole documents
until it runs out. Each of the k documents therefore receives an equal share of the
available tokens and contributes to the input, so increasing k genuinely changes what the
model sees (more documents, each proportionally shorter). This removes the saturation of
Part A and is the design used for the reported pipeline and for the Step 6 sweep.

This is a deliberate design choice, justified on experimental grounds: for a controlled
study of k, the independent variable (number of documents) must actually affect the model
input. Equal per-document budgeting guarantees this. (MAPLE did not describe such a scheme;
this is an addition of the present work, motivated by the Part A finding.) The trade-off is
that at larger k each document is truncated more aggressively — a tension between evidence
breadth and per-document depth that Step 6 examines directly.

### Part B results — SciFact (5,183-document corpus, 300 claims), k = 3

| Condition | Macro F1 | Precision | Recall |
|---|---|---|---|
| No retrieval (Model 1) | 0.5263 | 0.5287 | 0.5246 |
| BM25 + RoBERTa | 0.5127 | 0.5585 | 0.5139 |
| Dense + RoBERTa | **0.5583** | 0.5934 | 0.5537 |
| Dense + soft rerank + RoBERTa | 0.4879 | 0.4982 | 0.4909 |

Per-class F1 (SciFact, Part B):

| Condition | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|
| No retrieval | 0.56 | 0.37 | 0.65 |
| BM25 | 0.64 | 0.33 | 0.56 |
| Dense | 0.65 | 0.38 | 0.65 |
| Dense + rerank | 0.59 | 0.27 | 0.60 |

### Part B results — SciFact-Open (500,000-document corpus, 279 claims), k = 3

| Condition | Macro F1 | Precision | Recall |
|---|---|---|---|
| No retrieval (Model 1) | **0.6219** | 0.6236 | 0.6271 |
| BM25 + RoBERTa | 0.5229 | 0.5570 | 0.5162 |
| Dense + RoBERTa | 0.5560 | 0.5694 | 0.5512 |
| Dense + soft rerank + RoBERTa | 0.5075 | 0.5098 | 0.5131 |

Per-class F1 (SciFact-Open, Part B):

| Condition | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|
| No retrieval | 0.64 | 0.51 | 0.71 |
| BM25 | 0.60 | 0.49 | 0.48 |
| Dense | 0.61 | 0.47 | 0.59 |
| Dense + rerank | 0.57 | 0.43 | 0.53 |

---

## Part A vs Part B comparison

| Condition | SciFact A | SciFact B | SciFact-Open A | SciFact-Open B |
|---|---|---|---|---|
| No retrieval | 0.5263 | 0.5263 | 0.6219 | 0.6219 |
| BM25 | 0.5388 | 0.5127 | 0.5815 | 0.5229 |
| Dense | 0.5666 | 0.5583 | 0.5881 | 0.5560 |
| Dense + rerank | 0.4939 | 0.4879 | 0.4727 | 0.5075 |

The no-retrieval condition is identical in both parts (it uses no documents, so truncation
does not apply). The retrieval conditions are slightly **lower** under Part B on SciFact:
spreading the budget across three documents means each is truncated to roughly a third of an
abstract, so the model sees shorter fragments of more documents rather than the full text of
~2. That the naive scheme scores marginally higher at k = 3 is consistent with the
saturation finding — Part A effectively used ~2 near-complete abstracts, while Part B uses 3
partial ones. The value of Part B is not a higher score at a single k, but that it makes k a
real variable, which is required for the Step 6 depth study. On SciFact-Open the reranked
condition is actually higher under Part B, but the overall ordering of conditions is
unchanged.

## Consistency check

In both parts and both datasets, the no-retrieval condition reproduces the Step 2 baseline
exactly (SciFact 0.5263; SciFact-Open zero-shot 0.6219). Because the no-retrieval condition
uses the same Model 1 on the same claim-only inputs as Step 2, this exact match confirms the
pipeline is wired correctly to the earlier steps and that the four conditions are directly
comparable.

## Key findings (Part B, reported)

### 1. On the small corpus (SciFact), dense retrieval helps

Dense retrieval (0.5583) exceeds the no-retrieval baseline (0.5263) by +0.032, and is the
best condition on SciFact. BM25 (0.5127) sits just below the baseline, so on SciFact the
benefit of retrieval is carried specifically by the stronger dense retriever, consistent
with dense retrieval's higher recall in Step 3. When the corpus is small and dense retrieval
is reliable, feeding retrieved evidence to an evidence-trained classifier improves
verification.

### 2. On the large corpus (SciFact-Open), retrieval does NOT help

On the 500,000-document corpus, the no-retrieval baseline (0.6219) is *higher* than every
retrieval condition (BM25 0.5229, dense 0.5560, rerank 0.5075). Adding retrieved evidence
made classification worse. The interpretation is central to the project: when the corpus is
roughly 100× larger, retrieval is substantially harder (as quantified in Step 3, where
recall dropped at scale), so the evidence retrieved is noisier and less reliable, and that
noise misleads the classifier more than the absence of evidence would. Retrieval's benefit
is therefore **not universal — it is conditional on the retrieval task being tractable**, and
it inverts as corpus difficulty grows.

### 3. Stance reranking hurts on both datasets

The dense + soft rerank condition is the **worst** condition on both datasets (SciFact
0.4879, below the no-retrieval baseline; SciFact-Open 0.5075, below plain dense). This
confirms, at the classification level, the failure diagnosed at the retrieval level in
Step 4. Stance reranking degraded retrieval recall in Step 4 because the general-domain NLI
model rates hedged scientific abstracts as overwhelmingly neutral (93.6% of documents scored
neutral > 0.5, mean neutral score 0.924) and promotes a rare confidently-but-wrongly-stanced
document over the true evidence. Step 5 shows this recall damage propagates to final F1: the
classifier, fed reranking's misordered evidence, performs worse than with no reranking. The
CONTRADICT class is hit hard (F1 drops to 0.27 on SciFact), exactly the behaviour expected
when the evidence supplied is unreliable. This failure is consistent across Part A, Part B,
and both datasets, confirming it is a property of the reranker, not of a particular design or
classifier.

## Cross-dataset comparison (Part B, reported)

| Condition | SciFact F1 | SciFact-Open F1 | Change at scale |
|---|---|---|---|
| No retrieval | 0.5263 | 0.6219 | +0.096 |
| BM25 | 0.5127 | 0.5229 | +0.010 |
| Dense | 0.5583 | 0.5560 | −0.002 |
| Dense + rerank | 0.4879 | 0.5075 | +0.020 |

The **dense** row is the most informative: dense is the best condition on SciFact but on
SciFact-Open falls essentially level with its small-corpus value while dropping well below
the (higher) no-retrieval baseline there. The strong retriever that helps on a small corpus
loses its advantage on a large one. This crossover — retrieval helping when tractable and
not helping when the corpus makes it hard — is the empirical heart of the thesis.

## Relevance to the project hypotheses

**Hypothesis: retrieved evidence improves claim verification.** Partly supported, and
conditionally. Dense retrieval helps on SciFact but no retrieval condition helps on
SciFact-Open. The honest, evidenced conclusion is that retrieval's benefit is conditional on
corpus difficulty rather than universal.

**Hypothesis: stance-based reranking improves evidence quality and therefore verification.**
Refuted, consistently and with a diagnosed mechanism, at both the retrieval level (Step 4)
and the classification level (Step 5), across both parts and both datasets.

## Relevance to later steps

**Step 6 (controlled experimental matrix).** Step 5 fixes k = 3. Because Part B makes k a
genuine variable, Step 6 sweeps k ∈ {1, 3, 5, 10} across the retrieval conditions to study
sensitivity to retrieval depth — including the breadth-vs-depth trade-off introduced by
per-document budgeting (more documents, each shorter, as k grows). The Part A saturation
finding is precisely why this sweep uses the Part B design: under Part A the sweep would be
uninformative above k ≈ 2.

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
Several of the differences here — particularly the small gaps between retrieval conditions —
are of a comparable scale to run-to-run noise, and should be read with this floor in mind.
The *direction* of the findings (dense helps on SciFact, no retrieval condition helps on
SciFact-Open, reranking hurts on both) is consistent across parts, datasets, and two
independently-seeded classifier runs; the smaller absolute gaps are indicative rather than
precise. The reranking penalty and the SciFact-Open retrieval gap are the firmest effects.

## Result for the thesis

A project titled "Retrieval Effects and Failure Behaviour" is well served by these results.
Step 5 contributes: (i) a documented context-window saturation finding — nominal retrieval
depth overstates effective evidence when long documents are concatenated whole, which also
illuminates a limitation of prior work (MAPLE); (ii) a design response (per-document budget)
that makes retrieval depth a controllable variable; and (iii) the central conditional result
that retrieval helps on a tractable corpus but not at scale, together with a consistent,
mechanistically explained reranking failure. These are exactly the kind of conditional and
negative findings a failure-focused empirical study is designed to surface.

## Files

- Aggregate metrics: `results/step5_pipeline_scifact_thr0_5.json`,
  `results/step5_pipeline_scifact_open_thr0_5.json`
- Per-claim records (input to Steps 7 and 8): `results/step5_records_scifact_thr0_5.json`,
  `results/step5_records_scifact_open_thr0_5.json`
- Part A (naive concatenation) results are retained separately for the saturation finding.

## Note on run logs

A tokenizer warning ("Token indices sequence length is longer than 512") appears in the logs
when a claim+document pair exceeds 512 tokens before truncation. Truncation is applied, so
the model always receives a valid 512-token input; the warning is informational and does not
affect results.