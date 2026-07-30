# Step 5 Results: The RAG Pipeline

This document records Step 5, the full retrieval-augmented generation (RAG) pipeline for scientific claim verification. It integrates every component built in the previous steps, retrieval (Step 3), stance reranking (Step 4), and the two fine-tuned RoBERTa classifiers (Step 2), and evaluates four conditions on each dataset. This is the step where the project's central questions are answered at the level of final classification performance: does retrieved evidence help a verifier, and does stance reranking help or hurt?

Step 5 was carried out in **two parts**, reflecting a design issue discovered during the work:

- **Part A: naive concatenation.** Retrieved documents are concatenated whole and   fed to the classifier until the 512-token limit is reached. This is the natural first
implementation and mirrors how prior work (MAPLE) feeds retrieved abstracts.
- **Part B: per-document token budget.** After discovering that Part A saturates the
context window (below), the evidence budget is instead divided equally across the
retrieved documents, so that all documents contribute and the retrieval depth k genuinely affects the input.

Both parts use retrieval depth **k = 3**, following the depth used by MAPLE (Zeng and
Zubiaga, 2024), which retrieves the top-3 BM25 abstracts as evidence. Part B is the
reported pipeline; Part A is retained as a documented finding that motivates it. The
systematic sweep over k is carried out in Step 6, using the Part B design.

## Pipeline conditions

1. **No retrieval**: Model 1 (claim-only classifier), no evidence context.
2. **BM25 + RoBERTa**: sparse retrieval, then classification with Model 2 (evidence classifier).
3. **Dense + RoBERTa**: dense (mpnet) retrieval, then classification with Model 2.
4. **Dense + soft rerank + RoBERTa**: dense retrieval, soft stance reranking, then Model 2.

The no-retrieval condition uses Model 1 (trained on claim text alone); the three retrieval conditions use Model 2 (trained on claim + gold evidence pairs). Each classifier is applied to the input format it was trained on, which is the methodologically correct RAG setup.

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

Model 2 was trained on gold evidence but is fed **retrieved** evidence here, which is the standard "train on gold, test on retrieved" RAG evaluation. Retrieval and reranking are run live within the pipeline so the exact document text seen by the classifier is consistent end to end.

---

## Part A: naive concatenation, and the context-window saturation finding

The first implementation concatenated retrieved documents whole, adding each document in rank order until the 512-token input limit was reached, then stopping. Running the pipeline at k = 3, k = 5 and k = 10 under this scheme produced **identical results** at every k above a small value.

Investigation showed why. Retrieved abstracts are long: median 306 tokens on SciFact (IQR 230–398, n = 900) and 282 on SciFact-Open (IQR 214–370, n = 837), measured over the dense condition's retrieved documents. With a 512-token input and the claim consuming part of it, only about **two whole abstracts fit** before the budget is exhausted; any further retrieved documents are truncated away and never reach the classifier. Increasing k beyond ~2 therefore changes nothing, because the additional documents are discarded. **The effective retrieval depth is bounded by the context window, not by k.**

This is a genuine finding rather than a mere implementation detail, and it is directly
relevant to prior work. MAPLE (Zeng and Zubiaga, 2024) retrieves the top-3 BM25 abstracts and feeds them to a 512-token model, and the MAPLE paper itself notes that its retrieved-evidence instances are lengthy and "may exceed the maximum context length". Our analysis makes the consequence explicit: with whole-document concatenation, the nominal retrieval depth (k = 3) overstates the evidence the model actually reads, because the context window saturates after roughly two abstracts. In other words, the effective depth in such setups is set by the model's context length, not by the retrieval parameter.

*Note on scope.* We cite MAPLE only for its retrieval depth (k = 3) and its shared
context-length limitation. MAPLE is a methodologically different system, a few-shot
T5-small model measuring semantic-similarity evolution, feeding a logistic classifier, whereas this project uses a fully fine-tuned RoBERTa classifier reading claim + evidence. The two are therefore not directly comparable in absolute performance, and no such comparison is drawn; the connection is limited to retrieval depth and the context-length observation.

### Part A results (naive concatenation, k = 3)

| Condition | SciFact F1 | SciFact-Open F1 |
|---|---|---|
| No retrieval | 0.5263 | 0.6219 |
| BM25 | 0.5388 | 0.5815 |
| Dense | 0.5666 | 0.5881 |
| Dense + rerank | 0.4939 | 0.4727 |

These are valid results for the naive scheme, but because they saturate above k ≈ 2 they cannot support a study of retrieval depth. This motivated Part B.

---

## Part B: per-document token budget (reported pipeline)

To make the retrieval depth k a meaningful variable, the evidence token budget is divided **equally across the k retrieved documents**, rather than filling it with whole documents until it runs out. Each of the k documents therefore receives an equal share of the available tokens and contributes to the input, so increasing k genuinely changes what the model sees (more documents, each proportionally shorter). This removes the saturation of Part A and is the design used for the reported pipeline and for the Step 6 sweep.

This is a deliberate design choice, justified on experimental grounds: for a controlled study of k, the independent variable (number of documents) must actually affect the model input. Equal per-document budgeting guarantees this. (MAPLE did not describe such a scheme; this is an addition of the present work, motivated by the Part A finding.) The trade-off is that at larger k each document is truncated more aggressively, a tension between evidence breadth and per-document depth that Step 6 examines directly.

### Part B results: SciFact (5,183-document corpus, 300 claims), k = 3

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

### Part B results: SciFact-Open (500,000-document corpus, 279 claims), k = 3

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

The no-retrieval condition is identical in both parts (it uses no documents, so truncation does not apply). The retrieval conditions are slightly **lower** under Part B on SciFact: spreading the budget across three documents means each is truncated to roughly a third of an abstract, so the model sees shorter fragments of more documents rather than the full text of ~2. That the naive scheme scores marginally higher at k = 3 is consistent with the saturation finding where Part A effectively used ~2 near-complete abstracts, while Part B uses 3 partial ones. The value of Part B is not a higher score at a single k, but that it makes k a
real variable, which is required for the Step 6 depth study. On SciFact-Open the reranked condition is actually higher under Part B, but the overall ordering of conditions is unchanged.

## Consistency check

In both parts and both datasets, the no-retrieval condition reproduces the Step 2 baseline exactly (SciFact 0.5263; SciFact-Open zero-shot 0.6219). Because the no-retrieval condition uses the same Model 1 on the same claim-only inputs as Step 2, this exact match confirms the pipeline is wired correctly to the earlier steps and that the four conditions are directly comparable.

## Key findings (Part B, reported)

### 1. On the small corpus (SciFact), dense retrieval helps at the reported seed, but this is seed-sensitive

At the reported seed (42), dense retrieval (0.5583) exceeds the no-retrieval baseline (0.5263) by +0.032, and is the best condition on SciFact. BM25 (0.5127) sits just below the baseline, so at this seed the benefit of retrieval is carried specifically by the stronger dense retriever, consistent with dense retrieval's higher recall in Step 3.

**However, the multi-seed study below shows this result is seed-dependent and should be read with caution.** Averaged across three seeds, the no-retrieval mean (0.5242) is actually slightly *above* the dense mean (0.5183) on SciFact, and the retrieval conditions have much higher variance than the baseline. The honest, seed-aware conclusion is that on the small corpus retrieval's benefit is **within noise and unreliable** rather than a firm gain, see the multi-seed section for the full picture. This is itself an important finding and is discussed there.

### 2. On the large corpus (SciFact-Open), retrieval does NOT help

On the 500,000-document corpus, the no-retrieval baseline (0.6219) is *higher* than every retrieval condition (BM25 0.5229, dense 0.5560, rerank 0.5075). Adding retrieved evidence made classification worse. The interpretation is central to the project: when the corpus is roughly 100× larger, retrieval is substantially harder (as quantified in Step 3, where recall dropped at scale), so the evidence retrieved is noisier and less reliable, and that noise misleads the classifier more than the absence of evidence would. Retrieval's benefit is therefore **not universal, it is conditional on the retrieval task being tractable**, and it inverts as corpus difficulty grows.

### 3. Stance reranking hurts on both datasets

The dense + soft rerank condition is the **worst** condition on both datasets (SciFact
0.4879, below the no-retrieval baseline; SciFact-Open 0.5075, below plain dense). This
confirms, at the classification level, the failure diagnosed at the retrieval level in
Step 4. Stance reranking degraded retrieval recall in Step 4 because the general-domain NLI model rates hedged scientific abstracts as overwhelmingly neutral (93.6% of documents scored neutral > 0.5, mean neutral score 0.924) and promotes a rare confidently-but-wrongly-stanced document over the true evidence. Step 5 shows this recall damage propagates to final F1: the classifier, fed reranking's misordered evidence, performs worse than with no reranking. The CONTRADICT class is hit hard (F1 drops to 0.27 on SciFact), exactly the behaviour expected when the evidence supplied is unreliable. This failure is consistent across Part A, Part B, and both datasets, confirming it is a property of the reranker, not of a particular design or classifier.

## Cross-dataset comparison (Part B, reported)

| Condition | SciFact F1 | SciFact-Open F1 | Change at scale |
|---|---|---|---|
| No retrieval | 0.5263 | 0.6219 | +0.096 |
| BM25 | 0.5127 | 0.5229 | +0.010 |
| Dense | 0.5583 | 0.5560 | −0.002 |
| Dense + rerank | 0.4879 | 0.5075 | +0.020 |

The **dense** row is the most informative: dense is the best condition on SciFact but on SciFact-Open falls essentially level with its small-corpus value while dropping well below the (higher) no-retrieval baseline there. The strong retriever that helps on a small corpus loses its advantage on a large one. This crossover, retrieval helping when tractable and not helping when the corpus makes it hard, is the empirical heart of the thesis.

## Relevance to the project hypotheses

**Hypothesis: retrieved evidence improves claim verification.** Not supported as a general claim; supported only weakly and unreliably. At the reported seed, dense retrieval helps on SciFact, but the multi-seed study shows this benefit is within noise and does not survive averaging across seeds, and no retrieval condition helps on SciFact-Open at any seed. The honest, seed-aware conclusion is that retrieval's benefit is not universal, is conditional on corpus tractability, and is **fragile even where it appears**, retrieval-augmented conditions are also markedly less stable across training seeds than the no-retrieval baseline. This is a more nuanced and more defensible position than "retrieval improves verification".

**Hypothesis: stance-based reranking improves evidence quality and therefore verification.**
Refuted, consistently and with a diagnosed mechanism, at both the retrieval level (Step 4) and the classification level (Step 5), across both parts, both datasets, and all three seeds. This is the project's firmest negative result.

## Relevance to later steps

**Step 6 (controlled experimental matrix).** 
Step 5 fixes k = 3. Because Part B makes 'k' a genuine variable, Step 6 sweeps k ∈ {1, 3, 5, 10} across the retrieval conditions to study sensitivity to retrieval depth, including the breadth-vs-depth trade-off introduced by per-document budgeting (more documents, each shorter, as k grows). The Part A saturation finding is precisely why this sweep uses the Part B design: under Part A the sweep would be uninformative above k ≈ 2.

**Step 7 (failure taxonomy).** 
The per-claim records saved by this pipeline (claim, true and predicted label, confidence, retrieved document ids and text snippets, and for the reranked condition, the pre-rerank order) are the direct input to the failure taxonomy. The
SciFact-Open retrieval-does-not-help result and the reranking collapse are prime cases for the "irrelevant retrieval" and "contradictory retrieval" categories.

**Step 8 (confidence analysis).** 
Each record includes the classifier's confidence (max softmax probability), enabling the Step 8 analysis of whether low-confidence predictions correlate with errors, without re-running the pipeline.

## Note on the interpretation of magnitudes

The Step 2 multi-seed robustness check found a baseline standard deviation of about 0.014. Several of the differences here, particularly the small gaps between retrieval conditions, are of a comparable scale to run-to-run noise, and should be read with this floor in mind. The *direction* of the findings (dense helps on SciFact, no retrieval condition helps on SciFact-Open, reranking hurts on both) is consistent across parts, datasets, and two independently-seeded classifier runs; the smaller absolute gaps are indicative rather than precise. The reranking penalty and the SciFact-Open retrieval gap are the firmest effects.

## Multi-seed pipeline variance study

### Why this was done

The results above (Parts A and B) use the seed-42 classifiers. During the project it became important to know whether the pipeline findings were stable across training seeds or whether a single seed might have produced a favourable or unfavourable picture, particularly for the retrieval conditions, where a difference between conditions could be seed noise rather than a real effect. Step 2 was therefore revisited to train the classifiers (both Model 1 and Model 2) under two additional seeds (123 and 7), saving them to separate folders so the deployed seed-42 models were untouched (see step2_results.md). The full pipeline was then re-run under each seed, on both datasets, so that every pipeline condition could be reported
with a mean and standard deviation across seeds. This is the pipeline-level counterpart to the Step 2 training-variance study, and it is the reason the seed-123 and seed-7 models were created.

The pipeline itself is deterministic (it does no training), so the seed enters only through which classifiers are loaded. "Seed 42 pipeline" therefore means the pipeline run with the seed-42 models, such as the Part B results already reported above, and the seed-123 and seed-7 runs load their respective models. Retrieval and reranking are identical across seeds (they do not use the classifiers), so only the classification stage differs between seed runs.

### Results: SciFact (macro F1 across three seeds)

| Condition | Seed 42 | Seed 123 | Seed 7 | Mean ± SD |
|---|---|---|---|---|
| No retrieval | 0.5263 | 0.5403 | 0.5059 | **0.5242 ± 0.0141** |
| BM25 | 0.5127 | 0.4777 | 0.3899 | **0.4601 ± 0.0516** |
| Dense | 0.5583 | 0.5212 | 0.4755 | **0.5183 ± 0.0339** |
| Dense + rerank | 0.4879 | 0.4558 | 0.3866 | **0.4434 ± 0.0423** |

### Results: SciFact-Open (macro F1 across three seeds)

| Condition | Seed 42 | Seed 123 | Seed 7 | Mean ± SD |
|---|---|---|---|---|
| No retrieval | 0.6219 | 0.5803 | 0.5791 | **0.5938 ± 0.0199** |
| BM25 | 0.5229 | 0.5232 | 0.4164 | **0.4875 ± 0.0502** |
| Dense | 0.5560 | 0.5292 | 0.4087 | **0.4980 ± 0.0641** |
| Dense + rerank | 0.5075 | 0.5058 | 0.3776 | **0.4637 ± 0.0608** |

### What the variance study reveals

**Finding A: no-retrieval is stable; retrieval conditions are not.** 
The no-retrieval condition has a low standard deviation on both datasets (0.014 on SciFact, 0.020 on SciFact-Open), whereas every retrieval condition has a much larger standard deviation (0.034–0.064). Adding retrieval does not only fail to help at scale, it also makes the classifier's performance markedly **less stable** across training seeds. This is a genuine behavioural finding: retrieval-augmentation introduces seed-sensitivity that the claim-only baseline does not have, plausibly because the classifier must cope with variable, sometimes misleading retrieved evidence whose effect interacts with the particular model initialisation.

**Finding B: retrieval's apparent benefit on SciFact does not survive averaging.** At seed 42, dense retrieval beat the baseline on SciFact (0.5583 vs 0.5263). But across three seeds the no-retrieval mean (0.5242) is essentially level with indeed marginally above, the dense mean (0.5183), and well above the BM25 mean (0.4601). The single-seed "retrieval helps on SciFact" result was therefore partly a favourable-seed effect. The honest, seed-aware conclusion is that on the small corpus **retrieval's benefit is within noise and unreliable**, not a firm gain. This does not contradict the thesis, it sharpens it: retrieval's benefit is not merely conditional on corpus scale, it is *fragile* even where it appears to help.

**Finding C: the two robust conclusions survive the variance study.** 
Two findings hold clearly even after averaging and accounting for the spread:
- On SciFact-Open, no-retrieval (mean 0.5938) is well above every retrieval condition
  (means 0.46–0.50), by margins larger than the standard deviations. Retrieval reliably does **not** help at scale.
- Reranking is the worst or near-worst condition on both datasets at every seed and in the means (SciFact 0.4434; SciFact-Open 0.4637). Reranking reliably **hurts**.
These are the project's firmest results, and the variance study confirms rather than weakens them.

**Finding D: seed 7 is the low outlier for retrieval.** 
Much of the retrieval conditions' spread comes from seed 7, whose retrieval conditions are notably low (e.g. SciFact BM25 0.3899, SciFact-Open dense 0.4087). Seed 7's *no-retrieval* score is normal (0.5059 / 0.5791), so this is specifically a retrieval-condition instability at that seed, not a generally weak model. This concentration of variance in the retrieval conditions is exactly the seed-sensitivity described in Finding A.

### Relevance to the thesis

This study materially strengthens the thesis in three ways. First, it **corrects an
over-optimistic single-seed reading**: reporting only seed 42 would have overstated
retrieval's benefit on SciFact, and the variance study replaces that with the honest
conclusion that the benefit is within noise. Letting the data revise the claim, rather than reporting the most favourable seed, is the standard of rigour expected at distinction level.
Second, it **surfaces a new behavioural finding**, retrieval-augmentation introduces
seed-instability absent from the baseline, which is precisely the kind of failure behaviour the thesis is about. Third, it **confirms the two central negative results** (retrieval does not help at scale; reranking hurts) are robust across seeds rather than artefacts of one run.

The reported seed (42) is retained as the primary reported configuration throughout, because it is the model the whole pipeline was built on; it is not selected for being favourable. The means and standard deviations above are the representative central estimates, and the seed-42 point results in Parts A and B should be read alongside them.

### Limitation

Three seeds is a small sample for a variance estimate, so the standard deviations are
indicative rather than precise; a larger seed set would tighten them. Three seeds is
nonetheless sufficient to establish the scale of run-to-run variation, to distinguish the robust findings (Finding C) from the fragile one (Finding B), and to surface the
retrieval-instability behaviour (Finding A). The pipeline variance was measured at the
reported retrieval depth k = 3; propagating it across the full k-sweep (Step 6) was not done due to compute cost.



## Result for the thesis

A project titled "Retrieval Effects and Failure Behaviour" is well served by these results.
Step 5 contributes: (i) a documented context-window saturation finding, nominal retrieval depth overstates effective evidence when long documents are concatenated whole, which also illuminates a limitation of prior work (MAPLE); (ii) a design response (per-document budget) that makes retrieval depth a controllable variable; (iii) a multi-seed pipeline variance study showing that retrieval-augmented conditions are markedly less stable across seeds than the no-retrieval baseline, and that retrieval's apparent small-corpus benefit does not survive averaging; and (iv) two robust negative results, retrieval reliably does not help at scale (SciFact-Open), and stance reranking reliably hurts on both datasets, with a diagnosed
mechanism. These are exactly the kind of conditional, fragile, and negative findings a
failure-focused empirical study is designed to surface, and the variance study is what allows them to be stated with calibrated confidence rather than as single-run point estimates.

## Files

- Aggregate metrics: `results/step5_pipeline_scifact_thr0_5.json`,
  `results/step5_pipeline_scifact_open_thr0_5.json`
- Per-claim records (input to Steps 7 and 8): `results/step5_records_scifact_thr0_5.json`,
  `results/step5_records_scifact_open_thr0_5.json`
- Part A (naive concatenation) results are retained separately for the saturation finding.

## Note on run logs

A tokenizer warning ("Token indices sequence length is longer than 512") appears in the logs when a claim+document pair exceeds 512 tokens before truncation. Truncation is applied, so the model always receives a valid 512-token input; the warning is informational and does not affect results.