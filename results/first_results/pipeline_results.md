# Step 5: RAG Pipeline Evaluation Results

## Overview

This step connects all components built in previous steps into four complete end-to-end
pipeline conditions. Each pipeline takes a claim as input and produces a label prediction
of SUPPORT, CONTRADICT, or NEI (SciFact) or Supported/Refuted (SciClaimHunt). The four
conditions isolate the effect of each component so we can measure exactly what retrieval
and reranking contribute to final verification accuracy.

The four conditions evaluated are:

1. No Retrieval: the fine-tuned RoBERTa model classifies the claim alone with no retrieved context
2. BM25 + RoBERTa: top 5 BM25 documents are concatenated with the claim before classification
3. Dense + RoBERTa: top 5 dense retrieval documents are concatenated with the claim before classification
4. Dense + Stance Reranking + RoBERTa: top 10 dense documents are reranked by stance score and top 5 passed to the classifier

All four pipelines use the same fine-tuned RoBERTa checkpoint from Step 2.
Documents are concatenated to the claim with a [SEP] separator and truncated to fit within 512 tokens.
Evaluation is on the validation split for both datasets (SciFact has no public test set with gold labels).
top_k=5 documents are passed to the classifier, rerank_pool_size=10 for the reranked condition.

## SciFact Results (validation split, 450 claims)

| Pipeline | Macro F1 |
|---|---|
| No Retrieval (RoBERTa only) | 0.5596 |
| BM25 + RoBERTa | 0.2288 |
| Dense + RoBERTa | 0.2412 |
| Dense + Stance Reranking + RoBERTa | 0.3035 |

### Per-class breakdown

No Retrieval:

    SUPPORT:    precision 0.62  recall 0.65  f1 0.64
    CONTRADICT: precision 0.58  recall 0.40  f1 0.47
    NEI:        precision 0.52  recall 0.63  f1 0.57

BM25 + RoBERTa:

    SUPPORT:    precision 0.52  recall 0.18  f1 0.27
    CONTRADICT: precision 0.00  recall 0.00  f1 0.00
    NEI:        precision 0.27  recall 0.90  f1 0.42

Dense + RoBERTa:

    SUPPORT:    precision 0.52  recall 0.20  f1 0.29
    CONTRADICT: precision 0.00  recall 0.00  f1 0.00
    NEI:        precision 0.28  recall 0.91  f1 0.43

Dense + Stance Reranking + RoBERTa:

    SUPPORT:    precision 0.69  recall 0.30  f1 0.42
    CONTRADICT: precision 1.00  recall 0.02  f1 0.05
    NEI:        precision 0.29  recall 0.92  f1 0.44

## SciClaimHunt Results (validation split, 10872 claims)

| Pipeline | Macro F1 |
|---|---|
| No Retrieval (RoBERTa only) | 0.9870 |
| BM25 + RoBERTa | 0.7355 |
| Dense + RoBERTa | 0.7321 |
| Dense + Stance Reranking + RoBERTa | 0.7035 |

### Per-class breakdown

No Retrieval:

    Supported: precision 0.98  recall 0.99  f1 0.99
    Refuted:   precision 0.99  recall 0.99  f1 0.99

BM25 + RoBERTa:

    Supported: precision 0.63  recall 0.94  f1 0.75
    Refuted:   precision 0.93  recall 0.58  f1 0.72

Dense + RoBERTa:

    Supported: precision 0.63  recall 0.94  f1 0.75
    Refuted:   precision 0.93  recall 0.58  f1 0.71

Dense + Stance Reranking + RoBERTa:

    Supported: precision 0.60  recall 0.91  f1 0.73
    Refuted:   precision 0.89  recall 0.55  f1 0.68

## Key Findings

The central finding across both datasets is that adding retrieved evidence to a model
fine-tuned without retrieval augmentation consistently degrades classification performance. 

This is a direct empirical demonstration of retrieval-induced distributional shift:
the model was trained on plain claim text and is evaluated on claim plus concatenated
evidence, which is a fundamentally different input format.

On SciFact, BM25 and Dense retrieval cause macro F1 to drop from 0.56 to 0.23,
a collapse of more than 50 percent. The per-class breakdown reveals the mechanism:
both retrieval pipelines default to predicting NEI for almost all claims (NEI recall 0.90+), effectively abandoning the CONTRADICT class entirely. This is the model collapsing to the safest prediction when faced with an unfamiliar input format.

Stance-aware reranking partially recovers this degradation on SciFact (0.30 vs 0.23).
The reranked pipeline shows very high precision on CONTRADICT (1.00) but near-zero recall (0.02), indicating the reranker is highly selective and only fires when it is very confident. This conservative behaviour is consistent with the findings from Step 4, where zero-shot NLI on scientific text tends to assign high neutral scores, meaning few documents pass the stance threshold even in soft reranking mode.

On SciClaimHunt, the same degradation pattern holds but at a different scale. The no-retrieval baseline achieves near-perfect performance (0.987) on this binary task, and retrieval drops this to around 0.73 for BM25 and Dense. The reranked pipeline performs slightly worse than plain dense retrieval (0.70 vs 0.73), suggesting that stance reranking on SciClaimHunt evidence adds noise rather than signal. This is likely because SciClaimHunt evidence is already directly matched to claims by construction, so reranking by NLI stance score does not improve ordering.

The contrast between the two datasets is itself a meaningful finding: SciFact shows that
reranking helps relative to plain retrieval, while SciClaimHunt shows it does not.
This dataset-level difference is worth exploring further in the failure analysis steps.

## Notes

The 512 token truncation warning from transformers appears for some claim-document pairs
where the concatenated input exceeds the model maximum length. This is expected and handled by the truncate_and_concatenate function which gives priority to the claim and fits as many document tokens as the remaining budget allows. It is noted as a limitation of the fixed context window approach.

SciFact evaluation follows standard practice in the field (Wadden et al., 2020) by reporting on the development split, as the SciFact test set labels are not publicly available.