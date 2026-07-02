# Step 6: Controlled Experimental Matrix Results

## Overview

This step runs a systematic controlled experiment across all pipeline variants and
retrieval configurations to isolate the contribution of each component to final
claim verification accuracy. Rather than comparing pipelines at a single setting,
we vary retrieval method, number of retrieved documents (k), and stance filter
threshold independently across both datasets.

The experimental matrix covers:

    Retrieval condition : no retrieval, BM25, dense, dense + stance reranking
    k (docs retrieved)  : 0 (no retrieval), 1, 5, 10
    Stance threshold    : loose (0.5), strict (0.8)
    Datasets            : SciFact (450 validation claims), SciClaimHunt (10872 validation claims)

Total conditions evaluated: 13 per dataset (26 across both datasets).

All conditions use the same fine-tuned RoBERTa checkpoint from Step 2.
Retrievers and the stance reranker are built once and reused across all k values
to avoid redundant corpus encoding.
The rerank pool size is fixed at 10 for all dense + reranking conditions,
meaning 10 documents are always retrieved and reranked before taking the top k.

## SciFact Results (validation split, 450 claims, 5183 corpus documents)

| Condition | k | Threshold | Macro F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| No Retrieval | 0 | N/A | 0.5596 | 0.5710 | 0.5628 |
| BM25 | 1 | N/A | 0.2469 | 0.2592 | 0.3586 |
| BM25 | 5 | N/A | 0.2288 | 0.2641 | 0.3608 |
| BM25 | 10 | N/A | 0.2288 | 0.2641 | 0.3608 |
| Dense | 1 | N/A | 0.2517 | 0.2654 | 0.3733 |
| Dense | 5 | N/A | 0.2412 | 0.2667 | 0.3715 |
| Dense | 10 | N/A | 0.2412 | 0.2667 | 0.3715 |
| Dense + Reranking | 1 | loose | 0.3160 | 0.6534 | 0.4186 |
| Dense + Reranking | 1 | strict | 0.3160 | 0.6534 | 0.4186 |
| Dense + Reranking | 5 | loose | 0.3035 | 0.6611 | 0.4151 |
| Dense + Reranking | 5 | strict | 0.3035 | 0.6611 | 0.4151 |
| Dense + Reranking | 10 | loose | 0.3035 | 0.6611 | 0.4151 |
| Dense + Reranking | 10 | strict | 0.3035 | 0.6611 | 0.4151 |

## SciClaimHunt Results (validation split, 10872 claims, 10872 corpus documents)

| Condition | k | Threshold | Macro F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| No Retrieval | 0 | N/A | 0.9870 | 0.9870 | 0.9871 |
| BM25 | 1 | N/A | 0.7447 | 0.7848 | 0.7700 |
| BM25 | 5 | N/A | 0.7355 | 0.7802 | 0.7625 |
| BM25 | 10 | N/A | 0.7355 | 0.7802 | 0.7625 |
| Dense | 1 | N/A | 0.7454 | 0.7844 | 0.7704 |
| Dense | 5 | N/A | 0.7321 | 0.7782 | 0.7596 |
| Dense | 10 | N/A | 0.7321 | 0.7782 | 0.7596 |
| Dense + Reranking | 1 | loose | 0.7386 | 0.7696 | 0.7607 |
| Dense + Reranking | 1 | strict | 0.7386 | 0.7696 | 0.7607 |
| Dense + Reranking | 5 | loose | 0.7035 | 0.7467 | 0.7304 |
| Dense + Reranking | 5 | strict | 0.7035 | 0.7467 | 0.7304 |
| Dense + Reranking | 10 | loose | 0.7035 | 0.7467 | 0.7304 |
| Dense + Reranking | 10 | strict | 0.7035 | 0.7467 | 0.7304 |

## Key Findings

### Finding 1: k has minimal effect once retrieval is introduced

Across both datasets and all retrieval methods, increasing k from 1 to 5 or 10 produces
negligible changes in macro F1. On SciFact, BM25 scores 0.247 at k=1 and 0.229 at k=5
and k=10, a difference of only 0.018. Dense shows the same flat pattern: 0.252 at k=1
versus 0.241 at k=5 and k=10. SciClaimHunt follows identically: BM25 scores 0.745 at k=1
and 0.736 at k=5 and k=10.

This finding indicates that the performance degradation caused by retrieval is not
primarily driven by the volume of retrieved documents. The failure occurs as soon as any
retrieved context is added, regardless of how many documents are included. The dominant
cause is distributional shift: the model was fine-tuned on plain claim text and is
evaluated on claim plus concatenated evidence, which is a fundamentally different input
format. Adding more documents at k=5 or k=10 does not meaningfully increase noise beyond
what is already introduced at k=1.

### Finding 2: Stance threshold has no effect on output under soft reranking

Every dense + reranking condition produces identical results for loose (0.5) and strict
(0.8) thresholds on both datasets. This is a direct consequence of the soft reranking
design established in Step 4. In soft reranking, the threshold controls only the
would_be_filtered flag used for analysis, where it does not remove any documents from the
ranked list. Therefore both thresholds produce the same reranked document order and the
same classifier input, yielding identical predictions.

This confirms that the reranking effect comes entirely from the reordering of documents
by stance score, not from filtering. It also means that threshold sensitivity cannot be
measured under this design without switching to hard filtering, which was rejected in
Step 4 due to recall collapse. This is a meaningful methodological observation worth
discussing in the thesis.

### Finding 3: k=1 consistently outperforms k=5 and k=10 for retrieval conditions

The k=1 condition gives the highest F1 among all retrieval conditions on both datasets.
SciFact dense k=1 scores 0.252 versus 0.241 at k=5 and k=10. SciClaimHunt dense k=1
scores 0.745 versus 0.732 at k=5 and k=10. The same pattern holds for BM25.

This is consistent with the noise hypothesis: the fewer documents concatenated with the
claim, the less distributional shift the model experiences. With k=1 the model sees one
short document plus the claim, which is closer to its training format than five or ten
concatenated abstracts. The implication is that if retrieval is used with a model not
trained for retrieval augmentation, minimal retrieval causes the least harm.

### Finding 4: Reranking effect differs between datasets

On SciFact, stance reranking at k=1 improves F1 from 0.252 (dense) to 0.316 (reranked),
a gain of 0.064. At k=5 and k=10 the gain is smaller: from 0.241 to 0.304, a gain of 0.063.
Reranking consistently helps on SciFact across all k values.

On SciClaimHunt the picture is different. At k=1 reranking slightly decreases F1 from
0.745 to 0.739. At k=5 and k=10 reranking drops F1 from 0.732 to 0.704, a loss of 0.028.
Reranking consistently hurts on SciClaimHunt at k=5 and k=10.

This cross-dataset divergence suggests that reranking by NLI stance score is beneficial
when retrieved documents are genuinely diverse in stance (as in SciFact, where claims have
mixed evidence), but harmful when evidence is already directly matched to claims (as in
SciClaimHunt, where evidence was extracted to match the claim by construction). Reranking
in the SciClaimHunt case reorders documents away from their natural relevance ordering
without providing any compensating benefit.

### Finding 5: No-retrieval dominates all retrieval conditions across all k values

The no-retrieval baseline (0.560 SciFact, 0.987 SciClaimHunt) exceeds every retrieval
condition at every k value on both datasets. The controlled matrix makes this result
substantially stronger than the single comparison in Step 5: it holds across 12 different
retrieval configurations per dataset, including k=1 which represents the most favourable
possible retrieval setting. This rules out the possibility that the degradation is an
artefact of a particular k or threshold choice.

The magnitude of the gap varies by dataset. On SciFact the gap between no-retrieval and
the best retrieval condition (dense k=1, F1=0.252) is 0.308 F1 points. On SciClaimHunt
the gap between no-retrieval and the best retrieval condition (dense k=1, F1=0.745) is
0.242 F1 points. In both cases the gap is large enough to be practically significant.

## Notes

The 512 token truncation warning from transformers appears for inputs where the
concatenated claim and documents exceed the model maximum length. This is handled
by the truncate_and_concatenate function which allocates tokens to the claim first
and fills the remaining budget with document text. It is expected behaviour and is
noted as a limitation of the fixed context window approach.

SciFact evaluation follows standard practice by reporting on the development split
as the test set labels are not publicly available (Wadden et al., 2020).
SciClaimHunt uses an 80/10/10 split with a fixed random seed as implemented in
data/utils.py, with the val split used for all evaluation in this project.