# Step 7 Failure Taxonomy: Annotation Guide

This guide defines the failure categories and the decision precedence used to assign exactly
one primary category to each error during manual annotation. It is committed before the
manual annotation is carried out, so the taxonomy is hypothesis-driven rather than post-hoc.
Each error also carries a free-text `annotation_notes` field explaining the decision.

## What the annotator sees per error

Each exported row includes the claim, true and predicted labels, confidence, whether a
recognised gold document was retrieved (`gold_doc_retrieved`), correctness at each available
retrieval depth (`correct_by_k`), and any smaller-k correct to target-k wrong transition
(`harmful_transition_into_target_k`).

The row also contains the full retrieved documents returned by the retrieval pipeline
(`retrieved_documents`, with rank, document id, retrieval score, stance score, neutral score,
gold flag, and full text), together with the exact final model input when available
(`classifier_input_text`, token counts, and `was_truncated`). The cross-k context includes the
prediction, confidence, retrieved document ids, and classifier input at each available depth.

When records were generated using the current pipeline, each retrieval record carries
`classifier_input_text`, token counts, and `was_truncated`. Older records may lack these
fields; when absent, the export uses the weaker category-4 interpretation and explicitly
reports that the exact classifier input is unavailable.

Automatic diagnostic signals (`sig_*`) are included as aids only. The human annotator makes
the final category judgement.

## The four categories

### 1. `irrelevant_retrieval`

The retrieved context is unrelated, only broadly topical, or insufficiently claim-specific to
provide useful evidence.

Gold-evidence absence (`sig_gold_evidence_missing`) supports this decision but does not
determine it by itself. A retrieved document can be topically related yet still fail to
provide evidence that directly addresses the claim. The annotator must read the retrieved
documents and the classifier input before assigning this category.

### 2. `contradictory_retrieval`

The retrieved context materially favours a label opposing the gold label.

For example, the retrieved evidence may argue against a claim whose gold label is SUPPORT, or
strongly favour SUPPORT when the gold label is CONTRADICT.

This category is assigned by reading the evidence. It must not be inferred from `stance_score`
alone because the saved stance score measures stance strength but does not encode whether the
document entails or contradicts the claim.

### 3. `evidence_overload`

The prediction at the currently annotated target depth was wrong, but the same claim under the
same pipeline condition was correct at at least one smaller retrieval depth.

Read `correct_by_k` and `harmful_transition_into_target_k`. A non-empty
`harmful_transition_into_target_k` identifies a candidate increasing-depth failure.

Manual inspection must then determine whether the degradation is associated with:

- newly added documents introducing distracting or conflicting evidence;
- reduced token allocation to earlier documents because the evidence budget is divided across
  more retrieved documents;
- or both mechanisms together.

A correct-to-wrong transition is evidence of degradation as retrieval depth increases, but the
transition alone does not prove that the mechanism was evidence overload. The annotator should
inspect the cross-k retrieved document ids and the cross-k classifier inputs before assigning
this category.

### 4. `confident_wrong_prediction`

Clear, sufficient, label-consistent evidence was visible in the classifier input (or, for
older records without `classifier_input_text`, judged likely to have been visible), yet the
model made an incorrect prediction with confidence greater than or equal to 0.7.

This is the residual classifier-side category. It should only be assigned after retrieval-based
explanations have been excluded.

To assign `confident_wrong_prediction`, the annotator should verify that:

1. a gold or clearly relevant document was retrieved;
2. the relevant evidence was present in `classifier_input_text`;
3. the visible evidence favoured the correct label;
4. the error is not better explained by evidence overload, contradictory retrieval, or
   irrelevant retrieval; and
5. the prediction confidence was at least 0.7.

For older records without `classifier_input_text`, use the weaker criterion documented by the
export: relevant and label-consistent evidence must appear in the retrieved context and be
judged likely to have survived the input-construction process.

## Decision precedence

Because more than one description can fit the same error, assign the first category that
applies using the following order.

1. **Evidence overload.**
   If `harmful_transition_into_target_k` is non-empty, meaning that the same condition was
   correct at a smaller k and wrong at the currently annotated target k, inspect the cross-k
   context. Assign `evidence_overload` when the comparison supports an increasing-depth
   explanation, such as newly added distracting evidence, reduced token allocation to earlier
   evidence, or both. This category is checked first so that a high-confidence overload case is
   not incorrectly labelled as `confident_wrong_prediction`.

2. **Contradictory retrieval.**
   Otherwise, if the retrieved context materially favours a label opposing the gold label,
   assign `contradictory_retrieval`. The contradiction must be established by reading the
   evidence, not from the stance score alone.

3. **Irrelevant retrieval.**
   Otherwise, if the retrieved context is unrelated, only broadly topical, insufficiently
   claim-specific, or otherwise does not provide useful evidence for the claim, assign
   `irrelevant_retrieval`.

4. **Confident wrong prediction.**
   Otherwise, if clear, sufficient, label-consistent evidence was visible in
   `classifier_input_text`, the model was wrong, and confidence was at least 0.7, assign
   `confident_wrong_prediction`. This is the residual category after overload, contradiction,
   and irrelevance have been excluded.

## NEI-specific rule

For a gold NEI claim, the available evidence is insufficient to justify a definite SUPPORT or
CONTRADICT label. Relevant documents may still discuss the claim. Apply the categories as
follows:

- `contradictory_retrieval` means the retrieved context misleadingly pushes the model toward a
  definite SUPPORT or CONTRADICT decision despite the available evidence being insufficient for
  that conclusion.
- `irrelevant_retrieval` means the retrieved context does not directly or sufficiently address
  the claim.
- `confident_wrong_prediction` applies when relevant, appropriately non-conclusive evidence was
  visible in `classifier_input_text`, yet the model confidently overcommitted to SUPPORT or
  CONTRADICT.
- `evidence_overload` applies first when the prediction at the annotated target k was wrong
  after being correct at a smaller k, and manual inspection supports an increasing-depth
  explanation.

## Confidence threshold

The high-confidence threshold is 0.7, defined as the maximum softmax probability over the three
predicted classes.

The threshold was fixed before inspecting the failure-analysis results. It is well above the
three-class chance level of approximately 0.33.

The value is an operational threshold rather than a universally established calibration
standard. A sensitivity analysis at 0.6, 0.7, and 0.8 is reported separately, while 0.7 is
retained as the primary threshold.

The separate 0.5 cut-point used by `sig_strong_stance_document` is unrelated to the 0.7
confidence threshold. It is a descriptive threshold applied to the reranker's `stance_score`,
not to the classifier's prediction confidence.

## Automatic signals are aids, not labels

The `sig_*` columns are automatic diagnostics. They must not be treated as ground-truth failure
categories.

**`sig_gold_evidence_missing`.**
This is an evidence-recall signal indicating that the recognised gold document was not
retrieved. It is not proof that the retrieved documents were topically irrelevant. The
retrieved context may still be related to the claim while failing to contain the recognised
gold evidence.

**`sig_high_conf_error_with_gold_doc`.**
This indicates a high-confidence error for which a recognised gold document was present in the
retrieved set. It is consistent with a classifier-side failure after potentially useful
evidence was retrieved, but it does not prove one. The relevant evidence may have been
truncated, allocated too few tokens, ranked among distracting documents, or overwhelmed by
conflicting context. The annotator must inspect `classifier_input_text`.

**`sig_strong_stance_document`.**
This indicates that the top reranked document had a stance score of at least 0.5. It measures
stance strength only. It does not encode stance direction relative to the gold label and
therefore is not a contradiction detector.

**`correct_by_k` and `harmful_transition_into_target_k`.**
These fields identify correctness changes across retrieval depths. A correct-to-wrong
transition is evidence that increasing k degraded the final prediction, but it is only a
candidate overload signal. The annotator must inspect the documents and classifier inputs
across k to distinguish newly introduced noise from token-budget dilution or general prediction
instability.

## Annotation notes

Every annotated row should include a brief explanation in `annotation_notes`.

The note should identify the evidence that motivated the decision. For example:

- which retrieved document was irrelevant;
- which passage favoured the wrong label;
- which additional document or token-budget change appeared to cause overload;
- or which visible label-consistent evidence the classifier ignored despite high confidence.

The note should be specific enough that the decision can be reviewed during the second
annotation pass.

## Reliability

After the first annotation pass, a reproducibly selected subset of approximately 15–20 exported
errors should be copied into a second-pass CSV and re-annotated after a delay, without reference
to the first labels. The subset-selection procedure and random seed should be recorded
separately.

The script's `kappa` mode then compares the two supplied CSV files, matching rows by the stable
`condition::claim_id::k` annotation key. It reports:

- percentage agreement; and
- Cohen's kappa.

The reliability analysis is an intra-annotator consistency check rather than an inter-annotator
agreement study.
