# Step 7 Failure Taxonomy: Annotation Guide

This guide defines the failure categories and the decision precedence used to assign exactly
one primary category to each error during manual annotation. It is committed **before** the
manual annotation is carried out, so the taxonomy is hypothesis-driven, not post-hoc. Each
error also carries a free-text `annotation_notes` field explaining the decision.

## What the annotator sees per error

Each exported row includes: the claim, true and predicted labels, confidence, whether a gold
document was retrieved (`gold_doc_retrieved`), correctness at each retrieval depth
(`correct_by_k`) and the first harmful transition, all retrieved documents actually given to
the classifier (`retrieved_documents`, with rank, id, score, stance score, gold flag, text),
and automatic diagnostic signals (`sig_*`). The automatic signals are aids only; the human
makes the final judgement.

## The four categories

1. **irrelevant_retrieval**: the retrieved context is unrelated, only broadly topical, or
   insufficiently claim-specific to provide useful evidence. Gold-evidence absence
   (`sig_gold_evidence_missing`) supports this decision but does not determine it by itself:
   a retrieved document can be topically related yet non-evidential. The annotator reads the
   documents to decide.

2. **contradictory_retrieval**: the retrieved context materially favours a label opposing
   the gold label (for example, material arguing against a claim whose gold label is
   SUPPORT). Judged by reading the documents, not from the stance score alone (the saved
   stance score has no direction).

3. **evidence_overload**: adding documents broke a prediction that was correct at a smaller
   depth. Read `correct_by_k` / `first_harmful_transition`: if the claim was correct at a
   smaller k and wrong at a larger k for this condition, it is an overload candidate.

4. **confident_wrong_prediction**: clear, sufficient, label-consistent evidence was visible
   to the classifier, yet it erred with confidence ≥ 0.7. This is the residual
   classifier-side category, assigned only after the retrieval-based categories are excluded.

## Decision precedence (assign the FIRST that applies)

Because more than one description can fit one error, assign by this precedence:

1. **evidence_overload**: if `correct_by_k` shows the same condition was correct at a
   smaller k and wrong at a larger k, label `evidence_overload`. (Checked first so a
   high-confidence overload case is not mislabelled as confident_wrong_prediction.)

2. **contradictory_retrieval**: else, if retrieved context materially favours the wrong
   label, label `contradictory_retrieval`.

3. **irrelevant_retrieval**: else, if the context is unrelated, only broadly topical, or
   not claim-specific, label `irrelevant_retrieval`.

4. **confident_wrong_prediction**: else, if clear, sufficient, label-consistent evidence was
   visible AND confidence ≥ 0.7 but the prediction was wrong, label
   `confident_wrong_prediction`.

To assign `confident_wrong_prediction`, the annotator should verify: (a) a gold or clearly
relevant document was included; (b) its relevant content was actually visible in the input
(not truncated away, check the document's rank and the concatenation order); (c) the
evidence favoured the correct label; (d) the error is not better explained by overload or
contradiction; (e) confidence ≥ 0.7.

## NEI-specific rule

For a gold **NEI** claim, sufficient evidence is by definition unavailable, so:
- **contradictory_retrieval** means retrieved material strongly and misleadingly supports
  SUPPORT or CONTRADICT despite the evidence being insufficient for that conclusion.
- **irrelevant_retrieval** means the context does not directly address the claim.
- **confident_wrong_prediction** applies if clearly relevant material was visible and the
  model still confidently chose the wrong label.
- **evidence_overload** precedence still applies first if `correct_by_k` shows a
  correct-then-wrong transition.

## Confidence threshold

The high-confidence threshold is 0.7 (max softmax probability), fixed before inspecting the
failure results. It is well above the three-class chance level (~0.33). It is an operational
threshold, not a universally established value; a sensitivity check at 0.6 / 0.7 / 0.8 is
reported separately, with 0.7 kept as the primary threshold.

## Automatic signals are aids, not labels

The `sig_*` columns are automatic diagnostics. In particular:
- `sig_gold_evidence_missing` is an evidence-recall signal (the recognised gold document was
  not retrieved); it is **not** proof of topical irrelevance.
- `sig_high_conf_error_with_gold_doc` is consistent with a classifier-side failure after
  useful evidence was retrieved; it does **not** prove one (the gold text may be truncated or
  overwhelmed).
- `sig_strong_stance_document` is a strong-stance signal only; it carries **no** direction
  relative to the gold label and is not a contradiction detector.

## Reliability

After the first pass, a random subset (seed recorded) of roughly 15–20 errors is
re-annotated after a delay, without reference to the first labels. Intra-annotator agreement
(percentage agreement and Cohen's kappa, matched by `condition::claim_id`) is computed with
the `kappa` mode and reported as a consistency check.

