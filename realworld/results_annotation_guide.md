# Step 10: Results Annotation Guide

This guide fixes how each claim's pipeline output will be annotated, and it is written before the results are inspected. Pre-committing the annotation scheme is what keeps the qualitative analysis systematic rather than a set of after-the-fact anecdotes, in the same spirit as the Step 7 annotation guide. The categories deliberately reuse the Step 7 failure taxonomy rather than inventing a new scheme, so that the case-study findings sit on the same footing as the biomedical failure analysis and can be compared with it directly.

The run produces, for every claim, three conditions: no-retrieval (Model 1), plain dense into Model 2, and dense with soft reranking into Model 2, with the two retrieval conditions at k in {1, 3, 5, 10}. Each claim is annotated once at the anchor depth k = 3 (the SciFact-Open optimum used elsewhere in the project), and the other depths are consulted only where they change the reading. Every field below is recorded per claim in a case table.

## Fields recorded per claim

**claim_id.** The identifier from the claim set.

**corpus_fit.** The pre-registered near or far value from the claim set, carried over unchanged. This is the axis the hypotheses are tested along, so it is never reassigned after seeing the output.

**dense_evidence_relevance.** A human judgement of the evidence that plain dense retrieval selected. The first three values capture relevance (does the document bear on the claim at all); the fourth captures the one stance case that matters for the failure taxonomy, namely relevant-but-stance-opposed. This keeps a single field rather than a separate relevance-and-stance grid, which is more than a thirty-claim qualitative study needs. Use one of:
- `directly_evidential`: a retrieved document actually bears on the truth of the claim, in a way consistent with the true label.
- `topically_related_only`: retrieved documents are on a related topic but do not address the claim.
- `irrelevant`: retrieved documents are not even on topic.
- `contradictory_to_claim`: retrieved documents are relevant to the claim but stance-opposed: their stance would push the model toward an incorrect prediction relative to the assigned true label (for example, evidence supporting a claim whose true label is CONTRADICT, or contradicting a claim whose true label is SUPPORT).

**reranked_evidence_relevance.** The same judgement applied to the documents that dense-plus-rerank selected. Recording both is what allows the reranker's effect to be read off rather than inferred.

**reranker_effect.** How reranking changed the selected evidence relative to plain dense, using one of:
- `improved`: reranking brought more relevant evidence into the top-k.
- `unchanged`: reranking could have changed the selected set but the practical evidence quality is the same (for example it reordered documents of equal relevance).
- `worsened`: reranking pushed more relevant evidence out or promoted less relevant evidence.
- `reordering_only`: k equals the effective candidate-pool size, so reranking cannot change which documents are selected, but it may change their order and therefore the classifier input (including what survives truncation). Any observed effect is attributable to reordering rather than to document inclusion or exclusion. The `improved`, `unchanged` and `worsened` values are used only when reranking could actually change the selected set.

**reranker_score_note.** For cases where reranking promotes a stance-opposed document, record that document's reranker stance label and score, taken directly from the pipeline record (the `stance_score` and `neutral_score` fields). Left blank otherwise. This is what lets H2's word "confidently" rest on recorded evidence rather than impression.

**Per-condition prediction, confidence, and failure.** Because each claim is run under three conditions (no-retrieval, dense, dense+rerank) and a claim can be correct under one and wrong under another, these three fields are recorded separately for each condition rather than once per claim. At the anchor depth k = 3, record for each of the three conditions. Spelled out in full, the nine fields are: `no_retrieval_prediction`, `no_retrieval_confidence`, `no_retrieval_failure_category`, `dense_prediction`, `dense_confidence`, `dense_failure_category`, `dense_reranked_prediction`, `dense_reranked_confidence`, `dense_reranked_failure_category`. In the descriptions below `{condition}` stands for one of `no_retrieval`, `dense`, or `dense_reranked`:
- `{condition}_prediction`: the predicted label.
- `{condition}_confidence`: the maximum softmax probability over the three classes (the Step 8 definition; not a separate score).
- `{condition}_failure_category`: if that condition's prediction was wrong, the attributed failure, else blank.

The failure categories (shared across conditions) are the Step 7 taxonomy plus one addition for real-world claims:
1. `irrelevant_retrieval`: the model was given evidence that does not bear on the claim.
2. `contradictory_retrieval`: the model was given evidence pointing against the correct label.
3. `model_error_despite_adequate_input`: the model was wrong despite directly relevant retrieved evidence, or was wrong in the no-retrieval condition where retrieval quality cannot explain the error. This one category works across all three conditions.
4. `label_or_claim_limited`: the error arises from the claim's formulation or the reference label rather than the model: a vague, compound, or genuinely contested claim where the assigned true label is itself arguable.

Recording prediction, confidence, and failure per condition is what keeps H3, H4, and H5 unambiguous: a confidence or failure judgement always states which condition it refers to.

**ambiguity_type.** For claims that are hard to label or hard for the model because of how the claim itself is phrased, one short tag from: `vague_quantifier`, `overgeneralisation`, `compound_claim`, `causal_overclaim`, `scope_ambiguity`, `contested_definition`, `time_sensitive`, `geographically_variable`. Left blank for clean claims. This keeps the real-world messiness visible without turning the analysis into a taxonomy exercise.

**notes.** One or two sentences of interpretation: what happened on this claim and which hypothesis it speaks to.

## How the fields feed the hypotheses

- **H1** is read from `dense_evidence_relevance`, `reranked_evidence_relevance`, `dense_failure_category` and `dense_reranked_failure_category`, split by `corpus_fit`. The no-retrieval condition does not contribute, since H1 is about retrieval failures.
- **H2** is read from `reranker_effect`, the pairing of `dense_evidence_relevance` with `reranked_evidence_relevance`, and `reranker_score_note` for promoted stance-opposed documents.
- **H3** is read from the per-condition confidence fields on incorrect predictions for claims whose true label is CONTRADICT.
- **H4** is read from per-condition accuracy in the run summary (a between-condition comparison), primarily at the anchor depth k = 3.
- **H5** is read from the per-condition confidence fields across all incorrect predictions.

"High confidence" is defined comparatively, fixed before seeing results: H3 is supported directionally if the mean maximum-softmax probability among incorrect predictions on true-CONTRADICT claims exceeds that among incorrect predictions on true-SUPPORT and true-NEI claims. H5 is supported directionally if a majority of incorrect predictions have a maximum-softmax probability of at least 0.80.

## Discipline

- Annotation is performed after all pipeline runs are complete, not during, so that seeing partial results cannot subconsciously shape later judgements.
- The near/far value and the true label are fixed before annotation and are not revised to fit what the model did. If annotation reveals that a true label was genuinely wrong (not merely unfavourable to a hypothesis), that is recorded explicitly as a correction with its reason, not silently changed.
- Accuracy is an orienting count over about thirty deliberately chosen claims. It is never reported as an estimate of how the system would perform on social media at large.
- Where the pipeline was wrong for more than one reason, the single most proximate cause is chosen for `failure_category`, and the secondary reason goes in `notes`, so the counts stay clean.
