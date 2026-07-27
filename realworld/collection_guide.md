# Step 10: Real-World Claim Collection and Labelling Guide

This guide explains how the seafood and sustainability claims for the Step 10 case study were collected and labelled. It is written before collection starts, so that the choices are pre-committed rather than made up to fit the results afterwards, in the same spirit as the Step 7 annotation guide.

## What Step 10 is testing

Step 10 is not testing whether the pipeline "works" on seafood claims. The pipeline was trained on biomedical science and retrieves from a biomedical corpus (SciFact-Open), so it is expected to struggle on out-of-domain claims. The actual question is whether the findings from the earlier steps transfer to a genuinely different domain. Five directional hypotheses are carried in, each one drawn from a finding of an earlier step. They are written as expectations to be tested, not as results already established:

**H1 (from Step 7).** Retrieval-related failures are expected to involve topically irrelevant or merely related evidence more often for far-fit claims than for near-fit claims, because seafood claims mostly have no matching evidence in a biomedical corpus.

**H2 (from Step 4).** The stance reranker is expected to still promote confidently mis-stanced documents, so it is not expected to rescue out-of-domain retrieval and may reorder it unhelpfully.

**H3 (from Step 8).** On claims whose true label is CONTRADICT, wrong predictions are expected to arrive with relatively high model confidence (the maximum softmax probability), mirroring the inversion found in Step 8. The test is defined on the true label, matching the Step 8 definition.

**H4 (from Step 9).** Retrieval is hypothesised to be no better than, and potentially worse than, the no-retrieval baseline on these out-of-domain claims, so the no-retrieval baseline is run alongside the retrieval conditions rather than assumed to be worse.

**H5 (from Steps 7 and 8).** Confident wrong predictions are expected to persist, so errors are expected to arrive mostly at high model confidence rather than low.

Collecting the claims with these hypotheses already written down is what makes the case study hypothesis-driven rather than a demonstration. Throughout, "confidence" means the maximum softmax probability over the three classes, the same definition used in Step 8. No separate confidence-scoring model is trained.

## What counts as a claim

A claim is a factual assertion that can in principle be checked against evidence. It states that something is true about the world.

- "Farmed salmon contains more omega-3 than wild salmon" is a claim. It is checkable.
- "Farmed salmon tastes disgusting" is not a claim. It is an opinion, with nothing to verify.
- "The MSC blue label is meaningless greenwashing" is a claim (it asserts the label does not track real sustainability, which can be checked against how the certification works).
- "We should all stop eating fish" is not a claim. It is a recommendation, not an assertion of fact.

Only factual assertions are collected. Opinions, recommendations, questions and jokes are skipped.

## Where the claims come from

Claims are taken from public social media posts and comment sections: X (Twitter), Instagram, TikTok, Reddit (for example r/seafood, r/nutrition, r/Supplements), Quora, and Facebook, including the comment sections of posts about seafood, aquaculture and ocean health. Useful search phrases include things like "farmed salmon toxic", "wild fish healthier", "MSC label greenwashing", "omega-3 supplement better than fish", and "overfishing myth".

The posts are public, but they were written by real people, so the data ethics section below applies.

## Designing the set, not just grabbing it

The set is deliberately built to exercise the pipeline rather than collected at random. Aiming for about thirty claims, the set is balanced along three axes.

**True label:** The set needs a spread of SUPPORT (claims that are true), CONTRADICT (claims that are false) and NEI (claims where the science does not clearly settle it). False claims matter, because H3 (the confidence inversion on refutation) can only be tested on claims that are actually false. A set with no false claims could not test that hypothesis.

**Corpus fit:** Seafood and sustainability is not one uniform domain relative to a biomedical corpus. Some claims sit close to it: omega-3 and heart health, mercury levels in fish, farmed versus wild nutrition are health claims that genuinely have related evidence in SciFact-Open. Other claims sit far from it: MSC certification, overfishing economics, bycatch policy have no biomedical evidence at all. Including both ends gives a spectrum from "retrieves real evidence" to "completely out of domain", which is more informative than either extreme alone.

**Difficulty:** Some claims should be clear-cut and some genuinely ambiguous, so the case study shows how the system behaves on messy real-world claims and not only on tidy ones.

## How each claim is labelled

Every claim gets a true label, and the label is justified against a citable source rather than personal opinion. The label answers: relative to the current scientific or expert consensus, is this claim supported, contradicted, or not settled?

- **SUPPORT**: the consensus backs the claim.
- **CONTRADICT**: the consensus goes against the claim.
- **NEI (not enough information)**: the evidence is genuinely mixed, or the claim is too vague or too broad for the consensus to settle.

The source is the important part. Each label is backed by something citable: a review or meta-analysis, a Cochrane review, or a statement from a body such as the FDA, EFSA, FAO, NOAA or the Marine Stewardship Council. The `label_basis` field identifies the principal source or evidence base supporting the label. Where a label rests on a single document, it names that document (for example, “FDA Advice About Eating Fish, 2024”, not just “FDA guidance”); where it rests on a broader consensus or on the absence of supporting evidence, the field summarises that evidence base and the fuller references are recorded in the accompanying source register. This keeps the ground truth defensible and, as far as a qualitative study reasonably allows, reproducible. The label is the reviewer’s own judgement, but it has to be a judgement a cited source would support.

Where the consensus itself is uncertain, the honest label is NEI. Forcing a SUPPORT or CONTRADICT onto a genuinely unsettled claim would put a wrong ground-truth label into the study.

Two labelling disciplines are worth stating explicitly, because real-world claims break them more often than benchmark claims do:

- **One proposition per claim.** If a quote asserts two things (for example "freezing kills parasites, and farming makes no difference"), it is split into separate claims or the claim text is narrowed to the single proposition being labelled, so that one label is never applied to a conjunction where one part is true and the other false.
- **Meaning is preserved exactly.** Tidying slang and typos is allowed, but the normalised `claim_text` must assert the same thing as the `original_quote`. A quote about one substance is not relabelled as a claim about a different substance.

## Fields recorded for each claim

Recorded in `seafood_claims.csv`:

- `claim_id`: a short id (c01, c02, ...).
- `original_quote`: the assertion as it was actually written, kept verbatim so the normalisation can be checked against the source.
- `claim_text`: the assertion, transcribed as a clean factual statement. Slang and typos are tidied into plain English, but the meaning is kept exactly.
- `source_platform`: the platform and, where useful, the board (for example "Reddit r/seafood", "X"), with no username.
- `approx_date`: rough month or year, if visible.
- `true_label`: SUPPORT, CONTRADICT or NEI.
- `label_basis`: the principal citable source or evidence base supporting the label; fuller references are provided in the accompanying source register where the label rests on multiple sources or a broader consensus.
- `corpus_fit`: `near` if the claim plausibly has biomedical evidence in SciFact-Open judged by mechanism (not by fish species), `far` if it clearly does not. This is the reviewer's honest guess before running anything.
- `label_note`: one line on why this label was chosen, including any nuance a reader should know.

## Data ethics

The claims come from public social media posts written by real people. To respect that:

- Only publicly visible posts are used. No private, deleted or restricted content is used.
- Usernames, handles, profile details and links are not retained.
- The claims are used only to analyse system behaviour, never to single out or criticise any person who posted them.

An honest limitation: because a verbatim `original_quote` can sometimes be pasted into a search engine to locate the original post, the data are **pseudonymised rather than fully anonymous**. Removing direct identifiers reduces but does not eliminate the chance of re-identification. The `original_quote` is retained because it is what makes the normalisation checkable, which the study needs; the trade-off is accepted knowingly and stated here rather than overclaimed as full anonymity. Distinctive quotes from unusually small communities are avoided where an equivalent claim is available elsewhere.

## What happens next

Once the claims are collected and labelled, each one is run through three conditions: the no-retrieval baseline (Model 1, claim only), plain dense retrieval into Model 2, and dense retrieval with soft stance reranking into Model 2. The two retrieval conditions are run at matched depths k in {1, 3, 5, 10}. Running plain dense alongside dense-plus-rerank is what lets the analysis separate what retrieval selected from what reranking then did to it; without the plain-dense condition, an irrelevant document in the reranked output could not be attributed to retrieval versus reranking. Note that where k equals the rerank pool size, reranking can only reorder the pool rather than filter documents out of the final set; the effective pool size is recorded per k.

Each claim is then analysed qualitatively against the five hypotheses above, following the pre-committed protocol in `results_annotation_guide.md`: what was retrieved, whether it was relevant, whether reranking changed the selected evidence, what the model predicted, how confident it was, and which failure category (if any) the error falls into. Accuracy over these thirty deliberately chosen claims is treated as an orienting count, not as an estimate of real social-media performance. The write-up reports which of the five hypotheses held in this new domain and which did not.
