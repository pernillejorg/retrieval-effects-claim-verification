# Step 10: Real-World Claim Collection and Labelling Guide

This guide explains how the seafood and sustainability claims for the Step 10 case study were collected and labelled. It is written before collection starts, so that the choices are pre-committed rather than made up to fit the results afterwards, in the same spirit as the Step 7 annotation guide.

## What Step 10 is testing

Step 10 is not testing whether the pipeline "works" on seafood claims. The pipeline was trained on biomedical science and retrieves from a biomedical corpus (SciFact-Open), so it is expected to struggle on out-of-domain claims. The actual question is whether the findings from the earlier steps transfer to a genuinely different domain. Five specific predictions are carried in, each one a finding from an earlier step:

1. Irrelevant retrieval should dominate the failures (Step 7), because seafood claims mostly have no matching evidence in a biomedical corpus.
2. The stance reranker should still promote confidently mis-stanced documents (Step 4), so it should not rescue out-of-domain retrieval and may make it worse.
3. The confidence signal should be inverted on refutation claims (Step 8), so false seafood claims should tend to be predicted with high confidence.
4. Retrieval should be counterproductive compared with no retrieval (Step 9), so the no-retrieval baseline is run alongside the full pipeline rather than assumed to be worse.
5. Confident wrong predictions should persist (Steps 7 and 8), so errors should mostly arrive at high confidence rather than low.

Collecting the claims with these predictions already written down is what makes the case study hypothesis-driven rather than a demonstration.

## What counts as a claim

A claim is a factual assertion that can in principle be checked against evidence. It states that something is true about the world.

- "Farmed salmon contains more omega-3 than wild salmon" is a claim. It is checkable.
- "Farmed salmon tastes disgusting" is not a claim. It is an opinion, with nothing to verify.
- "The MSC blue label is meaningless greenwashing" is a claim (it asserts the label does not track real sustainability, which can be checked against how the certification works).
- "We should all stop eating fish" is not a claim. It is a recommendation, not an assertion of fact.

Only factual assertions are collected. Opinions, recommendations, questions and jokes are skipped. 

## Where the claims come from

Claims are taken from public social media posts and comment sections: X (Twitter), Instagram, TikTok, Reddit (for example r/sustainability, r/seafood, r/ZeroWaste), and the comment sections of news articles about overfishing, aquaculture and ocean health. Useful search phrases include things like "farmed salmon toxic", "wild fish healthier", "MSC label greenwashing", "omega-3 supplement better than fish", and "overfishing myth".

The posts are public, but they were written by real people, so the data ethics section below applies.

## Designing the set, not just grabbing it

The set is deliberately built to exercise the pipeline rather than collected at random. Aiming for 25 claims, the set is balanced along three axes.

**True label:** The set needs a spread of SUPPORT (claims that are true), CONTRADICT (claims that are false) and NEI (claims where the science does not clearly settle it). False claims matter most, because prediction 3 (the confidence inversion on refutation) can only be tested on claims that are actually false. A set with no false claims could not test that prediction.

**Corpus fit:** Seafood and sustainability is not one uniform domain relative to a biomedical corpus. Some claims sit close to it: omega-3 and heart health, mercury levels in fish, farmed versus wild nutrition are health claims that genuinely have related evidence in SciFact-Open. Other claims sit far from it: MSC certification, overfishing economics, bycatch policy have no biomedical evidence at all. Including both ends gives a spectrum from "retrieves real evidence" to "completely out of domain", which is more informative than either extreme alone.

**Difficulty:** Some claims should be clear-cut and some genuinely ambiguous, so the case study shows how the system behaves on messy real-world claims and not only on tidy ones.

## How each claim is labelled

Every claim gets a true label, and the label is justified against a citable source rather than personal opinion. The label answers: relative to the current scientific or expert consensus, is this claim supported, contradicted, or not settled?

- **SUPPORT**: the consensus backs the claim.
- **CONTRADICT**: the consensus goes against the claim.
- **NEI (not enough information)**: the evidence is genuinely mixed, or the claim is too vague or too broad for the consensus to settle.

The source is the important part. Each label is backed by something citable: a review or
meta-analysis, a Cochrane review, or a statement from a body such as the FDA, EFSA, FAO, NOAA or the Marine Stewardship Council. The label is the reviewer's own judgement, but it has to be a judgement a cited source would support, so that the ground truth is defensible and not just an assertion.

Where the consensus itself is uncertain, the honest label is NEI. Forcing a SUPPORT or CONTRADICT onto a genuinely unsettled claim would put a wrong ground-truth label into the study.

## Fields recorded for each claim

Recorded in `claims_template.csv`:

- `claim_id`: a short id (c01, c02, ...).
- `claim_text`: the assertion, transcribed as a clean factual statement. Slang and typos are tidied into plain English, but the meaning is kept exactly.
- `source_platform`: where it came from (e.g. "Reddit r/seafood", "X"), with no username.
- `approx_date`: rough month or year, if visible.
- `true_label`: SUPPORT, CONTRADICT or NEI.
- `label_basis`: the citable source the label rests on (e.g. "Cochrane review 2018 on omega-3 and cardiovascular outcomes"; "MSC certification standard").
- `corpus_fit`: `near` if the claim plausibly has biomedical evidence in SciFact-Open, `far` if it clearly does not. This is the reviewer's honest guess before running anything.
- `label_note`: one line on why this label was chosen.

## Data ethics

The claims come from public social media posts written by real people. To respect that:

- Only publicly visible posts are used.
- Claims are transcribed as standalone factual statements. No usernames, handles, profile details or links are recorded, so no individual is identifiable.
- No private, deleted or restricted content is used.
- The claims are used only to analyse system behaviour, never to single out or criticise any person who posted them.

This keeps the case study focused on the claims as linguistic objects rather than on the people, which is both the ethical choice and the scientifically relevant one.

## What happens next

Once the claims are collected and labelled, each one is run through two conditions: the full best pipeline (dense retrieval, stance reranking, RoBERTa, with a confidence score) at k=1, and the no-retrieval baseline. Each claim is then analysed qualitatively against the five predictions above: what was retrieved, whether it was relevant, what the model predicted, how confident it was, and which failure category (if any) the error falls into. The write-up reports which of the five predictions held in this new domain and which did not.