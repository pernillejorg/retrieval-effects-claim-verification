# Step 8 Results: Retrieval-Aware Confidence Scoring

This document records Step 8, the confidence analysis. When RoBERTa classifies a claim it
outputs a softmax distribution over the three labels, and the highest of those three
probabilities is taken as the model's confidence in its own answer. Step 8 asks one focused question about that number: does the model know when it is wrong, and does stance reranking change how well it knows?

## Why this step is needed after Steps 4 to 7

Everything before this point measures whether the system is right. Step 8 measures whether the system can tell when it is not, which is a different and arguably more useful property. A fact-checking system that is wrong 45% of the time is only dangerous if it is wrong *confidently*: if its own uncertainty flagged those cases, a practitioner could route them to a human and the system would still be usable.

Three earlier results make this question concrete rather than generic.

Step 7 defined `confident_wrong_prediction` as an error made at confidence of at least 0.7, and found it was the second largest failure category under plain dense retrieval (31.2% of annotated errors). That was measured on a manually annotated sample of 70 errors. Step 8 measures the same phenomenon across every prediction on both corpora, so the annotated finding can be checked against the full data rather than a sample.

Step 7 also had to set five errors aside as `excluded_below_threshold`, because they were
classifier errors in mechanism but sat at 0.53 to 0.64 confidence, below the 0.7 the category requires. Step 8 quantifies how large that moderate band actually is across all errors, which tells us whether those five were a quirk of the sample or a real population.

Step 6 found that accuracy falls as retrieval depth k grows, the evidence-overload result. It explicitly promised that Step 8 would test whether confidence tracks that decline. If confidence falls as the extra documents damage the prediction, the model is at least partly registering the harm. If confidence stays flat while accuracy falls, the model does not notice that retrieval is hurting it, which is a stronger and more troubling claim.

## Design

| Property | Value |
|---|---|
| Confidence definition | maximum softmax probability over the 3 classes |
| Chance floor | 0.333 (a 3-class maximum cannot be lower) |
| Primary high-confidence threshold | 0.7, carried over unchanged from Step 7 |
| Sensitivity thresholds | 0.5, 0.6, 0.8, 0.9 |
| Conditions | no retrieval, BM25, dense, dense + soft rerank |
| Datasets | SciFact (300 claims), SciFact-Open (279 claims) |
| Reported depth | k = 3, with a separate sweep over k in {1, 3, 5, 10} |
| Metric for correctness | accuracy (see the note below on why this is not macro F1) |
| Intervals | Wilson 95%, the same helper used in Step 7 |
| Seed | 42 (the Step 6 matrix seed) |

Step 8 re-reads the per-claim records the Step 6 matrix already saved. Nothing is retrained and nothing is retrieved again, because every record already stores both `confidence` and the full `probabilities` vector. The whole step runs on a CPU in about a minute.

### A note on accuracy versus macro F1

Steps 5 and 6 report macro F1, which averages per-class F1 and therefore weights the small CONTRADICT class heavily. Step 8 reports plain accuracy, because the question here is about the relationship between confidence and correctness at the level of individual predictions, and accuracy is the natural per-prediction measure. The two numbers are computed from the same records but are not interchangeable, so Step 8 accuracy figures should not be compared directly against Step 6 F1 figures. Where the two are compared below, it is the *ordering* of conditions that is compared, not the values.

### What was measured

1. **Separation.** Mean confidence on correct predictions minus mean confidence on wrong ones. If the model knows anything about its own reliability, this should be positive.

2. **AUROC.** The probability that a randomly chosen correct prediction carried higher confidence than a randomly chosen wrong one, with tied confidences counting as one half. 0.5 means confidence carries no discriminative information about correctness. This is a single discrimination summary, not a calibration curve.

3. **The confidence-accuracy gap.** Mean confidence minus accuracy, in percentage points.
Positive means the model claims more certainty in aggregate than its accuracy earns.

4. **The flagging rule.** Predictions below a threshold are marked unreliable. This reports how much gets flagged, how much more error-prone the flagged group is (the lift), what share of all errors the rule catches, and what accuracy survives on the predictions that are kept.

5. **A per-true-label breakdown**, so behaviour on one class cannot be hidden by the average.

6. **A paired dense-versus-reranked comparison**, matching the two conditions claim by claim.

### On the word calibration

This step measures confidence *discrimination*, not calibration. The confidence-accuracy gap is a single aggregate number and can hide opposing errors, because underconfidence on some predictions and overconfidence on others partly cancel. Establishing calibration properly would need a formal metric such as expected calibration error or a Brier score, which the project scope deliberately excludes. The findings below are therefore phrased in terms of discrimination and alignment, never as "better calibrated".

## Data integrity

Every check passed before any number was computed, on both corpora and at all four depths:

| Check | SciFact | SciFact-Open |
|---|---|---|
| Records validated | 1,200 | 1,116 |
| Confidence not equal to max softmax | 0 | 0 |
| Probability vectors not summing to 1 | 0 | 0 |
| Vectors not of length 3, non-finite, or outside [0, 1] | 0 | 0 |
| Duplicate claim ids within a condition | 0 | 0 |
| Conditions covering the same claim set | yes (300 each) | yes (279 each) |
| Same claims at every depth k | yes | yes |

The last two rows matter for the comparisons that follow. The paired reranking analysis and the cross-k analysis are only meaningful if the conditions and depths evaluate the same claims, and this confirms they do.

## Results: SciFact at k = 3

| Condition | Accuracy | Mean conf. correct | Mean conf. wrong | Separation | AUROC | Conf-acc gap |
|---|---|---|---|---|---|---|
| No retrieval | 55.3% | 0.9283 | 0.8876 | 0.0407 | 0.6469 | 35.7 pp |
| BM25 | 56.0% | 0.8686 | 0.8006 | 0.0680 | 0.6531 | 27.9 pp |
| Dense | **59.3%** | 0.8850 | 0.8056 | **0.0794** | **0.6947** | **26.0 pp** |
| Dense + soft rerank | 54.0% | 0.8976 | 0.8462 | 0.0514 | 0.6061 | 33.4 pp |

## Results: SciFact-Open at k = 3

| Condition | Accuracy | Mean conf. correct | Mean conf. wrong | Separation | AUROC | Conf-acc gap |
|---|---|---|---|---|---|---|
| No retrieval | **62.4%** | 0.9287 | 0.8784 | 0.0503 | 0.6420 | 28.6 pp |
| BM25 | 54.1% | 0.8410 | 0.7883 | 0.0527 | 0.5964 | 27.6 pp |
| Dense | 56.3% | 0.8581 | 0.8034 | **0.0547** | **0.6169** | **27.1 pp** |
| Dense + soft rerank | 51.6% | 0.8481 | 0.8095 | 0.0386 | 0.5658 | 31.3 pp |

## Errors by confidence band (the link back to Step 7)

| Condition | Errors at conf >= 0.7 (SciFact) | In the 0.5 to 0.7 band | Errors at conf >= 0.7 (SciFact-Open) | In the 0.5 to 0.7 band |
|---|---|---|---|---|
| No retrieval | 118/134 (88.1%) | 15/134 (11.2%) | 91/105 (86.7%) | 12/105 (11.4%) |
| BM25 | 97/132 (73.5%) | 35/132 (26.5%) | 88/128 (68.8%) | 37/128 (28.9%) |
| Dense | 91/122 (74.6%) | 29/122 (23.8%) | 85/122 (69.7%) | 37/122 (30.3%) |
| Dense + soft rerank | 108/138 (78.3%) | 27/138 (19.6%) | 100/135 (74.1%) | 33/135 (24.4%) |

The SciFact-Open column here reproduces the Step 7 diagnostic rates exactly (86.7, 68.8, 69.7 and 74.1 percent), which is a useful consistency check: two independently written analysis scripts, `failure_taxonomy.py` and `confidence_analysis.py`, computed the same quantity from the same records and agreed to the decimal. The SciFact error counts here (122 for dense, 138 for reranked) also match the counts `failure_taxonomy.py` reported when it drew the Step 7 annotation sample, which is a further check that the two scripts agree on what counts as an error. The full per-condition figures for both corpora are in `confidence_summary_*.csv`.

## The flagging rule at the primary 0.7 threshold

SciFact:

| Condition | Flagged | Error rate flagged | Error rate kept | Lift | Errors caught | Coverage | Retained accuracy |
|---|---|---|---|---|---|---|---|
| No retrieval | 8.7% | 61.5% | 43.1% | 1.43 | 11.9% | 91.3% | 56.9% |
| BM25 | 22.7% | 51.5% | 41.8% | 1.23 | 26.5% | 77.3% | 58.2% |
| Dense | 19.0% | 54.4% | 37.4% | **1.45** | 25.4% | 81.0% | **62.6%** |
| Dense + soft rerank | 17.7% | 56.6% | 43.7% | 1.29 | 21.7% | 82.3% | 56.3% |

SciFact-Open:

| Condition | Flagged | Error rate flagged | Error rate kept | Lift | Errors caught | Coverage | Retained accuracy |
|---|---|---|---|---|---|---|---|
| No retrieval | 9.3% | 53.8% | 36.0% | **1.50** | 13.3% | 90.7% | **64.0%** |
| BM25 | 26.5% | 54.1% | 42.9% | 1.26 | 31.2% | 73.5% | 57.1% |
| Dense | 25.1% | 52.9% | 40.7% | 1.30 | 30.3% | 74.9% | 59.3% |
| Dense + soft rerank | 23.7% | 53.0% | **46.9%** | **1.13** | 25.9% | 76.3% | 53.1% |

## Confidence by true label (SciFact-Open, k = 3)

| Condition | Label | n | Accuracy | Mean conf. correct | Mean conf. wrong | AUROC |
|---|---|---|---|---|---|---|
| No retrieval | SUPPORT | 116 | 67.2% | 0.9108 | 0.8277 | 0.7149 |
| No retrieval | CONTRADICT | 90 | 45.6% | 0.8946 | 0.9191 | **0.3549** |
| No retrieval | NEI | 73 | 75.3% | 0.9797 | 0.8749 | 0.8848 |
| BM25 | SUPPORT | 116 | 70.7% | 0.8809 | 0.7426 | 0.6804 |
| BM25 | CONTRADICT | 90 | 44.4% | 0.6668 | 0.8112 | **0.2535** |
| BM25 | NEI | 73 | 39.7% | 0.9684 | 0.7974 | 0.9067 |
| Dense | SUPPORT | 116 | 67.2% | 0.8890 | 0.7971 | 0.6275 |
| Dense | CONTRADICT | 90 | 43.3% | 0.6800 | 0.8155 | **0.2695** |
| Dense | NEI | 73 | 54.8% | 0.9714 | 0.7920 | 0.9258 |
| Rerank | SUPPORT | 116 | 58.6% | 0.8809 | 0.8086 | 0.5386 |
| Rerank | CONTRADICT | 90 | 37.8% | 0.6544 | 0.8312 | **0.2216** |
| Rerank | NEI | 73 | 57.5% | 0.9519 | 0.7715 | 0.8955 |

The same pattern appears on SciFact: for no retrieval, CONTRADICT AUROC is 0.1326 with mean
confidence 0.8008 when correct and 0.9389 when wrong; for BM25 it is 0.1696 with 0.6467 correct
and 0.8233 wrong. The complete per-label breakdown for both corpora is in
`confidence_by_label_*.csv`.

## Paired dense versus reranked, matched claim by claim

The condition-level comparison above is unpaired: the claims dense gets right are not the claims reranking gets right, so a difference in means could come from comparing different subsets rather than from reranking itself. The paired analysis matches the two conditions on claim id, so the same claim is compared with itself.

| Transition | SciFact (n) | SciFact-Open (n) | Meaning |
|---|---|---|---|
| correct to correct | 138 | 124 | stable correct |
| correct to wrong | **40** | **33** | harm introduced by reranking |
| wrong to correct | 24 | 20 | error repaired by reranking |
| wrong to wrong | 98 | 102 | persistent error |
| total matched | 300 | 279 | |

Mean per-claim confidence change (reranked minus dense):

| Subset | SciFact | SciFact-Open |
|---|---|---|
| All claims | +0.0213 | -0.0048 |
| Jointly correct | -0.0036 | -0.0086 |
| Jointly wrong | +0.0111 | +0.0215 |

## Confidence across retrieval depth (SciFact)

| Condition | k | Accuracy | Mean conf. | Separation | AUROC | Conf-acc gap |
|---|---|---|---|---|---|---|
| No retrieval | 1 to 10 | 55.3% | 0.9101 | 0.0407 | 0.6469 | 35.7 pp |
| BM25 | 1 | 62.7% | 0.8751 | 0.1007 | 0.6864 | 24.8 pp |
| BM25 | 3 | 56.0% | 0.8386 | 0.0680 | 0.6531 | 27.9 pp |
| BM25 | 5 | 52.7% | 0.8499 | 0.0614 | 0.6429 | 32.3 pp |
| BM25 | 10 | 50.3% | 0.8339 | 0.0612 | 0.6282 | 33.1 pp |
| Dense | 1 | 64.0% | 0.8598 | 0.1024 | 0.7036 | 22.0 pp |
| Dense | 3 | 59.3% | 0.8527 | 0.0794 | 0.6947 | 26.0 pp |
| Dense | 5 | 56.0% | 0.8530 | 0.0711 | 0.6692 | 29.3 pp |
| Dense | 10 | 52.7% | 0.8443 | 0.1002 | 0.6873 | 31.7 pp |
| Rerank | 1 | 48.0% | 0.9274 | **0.0003** | 0.5391 | **44.7 pp** |
| Rerank | 3 | 54.0% | 0.8740 | 0.0514 | 0.6061 | 33.4 pp |
| Rerank | 5 | 57.7% | 0.8624 | 0.0433 | 0.5982 | 28.5 pp |
| Rerank | 10 | 52.0% | 0.8404 | 0.0857 | 0.6691 | 32.0 pp |

## Confidence across retrieval depth (SciFact-Open)

| Condition | k | Accuracy | Mean conf. | Separation | AUROC | Conf-acc gap |
|---|---|---|---|---|---|---|
| No retrieval | 1 to 10 | 62.4% | 0.9098 | 0.0503 | 0.6420 | 28.6 pp |
| BM25 | 1 | 54.5% | 0.8594 | 0.0219 | 0.5411 | 31.4 pp |
| BM25 | 3 | 54.1% | 0.8168 | 0.0527 | 0.5964 | 27.6 pp |
| BM25 | 5 | 54.1% | 0.8186 | 0.0409 | 0.5902 | 27.8 pp |
| BM25 | 10 | 52.3% | 0.8090 | 0.0673 | 0.6232 | 28.6 pp |
| Dense | 1 | 55.6% | 0.8689 | 0.0217 | 0.5281 | 31.3 pp |
| Dense | 3 | 56.3% | 0.8342 | 0.0547 | 0.6169 | 27.1 pp |
| Dense | 5 | 55.2% | 0.8247 | 0.0592 | 0.6136 | 27.3 pp |
| Dense | 10 | 52.7% | 0.8100 | 0.0719 | 0.6335 | 28.3 pp |
| Rerank | 1 | 43.0% | 0.9004 | **-0.0057** | **0.4647** | **47.0 pp** |
| Rerank | 3 | 51.6% | 0.8294 | 0.0386 | 0.5658 | 31.3 pp |
| Rerank | 5 | 53.0% | 0.8181 | 0.0328 | 0.5689 | 28.8 pp |
| Rerank | 10 | 54.8% | 0.8124 | 0.0520 | 0.6060 | 26.4 pp |

## Key findings

### 1. The classifier is substantially overconfident in every condition

Mean confidence sits between 0.83 and 0.93 while accuracy sits between 51% and 62%. The gap between the two is 26 to 36 percentage points in every single condition on both corpora, and it is positive everywhere, meaning the model always claims more certainty in aggregate than its accuracy earns. The bluntest version of the same point is the error breakdown: between 69% and 88% of all errors were made at a confidence of at least 0.7.

This is the population-level confirmation of Step 7's `confident_wrong_prediction` category. Step 7 found that category accounted for 31.2% of the 70 manually annotated dense errors. Step 8 shows that across all 300 claims, 74.6% of dense errors (91 of 122) clear the same 0.7 bar. The manual sample was not unrepresentative.

It also puts Step 7's five excluded cases in context. Those were errors at 0.53 to 0.64
confidence, set aside because they fell below the category's threshold. Step 8 shows that the 0.5 to 0.7 band holds 19.6% to 26.5% of errors in the SciFact retrieval conditions and 24.4% to 30.3% on SciFact-Open. So that band is a real and sizeable population, not a handful of awkward cases, and excluding those five from the four-category counts rather than forcing them in was the right call.

### 2. Confidence discriminates errors, but only weakly

Separation is positive in every condition at k = 3, and AUROC ranges from 0.566 to 0.695. Both say the same thing: confidence carries genuine information about correctness, but not much. An AUROC of 0.62 means that if you pick one correct and one wrong prediction at random, confidence ranks them correctly only about 62% of the time, against 50% for a coin.

The flagging rule makes the practical consequence concrete. At the pre-specified 0.7 threshold, dense on SciFact flags 19% of predictions, and those flagged predictions are 1.45 times more error-prone than the kept ones. Abstaining on them lifts accuracy from 59.3% to 62.6%, but only at 81% coverage, and it catches just 25.4% of the errors. Three quarters of the mistakes survive the filter. The honest summary is that the model's own uncertainty is a usable but weak error signal, and it is nowhere near strong enough to be a safety mechanism on its own.

### 3. Confidence is inverted for CONTRADICT claims, which is the most striking result

The per-label breakdown reveals something the overall averages completely conceal. For CONTRADICT claims, AUROC is *below* 0.5 in every condition on both corpora: 0.3549, 0.2535, 0.2695 and 0.2216 on SciFact-Open, and 0.1326 and 0.1696 for the two SciFact conditions in the printed log. Below 0.5 means the relationship is reversed: on refutation claims the model tends to be *more* confident when it is wrong. The no-retrieval numbers show it plainly, with mean confidence of 0.8946 on correct CONTRADICT predictions and 0.9191 on incorrect ones.

The contrast with the other two labels is stark. NEI discriminates well (0.88 to 0.93) and SUPPORT moderately (0.54 to 0.71). Only CONTRADICT is inverted, and it is inverted everywhere.

The likely mechanism is class imbalance interacting with the softmax. CONTRADICT is the smallest class in both datasets (64 of 300 in SciFact, 90 of 279 in SciFact-Open), so the model predicts it rarely and hesitantly. When it does correctly identify a refutation it does so with modest confidence, and when it wrongly assigns a confident SUPPORT or NEI label to a refutation claim, that error arrives with high confidence. The result is a confidence signal that is actively misleading for exactly the class a fact-checking system most needs to get right.

This matters for the thesis because it qualifies the whole "confidence as an error detector" idea. The aggregate AUROC of around 0.62 is not a uniformly weak signal; it is a decent signal on two classes averaged with an actively harmful one on the third.

### 4. Stance reranking degrades confidence quality as well as accuracy

The reranked condition has the worst confidence discrimination of any retrieval condition on both corpora. On SciFact its AUROC is 0.6061 against dense's 0.6947, with lower separation (0.0514 against 0.0794) and a wider gap (33.4 pp against 26.0 pp). On SciFact-Open it is worse still: AUROC 0.5658 against dense's 0.6169.

The error-band table shows the same thing from another angle. On both corpora the reranked condition has the highest share of high-confidence errors of any retrieval condition: 78.3% of its SciFact errors were made at a confidence of at least 0.7, against dense's 74.6%, and 74.1% against dense's 69.7% on SciFact-Open. So reranking does not simply produce more errors, it produces errors the model is more sure of, which is the worse of the two failure modes for a system whose confidence a user might rely on.

The flagging rule shows the same thing from a practical angle. On SciFact-Open, reranking has the lowest lift (1.13) and, uniquely, its kept predictions are still wrong 46.9% of the time, which is barely better than the 53.0% error rate among the predictions the rule flagged. In other words, under reranking the confidence filter can hardly separate reliable predictions from unreliable ones at all.

The k = 1 result is the clearest demonstration of the mechanism diagnosed in Step 4. At k = 1 the reranked condition supplies exactly one document, the one the stance reranker ranked top, and Step 4 showed that this is usually a confidently-but-wrongly-stanced document rather than the gold evidence. At that depth, confidence separation collapses to 0.0003 on SciFact and to **-0.0057 on SciFact-Open, with an AUROC of 0.4647, below chance**, alongside the largest confidence-accuracy gap anywhere in the study (47.0 pp). Feeding the classifier a single confidently mis-stanced document makes it confidently wrong, and its confidence stops carrying any usable signal at all.

### 5. The paired comparison confirms reranking causes net harm

Matching claim by claim, reranking turns 40 correct dense predictions into wrong ones on SciFact while repairing only 24, and on SciFact-Open it breaks 33 while repairing 20. It breaks about 1.7 times as many predictions as it fixes on both corpora. This is the per-claim counterpart of the Step 6 F1 result and of Step 7's finding that reranking shifts the failure profile toward the categories it was designed to reduce.

The paired confidence changes are small and mostly not decisive, which is worth stating plainly rather than over-reading. On the jointly-correct subset, the cleanest comparison because both the claim and the correctness outcome are held constant, confidence falls very slightly under reranking on both corpora (-0.0036 and -0.0086). On the jointly-wrong subset it rises slightly (+0.0111 and +0.0215), which is the unhelpful direction: reranking makes the model marginally more confident in errors it was going to make anyway. These are descriptive differences on a conditional subset, not a tested effect.

### 6. On SciFact the model barely registers evidence overload, but on SciFact-Open it does

This is the question Step 6 asked Step 8 to answer, and the answer differs by corpus, which is worth reporting honestly rather than flattening into one claim.

On SciFact the model largely fails to notice the damage. Dense accuracy falls 11.3 percentage points from k = 1 to k = 10 (64.0% to 52.7%), while its mean confidence falls only 1.55 points (0.8598 to 0.8443). Confidence registers roughly one seventh of the degradation, and the confidence-accuracy gap consequently widens from 22.0 pp to 31.7 pp. BM25 behaves the same way, with accuracy down 12.4 points against a confidence drop of 4.1 and the gap widening from 24.8 pp to 33.1 pp. So on the tractable corpus, adding documents makes the model worse without making it meaningfully less sure of itself, which is precisely the retrieval-blind failure the step was looking for.

On SciFact-Open the picture reverses. Dense accuracy falls only 2.9 points from k = 1 to k = 10 while mean confidence falls 5.9 points, so the gap actually narrows from 31.3 pp to 28.3 pp. The reranked condition narrows most of all, from 47.0 pp to 26.4 pp. The reason is that on the harder corpus the k = 1 retrieval is much worse to begin with, so there is less accuracy left to lose, while the additional documents still dilute the input and pull confidence down.

The honest conclusion is therefore conditional: confidence fails to track overload on the
tractable corpus, where overload is the dominant failure mode, and tracks it adequately on the large corpus, where retrieval is already failing at every depth. Stating this as a single universal claim would misrepresent half the data.

### 7. The condition ordering matches Steps 5 to 7

At k = 3 on SciFact, Step 8 accuracy orders the conditions dense (59.3%) above BM25 (56.0%), above no retrieval (55.3%), above reranking (54.0%). Step 6 macro F1 at the same depth orders them dense (0.5583), no retrieval (0.5263), BM25 (0.5127), reranking (0.4879). The metrics differ so the middle two swap, but the two endpoints agree exactly: dense is best and reranking is worst under both. On SciFact-Open both steps agree that no retrieval leads and reranking trails. The confidence measures order the conditions the same way as accuracy, with dense highest on AUROC and reranking lowest on both corpora. The picture is consistent across metrics rather than an artefact of any one of them.

## Interpretation for the thesis

### This step moves the project from diagnostic to partly constructive

Steps 3 to 7 characterise how the system fails. Step 8 asks whether the system's own uncertainty could be used to detect those failures, which is the first question in the project with a practical rather than descriptive answer. The answer is a qualified no: confidence detects errors better than chance, but weakly (AUROC 0.57 to 0.69), it is actively misleading on refutation claims, and abstaining on the flagged predictions catches only a quarter to a third of errors while surrendering a fifth to a quarter of coverage. A practitioner could use this signal as one input to a triage rule, but not as a guarantee.

### It closes the loop on the reranker for the fourth time

The project's novel component has now been measured on four independent axes, and every one points the same way. Step 4 showed the stance reranker damages retrieval recall (R@1 from 0.533 to 0.120), with a diagnosed mechanism: a general-domain NLI model rates 93.6% of scientific abstracts as neutral, so reranking promotes whichever document triggers a spurious confident stance. Step 6 showed it never leads on F1 at any depth. Step 7 showed it shifts the failure profile toward the two retrieval-quality categories it was designed to reduce. Step 8 now shows it also degrades the model's ability to know when it is wrong, most dramatically at k = 1 where discrimination falls below chance.

Four measurements, one mechanism, no contradictions between them. That coherence is what makes the negative result a contribution rather than a disappointment: the hypothesis is refuted, and the refutation is explained.

### The CONTRADICT inversion is the most novel single finding

The inverted confidence on refutation claims is not something the project set out to look for, and it is not visible in any aggregate metric. It emerged only because the per-label breakdown was computed. For a fact-checking system it is the most consequential result in the step: the class that matters most for detecting misinformation is precisely the class where the model's confidence points the wrong way. It also suggests a concrete direction for future work, since class-balanced training or per-class confidence thresholds would be the obvious things to try.

## Limitations to state plainly

These are single-seed results. The Step 6 matrix was run with seed 42, and the project's
multi-seed work covers the Step 2 baseline and the Step 5 variance study, not this matrix. The confidence behaviour reported here characterises that run and has not been verified to replicate across training seeds. The saved JSON records this explicitly in its `seed_note` field.

No calibration metric is computed, by design. The confidence-accuracy gap is an aggregate that can conceal opposing errors within the distribution, so it supports claims about aggregate overconfidence but not about calibration in the formal sense.

No significance test is run between conditions. The comparisons are descriptive and are read alongside the Wilson intervals, in line with the project scope. Some of the differences between conditions, particularly on AUROC, are modest relative to the sample sizes involved.

The 0.7 threshold is analytical, not optimal. It was pre-specified and carried over from Step 7 so that both steps use one definition of high confidence. The other four thresholds are reported as sensitivity only, and none of them was selected by inspecting which produced the best retained accuracy, which would be post-hoc test-set optimisation. Note also that a threshold chosen to define a failure category is not automatically the best abstention threshold for a deployed system.

Accuracy and macro F1 are not interchangeable, as noted in the design section. Only the ordering of conditions is compared between Step 6 and Step 8.

## Relevance to later steps

**Step 9 (cross-corpus comparison).** Step 8 provides the third of the three questions Step 9 compares across corpora, after F1 and failure categories: does the confidence-correctness relationship hold under harder retrieval? The answer from the tables above is that it weakens (dense AUROC 0.695 on SciFact against 0.617 on SciFact-Open) but does not break, while the overload-tracking behaviour reverses between the two corpora. Both `confidence_*.json` files are saved for that comparison.

**Step 10 (real-world case study).** The flagging rule gives the case study a concrete mechanism for reporting uncertainty on out-of-domain seafood and sustainability claims, and the CONTRADICT inversion is a specific caution to check for there, since social-media claims that are false are exactly the CONTRADICT case.

## Files

- SciFact analysis: `results/step8_confidence/confidence_scifact_k3.json`
- SciFact-Open analysis: `results/step8_confidence/confidence_scifact_open_k3.json`
- Depth sweeps: `results/step8_confidence/confidence_by_k_{scifact,scifact_open}.json`
- Thesis tables as CSV: `confidence_summary_*`, `confidence_bins_*`, `flagging_tradeoff_*`,
  `confidence_by_label_*`, `paired_reranking_*`, `confidence_by_k_*`
- Analysis script: `analysis/confidence_analysis.py`
- Notebook: `notebooks/Step8_ConfidenceScoring.ipynb`
- Per-label AUROC values are in `results/step8_confidence/confidence_by_label_{scifact,scifact_open}_k3.csv`.

## Reproduction

```bash
# per-condition confidence analysis at the reported depth, one command per dataset
python analysis/confidence_analysis.py --mode analyse --dataset scifact --k 3 \
  --records_dir results/step6_matrix --out_dir results/step8_confidence

python analysis/confidence_analysis.py --mode analyse --dataset scifact_open --k 3 \
  --records_dir results/step6_matrix --out_dir results/step8_confidence

# the depth sweep (no --k here: cross-k mode reads every available depth)
python analysis/confidence_analysis.py --mode cross_k --dataset scifact \
  --records_dir results/step6_matrix --out_dir results/step8_confidence

python analysis/confidence_analysis.py --mode cross_k --dataset scifact_open \
  --records_dir results/step6_matrix --out_dir results/step8_confidence
```