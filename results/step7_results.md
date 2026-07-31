# Step 7: Failure Taxonomy

## What this step does and why

By this point in the project the pipeline has produced predictions across the different retrieval conditions, and a fair number of them are wrong. Step 7 is where those errors stop being just a lower F1 number and start being explained. The point is to describe why the system fails, and in particular to test whether the stance reranker changes the pattern of failures in the way it was designed to.

The four failure categories were written down and committed to before any error was looked at. This matters for the thesis because it keeps the analysis hypothesis-driven rather than post-hoc. I am not inventing categories to fit whatever I happened to see; I am applying a taxonomy that was fixed in advance. Two of the four categories, irrelevant retrieval and contradictory retrieval, are exactly the ones the stance reranker was built to reduce, so annotating the errors is also a direct test of my own contribution rather than a generic error count. That loop between the design of the reranker and the evaluation of it is the part that makes this more than descriptive error analysis.

Following the scope decision agreed earlier for a solo project on a ten-week timeline, SciFact gets the full manual annotation and is treated as the primary failure analysis. SciFact-Open gets only automatic quantitative signals across conditions, with no hand labelling. This is an honest limit on effort that still produces a meaningful cross-corpus comparison.

## The four categories (fixed before annotation)

1. irrelevant_retrieval. The retrieved documents are unrelated to the claim, or     only loosely on-topic, so they give the classifier nothing it could actually use to reach the right label.

2. contradictory_retrieval. The retrieved documents point towards a label that opposes the gold label. The error is caused by evidence pointing the wrong way, not by evidence being missing.

3. evidence_overload. Adding more documents broke a prediction that had been correct at a smaller k. This is read from the cross-k information saved with each row, and is only assigned when a real correct-to-wrong transition with increasing k actually exists for that claim.

4. confident_wrong_prediction. Label-consistent evidence was present in the retrieved context and was judged likely to have reached the classifier, but the model still predicted the wrong label with confidence of at least 0.7. This is a failure on the classifier side, not a retrieval problem.

The full operational rules, including how overlapping cases are decided and where the 0.7 threshold sits, are written up in results/step7_failure/annotation_guide.md, which was committed before annotation started.

## How the annotation was done

The errors were taken from the Step 6 matrix records at k=3, for the two conditions that get manual annotation: dense_roberta (plain dense retrieval) and dense_reranked_roberta (dense retrieval with the stance reranker on top). Plain dense produced 122 errors and the reranked condition produced 138. From each pool a fixed random sample of 35 was drawn with seed 42, which gives 70 annotated errors in total, balanced evenly across the two conditions and sitting inside the 50 to 75 range set for the primary analysis. Balancing the sample matters because the whole
comparison is dense against reranked, and an uneven split would make the two percentages harder to read against each other.

The random draw is reproducible (seed 42) and its label composition is recorded, because the mix of true labels in the sample affects how the category percentages should be read. The 35 errors sampled from each condition break down by true label as follows:

| condition              | SUPPORT | NEI | CONTRADICT |
|------------------------|---------|-----|------------|
| dense_roberta          | 8       | 19  | 8          |
| dense_reranked_roberta | 9       | 12  | 14         |

The two samples are not identically composed, which is expected because they are drawn independently from each condition's own error pool. This is worth keeping in view when comparing categories across conditions: the reranked sample contains more CONTRADICT-gold errors and fewer NEI-gold errors than the dense sample.

Each row was labelled with exactly one primary_category from the four above, using the annotation guide. Every exported row also carries the matching result from the other condition (its prediction, confidence, whether it was correct, and its retrieved documents), so a case where dense was wrong but reranked was right, or the reverse, can be read directly from a single row rather than cross-referencing two files.

Before any percentages are computed, the analysis script refuses to run on an annotation that is incomplete or internally inconsistent. It checks for blank categories, invalid category names, missing condition or claim id, duplicate keys, and logical inconsistencies (for example a row labelled confident_wrong_prediction that sits below the 0.7 confidence the category requires). This guard exists so that percentages are never computed over only the easy, already-labelled rows, which would bias the reported distribution towards the simpler cases. The partial-run flag was deliberately not used, because the project rule is that final thesis numbers must come from a complete, validated annotation.

Two data-integrity checks also run before the records are used, and both passed, which matters because the whole analysis rests on the confidence values and the gold lookup being trustworthy. The confidence-definition check confirmed that for all 1,200 SciFact records the reported confidence equals the maximum softmax probability over the three classes, with no record missing a probability vector, no mismatch, and no probability vector failing to sum to one. The gold-vs-records validation confirmed that all 300 SciFact claim ids were found in the gold lookup
and had retrievable gold document ids, so no error was annotated against a missing or unmatched gold reference. The reranker stance scores span effectively the full range (roughly 0.001 to 0.999), confirming the stance signal is live rather than collapsed.

One clarification on what the gold signal means. For SciFact, `gold_doc_retrieved` and `sig_gold_evidence_missing` are computed against the claim's cited doc ids rather than its annotated evidence set, which is why all 300 claims carry a gold reference above, including the 112 whose label is NEI. For an NEI claim the signal therefore means "the cited abstract was retrieved" rather than "the evidence was retrieved". The annotation guide already treats the sig_* columns as aids rather than labels, and the NEI-specific rule was applied by reading the retrieved documents directly, so no category assignment depends on this signal. It is recorded here so the signal is not over-read, and because the same field is defined
differently for SciFact-Open (annotated evidence docs), which is the asymmetry noted in Step 3.

## The five moderate-confidence cases, and why they are excluded rather than relabelled

During annotation, five errors turned out to be genuine classifier mistakes in mechanism. In each of them correct-direction evidence was present and the model simply got the direction or a qualifier wrong. In that sense they behave like confident_wrong_prediction. The problem is that their confidence sits between 0.53 and 0.64, which is below the 0.7 that the category requires by definition. Labelling them confident_wrong_prediction would break the category's own rule, and
the validator correctly flagged that as a logical inconsistency.

There were two honest ways to deal with this. I could have relabelled them into one of the retrieval categories, but that would be wrong, because the gold document was present, on-topic and supportive in these cases, so none of the retrieval categories actually fits. Or I could set them aside. I chose to set them aside, using a marker called excluded_below_threshold.

The important point, and the reason I did it this way, is that excluded_below_threshold is not a fifth category. It is not in the list of valid categories at all; it is a separate marker that tells the analysis to leave those rows out of the four-category counts and report them on their own. So the taxonomy still has exactly four categories. This is why the per-condition breakdown
below is computed over 65 rows and not 70. The choice keeps the four pre-registered categories clean and avoids quietly inventing a new category halfway through the project just to have somewhere to put five awkward rows.

The five cases are:

| annotation_key                  | true       | predicted  | confidence |
|---------------------------------|------------|------------|------------|
| dense_roberta::1012::k3         | SUPPORT    | CONTRADICT | 0.6402     |
| dense_roberta::148::k3          | SUPPORT    | CONTRADICT | 0.6001     |
| dense_roberta::1100::k3         | NEI        | SUPPORT    | 0.5765     |
| dense_reranked_roberta::781::k3 | CONTRADICT | SUPPORT    | 0.5443     |
| dense_reranked_roberta::42::k3  | CONTRADICT | SUPPORT    | 0.5335     |

This is itself a small finding worth reporting. It shows that the classifier fails even on cases where the correct evidence was in front of it, and that some of those failures are only moderately confident rather than confidently wrong. Because it is a deliberate decision about how the taxonomy handles sub-threshold errors, it should be mentioned to the supervisor and written into the thesis as an exclusion rule, not slipped in silently.

## Results on SciFact

The validation passed cleanly: no blank categories, no invalid category names, no missing condition or claim id, no duplicate keys, and no logical inconsistencies, with the five cases reported separately as excluded_below_threshold. Of the 70 annotated rows, 65 carry a valid four-category label, and the breakdown below is computed over those 65. Each percentage is given with a Wilson 95% confidence interval, which is the appropriate interval for proportions on small samples.

Plain dense retrieval (dense_roberta, n = 32):

| category                   | count | percentage | 95% CI        |
|----------------------------|-------|------------|---------------|
| irrelevant_retrieval       | 5     | 15.6%      | [6.9, 31.8]%  |
| contradictory_retrieval    | 3     | 9.4%       | [3.2, 24.2]%  |
| evidence_overload          | 14    | 43.8%      | [28.2, 60.7]% |
| confident_wrong_prediction | 10    | 31.2%      | [18.0, 48.6]% |

Dense retrieval with stance reranking (dense_reranked_roberta, n = 33):

| category                   | count | percentage | 95% CI        |
|----------------------------|-------|------------|---------------|
| irrelevant_retrieval       | 10    | 30.3%      | [17.4, 47.3]% |
| contradictory_retrieval    | 6     | 18.2%      | [8.6, 34.4]%  |
| evidence_overload          | 11    | 33.3%      | [19.8, 50.4]% |
| confident_wrong_prediction | 6     | 18.2%      | [8.6, 34.4]%  |

The effect of reranking on the two categories it was designed to reduce, reported descriptively with the intervals above:

| category                | dense | reranked |
|-------------------------|-------|----------|
| irrelevant_retrieval    | 15.6% | 30.3%    |
| contradictory_retrieval | 9.4%  | 18.2%    |

For completeness, the whole annotated set of 70 errors (both conditions combined) breaks down as follows. This is a descriptive summary only; the analysis itself is always read per condition, because pooling the two conditions would hide the very dense-versus-reranked contrast the study is about.

| category                   | count |
|----------------------------|-------|
| evidence_overload          | 25    |
| confident_wrong_prediction | 16    |
| irrelevant_retrieval       | 15    |
| contradictory_retrieval    | 9     |
| excluded_below_threshold   | 5     |
| total                      | 70    |

## What the results show

The first thing the SciFact breakdown shows is that when plain dense retrieval fails, it usually fails after retrieval rather than at retrieval. Evidence overload is the single largest category at 43.8%, and confident wrong prediction is second at 31.2%. Together those two downstream categories make up around 75% of the dense errors, while the two retrieval-quality categories, irrelevant and contradictory, only account for about 25% between them. In plain terms, dense retrieval mostly does find reasonable evidence, and the failures come from there being too much
of it, or from the classifier reading it wrongly. This is a useful result on its own, because it says the weak point of the plain pipeline is not really the retriever.

The second and more important result is what happens when the stance reranker is added. The reranker was built specifically to cut down irrelevant and contradictory retrieval, so the expectation going in was that those two categories should shrink. Instead they grow as a share of the errors. Irrelevant retrieval almost doubles, from 15.6% to 30.3%, and contradictory retrieval also roughly doubles, from 9.4% to 18.2%, while overload and confident-wrong both fall. Read carefully, this says
the reranker changes the character of the failures rather than reducing the two it was aimed at: there are fewer overload and misread errors, but more cases where the evidence is off-topic or points the wrong way.

The honest explanation for this, and it lines up with the Step 6 result that reranking never comes out ahead on F1, is that the reranker often promotes a document that takes a strong stance but the wrong one, and in doing so pushes out the good document that plain dense had retrieved. I saw this concretely during annotation, for example on the myocarditis claim where the reranker lifted an off-claim high-stance lupus document above the correct one. So the reranker is trading one kind of failure for another rather than fixing the categories it targets.

For the thesis this is a strong result even though it is a negative one. The project is framed around failure behaviour, not around getting the best F1, and a negative finding that is analysed honestly and has a mechanism behind it is worth more than a tidy confirmation would be. The claim I can now make and defend is that the stance reranker reshapes the failure profile, and by one reading makes its two target categories proportionally worse, because it tends to promote confidently mis-stanced documents. That connects the failure taxonomy directly back to the
central question of the project and to the earlier F1 result, so the pieces hold together instead of sitting apart.

There is one qualification that has to be stated alongside these numbers, because otherwise they can be over-read. The percentages are shares of each condition's own errors, and reranking changes the total number of errors. A larger share of irrelevant retrieval among reranked errors is not automatically a larger absolute number of irrelevant cases, so these proportions have to be read next to the Step 5 and Step 6 error rates rather than on their own. The intervals are also wide,
around 15 to 18 points at these sample sizes, so the differences are best treated as directional rather than exact.

## Results on SciFact-Open (automatic signals only)

For SciFact-Open the analysis reports automatic diagnostic signals across conditions, not manual taxonomy labels, which is the agreed scope decision. These are proxy signals, so they support the manual picture rather than proving it. The evaluation covers 279 claims, of which 206 have a recognised gold document id (the rest have none available in the corpus), and 15 claims that carried conflicting SUPPORT and CONTRADICT evidence were resolved to SUPPORT during loading. The same integrity checks that ran on SciFact passed here too: all 1,116 records had a valid
probability vector with confidence equal to the maximum softmax probability, and the reranker stance scores again spanned effectively the full range. The partial gold coverage (206 of 279) is why the gold-missing column below should be read as a rough recall proxy rather than an exact figure. Across the 279 claims:

| condition              | error rate | high-confidence error at 0.7 | gold missing |
|------------------------|------------|------------------------------|--------------|
| no_retrieval           | 37.6%      | 86.7%                        | not applicable |
| bm25_roberta           | 45.9%      | 68.8%                        | 50.0%        |
| dense_roberta          | 43.7%      | 69.7%                        | 46.1%        |
| dense_reranked_roberta | 48.4%      | 74.1%                        | 77.9%        |

The evidence-overload proxy, meaning claims that were correct at a lower k and then wrong at a higher k, comes out at 24.4% for BM25, 22.9% for dense, and 19.7% for reranked.

The high-confidence-error signal is also reported at three thresholds, so that the 0.7 cut used throughout is not treated as if it were the only defensible choice. This is the sensitivity analysis the annotation guide commits to. The share of each condition's errors that were made at or above the given confidence:

| condition              | at 0.60 | at 0.70 | at 0.80 |
|------------------------|---------|---------|---------|
| no_retrieval           | 93.3%   | 86.7%   | 75.2%   |
| bm25_roberta           | 81.2%   | 68.8%   | 49.2%   |
| dense_roberta          | 87.7%   | 69.7%   | 50.8%   |
| dense_reranked_roberta | 81.5%   | 74.1%   | 53.3%   |

The pattern is stable across the three cut-points rather than an artefact of choosing 0.7: the ordering of the conditions barely changes, and the reranked condition remains among the highest at every threshold. This is reassuring for the manual analysis too, since a large fraction of errors being high-confidence at any reasonable cut is what makes the confident-wrong and excluded-below-threshold discussion meaningful in the first place.

Two of these signals echo the manual SciFact findings and are worth pointing out, without pushing them further than proxy numbers allow. The reranked condition has both the highest error rate at 48.4% and by a clear margin the highest gold-missing rate at 77.9%. That is exactly what the mechanism from Finding 2 predicts: if the reranker is dropping the gold document in favour of a strongly stanced but wrong one, then the reranked condition should be the one that most often ends up without the gold document, and on the larger corpus it is. So the automatic signals on the
harder corpus corroborate the story the manual annotation tells on SciFact.

## Reliability (intra-annotator agreement)

Because this is a solo annotation, the obvious worry is that the labels reflect one person's shifting judgement rather than a scheme that can be applied consistently. To test this, I annotated the errors a second time. After a delay of a couple of days, and without looking at the first-pass labels or notes, I re-labelled all 70 errors from scratch in the annotation tool, working only from the committed annotation guide, and exported the result as annotation_scifact_pass2.csv. The script's kappa mode then compared the two passes, matching rows by the stable condition::claim_id::k key and computing agreement over the four categories.

The reliability is computed over 65 of the 70 rows. The five excluded_below_threshold cases are left out, because their exclusion is fixed mechanically by the sub-0.7 confidence rule rather than by a judgement, so there is no categorical decision there for a second pass to agree or disagree on. On the 65 double-annotated four-category errors, the reliability came out as:

| measure                          | value |
|----------------------------------|-------|
| double-annotated errors          | 65    |
| percentage agreement             | 93.8% |
| Cohen's kappa                    | 0.914 |

A kappa of 0.914 is "almost perfect" agreement on the Landis and Koch scale.

Two details make that number trustworthy rather than just high. First, the kappa sits close to the raw agreement rate, so it is not being distorted by the class-imbalance effect that can pull kappa well below percentage agreement when a single category dominates the sample. Second, the four rows I labelled differently across the two passes cluster on the boundary between contradictory_retrieval and confident_wrong_prediction, with three of the four falling on it and the fourth moving from contradictory_retrieval to irrelevant_retrieval. That is the one distinction that turns on weighing whether the retrieved evidence, taken as a whole, favours the wrong label or instead carries the correct direction that the classifier then ignored, so it is exactly where intra-annotator variation should be expected. Naming that boundary openly is more honest than presenting a suspiciously perfect figure, and it points to the definition most worth tightening if the taxonomy is ever extended.

The check is an intra-annotator consistency measure, not an inter-annotator agreement study, so it shows the scheme is stable in one annotator's hands rather than that it would replicate across independent annotators. A second independent annotator would be the stronger design, but that is beyond the scope of a solo project, so this is stated as the honest ceiling on what the reliability result claims.

## Limitations to state plainly

The per-condition samples are around 32 to 33 rows, so the Wilson intervals are wide and the category percentages are indicative rather than exact. The percentages are proportions of each condition's errors and not absolute counts, so they have to be read together with the Step 5 and Step 6 error rates. And no significance test is run between the two conditions, because the samples can overlap in the underlying claims; the dense against reranked comparison is therefore reported descriptively with intervals rather than as a tested difference.

## Why this matters for the thesis

Step 7 is the project's primary failure analysis and it is the point where the design of the stance reranker and the evaluation of it meet. It contributes four things. It applies a pre-registered four-category taxonomy to a balanced 70-error sample across two conditions, with per-condition breakdowns and confidence intervals. It produces the central qualitative result that plain dense failures are dominated by downstream problems (overload and confident-wrong), while stance reranking shifts the profile towards the two retrieval-quality categories it was
meant to reduce, which is a coherent negative finding that matches the Step 6 F1 result and has a concrete mechanism behind it. It documents a moderate-confidence sub-group of classifier errors below the 0.7 threshold, and handles them by exclusion so the scheme stays at exactly four categories rather than by relabelling them into a category they do not fit. And it backs the manual SciFact picture with automatic signals on the 100 times larger SciFact-Open corpus, where the reranked condition shows the highest error rate and the highest gold-missing rate. These are
the kind of conditional and negative findings a failure-focused study is meant to surface, and reporting them with intervals rather than as point claims is what lets them be stated with calibrated confidence. Underpinning all of it, the blind second annotation pass returns an almost-perfect intra-annotator kappa of 0.914, which answers the single-annotator objection by showing the four categories were applied consistently rather than ad hoc.

## Files

- Analysis result, and the input to Steps 8 and 9: results/step7_failure/analysis_scifact.json
- SciFact-Open diagnostic signals: results/step7_failure/rates_scifact_open.json
- Labelled annotation, kept safe in Drive and never overwritten: annotation_scifact.csv and .json
- Blind second-pass annotation for the reliability check: annotation_scifact_pass2.csv
- Operational taxonomy used for labelling: results/step7_failure/annotation_guide.md

## Reproduction

```bash
#exporting the balanced 70-error sample (seed 42, k=3)
python analysis/failure_taxonomy.py --mode export --dataset scifact --k 3 \
  --records_dir results/step6_matrix --out_dir results/step7_failure --max_errors 70

#automatic diagnostic signals on SciFact-Open (no manual annotation)
python analysis/failure_taxonomy.py --mode rates --dataset scifact_open --k 3 \
  --records_dir results/step6_matrix --out_dir results/step7_failure

#after filling primary_category, analyse (validates first, then breaks down per condition)
python analysis/failure_taxonomy.py --mode analyse --dataset scifact \
  --out_dir results/step7_failure

#extra option reliability check, once a blind second pass exists:
python analysis/failure_taxonomy.py --mode kappa --dataset scifact \
  --first_csv  results/step7_failure/annotation_scifact.csv \
  --second_csv results/step7_failure/annotation_scifact_pass2.csv \
  --out_dir results/step7_failure
```
