## Step 9: Cross-Corpus Generalisation (SciFact versus SciFact-Open)

This document records Step 9, the cross-corpus comparison. Everything before this point studied the two datasets largely one at a time. Step 9 puts them side by side and asks what changes when the candidate corpus grows from roughly 5,000 abstracts on SciFact to roughly 500,000 on SciFact-Open, an increase of about 100 times. The framing is deliberately about retrieval difficulty and corpus scale, not about domain: both datasets are biomedical, so this step is not a test of domain generalisation. Domain is handled separately in Step 10.

## Why this step is needed after Steps 6, 7 and 8

The earlier steps produced three separate pictures. Step 6 measured F1 across a retrieval matrix. Step 7 characterised the failure categories. Step 8 measured the confidence signal. Each was computed on both corpora, but each was mostly read within a corpus rather than across the two. The interesting scientific question, and the one the project plan sets for this step, is whether the conclusions hold as retrieval gets harder. A method that helps only on a small, clean corpus is far less useful than one that helps precisely when retrieval is difficult, which is exactly when help is most needed. So Step 9 asks the three plan questions as cross-corpus questions.

There is no new modelling here. Step 9 reads the result files Steps 6, 7 and 8 already saved and assembles them into direct comparisons. The one exception is a single command run at the start, explained below, which fills a genuine gap rather than computing anything new about the models.

## Method and one gap that had to be closed first

The comparison is assembled by a script (`analysis/cross_corpus.py`) rather than by hand, for a specific reason: putting this together means copying roughly sixty numbers out of four different files, which is exactly the kind of task where a digit gets transposed unnoticed. Reading every figure straight from the source JSON removes that risk, and it keeps Step 9 reproducible in the same way Steps 7 and 8 are.

The gap that had to be closed first concerns Step 7. The automatic diagnostic signals were originally computed only for SciFact-Open, because SciFact received the fuller manual annotation instead. That left the Q2 comparison lopsided: manual labels on one side, proxy signals on the other. So the Step 7 rates mode was run once more for SciFact, producing `rates_scifact.json`, so that the same proxy measures now exist for both corpora and can be compared like for like. This is the only thing Step 9 computes that did not already exist, and it is a re-reading of the existing records, not a new model run.

Three integrity choices carry through from earlier steps and are enforced by the script. The Step 3 retrieval recall figures are deliberately left out of the comparison, because on SciFact recall is measured over cited documents for all 300 claims while on SciFact-Open it is measured over annotated evidence for the 206 evidenced claims, so the two are not the same measurement. The macro F1 reader is pinned to the exact Step 6 schema so it can never silently read a precision or recall value by mistake. And each source file's seed is checked where the file records one: the Step 8 files carry a seed field and are verified against it, while the older Step 6 and Step 7 files predate that field, so their seed provenance is recorded as externally supplied rather than asserted. Every figure inherits the seed-42 run.

## Q1: Does stance reranking improve F1 consistently as retrieval difficulty scales?

The word that matters in the question is "consistently", and consistency can only be judged at matched retrieval depths, because comparing each method at its own best depth compares two different settings. So the primary Q1 result is reranked minus dense macro F1 at every matched k, signed so that a positive number means reranking helped.

| k | SciFact dense | SciFact reranked | reranked minus dense | SciFact-Open dense | SciFact-Open reranked | reranked minus dense |
|---|---|---|---|---|---|---|
| 1 | 0.5947 | 0.3852 | -0.2096 | 0.5429 | 0.4102 | -0.1327 |
| 3 | 0.5583 | 0.4879 | -0.0704 | 0.5560 | 0.5075 | -0.0484 |
| 5 | 0.5179 | 0.5389 | +0.0210 | 0.5454 | 0.5238 | -0.0217 |
| 10 | 0.4828 | 0.4824 | -0.0004 | 0.5200 | 0.5406 | +0.0206 |

Reranking beats dense at exactly one of the four depths on each corpus, so it is not consistent on either. The one depth where it wins differs by corpus (k=5 on SciFact, k=10 on SciFact-Open), and in both cases it is the depth where the reranker's own deficit happens to cross zero rather than a depth where it delivers a real advantage. The best-achievable comparison tells the same story: dense's best beats reranking's best by 0.0559 on SciFact and by 0.0154 on SciFact-Open.

It would be tempting to read that shrinking best-versus-best gap (0.0559 down to 0.0154) as reranking improving on the harder corpus, and that reading is exactly why the margin over the no-retrieval baseline is reported alongside. On SciFact, reranking at its best still clears the no-retrieval baseline (+0.0126). On SciFact-Open it falls below it (-0.0813). So the gap between dense and reranked narrows on the large corpus not because reranking gets better, but because dense gets worse and both sink toward, and past, the point where retrieving nothing would have been the better choice. The narrowing is convergence by shared decline, not by improvement.

A separate and cleaner finding falls out of the same table: the optimal retrieval depth increases with corpus difficulty. Dense peaks at k=1 on SciFact and k=3 on SciFact-Open; reranking peaks at k=5 then k=10; BM25 stays at k=1 on both. Only a controlled matrix run on two corpora of different scale can establish that, and it is a genuinely useful practical result: the harder the retrieval, the more documents are worth reading before the added noise outweighs the added evidence.

## Q2: How do the automatic failure indicators change as the corpus grows?

The original plan wording asks whether the dominant failure categories shift. That has to be narrowed honestly, because the manual four-category taxonomy was applied to SciFact only. Which manual category becomes dominant on SciFact-Open therefore cannot be established. What can be compared like for like is the set of automatic diagnostic signals, which exist for both corpora. These are proxy measures, not taxonomy labels, and are never reported as category percentages.

| Condition | Errors (SF) | Errors (SFO) | High-conf err (SF) | High-conf err (SFO) | Gold missing (SF) | Gold missing (SFO) |
|---|---|---|---|---|---|---|
| No retrieval | 44.7% | 37.6% | 88.1% | 86.7% | n/a | n/a |
| BM25 | 44.0% | 45.9% | 73.5% | 68.8% | 50.0% | 50.0% |
| Dense | 40.7% | 43.7% | 74.6% | 69.7% | 32.8% | 46.1% |
| Dense + rerank | 46.0% | 48.4% | 78.3% | 74.1% | 66.7% | 77.9% |

Two things stand out and both are consistent across the two corpora. First, the reranked
condition has the highest error rate and the highest gold-missing rate of any retrieval condition on both datasets. On dense retrieval, adding the reranker raises the gold-missing rate from 32.8% to 66.7% on SciFact and from 46.1% to 77.9% on SciFact-Open, a jump of roughly 32 points in each case. This is the Step 4 mechanism confirmed at scale: the stance reranker tends to promote a confidently but wrongly stanced document and push the gold document out of the retrieved set, and it does so just as much, proportionally, on the large corpus as on the small one.

A rising gold-missing rate has to be described carefully, and the script records the caveat alongside the number. It shows that recognised annotated evidence was more often absent from the retrieved set. It does not by itself show that the retrieved documents were topically irrelevant, because they may be relevant but non-gold, partially evidential, contradictory, or simply insufficient. Establishing that a rising gold-missing rate corresponds to the irrelevant_retrieval category would need document-level human judgement, which only SciFact received. So the honest
cross-corpus claim is about the automatic indicator, and the manual SciFact taxonomy is used only to interpret the likely mechanism behind it, not to assert category dominance on SciFact-Open.

The evidence-overload proxy actually falls slightly on the larger corpus (dense 29.3% on SciFact against 22.9% on SciFact-Open). That fits the retrieval picture: on the large corpus the k=1 retrieval is already so weak that there is less correct-at-low-k prediction left to break at higher k.

The manual SciFact taxonomy is retained as a separate layer for interpretation. It shows dense failures dominated by evidence overload (43.8%) and confident-wrong prediction (31.2%), and the reranked condition shifting toward the two retrieval-quality categories it was meant to reduce (irrelevant 15.6% to 30.3%, contradictory 9.4% to 18.2%). That shift is consistent with the gold-missing jump seen in the automatic signals, which is the value of keeping both layers.

## Q3: Does the confidence-correctness relationship hold under harder retrieval?

| Condition | Accuracy (SF) | AUROC (SF) | Accuracy (SFO) | AUROC (SFO) |
|---|---|---|---|---|
| No retrieval | 55.3% | 0.6469 | 62.4% | 0.6420 |
| BM25 | 56.0% | 0.6531 | 54.1% | 0.5964 |
| Dense | 59.3% | 0.6947 | 56.3% | 0.6169 |
| Dense + rerank | 54.0% | 0.6061 | 51.6% | 0.5658 |

Confidence discrimination weakens on the harder corpus but does not break. Dense AUROC falls from 0.6947 to 0.6169, a drop of 0.078, and every retrieval condition loses discriminative power moving to the large corpus. The reranked condition remains the weakest on both, consistent with the Step 8 finding that reranking degrades the confidence signal as well as accuracy. So the confidence signal generalises in the weak sense that it stays above chance and keeps the same condition ordering, but it becomes a poorer error detector exactly when retrieval is hardest.

Two sub-findings matter more than that headline.

The first is the CONTRADICT inversion, and Step 9 is what confirms it is real rather than a one-dataset oddity. Confidence discrimination for CONTRADICT claims is below chance in all eight condition-by-corpus cells, ranging from 0.1326 to 0.3549. In plain terms, for refutation claims the model tends to be more confident when it is wrong. NEI discriminates strongly everywhere (0.8303 to 0.9354) and SUPPORT moderately, so the problem is specific to CONTRADICT and it holds on both corpora. That makes it a property of the model and the task rather than of one dataset, which is a stronger claim than Step 8 alone could support. For a fact-checking system this is the most consequential single result in the project, because the class the model is least able to flag its own errors on is the refutation class, which is precisely the class that matters for catching false claims.

The second is how confidence responds to added documents, and here the two corpora differ, which is worth reporting honestly rather than flattening. On SciFact, accuracy falls as k grows while confidence barely moves, so the confidence-accuracy gap widens (dense accuracy down 11.3 points, confidence down only 1.55, gap widening 9.7 points): the model does not register the degradation that adding documents causes. On SciFact-Open, dense accuracy falls only slightly while confidence falls more, so the gap narrows: aggregate confidence tracks the decline. The reason is that on the large corpus retrieval is already weak at k=1, so there is little accuracy left to lose as k grows.
So the "model is blind to overload" result from Step 8 is corpus-specific: it holds on the tractable corpus where overload is the dominant failure mode, and does not hold on the large corpus where retrieval is failing at every depth. Stating it as a single universal claim would misrepresent half the data.

## The centrepiece: on the large corpus, retrieval is actively counterproductive

Pulling the F1 and confidence evidence together answers a sharper question than any of the three above: at each corpus, is the retrieval-augmented design worth using at all compared with retrieving nothing?

| Corpus | Best retrieval condition | Beats no retrieval on F1 | Beats no retrieval on accuracy |
|---|---|---|---|
| SciFact | dense (+0.0684 F1) | yes | yes (59.3% vs 55.3%) |
| SciFact-Open | dense (-0.0659 F1) | no | no (56.3% vs 62.4%) |

On SciFact the best retrieval condition clears the no-retrieval baseline on both F1 and accuracy, so retrieval earns its place. On SciFact-Open no tested retrieval configuration beats retrieving nothing, on either metric. The script labels dense there as "least harmful but still below baseline" rather than "best", so the wording cannot read as an endorsement.

This is the strongest cross-corpus claim in the project. It is not that retrieval is merely harder at scale, which Step 3 already showed. It is that on the large corpus retrieval becomes actively counterproductive: the retriever still finds the gold document a reasonable fraction of the time, but supplying the retrieved context makes the classifier worse than giving it no documents at all. For a diagnostic study of retrieval-augmented claim verification, that is a decision-relevant result, because it says the value of retrieval depends on corpus scale and can turn negative.

## Do the results defend the thesis, and are they valid?

They do, and the reason is coherence. The thesis is a diagnostic study of retrieval effects and failure behaviour, not a system-building project, so the value is in characterising how the system behaves, not in a method winning. Every Step 9 comparison points the same way as the steps that produced its inputs. Reranking is inconsistent here, as it was on F1 in Step 6, on failure categories in Step 7, and on confidence in Step 8. Retrieval degrades and then turns counterproductive at scale, which extends the Step 3 recall decline into a decision about whether to retrieve at all. The CONTRADICT inversion first seen in Step 8 is confirmed as a cross-corpus property. Four independent axes, one consistent picture, no contradictions between them.

The results are valid in the sense the project can defend. The comparison is assembled
programmatically from the saved results, so it cannot drift from the underlying numbers. The F1 reader is pinned to the real schema. The two layers of Q2 are kept separate so proxies are never dressed up as taxonomy labels. The intervals and the descriptive framing are inherited from the earlier steps. What the results are not is multi-seed: everything rests on the seed-42 run, and the cross-corpus differences have not been shown to replicate across training seeds. That is stated plainly rather than hidden.

## Limitations to state plainly

Corpus scale is a strong manipulation of retrieval difficulty, but it is not the only thing that differs between the two corpora, so the results are framed as corpus-scale or retrieval-difficulty generalisation rather than as isolating corpus size as the sole cause. Both corpora are biomedical, so this step says nothing about domain generalisation, which is Step 10's job.

The Q2 comparison is symmetric only at the level of automatic proxy signals. Manual taxonomy labels exist for SciFact alone, so no claim is made about which manual failure category dominates on SciFact-Open.

Every figure comes from the seed-42 run. The Step 8 source files record that seed and were checked against it; the older Step 6 and Step 7 files predate the seed field, so their provenance is externally supplied rather than verified from the file. The cross-corpus differences describe that run and do not establish stability across training seeds.

The cross-corpus comparisons are descriptive and are read alongside the intervals reported in the source steps. No significance test is run between corpora, in keeping with the project scope.

## Relevance to later steps

Step 10 (real-world case study) inherits two specific cautions from this step. The
counterproductive-retrieval result means the case study should report the no-retrieval baseline as a serious option rather than assuming retrieval helps. The CONTRADICT inversion, now confirmed on both corpora, is a concrete warning for the case study, because social-media claims that are false are exactly the refutation case where the model's confidence points the wrong way.

## Files

- Assembled comparison: `results/step9_comparison/cross_corpus_comparison.json`
- Q1 primary (matched depth): `results/step9_comparison/q1_matched_depth_reranking.csv`
- Q1 supplement (best depth): `results/step9_comparison/q1_f1_by_corpus.csv`
- Q2 automatic signals: `results/step9_comparison/q2_automatic_signals.csv`
- Q3 confidence: `results/step9_comparison/q3_confidence_by_corpus.csv`
- Q3 per-label AUROC: `results/step9_comparison/q3_per_label_auroc.csv`
- New symmetric input: `results/step7_failure/rates_scifact.json`
- Comparison script: `analysis/cross_corpus.py`

## Reproduction

```bash
# close the symmetry gap: automatic signals for SciFact (SciFact-Open already had them)
python analysis/failure_taxonomy.py --mode rates --dataset scifact --k 3 \
  --records_dir results/step6_matrix --out_dir results/step7_failure

# assemble the cross-corpus comparison from the saved Step 6, 7 and 8 results
python analysis/cross_corpus.py \
  --matrix_dir results/step6_matrix \
  --failure_dir results/step7_failure \
  --confidence_dir results/step8_confidence \
  --out_dir results/step9_comparison
```