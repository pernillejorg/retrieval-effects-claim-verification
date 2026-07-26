# Step 2 notebooks

Step 2 trains the two RoBERTa classifiers the rest of the project depends on:

- **Model 1**, the no-retrieval baseline (claim text only), which every
  retrieval-augmented result in later steps is measured against.
- **Model 2**, the evidence-aware classifier (claim + gold evidence), used only
  inside the retrieval-augmented conditions from Step 5 onward.

The step went through two documented corrections before the reported numbers
were settled: a **deduplication fix** (the SciFact validation split stores 450
rows for only 300 unique claims) and a **seeding fix** (the training seed was
not re-applied after the learning-rate search). The full story is in
`results/step2_results.md`. Because of those corrections the step was rerun
several times, so the notebooks are grouped by role rather than left as a flat
list. The two canonical notebooks sit at the top level; earlier and
additional_checks runs are in subfolders.

## What each notebook is

| Notebook (current name) | Model 1 F1 | Model 2 F1 | SciFact-Open | Val rows | Role |
|---|---|---|---|---|---|
| `Step2_reported_baseline.ipynb` | **0.5263** | **0.6438** | **0.6219** | 300 | **Reported (canonical).** Stage 3: 300 unique claims, corrected seeding, seed 42. Produces every headline Step 2 number. |
| `Step2_variance_study.ipynb` | 0.5263 / 0.5403 / 0.5059 | — | — | 300 | **Reported variance study.** Model 1 across seeds 42/123/7, mean 0.5242 ± 0.0141. This is the variance table in the results doc. |
| `additional_checks/Step2_variance_evidence_model.ipynb` | 0.5263 / 0.5403 / 0.5059 | 0.6438 / 0.5854 / 0.5595 | — | 300 | **Extra robustness.** Extends the variance study to Model 2 across the same three seeds (mean 0.5962). Supports the Model 2 variance section of the results doc. |
| `earlier_versions/Step2_stage1_raw_450.ipynb` | 0.5201 | — | 0.6143 | **450** | **Earlier_versions, Stage 1.** Raw validation split before deduplication, so metrics are computed over 450 duplicated rows. Model 1 only. |
| `earlier_versions/Step2_stage2_dedup_model1.ipynb` | 0.4570 | — | 0.5348 | 300 | **Earlier_versions, Stage 2.** Deduplicated to 300 unique claims but before the seeding fix. Model 1 only. |
| `earlier_versions/Step2_stage2_dedup_with_evidence.ipynb` | 0.4570 | 0.5828 | 0.5348 | 300 | **Earlier_versions, Stage 2.** Same deduplicated stage with Model 2 added, still before the seeding fix. |

## How to read this

- The two top-level notebooks produce every number reported in
  `results/step2_results.md`. Start there.
- `additional_checks/` is genuine extra work (the Model 2 multi-seed check) that
  supports the results doc but is not a headline result.
- `earlier_versions/` shows the correction history. These runs are kept for
  transparency, so the two data corrections and their effect on the numbers can
  be traced. They are **not** the reported results and should not be cited as
  such.

## Original filenames (for reference during upload)

| Current name | Original name |
|---|---|
| `Step2_reported_baseline.ipynb` | `Step2_baseline_evi_SciFact_seed.ipynb` |
| `Step2_variance_study.ipynb` | `Step2_baseline_SciFact_seed.ipynb` |
| `additional_checks/Step2_variance_evidence_model.ipynb` | `Step2_baseline_evi_SciFact_multipleseeds.ipynb` |
| `earlier_versions/Step2_stage1_raw_450.ipynb` | `Step2_baseline_SciFact_SciFact_Open1.ipynb` |
| `earlier_versions/Step2_stage2_dedup_model1.ipynb` | `Step2_baseline_SciFact_SciFact_Open2.ipynb` |
| `earlier_versions/Step2_stage2_dedup_with_evidence.ipynb` | `Step2_baseline_evi_SciFact_SciFact_Open.ipynb` |