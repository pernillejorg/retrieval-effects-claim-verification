"""
Step 8: Retrieval-aware confidence scoring for RAG claim verification.

This script answers one focused question: does the model know when it is wrong, and does
stance reranking change that? When RoBERTa classifies a claim it outputs a softmax
distribution over the three labels, and the highest probability is taken as the confidence
score. Step 8 asks whether that self-reported confidence carries usable information about
correctness.

Like Step 7, this script works ENTIRELY from the per-claim records already saved by the
Step 5 / Step 6 pipeline runs (records_*.json). It does NOT re-run retrieval, reranking, or
the classifier, and it needs no GPU. Every record already stores confidence and the full
probabilities vector, so the whole analysis is a re-reading of existing results.

WHAT IS COMPUTED (the four things Step 8 requires, plus what earlier steps promised):
  1. Confidence is recorded for every prediction in every pipeline condition. Before any
     number is computed the script re-validates the saved confidence (reusing the Step 7
     check) and adds stricter per-record checks of the probability vector itself.
  2. Whether low confidence correlates with error, reported four ways: mean confidence on
     correct versus wrong predictions and the gap between them, a coarse confidence-bin
     accuracy table, a single discrimination number (AUROC), and a per-true-label breakdown
     so label-specific behaviour (especially NEI) is not hidden by the overall average.
  3. Whether stance reranking changes confidence relative to plain dense retrieval. This is
     reported TWICE: an unpaired condition-level comparison, and a PAIRED per-claim
     comparison that matches the two conditions on claim_id. The paired version is the
     defensible one, because the unpaired means can differ simply by being computed over
     different subsets of claims. The paired version also reports the correct/wrong
     transition table, mirroring how Step 7's export already carries other-condition fields.
  4. A simple flagging rule: predictions below a confidence threshold are marked unreliable.
     The script measures whether flagged predictions really are disproportionately wrong, and
     what accuracy survives on the predictions that are kept.
  5. Cross-k behaviour, promised in step6_results.md: Step 6 found the conditions to be
     sensitive to retrieval depth, so this tests whether confidence moves alongside accuracy
     as k varies, in either direction.
  6. The link back to Step 7: the share of errors made at or above the 0.7 high-confidence
     threshold, and the size of the 0.5 to 0.7 moderate band that Step 7 documented as the
     excluded_below_threshold group.

ON SEED PROVENANCE, STATED EXPLICITLY. The Step 6 matrix records this script reads were
produced with the seed-42 classifiers, as recorded in step6_results.md ("Seed | 42 (fixed;
the seed axis is studied separately in the Step 5 variance study)"). Step 8 therefore
characterises the confidence behaviour of THAT run. It does not establish that the confidence
behaviour replicates across training seeds, and it must not be written up as if it did. The
multi-seed work in this project covers the Step 2 baseline and the Step 5 pipeline variance
study, not the Step 6 matrix, so no cross-seed aggregation is attempted here rather than
manufactured from per-seed files that do not exist.

ON SCOPE AND ON THE WORD "CALIBRATION". The project plan says not to attempt full calibration
curve analysis or complex statistical testing, so none is done. This script measures
confidence DISCRIMINATION (does confidence separate correct from wrong) and reports a single
aggregate confidence-minus-accuracy gap. It does NOT establish calibration: that aggregate gap
can hide opposing errors, because underconfidence on some predictions and overconfidence on
others partially cancel. Results should therefore be written as "confidence discriminated
errors better/worse" or "confidence was more closely aligned with correctness", NOT as "the
model was better calibrated", which would require a formal metric such as expected calibration
error or a Brier score. The field is named global_confidence_accuracy_gap_pp rather than
"overconfidence" for exactly this reason: positive means aggregate overconfidence, negative
means aggregate underconfidence, and neither says anything about the shape underneath.

ON AUROC, AND WHY IT IS NOT OVER-REACHING. AUROC here is used in one narrow sense: the
probability that a randomly chosen correct prediction carried higher confidence than a
randomly chosen wrong one, with tied confidence values counting as one half. The tie term
matters because the implementation below explicitly assigns average ranks to ties. It is a
single discrimination summary, not a calibration curve. 0.5 means confidence carries no
discriminative information about correctness. It is computed from the ranks
directly (no external dependency) so this script stays standard-library only, like Step 7.

ON THE THREE-CLASS FLOOR. Because confidence is the maximum of three softmax probabilities,
it cannot fall below about 0.333. The bins below therefore start at that chance level rather
than at zero, and a "low confidence" prediction means low relative to that floor.

Modes: analyse (one dataset at one depth), cross_k (the depth sweep for one dataset).
"""

#importing os for file path handling and directory creation
import os

#importing sys to add the project root to the import path
import sys

#adding the project root so the shared Step 7 helpers import correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing json for reading records and writing the analysis output
import json

#importing csv so the summary tables can also be written in a form that drops straight into
#the thesis or a plotting script without re-parsing nested json
import csv

#importing argparse for command line mode and dataset selection
import argparse

#importing math for the finiteness check on saved probability vectors
import math

#reusing the Step 7 helpers so the record handling, the confidence definition check and the
#Wilson interval are provably the SAME code in both steps rather than a second copy that
#could quietly drift. this is why the two steps' numbers can be compared directly.
#NOTE: failure_taxonomy.py is deliberately NOT modified by Step 8. Step 7 is finished and
#committed, so any extra validation Step 8 wants lives in this file instead.
from analysis.failure_taxonomy import (
    load_records,
    records_by_condition,
    is_error,
    wilson_interval,
    validate_confidence_definition,
    resolve_records_path,
    CONDITIONS,
    MATRIX_K_VALUES,
    CONFIDENT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#the lowest possible confidence for a 3-class max-softmax score, used as the first bin edge
#so the bins start at chance rather than at zero
CHANCE_LEVEL = 1.0 / 3.0

#the confidence bins used for the descriptive bin table. deliberately coarse: with a few
#hundred predictions per condition, finer bins would produce cells too small to read
#the first edge sits a hair below the chance floor and the last a hair above 1.0, so every
#every confidence that confidence_of() accepts lands in exactly one bin (the display still shows
#0.333 and 1.0). the coverage assertion in confidence_bin_table() enforces this
CONFIDENCE_BIN_EDGES = [CHANCE_LEVEL - 1e-6, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]

#the thresholds swept for the flagging rule. 0.7 is the PRIMARY pre-specified threshold,
#carried over unchanged from Step 7 so both steps use one definition of "high confidence".
#the others are reported as sensitivity only and are explicitly NOT used to pick a best
#operating point on the test set, which would be post-hoc test-set optimisation
FLAG_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

#the moderate confidence band that Step 7 documented as excluded_below_threshold: genuine
#classifier errors in mechanism that sat below the 0.7 bar the category requires
MODERATE_BAND = (0.5, CONFIDENT_THRESHOLD)

#the three labels the pipeline can predict, used for the record integrity checks
VALID_LABELS = ("SUPPORT", "CONTRADICT", "NEI")

#the default training seed for the Step 6 matrix records, recorded in step6_results.md. this
#is only a DEFAULT: it is overridable on the command line, because a records file supplied
#with --records_path could come from another seed and hard-coding 42 would then attach false
#provenance to the saved output
DEFAULT_RECORDS_SEED = 42

# ---------------------------------------------------------------------------
# Record identity and integrity
# ---------------------------------------------------------------------------

def record_id(record):
    """
    Return the stable per-claim identifier used to pair predictions across conditions. Raises
    rather than guessing, because a silent mismatch here would invalidate the paired analysis.
    """
    #trying the field names the pipeline may have used, in order of preference
    for key in ("claim_id", "id", "claim_index"):
        if key in record and record[key] is not None:
            return str(record[key])
    raise KeyError("Record has no claim_id / id / claim_index for paired comparison.")

def normalised_claim_text(record):
    """Return the claim text with whitespace normalised, or None when absent."""
    #used only to sanity check that two paired records really are the same claim
    claim = record.get("claim")
    if claim is None:
        return None
    return " ".join(str(claim).split())

def confidence_of(record):
    """
    Return the saved confidence, validated. A missing confidence is an error rather than a
    default of zero: silently substituting zero would put the record below the three-class
    floor, change the flagging results, and make missing data look like genuine uncertainty.
    """
    #refusing to invent a value when the field is absent
    if "confidence" not in record or record["confidence"] is None:
        raise KeyError(f"Record {record.get('claim_id')} is missing required field "
                       f"'confidence'.")
    conf = float(record["confidence"])
    #checking the value lies in the range a 3-class max-softmax score can occupy, with a small
    #tolerance so ordinary floating point noise does not trip the guard
    if not (CHANCE_LEVEL - 1e-6) <= conf <= (1.0 + 1e-6):
        raise ValueError(f"Confidence {conf} for claim {record.get('claim_id')} lies outside "
                         f"the valid three-class max-softmax range "
                         f"[{CHANCE_LEVEL:.4f}, 1.0].")
    return conf

def is_correct(record):
    """A record is correct when the prediction matches the true label (inverse of is_error)."""
    #reusing the Step 7 definition of an error so both steps agree on what counts as wrong
    return not is_error(record)

def validate_probability_vectors(records):
    """
    Check every saved probability vector more strictly than the Step 7 check does. Step 7
    confirms confidence equals the maximum and that the vector sums to one; this additionally
    confirms there are exactly three finite probabilities, each within [0, 1]. Together these
    mean the confidence analysis is standing on a genuine normalised distribution.
    """
    #walking every record and counting problems rather than failing on the first one
    bad_len = bad_finite = bad_range = 0
    for r in records:
        probs = r.get("probabilities")
        if not isinstance(probs, list) or len(probs) != 3:
            bad_len += 1
            continue
        #converting explicitly so a non-numeric entry reports the claim rather than a raw
        #float() traceback
        try:
            vals = [float(p) for p in probs]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric probability vector for claim "
                             f"{record_id(r)}: {probs}") from exc
        if not all(math.isfinite(p) for p in vals):
            bad_finite += 1
        if not all(0.0 <= p <= 1.0 for p in vals):
            bad_range += 1

    #reporting the outcome in the same style as the Step 7 checks
    print("Probability vector check (exactly 3 finite probabilities within [0, 1]):")
    print(f"  vectors not of length 3:        {bad_len}")
    print(f"  vectors with non-finite values: {bad_finite}")
    print(f"  vectors outside [0, 1]:         {bad_range}")
    if bad_len or bad_finite or bad_range:
        raise ValueError(
            "Saved probability vectors are malformed (wrong length, non-finite, or outside "
            "[0, 1]). The confidence analysis assumes a valid 3-class softmax distribution; "
            "fix the pipeline records before running Step 8.")

def validate_condition_records(by_cond):
    """
    Check that the per-condition record sets are usable: no duplicate claim ids within a
    condition, only valid label strings, and the same claim set across all conditions. A
    claim-set mismatch is treated as a pipeline error because all four conditions are designed
    to evaluate the same test claims.
    """
    id_sets = {}
    for cond, recs in by_cond.items():
        #skipping conditions absent from this records file
        if not recs:
            continue
        ids = [record_id(r) for r in recs]
        #duplicate ids are a genuine bug: pairing would silently drop or mismatch claims
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate claim identifiers within condition '{cond}'. "
                             f"The paired comparison requires one record per claim.")
        id_sets[cond] = set(ids)
        #checking the label strings so a schema change cannot pass unnoticed
        for r in recs:
            if r.get("true_label") not in VALID_LABELS:
                raise ValueError(f"Invalid true_label in '{cond}': {r.get('true_label')}")
            if r.get("predicted_label") not in VALID_LABELS:
                raise ValueError(f"Invalid predicted_label in '{cond}': "
                                 f"{r.get('predicted_label')}")

    #reporting the per-condition claim counts, then comparing the actual claim SETS rather
    #than just their sizes: two conditions can hold the same number of records while covering
    #different claims, which would silently invalidate every cross-condition comparison
    print("Record integrity check (unique claim ids, valid labels):")
    for cond, ids in id_sets.items():
        print(f"  {cond:<26} {len(ids)} unique claims")
    if id_sets:
        reference_cond = next(iter(id_sets))
        reference_ids = id_sets[reference_cond]
        for cond, ids in id_sets.items():
            if ids != reference_ids:
                #all four pipeline conditions run over the same test claims by design, so a
                #mismatch is a pipeline fault rather than something to analyse around
                raise ValueError(
                    f"Claim-set mismatch between '{reference_cond}' and '{cond}': "
                    f"{len(reference_ids - ids)} claims missing and {len(ids - reference_ids)} "
                    f"extra. All pipeline conditions should evaluate the same test claims.")
    print("  all conditions cover the same claim set")

def validate_cross_k_claim_sets(per_k):
    """
    Confirm each condition is evaluated on the same claims at every retrieval depth. Without
    this, an apparent change in accuracy or confidence across k could come from a different
    set of claims being evaluated rather than from the depth itself.
    """
    for cond in CONDITIONS:
        #collecting the claim id set this condition covers at each depth
        sets_by_k = {}
        for k, by_cond in per_k.items():
            recs = by_cond.get(cond, [])
            if recs:
                sets_by_k[k] = {record_id(r) for r in recs}
        #nothing to compare when the condition appears at fewer than two depths
        if len(sets_by_k) < 2:
            continue
        reference_k = min(sets_by_k)
        reference = sets_by_k[reference_k]
        for k, ids in sets_by_k.items():
            if ids != reference:
                raise ValueError(
                    f"Claim-set mismatch for '{cond}': k={k} differs from k={reference_k} by "
                    f"{len(reference - ids)} missing and {len(ids - reference)} additional "
                    f"claims. Cross-k comparison requires the same claims at every depth.")
    print("Cross-k claim-set check: every condition covers the same claims at all depths.")

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def mean_of(values):
    """Mean of a list of numbers, or None when the list is empty."""
    #guarding against an empty list so empty conditions report null rather than crashing
    if not values:
        return None
    return round(sum(values) / len(values), 4)

def median_of(values):
    """Median of a list of numbers, or None when the list is empty."""
    #guarding against an empty list
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    #averaging the two middle values for an even-length list
    if n % 2 == 0:
        return round((ordered[mid - 1] + ordered[mid]) / 2, 4)
    return round(ordered[mid], 4)

def percent(count, total):
    """Percentage of count out of total, rounded, or None when the total is zero."""
    #guarding against a zero denominator
    if not total:
        return None
    return round(100 * count / total, 1)

def fraction(count, total):
    """Unrounded proportion of count out of total, or None when the total is zero."""
    #kept separate from percent() so ratios are computed before any rounding is applied
    if not total:
        return None
    return count / total

# ---------------------------------------------------------------------------
# Discrimination: does confidence separate correct from wrong predictions?
# ---------------------------------------------------------------------------

def auroc_correct_vs_wrong(records):
    """
    Probability that a randomly chosen CORRECT prediction carried higher confidence than a
    randomly chosen WRONG one, with tied pairs counting as one half, computed from ranks (the
    Mann-Whitney U form of AUROC) with ties assigned average ranks. Returns None when either
    group is empty.

    Interpretation: 0.5 means confidence carries no discriminative information about whether
    the answer is right; higher values mean confidence is a usable error signal. A value below
    0.5 means the model tends to be more confident when it is wrong, which is actively misleading.
    """
    #splitting the confidences into the correct and wrong groups
    conf_correct = [confidence_of(r) for r in records if is_correct(r)]
    conf_wrong = [confidence_of(r) for r in records if is_error(r)]
    n_c, n_w = len(conf_correct), len(conf_wrong)
    #the statistic is undefined if either group is empty
    if n_c == 0 or n_w == 0:
        return None

    #pooling both groups and sorting by confidence so ranks can be assigned
    pooled = [(c, 1) for c in conf_correct] + [(c, 0) for c in conf_wrong]
    pooled.sort(key=lambda t: t[0])

    #assigning average ranks so tied confidences do not bias the statistic
    ranks = [0.0] * len(pooled)
    i = 0
    while i < len(pooled):
        j = i
        #walking forward over every entry sharing this confidence value
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        #the shared rank is the average of the 1-based positions in the tied block
        avg_rank = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            ranks[t] = avg_rank
        i = j + 1

    #summing the ranks belonging to the correct group
    rank_sum_correct = sum(ranks[idx] for idx, (_, lab) in enumerate(pooled) if lab == 1)
    #converting the rank sum into the U statistic and then into AUROC
    u_correct = rank_sum_correct - n_c * (n_c + 1) / 2.0
    return round(u_correct / (n_c * n_w), 4)

# ---------------------------------------------------------------------------
# Per-condition confidence summary
# ---------------------------------------------------------------------------

def confidence_summary(records):
    """
    Summarise, for one condition's predictions: how accurate it was, how confident it was on
    correct versus wrong predictions, how far apart those two are, how well confidence
    discriminates correct from wrong, and how far mean confidence sits above accuracy in
    aggregate.
    """
    #counting the predictions and splitting them by correctness
    n = len(records)
    correct = [r for r in records if is_correct(r)]
    wrong = [r for r in records if is_error(r)]
    n_correct, n_wrong = len(correct), len(wrong)

    #pulling the confidence values for each group
    conf_all = [confidence_of(r) for r in records]
    conf_correct = [confidence_of(r) for r in correct]
    conf_wrong = [confidence_of(r) for r in wrong]

    #computing accuracy with a Wilson interval, matching the interval used in Step 7. the
    #Step 7 helper already returns percentages, so these are on the same scale as accuracy_pct
    acc = percent(n_correct, n)
    acc_ci = wilson_interval(n_correct, n)

    #the separation is the headline "does the model know when it is wrong" number: how much
    #more confident it is when right than when wrong
    mean_c = mean_of(conf_correct)
    mean_w = mean_of(conf_wrong)
    separation = round(mean_c - mean_w, 4) if (mean_c is not None and mean_w is not None) else None

    #the aggregate gap between mean confidence and accuracy, both on the 0-100 scale. positive
    #means aggregate overconfidence, negative means aggregate underconfidence. this is NOT a
    #calibration measure: opposing errors within the distribution can cancel each other here
    mean_all = mean_of(conf_all)
    conf_acc_gap = None
    if mean_all is not None and acc is not None:
        conf_acc_gap = round(100 * mean_all - acc, 1)

    #counting how many of the errors were made at or above the Step 7 high-confidence bar,
    #and how many sat in the moderate band Step 7 recorded as excluded_below_threshold
    hc_errors = sum(1 for r in wrong if confidence_of(r) >= CONFIDENT_THRESHOLD)
    mod_errors = sum(1 for r in wrong
                     if MODERATE_BAND[0] <= confidence_of(r) < MODERATE_BAND[1])

    return {
        "n_predictions": n,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy_pct": acc,
        "accuracy_ci95_pct": acc_ci,
        "mean_confidence_all": mean_all,
        "mean_confidence_correct": mean_c,
        "mean_confidence_wrong": mean_w,
        "median_confidence_correct": median_of(conf_correct),
        "median_confidence_wrong": median_of(conf_wrong),
        "separation_correct_minus_wrong": separation,
        "auroc_confidence_detects_correct": auroc_correct_vs_wrong(records),
        "global_confidence_accuracy_gap_pp": conf_acc_gap,
        "high_confidence_errors": hc_errors,
        "high_confidence_error_pct_of_errors": percent(hc_errors, n_wrong),
        "moderate_band_errors": mod_errors,
        "moderate_band_error_pct_of_errors": percent(mod_errors, n_wrong),
    }

# ---------------------------------------------------------------------------
# Per-true-label breakdown (supplementary, so class-specific behaviour is visible)
# ---------------------------------------------------------------------------

def per_label_summary(records):
    """
    Break the confidence behaviour down by TRUE label. The overall averages can hide very
    different behaviour per class, and in a 3-class claim verification task NEI often behaves
    differently from the evidence-bearing labels: the model may be sensible on SUPPORT and
    CONTRADICT while being confidently wrong on NEI. Reported as supplementary analysis
    rather than as another headline result.
    """
    rows = {}
    for label in VALID_LABELS:
        subset = [r for r in records if r.get("true_label") == label]
        #skipping a label that does not occur in this dataset split
        if not subset:
            continue
        n = len(subset)
        n_correct = sum(1 for r in subset if is_correct(r))
        rows[label] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy_pct": percent(n_correct, n),
            "accuracy_ci95_pct": wilson_interval(n_correct, n),
            "mean_confidence_all": mean_of([confidence_of(r) for r in subset]),
            "mean_confidence_correct": mean_of([confidence_of(r) for r in subset
                                                if is_correct(r)]),
            "mean_confidence_wrong": mean_of([confidence_of(r) for r in subset
                                              if is_error(r)]),
            "auroc_confidence_detects_correct": auroc_correct_vs_wrong(subset),
        }
    return rows

# ---------------------------------------------------------------------------
# Confidence-bin accuracy (descriptive, not a fitted calibration curve)
# ---------------------------------------------------------------------------

def confidence_bin_table(records):
    """
    Group predictions into coarse confidence bins and report accuracy in each, with Wilson
    intervals. If confidence is informative, accuracy should rise across the bins. This is a
    descriptive breakdown rather than a calibration analysis, and the bins are kept wide
    because the per-condition samples are only a few hundred predictions.
    """
    rows = []
    #walking each adjacent pair of bin edges
    for lo, hi in zip(CONFIDENCE_BIN_EDGES[:-1], CONFIDENCE_BIN_EDGES[1:]):
        #selecting the predictions whose confidence falls in this half-open bin
        in_bin = [r for r in records if lo <= confidence_of(r) < hi]
        n = len(in_bin)
        n_correct = sum(1 for r in in_bin if is_correct(r))
        #the top bin's upper edge sits just above 1.0 so a confidence of exactly 1.0 is
        #included; flagging that keeps the displayed interval from looking wrongly exclusive
        rows.append({
            #displaying the nominal 0.333 floor rather than the tolerant edge used internally
            "bin_low": round(max(lo, CHANCE_LEVEL), 3),
            "bin_high": round(min(hi, 1.0), 3),
            "is_final_bin_inclusive_of_1": hi > 1.0,
            "n": n,
            "n_correct": n_correct,
            "accuracy_pct": percent(n_correct, n),
            "accuracy_ci95_pct": wilson_interval(n_correct, n) if n else (None, None),
        })
    #confirming the bins partition the predictions exactly once each, so no prediction can be
    #silently dropped if the bin edges are ever changed
    if sum(row["n"] for row in rows) != len(records):
        raise ValueError("Confidence bins do not cover every prediction exactly once; check "
                         "CONFIDENCE_BIN_EDGES against the accepted confidence range.")
    return rows

# ---------------------------------------------------------------------------
# The flagging rule: mark low-confidence predictions as unreliable
# ---------------------------------------------------------------------------

def flagging_table(records):
    """
    Apply the simple rule "flag any prediction below threshold as unreliable" at several
    thresholds and measure whether it works. For each threshold this reports how much of the
    output gets flagged, how much more error-prone the flagged predictions are than the kept
    ones, how many of all errors the rule manages to catch, and what accuracy survives on the
    predictions that are kept.

    The last number is the practically useful one: it is what a deployed system would achieve
    if it abstained on everything the rule flags, and it must be read together with coverage.
    """
    rows = []
    n_total = len(records)
    n_errors_total = sum(1 for r in records if is_error(r))

    #sweeping the flag threshold rather than fixing one, so the sensitivity of the rule is
    #visible. 0.7 remains the pre-specified primary threshold; the sweep is NOT scanned to
    #choose a best operating point, which would be post-hoc test-set optimisation
    for thr in FLAG_THRESHOLDS:
        flagged = [r for r in records if confidence_of(r) < thr]
        kept = [r for r in records if confidence_of(r) >= thr]

        n_flagged, n_kept = len(flagged), len(kept)
        flagged_errors = sum(1 for r in flagged if is_error(r))
        kept_errors = sum(1 for r in kept if is_error(r))

        #computing the lift from UNROUNDED proportions and rounding only for presentation, so
        #the ratio is not distorted by rounding each rate to one decimal place first
        flagged_frac = fraction(flagged_errors, n_flagged)
        kept_frac = fraction(kept_errors, n_kept)
        lift = None
        lift_note = None
        if flagged_frac is not None and kept_frac is not None and kept_frac > 0:
            lift = round(flagged_frac / kept_frac, 2)
        elif n_kept and kept_frac == 0:
            #the ratio is mathematically unbounded here rather than simply unavailable
            lift_note = "Undefined (unbounded): the kept group contained zero errors."
        else:
            lift_note = "Undefined: one of the two groups was empty."

        rows.append({
            "threshold": thr,
            "is_primary_threshold": (thr == CONFIDENT_THRESHOLD),
            "n_flagged": n_flagged,
            "pct_flagged": percent(n_flagged, n_total),
            "flagged_error_rate_pct": percent(flagged_errors, n_flagged),
            "flagged_error_rate_ci95_pct": wilson_interval(flagged_errors, n_flagged) if n_flagged else (None, None),
            "kept_error_rate_pct": percent(kept_errors, n_kept),
            "kept_error_rate_ci95_pct": wilson_interval(kept_errors, n_kept) if n_kept else (None, None),
            "error_rate_lift_flagged_over_kept": lift,
            "lift_note": lift_note,
            #the share of ALL errors that the rule successfully flags (its recall)
            "pct_of_all_errors_caught": percent(flagged_errors, n_errors_total),
            #what a system would achieve if it answered only on the kept predictions
            "retained_coverage_pct": percent(n_kept, n_total),
            "retained_accuracy_pct": percent(n_kept - kept_errors, n_kept),
            "retained_accuracy_ci95_pct": wilson_interval(n_kept - kept_errors, n_kept) if n_kept else (None, None),
        })
    return rows

# ---------------------------------------------------------------------------
# Reranking comparison: unpaired (descriptive) and paired (the defensible one)
# ---------------------------------------------------------------------------

def unpaired_reranking_comparison(by_cond):
    """
    Compare plain dense retrieval against dense + stance reranking at the CONDITION level.

    IMPORTANT LIMITATION, stated here because it is easy to over-read. The two means are
    computed over different sets of claims: the claims dense got right are not the same claims
    the reranked condition got right. So this supports the statement "correct predictions under
    reranking carried higher or lower average confidence than correct predictions under dense",
    but it does NOT support "reranking raised confidence on correct predictions", which is a
    within-claim statement. The paired comparison below is the one that supports that.
    """
    dense = by_cond.get("dense_roberta", [])
    rerank = by_cond.get("dense_reranked_roberta", [])
    #returning nothing when either condition is missing from these records
    if not dense or not rerank:
        return None

    d = confidence_summary(dense)
    r = confidence_summary(rerank)

    #differencing the measures that matter, reranked minus dense, so a positive number always
    #means "the reranked condition was higher on this measure"
    def diff(key, nd):
        if d.get(key) is None or r.get(key) is None:
            return None
        return round(r[key] - d[key], nd)

    keys = [
        ("mean_confidence_correct", 4),
        ("mean_confidence_wrong", 4),
        ("separation_correct_minus_wrong", 4),
        ("auroc_confidence_detects_correct", 4),
        ("global_confidence_accuracy_gap_pp", 1),
        ("accuracy_pct", 1),
    ]
    out = {k: {"dense": d[k], "reranked": r[k], "difference": diff(k, nd)} for k, nd in keys}
    out["comparison_type"] = ("unpaired: the two conditions' means are computed over "
                              "different subsets of claims")
    return out

def pair_records(left_records, right_records):
    """
    Pair two conditions' records by claim id so the same claim is compared with itself. Raises
    on duplicate ids (which would make the pairing ambiguous) and on no overlap at all.
    """
    left = {record_id(r): r for r in left_records}
    right = {record_id(r): r for r in right_records}
    #a length mismatch after keying means duplicate ids collapsed records together
    if len(left) != len(left_records):
        raise ValueError("Duplicate claim identifiers in the left condition.")
    if len(right) != len(right_records):
        raise ValueError("Duplicate claim identifiers in the right condition.")
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("No matching claim identifiers between the two conditions.")

    #checking that a matched id really is the same claim in both conditions, so a recycled or
    #corrupted id cannot produce a convincing but meaningless paired comparison
    pairs = []
    for cid in common:
        lrec, rrec = left[cid], right[cid]
        if lrec.get("true_label") != rrec.get("true_label"):
            raise ValueError(f"Gold-label mismatch for paired claim {cid}: "
                             f"{lrec.get('true_label')} versus {rrec.get('true_label')}.")
        lclaim, rclaim = normalised_claim_text(lrec), normalised_claim_text(rrec)
        if lclaim is not None and rclaim is not None and lclaim != rclaim:
            raise ValueError(f"Claim-text mismatch for paired claim {cid}.")
        pairs.append((lrec, rrec))
    return pairs

def paired_reranking_comparison(by_cond):
    """
    Compare dense against dense + stance reranking PER CLAIM, which is the comparison that most
    directly isolates the within-claim confidence difference associated with reranking, because
    the claim is held constant.

    Also reports the correct/wrong transition table, which is the confidence-side counterpart
    of the paired information Step 7's annotation export already carries per row:
      correct -> correct : stable correct
      correct -> wrong   : harm introduced by reranking
      wrong   -> correct : error repaired by reranking
      wrong   -> wrong   : persistent error
    The jointly-correct subset gives the most directly comparable contrast, because correctness
    is held constant as well as the claim. It remains descriptive: the subset is selected on
    both systems' outcomes, so no causal claim is made.
    """
    dense = by_cond.get("dense_roberta", [])
    rerank = by_cond.get("dense_reranked_roberta", [])
    #returning nothing when either condition is missing from these records
    if not dense or not rerank:
        return None

    pairs = pair_records(dense, rerank)

    #the per-claim confidence change, reranked minus dense, over every matched claim
    all_deltas = [confidence_of(rr) - confidence_of(dd) for dd, rr in pairs]

    #splitting the pairs into the four transition cells
    joint_correct = [(d, r) for d, r in pairs if is_correct(d) and is_correct(r)]
    joint_wrong = [(d, r) for d, r in pairs if is_error(d) and is_error(r)]
    correct_to_wrong = [(d, r) for d, r in pairs if is_correct(d) and is_error(r)]
    wrong_to_correct = [(d, r) for d, r in pairs if is_error(d) and is_correct(r)]

    def mean_delta(subset):
        """Mean per-claim confidence change over a subset of pairs."""
        #guarding against an empty subset
        if not subset:
            return None
        return round(sum(confidence_of(r) - confidence_of(d)
                         for d, r in subset) / len(subset), 4)

    return {
        "comparison_type": "paired by claim_id: the same claim under both conditions",
        "n_matched_claims": len(pairs),
        "mean_paired_confidence_change_all_claims": mean_of(all_deltas),
        "jointly_correct": {"n": len(joint_correct),
                            "mean_paired_confidence_change": mean_delta(joint_correct)},
        "jointly_wrong": {"n": len(joint_wrong),
                          "mean_paired_confidence_change": mean_delta(joint_wrong)},
        "correct_to_wrong": {"n": len(correct_to_wrong),
                             "mean_paired_confidence_change": mean_delta(correct_to_wrong)},
        "wrong_to_correct": {"n": len(wrong_to_correct),
                             "mean_paired_confidence_change": mean_delta(wrong_to_correct)},
        "prediction_transitions": {
            "correct_to_correct": len(joint_correct),
            "correct_to_wrong": len(correct_to_wrong),
            "wrong_to_correct": len(wrong_to_correct),
            "wrong_to_wrong": len(joint_wrong),
        },
        "note": ("The jointly-correct subset gives the most directly comparable descriptive "
                 "contrast, because the claim and the correctness outcome are both held "
                 "constant. The remaining difference is ASSOCIATED with the change in "
                 "retrieved evidence; it is a conditional descriptive subset selected on both "
                 "systems' outcomes, so it is not presented as a formal causal estimate or a "
                 "statistically tested effect."),
    }

# ---------------------------------------------------------------------------
# CSV writers, so the tables drop straight into the thesis or a plotting script
# ---------------------------------------------------------------------------

def write_csv(path, fieldnames, rows):
    """Write a list of dicts to CSV with a fixed column order."""
    #writing with utf-8 and no extra blank lines on any platform
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def export_csv_tables(out, out_dir, dataset, k):
    """
    Write the summary tables as CSV alongside the json. The json stays the machine-readable
    record for Step 9; these are for reading, for the thesis tables, and for plotting without
    having to re-parse nested json.
    """
    tag = f"{dataset}_k{k}"

    #the per-condition summary, one row per condition
    summary_rows = []
    for cond, s in out["per_condition"].items():
        row = dict(s)
        row["condition"] = cond
        row["accuracy_ci95_low_pct"] = s["accuracy_ci95_pct"][0]
        row["accuracy_ci95_high_pct"] = s["accuracy_ci95_pct"][1]
        summary_rows.append(row)
    write_csv(os.path.join(out_dir, f"confidence_summary_{tag}.csv"),
              ["condition", "n_predictions", "accuracy_pct", "accuracy_ci95_low_pct",
               "accuracy_ci95_high_pct", "mean_confidence_all", "mean_confidence_correct",
               "mean_confidence_wrong", "separation_correct_minus_wrong",
               "auroc_confidence_detects_correct", "global_confidence_accuracy_gap_pp",
               "high_confidence_errors", "high_confidence_error_pct_of_errors",
               "moderate_band_errors", "moderate_band_error_pct_of_errors"],
              summary_rows)

    #the confidence-bin table, one row per condition and bin
    bin_rows = []
    for cond, rows in out["confidence_bins"].items():
        for r in rows:
            row = dict(r)
            row["condition"] = cond
            #splitting the interval into two columns so the CSV can be used as a thesis table
            row["accuracy_ci95_low_pct"] = r["accuracy_ci95_pct"][0]
            row["accuracy_ci95_high_pct"] = r["accuracy_ci95_pct"][1]
            bin_rows.append(row)
    write_csv(os.path.join(out_dir, f"confidence_bins_{tag}.csv"),
              ["condition", "bin_low", "bin_high", "is_final_bin_inclusive_of_1", "n",
               "n_correct", "accuracy_pct", "accuracy_ci95_low_pct",
               "accuracy_ci95_high_pct"], bin_rows)

    #the flagging trade-off table, one row per condition and threshold
    flag_rows = []
    for cond, rows in out["flagging_rule"].items():
        for r in rows:
            row = dict(r)
            row["condition"] = cond
            flag_rows.append(row)
    write_csv(os.path.join(out_dir, f"flagging_tradeoff_{tag}.csv"),
              ["condition", "threshold", "is_primary_threshold", "pct_flagged",
               "flagged_error_rate_pct", "kept_error_rate_pct",
               "error_rate_lift_flagged_over_kept", "pct_of_all_errors_caught",
               "retained_coverage_pct", "retained_accuracy_pct"], flag_rows)

    #the per-true-label breakdown, one row per condition and label
    label_rows = []
    for cond, labels in out["per_true_label"].items():
        for label, s in labels.items():
            row = dict(s)
            row["condition"] = cond
            row["true_label"] = label
            row["accuracy_ci95_low_pct"] = s["accuracy_ci95_pct"][0]
            row["accuracy_ci95_high_pct"] = s["accuracy_ci95_pct"][1]
            label_rows.append(row)
    write_csv(os.path.join(out_dir, f"confidence_by_label_{tag}.csv"),
              ["condition", "true_label", "n", "n_correct", "accuracy_pct",
               "accuracy_ci95_low_pct", "accuracy_ci95_high_pct",
               "mean_confidence_all", "mean_confidence_correct", "mean_confidence_wrong",
               "auroc_confidence_detects_correct"], label_rows)

    #the paired dense-vs-reranked comparison, one row per subset, so the transition table and
    #the per-claim confidence changes are readable without opening the json
    paired = out.get("reranking_comparison_paired")
    if paired:
        paired_rows = [{"subset": "all_claims",
                        "n": paired["n_matched_claims"],
                        "mean_paired_confidence_change":
                            paired["mean_paired_confidence_change_all_claims"]}]
        for key in ("jointly_correct", "jointly_wrong", "correct_to_wrong",
                    "wrong_to_correct"):
            paired_rows.append({"subset": key,
                                "n": paired[key]["n"],
                                "mean_paired_confidence_change":
                                    paired[key]["mean_paired_confidence_change"]})
        write_csv(os.path.join(out_dir, f"paired_reranking_{tag}.csv"),
                  ["subset", "n", "mean_paired_confidence_change"], paired_rows)

    print(f"Wrote CSV tables to {out_dir} (confidence_summary, confidence_bins, "
          f"flagging_tradeoff, confidence_by_label, paired_reranking for {tag})")

# ---------------------------------------------------------------------------
# Mode: analyse one dataset at one retrieval depth
# ---------------------------------------------------------------------------

def analyse_confidence(records_path, dataset, k, records_seed, out_json, out_dir):
    """
    Run the full confidence analysis for one dataset at one retrieval depth and save it.
    Prints a readable report as it goes so the notebook output is self-explanatory.
    """
    #loading the saved per-claim records for this dataset and depth
    records = load_records(records_path)
    print(f"Loaded {len(records)} records from {records_path}")

    #re-validating the saved confidence BEFORE any confidence number is computed. the first
    #check is the Step 7 one (confidence equals max softmax, probabilities sum to 1); the
    #second adds the stricter structural checks. the whole step is meaningless if either fails
    validate_confidence_definition(records)
    validate_probability_vectors(records)

    #splitting the records into the four pipeline conditions and checking their integrity
    by_cond = records_by_condition(records)
    validate_condition_records(by_cond)

    out = {
        "dataset": dataset,
        "k": k,
        "records_path": records_path,
        "records_seed": records_seed,
        "seed_note": (
            f"These records were produced with the seed-{records_seed} classifiers "
            f"(the Step 6 matrix is seed 42, per step6_results.md). The "
            "confidence behaviour reported here characterises that run and is NOT verified "
            "across training seeds; the project's multi-seed work covers the Step 2 baseline "
            "and the Step 5 variance study, not the Step 6 matrix."),
        "confidence_definition": "max softmax probability over the 3 classes (validated above)",
        "chance_level": round(CHANCE_LEVEL, 4),
        "threshold_policy": {
            "primary_threshold": CONFIDENT_THRESHOLD,
            "selection_basis": ("Pre-specified before Step 8 and shared with the Step 7 "
                                "high-confidence error definition, so both steps use one "
                                "definition of high confidence."),
            "other_thresholds": ("Reported as sensitivity analysis only. They are NOT used to "
                                 "select an optimal operating threshold on the test set, "
                                 "which would be post-hoc test-set optimisation."),
            "status": ("An analytical threshold chosen for consistency with Step 7, not an "
                       "empirically optimal abstention threshold for deployment."),
        },
        "per_condition": {},
        "per_true_label": {},
        "confidence_bins": {},
        "flagging_rule": {},
    }

    #walking the conditions in a fixed order so the printed report is stable
    for cond in CONDITIONS:
        recs = by_cond.get(cond, [])
        #skipping a condition that is absent from this records file
        if not recs:
            continue

        summary = confidence_summary(recs)
        out["per_condition"][cond] = summary
        out["per_true_label"][cond] = per_label_summary(recs)
        out["confidence_bins"][cond] = confidence_bin_table(recs)
        out["flagging_rule"][cond] = flagging_table(recs)

        #printing the per-condition summary
        print(f"\n{cond} (n={summary['n_predictions']}):")
        print(f"  accuracy                       {summary['accuracy_pct']}%  "
              f"95% CI {summary['accuracy_ci95_pct']}")
        print(f"  mean confidence when correct   {summary['mean_confidence_correct']}")
        print(f"  mean confidence when wrong     {summary['mean_confidence_wrong']}")
        print(f"  separation (correct - wrong)   {summary['separation_correct_minus_wrong']}")
        print(f"  AUROC (confidence detects correct) {summary['auroc_confidence_detects_correct']}")
        print(f"  confidence minus accuracy gap  {summary['global_confidence_accuracy_gap_pp']} pp "
              f"(aggregate only, not a calibration measure)")
        print(f"  errors at confidence >= {CONFIDENT_THRESHOLD}   "
              f"{summary['high_confidence_errors']}/{summary['n_wrong']} "
              f"({summary['high_confidence_error_pct_of_errors']}%)")
        print(f"  errors in the {MODERATE_BAND[0]}-{MODERATE_BAND[1]} moderate band  "
              f"{summary['moderate_band_errors']}/{summary['n_wrong']} "
              f"({summary['moderate_band_error_pct_of_errors']}%)")

        #printing the per-true-label breakdown so class-specific behaviour is visible
        print("  by true label:")
        for label, s in out["per_true_label"][cond].items():
            print(f"    {label:<11} n={s['n']:<4} accuracy {s['accuracy_pct']}%  "
                  f"mean conf correct {s['mean_confidence_correct']}  "
                  f"wrong {s['mean_confidence_wrong']}  "
                  f"AUROC {s['auroc_confidence_detects_correct']}")

        #printing the confidence-bin accuracy table for this condition
        print("  accuracy by confidence bin:")
        for row in out["confidence_bins"][cond]:
            if row["n"] == 0:
                continue
            #closing the bracket on the final bin, which does include a confidence of 1.0
            closing = "]" if row["is_final_bin_inclusive_of_1"] else ")"
            print(f"    [{row['bin_low']:.3f}, {row['bin_high']:.3f}{closing}  "
                  f"n={row['n']:<4} accuracy {row['accuracy_pct']}%  "
                  f"95% CI {row['accuracy_ci95_pct']}")

        #printing the flagging rule sweep for this condition
        print("  flagging rule (flag predictions below threshold as unreliable):")
        for row in out["flagging_rule"][cond]:
            #showing "undefined" rather than a bare None when the ratio cannot be formed
            lift_text = row["error_rate_lift_flagged_over_kept"]
            if lift_text is None:
                lift_text = "undefined"
            primary = "  <- primary" if row["is_primary_threshold"] else ""
            print(f"    thr {row['threshold']}: flagged {row['pct_flagged']}% of predictions, "
                  f"error rate flagged {row['flagged_error_rate_pct']}% vs kept "
                  f"{row['kept_error_rate_pct']}% (lift {lift_text}), "
                  f"catches {row['pct_of_all_errors_caught']}% of errors, "
                  f"retained accuracy {row['retained_accuracy_pct']}% "
                  f"at {row['retained_coverage_pct']}% coverage{primary}")

    #comparing dense against dense + stance reranking, first unpaired then paired
    out["reranking_comparison_unpaired"] = unpaired_reranking_comparison(by_cond)
    out["reranking_comparison_paired"] = paired_reranking_comparison(by_cond)

    if out["reranking_comparison_unpaired"]:
        print("\nReranking effect on confidence, UNPAIRED (different claim subsets, "
              "descriptive only):")
        for key, vals in out["reranking_comparison_unpaired"].items():
            #skipping the explanatory string entry
            if not isinstance(vals, dict):
                continue
            print(f"  {key:<36} dense {vals['dense']}  ->  reranked {vals['reranked']}  "
                  f"(difference {vals['difference']})")

    if out["reranking_comparison_paired"]:
        p = out["reranking_comparison_paired"]
        print(f"\nReranking effect on confidence, PAIRED by claim "
              f"(n={p['n_matched_claims']} matched claims):")
        print(f"  mean per-claim confidence change (all claims)   "
              f"{p['mean_paired_confidence_change_all_claims']}")
        print(f"  jointly correct  n={p['jointly_correct']['n']:<5} "
              f"mean change {p['jointly_correct']['mean_paired_confidence_change']}")
        print(f"  jointly wrong    n={p['jointly_wrong']['n']:<5} "
              f"mean change {p['jointly_wrong']['mean_paired_confidence_change']}")
        print("  prediction transitions (dense -> reranked):")
        t = p["prediction_transitions"]
        print(f"    correct -> correct  {t['correct_to_correct']:<5} (stable correct)")
        print(f"    correct -> wrong    {t['correct_to_wrong']:<5} (harm introduced by reranking)")
        print(f"    wrong   -> correct  {t['wrong_to_correct']:<5} (error repaired by reranking)")
        print(f"    wrong   -> wrong    {t['wrong_to_wrong']:<5} (persistent error)")

    #recording the interpretation notes in the saved file so the numbers cannot be read out of
    #context later, in the same spirit as the Step 7 notes
    out["statistics_note"] = (
        "95% intervals are Wilson score intervals returned as percentages, the same helper "
        "used in Step 7. No significance test is run between conditions, in line with the "
        "project scope: comparisons are descriptive and read alongside the intervals. AUROC is "
        "the probability that a randomly chosen correct prediction carried higher confidence "
        "than a randomly chosen wrong one, with tied confidence values counting as one half. "
        "It is computed from ranks with ties assigned average ranks; 0.5 means confidence "
        "carries no discriminative information about correctness.")
    out["note"] = (
        "Confidence is the maximum softmax probability over 3 classes, so it cannot fall below "
        "about 0.333 and 'low confidence' means low relative to that floor. "
        "global_confidence_accuracy_gap_pp is mean confidence minus accuracy in percentage "
        "points: positive is aggregate overconfidence, negative is aggregate underconfidence. "
        "It is NOT a calibration measure, because opposing errors within the distribution can "
        "cancel; results should be described in terms of discrimination or alignment, not "
        "calibration, unless a formal metric such as ECE or a Brier score is added. The "
        "flagging rule's retained accuracy is what a system would achieve if it abstained on "
        "every flagged prediction, so it must be read together with retained coverage: high "
        "accuracy on a small kept fraction is not automatically a good trade. The unpaired "
        "reranking comparison is computed over different claim subsets, so it cannot speak to "
        "within-claim change; the paired comparison can, descriptively.")

    #saving the analysis so Step 9 can compare the two corpora without recomputing
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved confidence analysis to {out_json}")

    #writing the same tables as CSV for the thesis and for plotting
    export_csv_tables(out, out_dir, dataset, k)
    return out

# ---------------------------------------------------------------------------
# Mode: does confidence move with accuracy as k changes?
# ---------------------------------------------------------------------------

def cross_k_confidence(records_dir, dataset, records_seed, out_json, out_dir):
    """
    Step 6 found the pipeline conditions to be sensitive to retrieval depth, with plain
    retrieval and the reranked condition behaving differently from each other as k grows.
    This analysis tests whether confidence moves alongside accuracy as k varies, without
    assuming a direction in advance.

    Read the columns together rather than any one alone. Mean confidence can stay flat while
    the correct and wrong distributions converge, which is why separation and AUROC are
    reported next to it: a falling AUROC means confidence is becoming a worse error signal
    even if the average has not moved.
    """
    out = {"dataset": dataset, "records_seed": records_seed, "per_condition": {}}

    #walking each depth in the Step 6 matrix and loading that depth's records. the path is
    #resolved by the shared Step 7 resolver so the matrix naming convention lives in exactly
    #one place rather than being duplicated here
    per_k = {}
    for k in MATRIX_K_VALUES:
        try:
            path = resolve_records_path(records_dir, dataset, k, explicit=None)
        except FileNotFoundError:
            #skipping a depth whose records file is not present rather than failing outright
            print(f"  (no records file for k={k}, skipping)")
            continue
        #running the same validation as analyse mode, so a malformed depth cannot slip into
        #the cross-k table just because it was never opened in analyse mode
        print(f"\nValidating k={k} records ({path}):")
        recs = load_records(path)
        validate_confidence_definition(recs)
        validate_probability_vectors(recs)
        by_cond_k = records_by_condition(recs)
        validate_condition_records(by_cond_k)
        per_k[k] = by_cond_k

    #failing loudly rather than producing a vacuous result when the records are not there
    if not per_k:
        raise FileNotFoundError(
            f"No Step 6 records were found for dataset '{dataset}' in {records_dir}.")
    if len(per_k) < 2:
        raise ValueError(
            f"Cross-k analysis needs at least two retrieval depths; only k={sorted(per_k)} "
            f"was found. A single depth is not a cross-k comparison.")

    #recording which depths were actually analysed, rather than claiming all four were, since
    #missing files are skipped above
    out["k_values_requested"] = list(MATRIX_K_VALUES)
    out["k_values_analysed"] = sorted(per_k)
    out["k_values_missing"] = [k for k in MATRIX_K_VALUES if k not in per_k]

    #confirming every condition covers the same claims at every depth before comparing across k
    validate_cross_k_claim_sets(per_k)

    #building the per-condition trend across the depths that were found
    for cond in CONDITIONS:
        rows = []
        for k in sorted(per_k.keys()):
            recs = per_k[k].get(cond, [])
            if not recs:
                continue
            s = confidence_summary(recs)
            rows.append({
                "condition": cond,
                "k": k,
                "n": s["n_predictions"],
                "accuracy_pct": s["accuracy_pct"],
                "mean_confidence_all": s["mean_confidence_all"],
                "mean_confidence_correct": s["mean_confidence_correct"],
                "mean_confidence_wrong": s["mean_confidence_wrong"],
                "separation_correct_minus_wrong": s["separation_correct_minus_wrong"],
                "auroc_confidence_detects_correct": s["auroc_confidence_detects_correct"],
                "global_confidence_accuracy_gap_pp": s["global_confidence_accuracy_gap_pp"],
            })
        if rows:
            out["per_condition"][cond] = rows

    #printing the trend so the notebook shows the accuracy and confidence columns side by side
    print(f"\nConfidence across retrieval depth k ({dataset}):")
    for cond, rows in out["per_condition"].items():
        print(f"\n  {cond}:")
        print(f"    {'k':<4}{'accuracy':<12}{'mean conf':<12}{'separation':<13}"
              f"{'AUROC':<9}{'conf-acc gap':<13}")
        for r in rows:
            print(f"    {r['k']:<4}{str(r['accuracy_pct']) + '%':<12}"
                  f"{str(r['mean_confidence_all']):<12}"
                  f"{str(r['separation_correct_minus_wrong']):<13}"
                  f"{str(r['auroc_confidence_detects_correct']):<9}"
                  f"{str(r['global_confidence_accuracy_gap_pp']) + ' pp':<13}")

    out["note"] = (
        "Read the accuracy, mean-confidence, separation and AUROC columns together, and do "
        "not reduce the interpretation to the two aggregate columns. Several patterns are "
        "possible and each means something different: accuracy falling while mean confidence "
        "stays flat means the model does not register the degradation; accuracy and confidence "
        "falling together means it partly does; a falling AUROC with flat mean confidence "
        "means the correct and wrong distributions are converging, so confidence is becoming a "
        "worse error signal regardless of its average. The no_retrieval condition uses no "
        "documents, so it is identical at every k by construction and acts as the flat control.")
    out["seed_note"] = (
        f"Seed-{records_seed} records (the Step 6 matrix is seed 42, per step6_results.md); "
        f"not verified across training seeds.")

    #saving the depth sweep for the Step 9 cross-corpus comparison
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved cross-k confidence analysis to {out_json}")

    #writing the flat version as CSV for plotting accuracy and confidence against k
    flat = [row for rows in out["per_condition"].values() for row in rows]
    write_csv(os.path.join(out_dir, f"confidence_by_k_{dataset}.csv"),
              ["condition", "k", "n", "accuracy_pct", "mean_confidence_all",
               "mean_confidence_correct", "mean_confidence_wrong",
               "separation_correct_minus_wrong", "auroc_confidence_detects_correct",
               "global_confidence_accuracy_gap_pp"], flat)
    print(f"Wrote CSV table to {out_dir} (confidence_by_k_{dataset}.csv)")
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    #setting up the command line arguments, mirroring the Step 7 script's conventions
    parser = argparse.ArgumentParser(description="Step 8 retrieval-aware confidence scoring")
    parser.add_argument("--mode", required=True, choices=["analyse", "cross_k"])
    parser.add_argument("--dataset", default="scifact",
                        choices=["scifact", "scifact_open"])
    parser.add_argument("--records_dir", default="results/step6_matrix")
    parser.add_argument("--records_path", default=None,
                        help="Explicit records file; overrides --records_dir/--k for locating "
                             "the file.")
    parser.add_argument("--k", type=int, default=None, choices=list(MATRIX_K_VALUES),
                        help="Retrieval depth to analyse, one of the Step 6 matrix depths: "
                             "1, 3, 5, or 10. Required in analyse mode, so an explicit "
                             "--records_path cannot be mislabelled with a default depth.")
    parser.add_argument("--records_seed", type=int, default=None,
                        help="Training seed that produced the supplied records. Defaults to "
                             "42, the Step 6 matrix seed, when reading the standard records "
                             "directory. REQUIRED with --records_path, because the seed "
                             "cannot be inferred from an arbitrary file and a wrong default "
                             "would attach false provenance to the saved output.")
    parser.add_argument("--out_dir", default="results/step8_confidence")
    args = parser.parse_args()

    #rejecting arguments the chosen mode would silently ignore, so nobody can believe they
    #restricted a cross-k run to one file or one depth when the mode does neither
    if args.mode == "cross_k":
        if args.records_path is not None:
            parser.error("--records_path is only valid in analyse mode. Cross-k mode reads "
                         "each depth from --records_dir.")
        if args.k is not None:
            parser.error("--k is only valid in analyse mode. Cross-k mode analyses all "
                         "available matrix depths.")

    if args.mode == "analyse":
        #the depth is metadata on the saved output, so it must be stated rather than defaulted
        if args.k is None:
            parser.error("--k is required in analyse mode.")
        #refusing to guess provenance for an explicitly supplied records file: the seed cannot
        #be read off an arbitrary path, and silently writing "seed 42" would be a false claim
        if args.records_path is not None and args.records_seed is None:
            parser.error("--records_seed must be supplied when using --records_path, because "
                         "the seed cannot be inferred safely from an arbitrary records file.")
    records_seed = args.records_seed if args.records_seed is not None else DEFAULT_RECORDS_SEED

    #ensuring the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "analyse":
        #resolving the records file for this dataset and depth, reusing the Step 7 resolver
        records_path = resolve_records_path(args.records_dir, args.dataset, args.k,
                                            explicit=args.records_path)
        analyse_confidence(
            records_path=records_path,
            dataset=args.dataset,
            k=args.k,
            records_seed=records_seed,
            out_json=os.path.join(args.out_dir,
                                  f"confidence_{args.dataset}_k{args.k}.json"),
            out_dir=args.out_dir,
        )
    elif args.mode == "cross_k":
        cross_k_confidence(
            records_dir=args.records_dir,
            dataset=args.dataset,
            records_seed=records_seed,
            out_json=os.path.join(args.out_dir, f"confidence_by_k_{args.dataset}.json"),
            out_dir=args.out_dir,
        )

if __name__ == "__main__":
    main()