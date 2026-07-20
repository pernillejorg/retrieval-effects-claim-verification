"""
Step 8: Retrieval-aware confidence scoring for RAG claim verification.

This script answers one focused question: does the model know when it is wrong, and does
stance reranking change that? When RoBERTa classifies a claim it outputs a softmax
distribution over the three labels, and the highest probability is taken as the confidence
score. Step 8 asks whether that self-reported confidence carries usable information about
correctness.

Like Step 7, this script works ENTIRELY from the per-claim records already saved by the
Step 5 / Step 6 pipeline runs (records_*.json). It does NOT re-run retrieval, reranking, or
the classifier, and it needs no GPU. Every record already stores `confidence` and the full
`probabilities` vector, so the whole analysis is a re-reading of existing results.

WHAT IS COMPUTED (the four things Step 8 requires, plus two the earlier steps promised):
  1. Confidence is recorded for every prediction in every pipeline condition. The script
     first re-validates that the saved confidence really is the maximum softmax probability,
     reusing the same check as Step 7, so the whole analysis rests on a verified definition.
  2. Whether low confidence correlates with error. This is reported three ways: the mean
     confidence on correct versus wrong predictions and the gap between them, a coarse
     confidence-bin accuracy table, and a single discrimination number (AUROC).
  3. Whether stance reranking produces higher confidence on correct predictions than plain
     dense retrieval, which is the specific comparison the project hypothesis needs.
  4. A simple flagging rule: predictions below a confidence threshold are marked unreliable.
     The script measures whether flagged predictions really are disproportionately wrong, and
     what accuracy survives on the predictions that are kept.
  5. Cross-k behaviour, promised in step6_results.md: Step 6 found accuracy falls as k grows,
     so this asks whether confidence falls with it (the model noticing the degradation) or
     stays flat (the model failing to notice).
  6. The link back to Step 7: the share of errors made at or above the 0.7 high-confidence
     threshold, and the size of the 0.5 to 0.7 moderate band that Step 7 documented as the
     excluded_below_threshold group.

ON SCOPE, DELIBERATELY LIMITED. The project plan says not to attempt full calibration curve
analysis or complex statistical testing, so none is done here. The confidence-bin table and
the single overconfidence gap (mean confidence minus accuracy) are reported as plain
descriptive summaries, not as a fitted calibration analysis, and no significance test is run
between conditions. Proportions carry Wilson 95% intervals, the same interval used in Step 7,
so the comparisons stay descriptive but honest about precision.

ON AUROC, AND WHY IT IS NOT OVER-REACHING. AUROC here is used in one narrow sense: the
probability that a randomly chosen correct prediction carried higher confidence than a
randomly chosen wrong one. It is a single discrimination summary, not a calibration curve.
0.5 means confidence carries no information about correctness; above 0.5 means higher
confidence really does indicate a more likely correct answer. It is computed from the ranks
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

#importing argparse for command line mode and dataset selection
import argparse

#reusing the Step 7 helpers so the record handling, the confidence definition check and the
#Wilson interval are provably the SAME code in both steps rather than a second copy that
#could quietly drift. this is why the two steps' numbers can be compared directly.
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
CONFIDENCE_BIN_EDGES = [CHANCE_LEVEL, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]

#the thresholds swept for the flagging rule. 0.7 is the primary threshold carried over from
#Step 7 so the two steps use one consistent definition of "high confidence"
FLAG_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

#the moderate confidence band that Step 7 documented as excluded_below_threshold: genuine
#classifier errors in mechanism that sat below the 0.7 bar the category requires
MODERATE_BAND = (0.5, CONFIDENT_THRESHOLD)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def confidence_of(record):
    """Return the saved confidence for a record as a float, defaulting to 0.0 if absent."""
    #reading the saved confidence value defensively
    return float(record.get("confidence", 0.0))

def is_correct(record):
    """A record is correct when the prediction matches the true label (inverse of is_error)."""
    #reusing the Step 7 definition of an error so both steps agree on what counts as wrong
    return not is_error(record)

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

# ---------------------------------------------------------------------------
# Discrimination: does confidence separate correct from wrong predictions?
# ---------------------------------------------------------------------------

def auroc_correct_vs_wrong(records):
    """
    Probability that a randomly chosen CORRECT prediction carried higher confidence than a
    randomly chosen WRONG one, computed from ranks (the Mann-Whitney U form of AUROC) with
    ties handled by average ranks. Returns None when either group is empty.

    Interpretation: 0.5 means confidence tells you nothing about whether the answer is right;
    higher means confidence is a usable error signal; below 0.5 would mean the model is more
    confident when it is wrong, which would be actively misleading.
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
    discriminates correct from wrong, and how far mean confidence sits above accuracy.
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

    #computing accuracy with a Wilson interval, matching the interval used in Step 7
    acc = percent(n_correct, n)
    acc_ci = wilson_interval(n_correct, n)

    #the separation is the headline "does the model know when it is wrong" number: how much
    #more confident it is when right than when wrong
    mean_c = mean_of(conf_correct)
    mean_w = mean_of(conf_wrong)
    separation = round(mean_c - mean_w, 4) if (mean_c is not None and mean_w is not None) else None

    #the overconfidence gap is mean confidence minus accuracy, both on the same 0-100 scale.
    #a positive gap means the model claims more certainty than its accuracy earns. this is a
    #single descriptive number, NOT a fitted calibration analysis
    mean_all = mean_of(conf_all)
    overconfidence = None
    if mean_all is not None and acc is not None:
        overconfidence = round(100 * mean_all - acc, 1)

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
        "accuracy_ci95": acc_ci,
        "mean_confidence_all": mean_all,
        "mean_confidence_correct": mean_c,
        "mean_confidence_wrong": mean_w,
        "median_confidence_correct": median_of(conf_correct),
        "median_confidence_wrong": median_of(conf_wrong),
        "separation_correct_minus_wrong": separation,
        "auroc_confidence_detects_correct": auroc_correct_vs_wrong(records),
        "overconfidence_gap_pp": overconfidence,
        "high_confidence_errors": hc_errors,
        "high_confidence_error_pct_of_errors": percent(hc_errors, n_wrong),
        "moderate_band_errors": mod_errors,
        "moderate_band_error_pct_of_errors": percent(mod_errors, n_wrong),
    }

# ---------------------------------------------------------------------------
# Confidence-bin accuracy (descriptive, not a fitted calibration curve)
# ---------------------------------------------------------------------------

def confidence_bin_table(records):
    """
    Group predictions into coarse confidence bins and report accuracy in each, with Wilson
    intervals. If confidence is informative, accuracy should rise across the bins. This is a
    descriptive breakdown rather than a calibration analysis, and bins are kept wide because
    the per-condition samples are only a few hundred predictions.
    """
    rows = []
    #walking each adjacent pair of bin edges
    for lo, hi in zip(CONFIDENCE_BIN_EDGES[:-1], CONFIDENCE_BIN_EDGES[1:]):
        #selecting the predictions whose confidence falls in this half-open bin
        in_bin = [r for r in records if lo <= confidence_of(r) < hi]
        n = len(in_bin)
        n_correct = sum(1 for r in in_bin if is_correct(r))
        rows.append({
            "bin_low": round(lo, 3),
            "bin_high": round(min(hi, 1.0), 3),
            "n": n,
            "n_correct": n_correct,
            "accuracy_pct": percent(n_correct, n),
            "accuracy_ci95": wilson_interval(n_correct, n) if n else (None, None),
        })
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
    if it abstained on everything the rule flags.
    """
    rows = []
    n_total = len(records)
    n_errors_total = sum(1 for r in records if is_error(r))

    #sweeping the flag threshold rather than fixing one, so the result is not an artefact of a
    #single arbitrary cut-point
    for thr in FLAG_THRESHOLDS:
        flagged = [r for r in records if confidence_of(r) < thr]
        kept = [r for r in records if confidence_of(r) >= thr]

        n_flagged, n_kept = len(flagged), len(kept)
        flagged_errors = sum(1 for r in flagged if is_error(r))
        kept_errors = sum(1 for r in kept if is_error(r))

        #the error rate inside each group, which is what tells us if the rule separates them
        flagged_err_rate = percent(flagged_errors, n_flagged)
        kept_err_rate = percent(kept_errors, n_kept)

        #the lift says how many times more error-prone the flagged group is than the kept
        #group. above 1 means the rule is doing something useful
        lift = None
        if flagged_err_rate is not None and kept_err_rate:
            lift = round(flagged_err_rate / kept_err_rate, 2)

        rows.append({
            "threshold": thr,
            "n_flagged": n_flagged,
            "pct_flagged": percent(n_flagged, n_total),
            "flagged_error_rate_pct": flagged_err_rate,
            "flagged_error_rate_ci95": wilson_interval(flagged_errors, n_flagged) if n_flagged else (None, None),
            "kept_error_rate_pct": kept_err_rate,
            "kept_error_rate_ci95": wilson_interval(kept_errors, n_kept) if n_kept else (None, None),
            "error_rate_lift_flagged_over_kept": lift,
            #the share of ALL errors that the rule successfully flags (its recall)
            "pct_of_all_errors_caught": percent(flagged_errors, n_errors_total),
            #what a system would achieve if it answered only on the kept predictions
            "retained_coverage_pct": percent(n_kept, n_total),
            "retained_accuracy_pct": percent(n_kept - kept_errors, n_kept),
            "retained_accuracy_ci95": wilson_interval(n_kept - kept_errors, n_kept) if n_kept else (None, None),
        })
    return rows

# ---------------------------------------------------------------------------
# Reranking comparison (the specific hypothesis Step 8 has to test)
# ---------------------------------------------------------------------------

def reranking_confidence_comparison(by_cond):
    """
    Compare plain dense retrieval against dense + stance reranking on the confidence measures,
    which is the question the project plan asks directly: does stance reranking produce higher
    confidence on correct predictions, and does it make confidence a better error signal?
    """
    dense = by_cond.get("dense_roberta", [])
    rerank = by_cond.get("dense_reranked_roberta", [])
    #returning nothing when either condition is missing from these records
    if not dense or not rerank:
        return None

    d = confidence_summary(dense)
    r = confidence_summary(rerank)

    #differencing the measures that matter for the hypothesis, reranked minus dense, so a
    #positive number always means "reranking increased this"
    def diff(key, nd=4):
        if d.get(key) is None or r.get(key) is None:
            return None
        return round(r[key] - d[key], nd)

    return {
        "mean_confidence_correct": {"dense": d["mean_confidence_correct"],
                                    "reranked": r["mean_confidence_correct"],
                                    "difference": diff("mean_confidence_correct")},
        "mean_confidence_wrong": {"dense": d["mean_confidence_wrong"],
                                  "reranked": r["mean_confidence_wrong"],
                                  "difference": diff("mean_confidence_wrong")},
        "separation_correct_minus_wrong": {"dense": d["separation_correct_minus_wrong"],
                                           "reranked": r["separation_correct_minus_wrong"],
                                           "difference": diff("separation_correct_minus_wrong")},
        "auroc_confidence_detects_correct": {"dense": d["auroc_confidence_detects_correct"],
                                             "reranked": r["auroc_confidence_detects_correct"],
                                             "difference": diff("auroc_confidence_detects_correct")},
        "overconfidence_gap_pp": {"dense": d["overconfidence_gap_pp"],
                                  "reranked": r["overconfidence_gap_pp"],
                                  "difference": diff("overconfidence_gap_pp", nd=1)},
        "accuracy_pct": {"dense": d["accuracy_pct"],
                         "reranked": r["accuracy_pct"],
                         "difference": diff("accuracy_pct", nd=1)},
    }

# ---------------------------------------------------------------------------
# Mode: analyse one dataset at one retrieval depth
# ---------------------------------------------------------------------------

def analyse_confidence(records_path, dataset, k, out_json):
    """
    Run the full confidence analysis for one dataset at one retrieval depth and save it.
    Prints a readable report as it goes so the notebook output is self-explanatory.
    """
    #loading the saved per-claim records for this dataset and depth
    records = load_records(records_path)
    print(f"Loaded {len(records)} records from {records_path}")

    #re-validating that the saved confidence really is the max softmax probability BEFORE any
    #confidence number is computed. this is the same check Step 7 runs, and the whole step is
    #meaningless if it fails
    validate_confidence_definition(records)

    #splitting the records into the four pipeline conditions
    by_cond = records_by_condition(records)

    out = {
        "dataset": dataset,
        "k": k,
        "records_path": records_path,
        "confidence_definition": "max softmax probability over the 3 classes (validated above)",
        "chance_level": round(CHANCE_LEVEL, 4),
        "primary_high_confidence_threshold": CONFIDENT_THRESHOLD,
        "per_condition": {},
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
        out["confidence_bins"][cond] = confidence_bin_table(recs)
        out["flagging_rule"][cond] = flagging_table(recs)

        #printing the per-condition summary
        print(f"\n{cond} (n={summary['n_predictions']}):")
        print(f"  accuracy                       {summary['accuracy_pct']}%  "
              f"95% CI {summary['accuracy_ci95']}")
        print(f"  mean confidence when correct   {summary['mean_confidence_correct']}")
        print(f"  mean confidence when wrong     {summary['mean_confidence_wrong']}")
        print(f"  separation (correct - wrong)   {summary['separation_correct_minus_wrong']}")
        print(f"  AUROC (confidence detects correct) {summary['auroc_confidence_detects_correct']}")
        print(f"  overconfidence gap             {summary['overconfidence_gap_pp']} pp "
              f"(mean confidence minus accuracy)")
        print(f"  errors at confidence >= {CONFIDENT_THRESHOLD}   "
              f"{summary['high_confidence_errors']}/{summary['n_wrong']} "
              f"({summary['high_confidence_error_pct_of_errors']}%)")
        print(f"  errors in the {MODERATE_BAND[0]}-{MODERATE_BAND[1]} moderate band  "
              f"{summary['moderate_band_errors']}/{summary['n_wrong']} "
              f"({summary['moderate_band_error_pct_of_errors']}%)")

        #printing the confidence-bin accuracy table for this condition
        print("  accuracy by confidence bin:")
        for row in out["confidence_bins"][cond]:
            if row["n"] == 0:
                continue
            print(f"    [{row['bin_low']:.3f}, {row['bin_high']:.3f})  "
                  f"n={row['n']:<4} accuracy {row['accuracy_pct']}%  "
                  f"95% CI {row['accuracy_ci95']}")

        #printing the flagging rule sweep for this condition
        print("  flagging rule (flag predictions below threshold as unreliable):")
        for row in out["flagging_rule"][cond]:
            #showing n/a when the kept group has no errors at all, which makes the ratio undefined
            lift_text = ("n/a" if row["error_rate_lift_flagged_over_kept"] is None
                         else row["error_rate_lift_flagged_over_kept"])
            print(f"    thr {row['threshold']}: flagged {row['pct_flagged']}% of predictions, "
                  f"error rate flagged {row['flagged_error_rate_pct']}% vs kept "
                  f"{row['kept_error_rate_pct']}% (lift {lift_text}), "
                  f"catches {row['pct_of_all_errors_caught']}% of errors, "
                  f"retained accuracy {row['retained_accuracy_pct']}% "
                  f"at {row['retained_coverage_pct']}% coverage")

    #comparing dense against dense + stance reranking, which is the project's own hypothesis
    comparison = reranking_confidence_comparison(by_cond)
    out["reranking_comparison"] = comparison
    if comparison:
        print("\nReranking effect on confidence (dense -> dense + stance rerank):")
        for key, vals in comparison.items():
            print(f"  {key:<32} dense {vals['dense']}  ->  reranked {vals['reranked']}  "
                  f"(difference {vals['difference']})")
    else:
        print("\nReranking comparison unavailable: one of the two conditions is missing.")

    #recording the interpretation notes in the saved file so the numbers cannot be read out of
    #context later, in the same spirit as the Step 7 notes
    out["statistics_note"] = (
        "95% intervals are Wilson score intervals, the same interval used in Step 7. No "
        "significance test is run between conditions, in line with the project scope: the "
        "comparisons are descriptive and are read alongside the intervals. AUROC is the "
        "probability that a randomly chosen correct prediction carried higher confidence than "
        "a randomly chosen wrong one, computed from ranks with ties averaged; 0.5 means "
        "confidence carries no information about correctness.")
    out["note"] = (
        "Confidence is the maximum softmax probability over 3 classes, so it cannot fall "
        "below about 0.333 and 'low confidence' means low relative to that floor. The "
        "overconfidence gap is mean confidence minus accuracy, a single descriptive number "
        "rather than a fitted calibration analysis, which the project scope excludes. The "
        "flagging rule's retained accuracy is what a system would achieve if it abstained on "
        "every flagged prediction, so it must be read together with retained coverage: high "
        "accuracy on a small kept fraction is not automatically a good trade.")

    #saving the analysis so Step 9 can compare the two corpora without recomputing
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved confidence analysis to {out_json}")
    return out

# ---------------------------------------------------------------------------
# Mode: does confidence track the accuracy decline as k grows?
# ---------------------------------------------------------------------------

def cross_k_confidence(records_dir, dataset, out_json):
    """
    Step 6 found that accuracy falls as retrieval depth k grows (the evidence-overload
    result). This asks the natural follow-up promised in step6_results.md: does the model's
    confidence fall with it? If confidence drops alongside accuracy, the model is at least
    partly aware that the extra documents are hurting it. If confidence stays flat or rises
    while accuracy falls, the model is blind to its own degradation, which is a stronger and
    more concerning claim about retrieval-aware confidence.
    """
    out = {"dataset": dataset, "k_values": list(MATRIX_K_VALUES), "per_condition": {}}

    #walking each depth in the Step 6 matrix and loading that depth's records
    per_k = {}
    for k in MATRIX_K_VALUES:
        path = os.path.join(records_dir, f"records_{dataset}_k{k}_thr0_5.json")
        #skipping a depth whose records file is not present rather than failing outright
        if not os.path.exists(path):
            print(f"  (no records file for k={k}, skipping: {path})")
            continue
        per_k[k] = records_by_condition(load_records(path))

    #building the per-condition trend across the depths that were found
    for cond in CONDITIONS:
        rows = []
        for k in sorted(per_k.keys()):
            recs = per_k[k].get(cond, [])
            if not recs:
                continue
            s = confidence_summary(recs)
            rows.append({
                "k": k,
                "n": s["n_predictions"],
                "accuracy_pct": s["accuracy_pct"],
                "mean_confidence_all": s["mean_confidence_all"],
                "mean_confidence_correct": s["mean_confidence_correct"],
                "mean_confidence_wrong": s["mean_confidence_wrong"],
                "separation_correct_minus_wrong": s["separation_correct_minus_wrong"],
                "auroc_confidence_detects_correct": s["auroc_confidence_detects_correct"],
                "overconfidence_gap_pp": s["overconfidence_gap_pp"],
            })
        if rows:
            out["per_condition"][cond] = rows

    #printing the trend so the notebook shows the accuracy and confidence columns side by side
    print(f"\nConfidence across retrieval depth k ({dataset}):")
    for cond, rows in out["per_condition"].items():
        print(f"\n  {cond}:")
        print(f"    {'k':<4}{'accuracy':<12}{'mean conf':<12}{'separation':<13}"
              f"{'AUROC':<9}{'overconf gap':<13}")
        for r in rows:
            print(f"    {r['k']:<4}{str(r['accuracy_pct']) + '%':<12}"
                  f"{str(r['mean_confidence_all']):<12}"
                  f"{str(r['separation_correct_minus_wrong']):<13}"
                  f"{str(r['auroc_confidence_detects_correct']):<9}"
                  f"{str(r['overconfidence_gap_pp']) + ' pp':<13}")

    out["note"] = (
        "Read the accuracy and mean-confidence columns together. Accuracy falling while mean "
        "confidence stays flat means the model does not register the degradation that adding "
        "documents causes, which is the retrieval-aware-confidence failure this step is "
        "looking for. The no_retrieval condition uses no documents, so it is identical at "
        "every k by construction and acts as the flat control.")

    #saving the depth sweep for the Step 9 cross-corpus comparison
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved cross-k confidence analysis to {out_json}")
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
    parser.add_argument("--k", type=int, default=3, choices=list(MATRIX_K_VALUES),
                        help="Retrieval depth to analyse. Must be one of the Step 6 matrix "
                             "depths: 1, 3, 5, or 10.")
    parser.add_argument("--out_dir", default="results/step8_confidence")
    args = parser.parse_args()

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
            out_json=os.path.join(args.out_dir,
                                  f"confidence_{args.dataset}_k{args.k}.json"),
        )
    elif args.mode == "cross_k":
        cross_k_confidence(
            records_dir=args.records_dir,
            dataset=args.dataset,
            out_json=os.path.join(args.out_dir, f"confidence_by_k_{args.dataset}.json"),
        )


if __name__ == "__main__":
    main()