"""
Step 9: Cross-corpus generalisation, SciFact versus SciFact-Open.

This script does no new experiments and no modelling. It reads the result files the earlier
steps already saved and puts the two corpora side by side, so that single-corpus observations
become claims about how retrieval behaviour changes when the candidate pool grows roughly 100
times (about 5,000 abstracts on SciFact against about 500,000 on SciFact-Open).

The reason it exists as a script rather than as hand-written tables is transcription safety.
Assembling this comparison by hand means copying roughly sixty numbers out of four different
files, which is exactly the kind of task where a digit gets transposed and nobody notices. Every
figure below is read straight from the source JSON, so the write-up cannot drift from the data.

WHAT IT ANSWERS (the three Step 9 questions from the project plan):

  Q1. Does stance reranking improve F1 consistently as retrieval difficulty scales up?
      Read from the Step 6 matrix files. The PRIMARY comparison is reranked minus dense macro
      F1 at each matched retrieval depth, because "consistently" can only be judged with the
      depth held constant. A SEPARATE best-achievable comparison reports each condition's
      maximum macro F1 and the depth it occurs at. Each retrieval condition's margin over the
      no-retrieval baseline is also reported, which distinguishes genuine reranking improvement
      from cases where both retrieval pipelines simply deteriorate together.

  Q2. How do comparable automatic failure indicators change when the candidate corpus grows
      roughly 100 times, and how do those shifts relate to the manually observed SciFact
      taxonomy? This is deliberately narrower than "do the dominant failure categories shift",
      because manual four-category labels exist for SciFact only, so which manual category
      becomes dominant on SciFact-Open cannot be established. Reported in two separate layers
      that are never mixed. The SYMMETRIC layer compares the automatic diagnostic signals, which
      exist for both corpora: error rate, high-confidence error rate, gold-missing rate and the
      evidence-overload proxy. The ASYMMETRIC layer is the manual four-category annotation, which
      exists for SciFact only. Proxy signals are not taxonomy labels and are never reported as
      category percentages.

  Q3. Does the confidence-correctness correlation hold under harder retrieval?
      Read from the Step 8 confidence files, which cover both corpora, so this comparison is
      symmetric already. Reported as accuracy, separation, AUROC and the aggregate
      confidence-accuracy gap per condition, plus the per-label AUROC so that the CONTRADICT
      behaviour can be checked on both corpora rather than assumed to generalise.

ON WHAT IS NOT COMPARED. The Step 3 retrieval recall figures are deliberately left out of the
assembled tables. For SciFact, `evidence_doc_ids` holds the claim's cited doc ids, so recall
there is cited-document recall over all 300 claims; for SciFact-Open it holds annotated evidence
doc ids, so recall there is evidence recall over the 206 evidenced claims. Those are two
different measurements and putting them in one table would invite a false comparison. The
asymmetry is documented in step3_results.md and is restated in the Step 9 write-up instead.

ON SEEDS. Every number here inherits the seed-42 Step 6 matrix. The cross-corpus differences
reported are not verified across training seeds, and the write-up must say so.

Usage:
  python analysis/cross_corpus.py --out_dir results/step9_comparison
"""

#importing os for file path handling and directory creation
import os

#importing sys to add the project root to the import path
import sys

#adding the project root so the shared helpers import correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing json for reading the earlier steps' result files and writing the comparison
import json

#importing csv so the assembled tables drop straight into the thesis without re-parsing json
import csv

#importing argparse for the command line interface
import argparse

#reusing the shared constants so the condition names and depths are the SAME here as in Steps 7
#and 8, rather than a third hand-typed copy that could quietly drift out of step
from analysis.failure_taxonomy import CONDITIONS, MATRIX_K_VALUES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#the two corpora being compared, in the order they should appear in every table
DATASETS = ("scifact", "scifact_open")

#a readable label and the approximate corpus size for each, used in the printed report
DATASET_LABELS = {
    "scifact": "SciFact (about 5,000 abstracts)",
    "scifact_open": "SciFact-Open (about 500,000 abstracts)",
}

#the depth the pipeline reports throughout the project, kept for the signal and confidence
#comparisons so Step 9 lines up with Steps 7 and 8
REPORTED_K = 3

#the four labels the pipeline can predict, used for the per-label AUROC comparison
VALID_LABELS = ("SUPPORT", "CONTRADICT", "NEI")

#the seed every input file inherits, restated in the output so provenance travels with the data
RECORDS_SEED = 42

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def verify_seed(obj, source_name, verified, unverified):
    """
    Check a source file's recorded seed against the seed this comparison claims, where the file
    records one. Step 8 writes records_seed, so those are verified; the Step 6 matrix files
    predate the field, so they are listed as externally supplied rather than silently trusted.
    """
    #the field Step 8 writes; older files simply do not have it
    seed = obj.get("records_seed")
    if seed is None:
        unverified.append(source_name)
    elif seed != RECORDS_SEED:
        raise ValueError(
            f"Seed mismatch in {source_name}: file records seed {seed} but this comparison is "
            f"assembled as seed {RECORDS_SEED}. Mixing training seeds would make the "
            f"cross-corpus differences uninterpretable.")
    else:
        verified.append(source_name)

def read_json(path, what):
    """Read a JSON file, failing with a message that names the missing step if it is absent."""
    #failing early and clearly rather than part way through assembling a table
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {what}: {path}\n"
            f"Step 9 assembles existing results and cannot regenerate them. Run the step that "
            f"produces this file first, then re-run the comparison.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_macro_f1(matrix_obj, condition):
    """
    Pull one condition's macro F1 out of a Step 6 matrix file.

    The matrix files were written by an earlier step, so rather than assume one key layout this
    walks the object looking for the condition and a macro-F1-like field. If it cannot find one
    it raises with the keys it DID see, which turns a silent wrong number into a clear message.
    """
    #the field names a macro F1 might plausibly have been saved under
    #ONLY macro-F1-specific names. bare "f1" is deliberately excluded, because it could mean
    #binary, micro or weighted F1, and silently reading the wrong metric is exactly the failure
    #this script exists to prevent. once the real Step 6 schema is confirmed this should be
    #pinned to that single exact path rather than probed
    f1_keys = ("macro_f1", "f1_macro", "macro_F1", "macroF1")

    #trying the layout {condition: {"macro_f1": value}} first, at the top level or one level in
    candidates = [matrix_obj]
    for key in ("results", "conditions", "per_condition", "metrics"):
        if isinstance(matrix_obj.get(key), dict):
            candidates.append(matrix_obj[key])

    for holder in candidates:
        entry = holder.get(condition)
        #the condition maps straight to a number
        if isinstance(entry, (int, float)):
            return float(entry)
        #the condition maps to a dict containing a macro F1 field
        if isinstance(entry, dict):
            for fk in f1_keys:
                if isinstance(entry.get(fk), (int, float)):
                    return float(entry[fk])

    #trying the transposed layout {"macro_f1": {condition: value}}
    for fk in f1_keys:
        holder = matrix_obj.get(fk)
        if isinstance(holder, dict) and isinstance(holder.get(condition), (int, float)):
            return float(holder[condition])

    raise KeyError(
        f"Could not find a macro F1 for condition '{condition}' in the matrix file. "
        f"Top-level keys present: {sorted(matrix_obj.keys())}. "
        f"Add the correct key name to find_macro_f1() rather than transcribing the value by "
        f"hand, so the script stays the single source of truth.")

# ---------------------------------------------------------------------------
# Q1: F1 across retrieval difficulty
# ---------------------------------------------------------------------------

def collect_f1(matrix_dir, verified, unverified):
    """
    Build, for each corpus and condition, the macro F1 at every depth, the best F1 and the depth
    it occurs at. Each matrix file is read exactly once (all conditions are extracted from it in
    one pass), so a file is opened and provenance-checked a single time rather than once per
    condition.
    """
    out = {}
    for dataset in DATASETS:
        #reading every depth's matrix once, up front
        matrices_by_k = {}
        for k in MATRIX_K_VALUES:
            path = os.path.join(matrix_dir, f"matrix_{dataset}_k{k}_thr0_5.json")
            matrix = read_json(path, f"Step 6 matrix for {dataset} at k={k}")
            verify_seed(matrix, f"Step 6 matrix {dataset} k={k}", verified, unverified)
            matrices_by_k[k] = matrix

        per_condition = {}
        for cond in CONDITIONS:
            #extracting this condition's F1 from each already-loaded matrix, unrounded, so the
            #maximisation cannot be decided by a rounding tie
            by_k_raw = {k: find_macro_f1(matrix, cond)
                        for k, matrix in matrices_by_k.items()}

            if cond == "no_retrieval":
                #the baseline uses no documents, so it must be identical at every depth
                if len({round(v, 10) for v in by_k_raw.values()}) != 1:
                    raise ValueError(
                        f"No-retrieval macro F1 varies across the k-specific matrix files for "
                        f"{dataset}: {by_k_raw}. The baseline uses no retrieved documents and "
                        f"must be invariant to depth, so this indicates a matrix file problem.")
                #a best depth is meaningless for a condition that does not vary with depth
                best_k = None
                best_f1 = next(iter(by_k_raw.values()))
            else:
                best_k = max(by_k_raw, key=by_k_raw.get)
                best_f1 = by_k_raw[best_k]

            per_condition[cond] = {
                "f1_by_k": {k: round(v, 4) for k, v in by_k_raw.items()},
                "f1_by_k_unrounded": by_k_raw,
                "best_f1": round(best_f1, 4),
                "best_f1_unrounded": best_f1,
                "best_k": best_k,
            }
        out[dataset] = per_condition
    return out

def summarise_f1(f1):
    """
    Turn the raw F1 table into the Q1 answers.

    The project question asks whether reranking improves F1 CONSISTENTLY as retrieval difficulty
    scales, and "consistently" means at matched depths, not just when each method is allowed to
    pick its own best depth. So two different comparisons are reported and kept apart:

      1. The matched-depth effect: reranked minus dense at each k. This is what answers
         "consistently", because it holds the retrieval depth constant.
      2. The best-achievable effect: each condition's maximum F1 and the depth it occurs at.
         This answers "how good can each method get", which is a different question.

    Both are signed as RERANKED MINUS DENSE throughout, so a positive number always means
    reranking helped and a negative number always means it hurt. The earlier draft used the
    opposite sign and it was too easy to misread.

    The margin over no retrieval is reported alongside, because a shrinking distance between
    dense and reranked could mean either that reranking improved or that both sank together, and
    only the baseline margin distinguishes those two very different stories.
    """
    summary = {}
    for dataset in DATASETS:
        d = f1[dataset]
        no_ret = d["no_retrieval"]["best_f1_unrounded"]

        #the matched-depth comparison, which is the one that speaks to "consistently"
        by_k = {k: round(d["dense_reranked_roberta"]["f1_by_k_unrounded"][k]
                         - d["dense_roberta"]["f1_by_k_unrounded"][k], 4)
                for k in MATRIX_K_VALUES}
        wins = sum(1 for v in by_k.values() if v > 0)

        #the margin over no retrieval says whether retrieval earns its place at all
        margins = {cond: round(d[cond]["best_f1_unrounded"] - no_ret, 4)
                   for cond in CONDITIONS if cond != "no_retrieval"}

        summary[dataset] = {
            "best_f1_by_condition": {c: d[c]["best_f1"] for c in CONDITIONS},
            "best_k_by_condition": {c: d[c]["best_k"] for c in CONDITIONS},
            "reranked_minus_dense_by_k": by_k,
            "reranking_beats_dense_at_all_k": wins == len(MATRIX_K_VALUES),
            "n_k_where_reranking_beats_dense": wins,
            "n_k_compared": len(MATRIX_K_VALUES),
            "reranked_best_minus_dense_best": round(
                d["dense_reranked_roberta"]["best_f1_unrounded"]
                - d["dense_roberta"]["best_f1_unrounded"], 4),
            "margin_over_no_retrieval_at_best_k": margins,
            "any_retrieval_beats_no_retrieval": any(v > 0 for v in margins.values()),
        }

    #how the reranking effect itself changes with corpus difficulty. reported as an absolute
    #change in F1 points rather than a percentage: a percentage reduction is only meaningful
    #when both quantities are comparable positive distances, and it would badly misrepresent a
    #sign change, which would be the more important finding
    eff_small = summary["scifact"]["reranked_best_minus_dense_best"]
    eff_large = summary["scifact_open"]["reranked_best_minus_dense_best"]
    summary["reranking_effect_change"] = {
        "scifact": eff_small,
        "scifact_open": eff_large,
        "absolute_change_f1_points": round(eff_large - eff_small, 4),
        "direction_changed": (eff_small != 0 and eff_large != 0
                              and (eff_small > 0) != (eff_large > 0)),
        "note": ("Signed as reranked minus dense, so positive means reranking helped. Reported "
                 "in F1 points rather than as a percentage change, because a percentage is "
                 "undefined or misleading when either quantity is negative or near zero."),
    }

    #the optimal-depth shift, reported separately because it is a finding in its own right
    summary["optimal_depth_shift"] = {
        cond: {"scifact": f1["scifact"][cond]["best_k"],
               "scifact_open": f1["scifact_open"][cond]["best_k"]}
        for cond in CONDITIONS
    }
    return summary

# ---------------------------------------------------------------------------
# Q2: failure signals across corpora
# ---------------------------------------------------------------------------

def collect_signals(failure_dir, verified, unverified):
    """
    Read the automatic diagnostic signals for both corpora. These are the SYMMETRIC layer of Q2:
    the same proxy measures computed the same way on both datasets, so they can be compared
    directly. They are proxies, not taxonomy labels, and the output says so.
    """
    out = {}
    for dataset in DATASETS:
        path = os.path.join(failure_dir, f"rates_{dataset}.json")
        rates = read_json(path, f"Step 7 automatic signals for {dataset} "
                                f"(run: --mode rates --dataset {dataset})")
        verify_seed(rates, f"Step 7 rates {dataset}", verified, unverified)
        signals_block = rates.get("per_condition_signals")
        if not isinstance(signals_block, dict):
            raise KeyError(f"{path} is missing the per_condition_signals object.")
        missing = set(CONDITIONS) - set(signals_block)
        unexpected = set(signals_block) - set(CONDITIONS)
        if missing:
            raise KeyError(f"{path} is missing expected conditions: {sorted(missing)}")
        if unexpected:
            raise KeyError(f"{path} contains unexpected conditions: {sorted(unexpected)}")
        per_condition = {}
        for cond, sig in signals_block.items():
            #pulling the overload proxy for this condition, which lives in a separate block
            overload = rates.get("evidence_overload_across_k", {}).get(cond, {})
            per_condition[cond] = {
                "n_claims": sig.get("n_claims"),
                "n_errors": sig.get("n_errors"),
                "error_rate_pct": sig.get("error_rate_pct"),
                "high_confidence_error_pct_of_errors":
                    sig.get("high_confidence_error_pct_of_errors"),
                "gold_evidence_missing_pct_of_defined_errors":
                    sig.get("gold_evidence_missing_pct_of_defined_errors"),
                "high_conf_error_with_gold_doc_pct_of_defined_errors":
                    sig.get("high_conf_error_with_gold_doc_pct_of_defined_errors"),
                "overload_pct_of_eligible": overload.get("overload_pct_of_eligible"),
            }
        out[dataset] = per_condition
    return out

def collect_manual_categories(failure_dir, verified, unverified):
    """
    Read the Step 7 manual four-category breakdown. This is the ASYMMETRIC layer of Q2: it
    exists for SciFact only, because manual annotation of the second corpus was outside the
    scope of a solo project. It is returned separately from the signals so the two layers can
    never be accidentally combined into one table.
    """
    path = os.path.join(failure_dir, "analysis_scifact.json")
    analysis = read_json(path, "Step 7 manual annotation analysis for SciFact")
    verify_seed(analysis, "Step 7 manual analysis scifact", verified, unverified)
    breakdown = analysis.get("per_condition_manual_breakdown")
    if not isinstance(breakdown, dict):
        raise KeyError(f"{path} is missing the per_condition_manual_breakdown object.")
    return {
        "dataset": "scifact",
        "available_for": ["scifact"],
        "not_available_for": ["scifact_open"],
        "reason": ("Full manual annotation was scoped to the primary dataset only. SciFact-Open "
                   "is covered by the automatic signals above, which are proxies rather than "
                   "taxonomy labels."),
        "what_cannot_be_claimed": (
            "Because manual taxonomy annotation was carried out on SciFact only, it CANNOT be "
            "established which manual failure category becomes dominant on SciFact-Open. The "
            "answerable question is narrower than the original Q2 wording: how do comparable "
            "AUTOMATIC failure indicators change as the candidate corpus grows roughly 100 "
            "times, and how do those shifts relate to the manually observed SciFact taxonomy. "
            "A rising gold-missing rate in particular shows that recognised annotated evidence "
            "was more often absent; it does NOT establish that the retrieved documents were "
            "topically irrelevant, since they may be relevant but non-gold, partially "
            "evidential, contradictory, or insufficient. The irrelevant_retrieval label needs "
            "document-level human judgement, which only SciFact received."),
        "per_condition": {cond: {"n_annotated": v.get("n_annotated"),
                                 "counts": v.get("counts"),
                                 "pct": v.get("pct")}
                          for cond, v in breakdown.items()},
        "excluded_below_threshold_count":
            analysis.get("validation", {}).get("excluded_below_threshold_count"),
    }

# ---------------------------------------------------------------------------
# Q3: confidence behaviour across corpora
# ---------------------------------------------------------------------------

def collect_confidence(confidence_dir, verified, unverified):
    """
    Read the Step 8 confidence analysis for both corpora at the reported depth, plus the depth
    sweeps. This comparison is symmetric already, since Step 8 ran on both datasets.
    """
    out = {"at_reported_k": {}, "per_label_auroc": {}, "across_k": {}}
    for dataset in DATASETS:
        path = os.path.join(confidence_dir, f"confidence_{dataset}_k{REPORTED_K}.json")
        conf = read_json(path, f"Step 8 confidence analysis for {dataset}")
        verify_seed(conf, f"Step 8 confidence {dataset} k={REPORTED_K}", verified, unverified)
        #validating the confidence blocks the same way the signals are validated, so a missing
        #condition cannot be silently dropped from the comparison
        for block_name in ("per_condition", "per_true_label"):
            block = conf.get(block_name)
            if not isinstance(block, dict):
                raise KeyError(f"{path} is missing the {block_name} object.")
            miss = set(CONDITIONS) - set(block)
            extra = set(block) - set(CONDITIONS)
            if miss:
                raise KeyError(f"{path} {block_name} missing conditions: {sorted(miss)}")
            if extra:
                raise KeyError(f"{path} {block_name} has unexpected conditions: {sorted(extra)}")

        #the headline per-condition confidence measures
        out["at_reported_k"][dataset] = {
            cond: {
                "accuracy_pct": s.get("accuracy_pct"),
                "separation_correct_minus_wrong": s.get("separation_correct_minus_wrong"),
                "auroc_confidence_detects_correct": s.get("auroc_confidence_detects_correct"),
                "global_confidence_accuracy_gap_pp": s.get("global_confidence_accuracy_gap_pp"),
                "high_confidence_error_pct_of_errors":
                    s.get("high_confidence_error_pct_of_errors"),
            }
            for cond, s in conf.get("per_condition", {}).items()
        }

        #the per-label AUROC, so the CONTRADICT behaviour can be CHECKED on both corpora rather
        #than assumed to generalise from one
        out["per_label_auroc"][dataset] = {
            cond: {label: stats.get("auroc_confidence_detects_correct")
                   for label, stats in labels.items()}
            for cond, labels in conf.get("per_true_label", {}).items()
        }

        #the depth sweep, used to compare how confidence responds to added documents
        sweep_path = os.path.join(confidence_dir, f"confidence_by_k_{dataset}.json")
        sweep = read_json(sweep_path, f"Step 8 depth sweep for {dataset}")
        verify_seed(sweep, f"Step 8 depth sweep {dataset}", verified, unverified)
        sweep_block = sweep.get("per_condition")
        if not isinstance(sweep_block, dict):
            raise KeyError(f"{sweep_path} is missing the per_condition object.")
        #confirming each retrieval condition covers exactly the expected depths, so a cross-k
        #trend cannot be drawn from a condition that is missing a depth
        #confirming the sweep contains exactly the expected conditions, so a missing one cannot
        #slip through the depth check below just because it was never iterated
        miss = set(CONDITIONS) - set(sweep_block)
        extra = set(sweep_block) - set(CONDITIONS)
        if miss:
            raise KeyError(f"{sweep_path} is missing expected conditions: {sorted(miss)}")
        if extra:
            raise KeyError(f"{sweep_path} contains unexpected conditions: {sorted(extra)}")
        expected_k = set(MATRIX_K_VALUES)
        for cond, rows in sweep_block.items():
            if cond == "no_retrieval":
                continue
            observed_k = {r.get("k") for r in rows}
            if observed_k != expected_k:
                raise ValueError(
                    f"{sweep_path}, condition {cond}, has depths {sorted(observed_k)}; "
                    f"expected {sorted(expected_k)}.")
        out["across_k"][dataset] = sweep_block
    return out

def summarise_confidence(conf):
    """
    Turn the confidence tables into the Q3 answers: does discrimination survive the harder
    corpus, is the CONTRADICT inversion a property of one dataset or of the model, and does
    confidence respond to added documents the same way on both corpora.
    """
    summary = {}

    #how each condition's discrimination changes from the small corpus to the large one
    auroc_change = {}
    for cond in CONDITIONS:
        small = conf["at_reported_k"]["scifact"].get(cond, {}).get(
            "auroc_confidence_detects_correct")
        large = conf["at_reported_k"]["scifact_open"].get(cond, {}).get(
            "auroc_confidence_detects_correct")
        if small is not None and large is not None:
            auroc_change[cond] = {"scifact": small, "scifact_open": large,
                                  "change": round(large - small, 4)}
    summary["auroc_change_by_condition"] = auroc_change

    #checking whether below-chance discrimination on a label holds on BOTH corpora, which is
    #what turns a single-dataset oddity into a claim about the model and task
    inverted = {}
    for label in VALID_LABELS:
        per_dataset = {}
        for dataset in DATASETS:
            vals = [labels.get(label)
                    for labels in conf["per_label_auroc"][dataset].values()
                    if labels.get(label) is not None]
            if vals:
                per_dataset[dataset] = {
                    "min_auroc": round(min(vals), 4),
                    "max_auroc": round(max(vals), 4),
                    "all_below_chance": all(v < 0.5 for v in vals),
                    "n_conditions": len(vals),
                    "all_expected_conditions_present": len(vals) == len(CONDITIONS),
                }
        inverted[label] = per_dataset
        #a label is only called inverted if BOTH corpora are present, every expected condition
        #is present in each, and every one is below chance. requiring this stops missing data
        #from making the claim look stronger than the evidence supports
        has_both = all(ds in per_dataset for ds in DATASETS)
        inverted[label]["inverted_on_both_corpora"] = bool(
            has_both and all(per_dataset[ds]["all_below_chance"]
                             and per_dataset[ds]["all_expected_conditions_present"]
                             for ds in DATASETS))
    summary["per_label_discrimination"] = inverted

    #how confidence responds to added documents on each corpus, read as the change in the
    #confidence-accuracy gap between the shallowest and deepest retrieval
    overload_response = {}
    for dataset in DATASETS:
        per_cond = {}
        for cond, rows in conf["across_k"][dataset].items():
            #the no-retrieval control does not vary with k, so it carries no information here
            if cond == "no_retrieval" or len(rows) < 2:
                continue
            #sorting by depth explicitly rather than trusting the saved row order
            ordered = sorted(rows, key=lambda r: r["k"])
            #confirming the confidence scale before treating it as a proportion
            for r in ordered:
                if not 0.0 <= r["mean_confidence_all"] <= 1.0:
                    raise ValueError(
                        f"Expected mean_confidence_all on the 0 to 1 scale, received "
                        f"{r['mean_confidence_all']} for {cond} at k={r['k']}.")

            first, last = ordered[0], ordered[-1]
            acc_change = round(last["accuracy_pct"] - first["accuracy_pct"], 1)
            conf_change = round(100 * (last["mean_confidence_all"]
                                       - first["mean_confidence_all"]), 2)
            gap_change = round(last["global_confidence_accuracy_gap_pp"]
                               - first["global_confidence_accuracy_gap_pp"], 1)

            #only judging whether confidence registered degradation when accuracy actually
            #degraded. if accuracy rose there is nothing to register, and calling that a success
            #would be meaningless
            if acc_change < 0:
                response = ("aggregate_confidence_tracked_the_decline" if gap_change <= 0
                            else "aggregate_confidence_did_not_track_the_decline")
            elif acc_change > 0:
                response = "accuracy_improved_so_no_degradation_to_track"
            else:
                response = "accuracy_unchanged"

            per_cond[cond] = {
                "k_low": first["k"], "k_high": last["k"],
                "accuracy_change_pp": acc_change,
                "mean_confidence_change_pp": conf_change,
                "confidence_accuracy_gap_change_pp": gap_change,
                #a widening gap means accuracy fell faster than confidence did. note this is an
                #AGGREGATE statement: it says mean confidence moved roughly in line with mean
                #accuracy, not that the model was right about any individual prediction
                "confidence_response": response,
            }
        overload_response[dataset] = per_cond
    summary["overload_response"] = overload_response
    return summary

# ---------------------------------------------------------------------------
# Do any tested retrieval configurations outperform the no-retrieval baseline?
# ---------------------------------------------------------------------------

def retrieval_worth_it(f1_summary, conf):
    """
    Bring the F1 and confidence evidence together on one question: at each corpus, does the best
    retrieval condition actually beat retrieving nothing? This is the sharpest cross-corpus
    claim available, because it is not about which retriever is better but about whether the
    retrieval-augmented design is justified at all once the corpus is large.
    """
    out = {}
    for dataset in DATASETS:
        margins = f1_summary[dataset]["margin_over_no_retrieval_at_best_k"]
        best_cond = max(margins, key=lambda c: margins[c])

        #the same question asked of the Step 8 accuracy figures, as an independent check that
        #the answer is not an artefact of macro F1 weighting the small CONTRADICT class
        acc = conf["at_reported_k"][dataset]
        acc_no_ret = acc.get("no_retrieval", {}).get("accuracy_pct")
        acc_best_retrieval = max(
            (v.get("accuracy_pct") for c, v in acc.items()
             if c != "no_retrieval" and v.get("accuracy_pct") is not None), default=None)

        best_margin = margins[best_cond]
        out[dataset] = {
            #naming the depth rule explicitly, because the F1 figure is each condition's best
            #across all depths while the accuracy figure is at the single reported depth. those
            #are different selection rules and must not read as one matched comparison
            "best_retrieval_condition_across_k": best_cond,
            "best_margin_over_no_retrieval_f1_across_k": best_margin,
            #saying plainly when the "best" retrieval condition is still worse than retrieving
            #nothing, so the label cannot read as an endorsement
            "best_retrieval_condition_status": (
                "outperforms_baseline" if best_margin > 0
                else "ties_baseline" if best_margin == 0
                else "least_harmful_but_still_below_baseline"),
            "retrieval_beats_no_retrieval_on_f1_across_k": best_margin > 0,
            "accuracy_no_retrieval_at_reported_k_pct": acc_no_ret,
            "best_retrieval_accuracy_at_reported_k_pct": acc_best_retrieval,
            "retrieval_beats_no_retrieval_on_accuracy_at_reported_k": (
                acc_best_retrieval > acc_no_ret
                if (acc_best_retrieval is not None and acc_no_ret is not None) else None),
        }
    return out

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(path, fieldnames, rows):
    """Write a list of dicts to CSV with a fixed column order."""
    #writing with utf-8 and no extra blank lines on any platform
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def export_tables(out, out_dir):
    """Write the assembled comparisons as CSV so they drop straight into the thesis."""
    #Q1: best F1 and best depth per condition, both corpora on one row each
    rows = []
    for cond in CONDITIONS:
        row = {"condition": cond}
        for dataset in DATASETS:
            s = out["q1_f1"]["summary"][dataset]
            row[f"best_f1_{dataset}"] = s["best_f1_by_condition"].get(cond)
            row[f"best_k_{dataset}"] = s["best_k_by_condition"].get(cond)
            row[f"margin_over_no_retrieval_{dataset}"] = (
                s["margin_over_no_retrieval_at_best_k"].get(cond))
        rows.append(row)
    write_csv(os.path.join(out_dir, "q1_f1_by_corpus.csv"),
              ["condition", "best_f1_scifact", "best_k_scifact",
               "margin_over_no_retrieval_scifact", "best_f1_scifact_open",
               "best_k_scifact_open", "margin_over_no_retrieval_scifact_open"], rows)

    #Q2: the symmetric automatic signals, one row per corpus and condition
    rows = []
    for dataset in DATASETS:
        for cond, sig in out["q2_signals"][dataset].items():
            row = dict(sig)
            row["dataset"] = dataset
            row["condition"] = cond
            rows.append(row)
    write_csv(os.path.join(out_dir, "q2_automatic_signals.csv"),
              ["dataset", "condition", "n_claims", "n_errors", "error_rate_pct",
               "high_confidence_error_pct_of_errors",
               "gold_evidence_missing_pct_of_defined_errors",
               "high_conf_error_with_gold_doc_pct_of_defined_errors",
               "overload_pct_of_eligible"], rows)

    #Q3: the confidence measures, one row per corpus and condition
    rows = []
    for dataset in DATASETS:
        for cond, s in out["q3_confidence"]["at_reported_k"][dataset].items():
            row = dict(s)
            row["dataset"] = dataset
            row["condition"] = cond
            rows.append(row)
    write_csv(os.path.join(out_dir, "q3_confidence_by_corpus.csv"),
              ["dataset", "condition", "accuracy_pct", "separation_correct_minus_wrong",
               "auroc_confidence_detects_correct", "global_confidence_accuracy_gap_pp",
               "high_confidence_error_pct_of_errors"], rows)

    #Q3 supplement: per-label AUROC, which is where the CONTRADICT behaviour shows up
    rows = []
    for dataset in DATASETS:
        for cond, labels in out["q3_confidence"]["per_label_auroc"][dataset].items():
            for label, auroc in labels.items():
                rows.append({"dataset": dataset, "condition": cond,
                             "true_label": label, "auroc": auroc})
    write_csv(os.path.join(out_dir, "q3_per_label_auroc.csv"),
              ["dataset", "condition", "true_label", "auroc"], rows)

    #Q1 primary table: matched-depth reranking effect, which is what answers "consistently".
    #the best-depth table above is its supplement
    matched = []
    for dataset in DATASETS:
        summ = out["q1_f1"]["summary"][dataset]
        byk = out["q1_f1"]["by_k"][dataset]
        for k, effect in summ["reranked_minus_dense_by_k"].items():
            matched.append({
                "dataset": dataset, "k": k,
                "dense_macro_f1": byk["dense_roberta"]["f1_by_k"][k],
                "reranked_macro_f1": byk["dense_reranked_roberta"]["f1_by_k"][k],
                "reranked_minus_dense": effect, "reranking_helped": effect > 0})
    write_csv(os.path.join(out_dir, "q1_matched_depth_reranking.csv"),
              ["dataset", "k", "dense_macro_f1", "reranked_macro_f1",
               "reranked_minus_dense", "reranking_helped"], matched)

    print(f"Wrote CSV tables to {out_dir} (q1_matched_depth_reranking, q1_f1_by_corpus, "
          f"q2_automatic_signals, q3_confidence_by_corpus, q3_per_label_auroc)")

# ---------------------------------------------------------------------------
# Assembling and reporting
# ---------------------------------------------------------------------------

def compare(matrix_dir, failure_dir, confidence_dir, out_dir):
    """Assemble the whole cross-corpus comparison, print it, and save it."""
    #tracking which sources had their seed checked and which predate the provenance field
    verified, unverified = [], []

    #Q1: F1 across the two corpora
    f1 = collect_f1(matrix_dir, verified, unverified)
    f1_summary = summarise_f1(f1)

    #Q2: the two layers, kept separate by construction
    signals = collect_signals(failure_dir, verified, unverified)
    manual = collect_manual_categories(failure_dir, verified, unverified)

    #Q3: confidence on both corpora
    conf = collect_confidence(confidence_dir, verified, unverified)
    conf_summary = summarise_confidence(conf)

    out = {
        "comparison": "SciFact versus SciFact-Open",
        "corpora": DATASET_LABELS,
        "reported_k_for_signals_and_confidence": REPORTED_K,
        "records_seed": RECORDS_SEED,
        "seed_provenance": {"seed_verified_sources": verified,
                            "seed_unverified_sources": unverified},
        "q1_f1": {"by_k": f1, "summary": f1_summary},
        "q2_signals": signals,
        "q2_manual_categories_scifact_only": manual,
        "q3_confidence": conf,
        "q3_summary": conf_summary,
        "retrieval_worth_it": retrieval_worth_it(f1_summary, conf),
    }

    # ---- printed report ----
    print("=" * 78)
    print("Q1: macro F1 across retrieval difficulty (best depth per condition)")
    print("=" * 78)
    for dataset in DATASETS:
        st = f1_summary[dataset]
        print(f"\n{DATASET_LABELS[dataset]}:")
        for cond in CONDITIONS:
            #the baseline has no meaningful best depth, so it prints without one
            bk = st["best_k_by_condition"][cond]
            depth = "n/a (invariant)" if bk is None else f"k={bk}"
            if cond == "no_retrieval":
                print(f"  {cond:<26} best F1 {st['best_f1_by_condition'][cond]:.4f} "
                      f"at {depth:<16}(baseline)")
            else:
                print(f"  {cond:<26} best F1 {st['best_f1_by_condition'][cond]:.4f} "
                      f"at {depth:<16}margin over no retrieval "
                      f"{st['margin_over_no_retrieval_at_best_k'][cond]:+.4f}")
        print(f"  reranked minus dense at matched depths: "
              f"{st['reranked_minus_dense_by_k']}")
        print(f"  reranking beats dense at {st['n_k_where_reranking_beats_dense']} of "
              f"{st['n_k_compared']} depths "
              f"(consistently: {st['reranking_beats_dense_at_all_k']})")
        print(f"  reranked best minus dense best: "
              f"{st['reranked_best_minus_dense_best']:+.4f}")
        print(f"  any retrieval condition beats no retrieval: "
              f"{st['any_retrieval_beats_no_retrieval']}")

    g = f1_summary["reranking_effect_change"]
    print(f"\nReranking effect (reranked minus dense, at each corpus's best depths): "
          f"{g['scifact']:+.4f} on SciFact, {g['scifact_open']:+.4f} on SciFact-Open "
          f"(change {g['absolute_change_f1_points']:+.4f} F1 points, "
          f"direction changed: {g['direction_changed']})")
    print("Optimal depth by condition:")
    for cond, d in f1_summary["optimal_depth_shift"].items():
        sf = "n/a" if d["scifact"] is None else f"k={d['scifact']}"
        so = "n/a" if d["scifact_open"] is None else f"k={d['scifact_open']}"
        print(f"  {cond:<26} SciFact {sf:<8} SciFact-Open {so}")

    print("\n" + "=" * 78)
    print("Q2 (symmetric layer): automatic diagnostic signals, proxies not taxonomy labels")
    print("=" * 78)
    for dataset in DATASETS:
        print(f"\n{DATASET_LABELS[dataset]}:")
        print(f"  {'condition':<26}{'errors':<9}{'high-conf err':<15}{'gold missing':<14}"
              f"{'overload':<10}")
        for cond, sig in signals[dataset].items():
            print(f"  {cond:<26}{str(sig['error_rate_pct']) + '%':<9}"
                  f"{str(sig['high_confidence_error_pct_of_errors']) + '%':<15}"
                  f"{str(sig['gold_evidence_missing_pct_of_defined_errors']):<14}"
                  f"{str(sig['overload_pct_of_eligible']):<10}")

    print("\n" + "-" * 78)
    print("Q2 (asymmetric layer): manual four-category annotation, SciFact only")
    print("-" * 78)
    for cond, v in manual["per_condition"].items():
        print(f"  {cond} (n={v['n_annotated']}): {v['pct']}")
    print(f"  excluded_below_threshold: {manual['excluded_below_threshold_count']}")
    print(f"  {manual['reason']}")

    print("\n" + "=" * 78)
    print("Q3: confidence behaviour across corpora")
    print("=" * 78)
    for dataset in DATASETS:
        print(f"\n{DATASET_LABELS[dataset]}:")
        print(f"  {'condition':<26}{'accuracy':<11}{'separation':<13}{'AUROC':<9}"
              f"{'conf-acc gap':<13}")
        for cond, s in conf["at_reported_k"][dataset].items():
            print(f"  {cond:<26}{str(s['accuracy_pct']) + '%':<11}"
                  f"{str(s['separation_correct_minus_wrong']):<13}"
                  f"{str(s['auroc_confidence_detects_correct']):<9}"
                  f"{str(s['global_confidence_accuracy_gap_pp']) + ' pp':<13}")

    print("\nAUROC change from SciFact to SciFact-Open:")
    for cond, c in conf_summary["auroc_change_by_condition"].items():
        print(f"  {cond:<26}{c['scifact']:.4f} -> {c['scifact_open']:.4f} "
              f"({c['change']:+.4f})")

    print("\nPer-label discrimination (is any label below chance on BOTH corpora?):")
    for label, d in conf_summary["per_label_discrimination"].items():
        flag = "  <- INVERTED on both corpora" if d.get("inverted_on_both_corpora") else ""
        ranges = ", ".join(f"{ds}: {d[ds]['min_auroc']:.4f} to {d[ds]['max_auroc']:.4f}"
                           for ds in DATASETS if ds in d)
        print(f"  {label:<12}{ranges}{flag}")

    print("\nHow confidence responds to added documents (change from lowest to highest k):")
    for dataset in DATASETS:
        print(f"  {dataset}:")
        for cond, r in conf_summary["overload_response"][dataset].items():
            verdict = r["confidence_response"].replace("_", " ")
            print(f"    {cond:<26}accuracy {r['accuracy_change_pp']:+.1f} pp, "
                  f"confidence {r['mean_confidence_change_pp']:+.2f} pp, "
                  f"gap {r['confidence_accuracy_gap_change_pp']:+.1f} pp ({verdict})")

    print("\n" + "=" * 78)
    print("Does any tested retrieval configuration outperform the no-retrieval baseline?")
    print("=" * 78)
    for dataset, r in out["retrieval_worth_it"].items():
        print(f"  {DATASET_LABELS[dataset]}:")
        print(f"    best retrieval condition across k: {r['best_retrieval_condition_across_k']} "
              f"(margin {r['best_margin_over_no_retrieval_f1_across_k']:+.4f} F1)")
        print(f"    status: {r['best_retrieval_condition_status']}")
        print(f"    beats no retrieval on F1 (best across k):        "
              f"{r['retrieval_beats_no_retrieval_on_f1_across_k']}")
        print(f"    beats no retrieval on accuracy (at k={REPORTED_K}):        "
              f"{r['retrieval_beats_no_retrieval_on_accuracy_at_reported_k']} "
              f"({r['best_retrieval_accuracy_at_reported_k_pct']}% vs "
              f"{r['accuracy_no_retrieval_at_reported_k_pct']}%)")

    #recording the interpretation limits with the data, as in Steps 7 and 8
    out["note"] = (
        "All analyses use records intended to derive from seed 42. Where a source file carries "
        "records_seed metadata that value was checked (seed_verified_sources); older files "
        "without the field are listed under seed_unverified_sources, so their provenance is "
        "externally supplied rather than verified from the JSON. Cross-corpus differences were "
        "not tested across multiple training seeds. "
        "Step 3 retrieval recall is deliberately excluded from these tables: on SciFact it is "
        "cited-document recall over all 300 claims, on SciFact-Open it is evidence recall over "
        "the 206 evidenced claims, so the two are not the same measurement. The Q2 layers are "
        "kept separate because the automatic signals are proxies, not taxonomy labels, and only "
        "SciFact carries manual category labels. The comparison therefore "
        "describes the seed-42 run and does not establish stability across training seeds.")

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "cross_corpus_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved cross-corpus comparison to {out_json}")

    export_tables(out, out_dir)
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    #setting up the command line arguments, mirroring the Step 7 and 8 scripts' conventions
    parser = argparse.ArgumentParser(
        description="Step 9 cross-corpus generalisation (SciFact versus SciFact-Open)")
    parser.add_argument("--matrix_dir", default="results/step6_matrix",
                        help="Directory holding the Step 6 matrix_*.json files.")
    parser.add_argument("--failure_dir", default="results/step7_failure",
                        help="Directory holding rates_*.json and analysis_scifact.json.")
    parser.add_argument("--confidence_dir", default="results/step8_confidence",
                        help="Directory holding the Step 8 confidence_*.json files.")
    parser.add_argument("--out_dir", default="results/step9_comparison")
    args = parser.parse_args()

    #ensuring the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    compare(matrix_dir=args.matrix_dir,
            failure_dir=args.failure_dir,
            confidence_dir=args.confidence_dir,
            out_dir=args.out_dir)


if __name__ == "__main__":
    main()