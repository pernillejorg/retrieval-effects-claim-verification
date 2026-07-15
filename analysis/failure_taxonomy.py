"""
Step 7: Failure taxonomy analysis for RAG claim verification.

This script performs the failure analysis component of the thesis. It works entirely from
the per-claim records already saved by the Step 5 / Step 6 pipeline runs (records_*.json),
so it does NOT re-run retrieval, reranking, or the classifier. Gold evidence is joined in
fresh from the dataset loader (a fast lookup, no GPU) so that automatic signals can be
grounded in whether the recognised gold document was retrieved.

IMPORTANT ON INTERPRETATION. The automatic `sig_*` signals below are diagnostic aids, not
ground-truth taxonomy labels, and they are named and described to avoid over-claiming:
  - Gold evidence being absent is an evidence-recall failure signal, NOT proof that the
    retrieved context is topically irrelevant.
  - A gold document being retrieved shows document-level retrieval succeeded, NOT that the
    classifier received adequate label-consistent evidence (it may be truncated, ranked low,
    or overwhelmed).
The four-category taxonomy is assigned by MANUAL annotation on SciFact, using the decision
precedence in results/step7_failure/annotation_guide.md. Automatic signals only support that
judgement. For SciFact-Open, only automatic diagnostic signals are reported (no manual
annotation), and they are explicitly proxy signals, not taxonomy rates.

ON WHAT THE CLASSIFIER ACTUALLY SAW. The saved records contain the full RETRIEVED
documents, which are not necessarily identical to the exact text that survived equal
per-document budgeting and the final 512-token input limit. A gold document may appear
in the retrieved set while its label-consistent sentence was removed by per-document
budgeting.

When records were generated using the current pipeline, each retrieval-condition record
contains `classifier_input_text`, token counts, and `was_truncated`, allowing the annotator
to inspect the exact final classifier input. Older records may lack these fields; when they
are absent, category 4 uses the weaker interpretation that sufficient evidence appeared in
the retrieved context and was judged likely to have been available to the classifier. The 
export explicitly reports when the exact classifier input is unavailable.

Four failure categories (defined upfront; formal precedence is in annotation_guide.md):
    1. irrelevant_retrieval        - context unrelated, only broadly topical, or not
                                     claim-specific; gold-evidence absence supports but does
                                     not by itself determine this.
    2. contradictory_retrieval     - retrieved context materially favours a label opposing
                                     the gold label.
    3. evidence_overload           - increasing retrieval depth produced a correct-to-wrong
                                     transition; manual inspection determines whether this
                                     resulted from additional documents, dilution of each
                                     document's token budget, or both.
    4. confident_wrong_prediction  - clear, sufficient and label-consistent evidence appeared
                                     in the retrieved context and was judged likely to have
                                     been available to the classifier, yet it erred with
                                     confidence >= 0.7. (Strengthened to "verified visible in
                                     classifier_input_text" only when that field is present.)

Manual annotation covers BOTH the dense and dense+rerank conditions on SciFact, so the
reranker's effect on categories 1 and 2 can be measured under the same taxonomy.

Modes: export (build the SciFact annotation sample), analyse (read completed annotation),
rates (SciFact-Open automatic diagnostics), kappa (intra-annotator agreement).
"""

#importing os for file path handling and directory creation
import os

#importing sys to add the project root to the import path
import sys

#adding the project root so any project imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing json for reading records and writing analysis output
import json

#importing argparse for command line mode and dataset selection
import argparse

#importing csv for writing and reading the manual annotation file
import csv

#importing random for reproducible sampling
import random

#importing math for the Wilson confidence interval computation
import math

# ---------------------------------------------------------------------------
# Failure categories and configuration
# ---------------------------------------------------------------------------

#defining the four failure category labels used for annotation and analysis
CATEGORY_IRRELEVANT = "irrelevant_retrieval"
CATEGORY_CONTRADICTORY = "contradictory_retrieval"
CATEGORY_OVERLOAD = "evidence_overload"
CATEGORY_CONFIDENT_WRONG = "confident_wrong_prediction"
EXCLUDED_BELOW_THRESHOLD = "excluded_below_threshold"
ACCEPTED_NON_CATEGORY = [EXCLUDED_BELOW_THRESHOLD]

#listing all valid categories for validation of the completed annotation
VALID_CATEGORIES = [
    CATEGORY_IRRELEVANT,
    CATEGORY_CONTRADICTORY,
    CATEGORY_OVERLOAD,
    CATEGORY_CONFIDENT_WRONG,
]

#matching the four pipeline conditions to the keys used in the saved records
CONDITIONS = ["no_retrieval", "bm25_roberta", "dense_roberta", "dense_reranked_roberta"]

#naming the two retrieval conditions that are manually annotated and compared
ANNOTATED_CONDITIONS = ["dense_roberta", "dense_reranked_roberta"]

#the k depths that make up the Step 6 controlled matrix
#(BM25/dense/dense+rerank x {1,3,5,10}).
#cross-k / overload analysis reads exactly these depths; missing files are skipped.
#MATRIX_K_VALUES = (1, 5, 10)
MATRIX_K_VALUES = (1, 3, 5, 10)

#setting the high-confidence threshold; fixed BEFORE inspecting results, and reported with a
#sensitivity check at 0.6/0.7/0.8 (see SENSITIVITY_THRESHOLDS). it is well above the
#three-class chance level (~0.33). The 0.7 value is the predefined PRIMARY threshold.
CONFIDENT_THRESHOLD = 0.7

#the thresholds over which the high-confidence-error diagnostic is reported, so a reader can
#see whether that automatic diagnostic is sensitive to the exact cut-point.
SENSITIVITY_THRESHOLDS = (0.6, 0.7, 0.8)

#fixing the random seed so the manual sample and any reliability subset are reproducible
SAMPLE_SEED = 42

# ---------------------------------------------------------------------------
# Record loading and gold-evidence join
# ---------------------------------------------------------------------------

def load_records(path):
    """Load a saved records file (a list of per-claim record dicts)."""
    #reading the saved records json from disk as utf-8
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    #handling both a plain list and a dict wrapping the records under a "records" key
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return data

def load_gold_evidence(dataset):
    """
    Load gold evidence document ids per claim id, fresh from the dataset loader. Returns a
    dict {claim_id (str): set(gold_doc_ids)} and prints validation counts so a silent
    mismatch (which would make gold-grounded signals uninformative) is caught early.
    """
    #importing the dataset loaders lazily so the analyse/kappa modes, which do not need the
    #dataset package, still run in environments where it is unavailable
    from data.utils import load_scifact, load_scifact_open

    #loading the appropriate dataset's claims (loaders return (claims, corpus))
    if dataset == "scifact":
        claims, _ = load_scifact(split="validation")
    else:
        claims, _ = load_scifact_open(corpus_file="full")
    #building a claim-id -> gold-doc-id-set map, normalising ids to strings
    gold = {}
    with_gold = 0
    for c in claims:
        ids = set(str(d) for d in c.get("evidence_doc_ids", []))
        gold[str(c["id"])] = ids
        if ids:
            with_gold += 1
    #reporting the validation counts
    print(f"Gold lookup loaded ({dataset}): {len(gold)} claims, "
          f"{with_gold} with gold document ids.")
    if with_gold == 0:
        print(f"  NOTE: {dataset} supplied no gold document ids at all; gold-based "
              f"diagnostics will be UNAVAILABLE (reported as null), not zero.")
    return gold

def gold_has_any_doc_ids(gold):
    """True if at least one claim in the gold lookup carries any gold document id."""
    #checking whether gold-based diagnostics are meaningful for this dataset at all
    return any(len(v) > 0 for v in gold.values())

def record_retrieved_ids(record):
    """
    Return the set of retrieved document ids for a record, robust to schema: prefer the
    explicit retrieved_doc_ids field, else fall back to ids inside retrieved_docs. Ids are
    normalised to strings so integer/string mismatches across files do not silently fail.
    """
    #trying the explicit retrieved_doc_ids field first
    ids = record.get("retrieved_doc_ids")
    if ids is None:
        #falling back to ids embedded in the retrieved_docs list
        ids = [d.get("doc_id") for d in record.get("retrieved_docs", [])
               if d.get("doc_id") is not None]
    return set(str(i) for i in ids)

def records_by_condition(records):
    """Split a flat list of records into a dict keyed by condition."""
    #initialising an empty list per condition
    by_cond = {c: [] for c in CONDITIONS}
    #assigning each record to its condition bucket
    for r in records:
        cond = r.get("condition")
        if cond in by_cond:
            by_cond[cond].append(r)
    return by_cond

def is_error(record):
    """A record is an error when the predicted label differs from the true label."""
    #comparing predicted against true label
    return record.get("predicted_label") != record.get("true_label")

def gold_doc_retrieved(record, gold):
    """
    Return True if at least one recognised gold evidence document for this claim was among the
    retrieved documents, False if the claim has gold evidence but none was retrieved, and None
    if the claim has gold evidence recorded as empty (signal undefined, e.g. NEI).

    Raises KeyError if the claim id is absent from the gold lookup entirely. That is a genuine
    split/id mismatch (records and gold loaded from different splits, id fields, or dataset
    releases) and must not be silently treated like a valid NEI claim with no gold evidence.
    validate_gold_against_records() is called first in every mode so this KeyError should never
    fire in practice; it remains as a defensive guard.
    """
    #reading this claim's id (already normalised elsewhere, normalise again defensively)
    cid = str(record.get("claim_id"))
    if cid not in gold:
        raise KeyError(f"Claim {cid} not found in gold lookup - possible split/id mismatch.")
    gold_ids = gold[cid]
    #distinguishing "no gold evidence recorded" (undefined signal) from "not in lookup" above
    if not gold_ids:
        return None
    #checking for any overlap between retrieved and gold, using robust id extraction
    return len(record_retrieved_ids(record) & gold_ids) > 0

def validate_gold_against_records(records, gold, dataset):
    """
    Confirm the saved record claim ids actually match the loaded gold lookup, so a split/id
    mismatch cannot hide behind gold.get(cid, empty). Distinguishes 'cid absent from lookup'
    (a mismatch, fatal) from 'cid present with empty gold set' (a legitimate NEI claim).
    Also reports whether gold document ids are available at all for this dataset.
    """
    #collecting the sets needed for the report
    rec_cids = set(str(r.get("claim_id")) for r in records)
    in_gold = {c for c in rec_cids if c in gold}
    absent = rec_cids - in_gold
    with_gold_ev = {c for c in in_gold if gold[c]}
    with_retrievable = set(str(r.get("claim_id")) for r in records if record_retrieved_ids(r))

    #printing the validation report
    print(f"Gold-vs-records validation ({dataset}):")
    print(f"  unique record claim ids:               {len(rec_cids)}")
    print(f"  record claims found in gold lookup:    {len(in_gold)}")
    print(f"  record claims ABSENT from gold lookup: {len(absent)}")
    print(f"  claims with gold document ids:         {len(with_gold_ev)}")
    print(f"  records with retrievable doc ids:      {len(with_retrievable)}")

    #a severe mismatch: some evaluated claim is not present in the gold lookup at all
    if absent:
        sample_absent = sorted(absent)[:10]
        raise ValueError(
            f"{len(absent)} record claim id(s) are absent from the {dataset} gold lookup "
            f"(e.g. {sample_absent}). This indicates a split/id mismatch - records and gold "
            f"were loaded from different splits, id fields, or dataset releases. gold.get(cid) "
            f"would otherwise treat these as NEI claims with no gold evidence and hide the "
            f"problem. Fix the loader or the records before analysis.")

    #for SciFact-Open (or any dataset) with no gold doc ids, say so explicitly
    if not gold_has_any_doc_ids(gold):
        print(f"  NOTE: {dataset} gold lookup contains no gold document ids; gold-based "
              f"diagnostics are UNAVAILABLE for this dataset and are reported as null.")
    return {
        "n_record_claims": len(rec_cids),
        "n_absent_from_gold": len(absent),
        "n_claims_with_gold_doc_ids": len(with_gold_ev),
        "gold_doc_ids_available": gold_has_any_doc_ids(gold),
    }

def validate_confidence_definition(records):
    """
    Confirm the saved `confidence` really is the maximum softmax probability over the three
    classes (not a raw logit or unnormalised score), by cross-checking against the saved
    `probabilities` vector. All saved records are checked so every pipeline condition is validated.
    Raises if confidence disagrees with max(probabilities).
    """
    #scanning all records that carry a probability vector
    checked = missing = bad_max = bad_sum = 0
    for r in records:
        probs = r.get("probabilities")
        conf = r.get("confidence")
        if probs is None or conf is None:
            missing += 1
            continue
        checked += 1
        if abs(max(probs) - float(conf)) > 1e-4:
            bad_max += 1
        if abs(sum(probs) - 1.0) > 1e-3:
            bad_sum += 1

    #reporting the outcome
    print("Confidence definition check (confidence == max softmax prob over 3 classes):")
    print(f"  records checked: {checked}   (records without a probability vector: {missing})")
    print(f"  confidence != max softmax probability: {bad_max}")
    print(f"  probabilities not summing to 1:        {bad_sum}")
    if checked and (bad_max or bad_sum):
        raise ValueError(
            "Saved `confidence` does not match the maximum softmax probability over the three "
            "classes (or probabilities do not sum to 1). The high-confidence diagnostics assume "
            "confidence is a normalised max-softmax value in [0, 1]; fix the pipeline before "
            "running the taxonomy analysis.")
    if checked == 0:
        print("  (no probability vectors saved in these records; skipping the definition check)")

# ---------------------------------------------------------------------------
# Automatic diagnostic signals (aids, NOT taxonomy labels)
# ---------------------------------------------------------------------------

def signal_high_confidence_error(record, threshold=CONFIDENT_THRESHOLD):
    """An incorrect prediction made with confidence >= the threshold. Descriptive only."""
    #flagging errors made at or above the confidence threshold
    return is_error(record) and record.get("confidence", 0.0) >= threshold

def signal_gold_evidence_missing(record, gold):
    """
    The recognised gold evidence document was absent from the retrieved set. This is an
    evidence-RECALL failure signal. It supports, but does not by itself determine, the
    manual irrelevant_retrieval category (retrieved docs could still be topically related).
    Returns None when the claim has no gold evidence recorded.
    """
    #reading whether a gold doc was retrieved
    got_gold = gold_doc_retrieved(record, gold)
    if got_gold is None:
        return None
    #flagging errors where the gold document was not retrieved
    return is_error(record) and (got_gold is False)

def signal_high_confidence_error_with_gold_doc(record, gold):
    """
    A high-confidence error for which a gold document WAS retrieved. This is consistent with a
    classifier-side failure after useful evidence was retrieved, but does NOT prove it (the
    gold text may be truncated, ranked low, or overwhelmed). Returns None when the claim has
    no gold evidence recorded.
    """
    #checking gold availability FIRST so an undefined case (no gold) stays undefined
    got_gold = gold_doc_retrieved(record, gold)
    if got_gold is None:
        return None
    #then requiring a high-confidence error and that a gold document was retrieved
    if not signal_high_confidence_error(record):
        return False
    return got_gold

def signal_strong_stance_document(record):
    """
    The top retrieved document is strongly stance-bearing according to the reranker's stance
    score. This is a strong-stance signal only: the saved records provide a single stance_score
    (and neutral_score) but NOT separate entailment/contradiction probabilities, so it does not
    establish stance DIRECTION relative to the gold label and is NOT a contradiction detector.
    The 0.5 cut-point is a descriptive heuristic that is only meaningful if stance_score lies in
    [0, 1]; see check_stance_score_range(), which validates that assumption on the records.
    Returns None when no stance score is present (non-reranked conditions).
    """
    #reading the retrieved documents and checking for a stance score
    docs = record.get("retrieved_docs", [])
    if not docs or "stance_score" not in docs[0]:
        return None
    #flagging when the top document carries a strong stance (direction unknown)
    return is_error(record) and float(docs[0].get("stance_score", 0.0)) >= 0.5

def check_stance_score_range(records):
    """
    Inspect the range of stance_score values actually present in the reranked records. The
    0.5 strong-stance cut-point is only interpretable if the score is a bounded [0, 1]
    probability. Print the observed min/max and warn if any value falls outside [0, 1].
    """
    #collecting every stance score across reranked documents
    values = []
    for r in records:
        for d in r.get("retrieved_docs", []):
            if "stance_score" in d and d.get("stance_score") is not None:
                values.append(float(d["stance_score"]))
    if not values:
        return
    lo, hi = min(values), max(values)
    print(f"Stance-score range across reranked records: min={lo:.3f} max={hi:.3f} "
          f"(n={len(values)}).")
    if lo < 0.0 or hi > 1.0:
        print("  WARNING: stance_score falls outside [0, 1]; the 0.5 strong-stance cut-point "
              "has no stable meaning. Treat sig_strong_stance_document as non-interpretable and "
              "document how the reranker computes stance_score.")

def detect_overload_claims(by_k):
    """
    Evidence overload compares the SAME CLAIM across k for one condition. A claim is an
    overload candidate when there exists any pair k_i < k_j with the prediction correct at
    k_i and wrong at k_j. Returns (overload_claim_ids, eligible_claim_ids) where eligible =
    claims present at two or more depths (the fair denominator).
    """
    #sorting the available k values
    ks = sorted(by_k.keys())
    if len(ks) < 2:
        return set(), set()

    #recording correctness of each claim at each k, normalising ids to strings
    correct_at = {}
    for k in ks:
        for r in by_k[k]:
            cid = str(r.get("claim_id"))
            correct_at.setdefault(cid, {})[k] = not is_error(r)

    #collecting overload candidates and the eligible denominator
    overload_claims = set()
    eligible_claims = set()
    for cid, per_k in correct_at.items():
        present_ks = sorted(per_k.keys())
        if len(present_ks) < 2:
            continue
        eligible_claims.add(cid)
        #flagging if correctness drops between any earlier and later depth
        found = False
        for i in range(len(present_ks)):
            for j in range(i + 1, len(present_ks)):
                if per_k[present_ks[i]] and not per_k[present_ks[j]]:
                    overload_claims.add(cid)
                    found = True
                    break
            if found:
                break
    return overload_claims, eligible_claims

def harmful_transition_into_k(correct_by_k, target_k):
    """
    Return the first smaller-k correct -> target-k wrong transition.
    Empty when the target depth is absent, correct, or no smaller depth was correct.
    """
    if target_k not in correct_by_k or correct_by_k[target_k]:
        return ""
    for smaller_k in sorted(k for k in correct_by_k if k < target_k):
        if correct_by_k[smaller_k]:
            return f"k={smaller_k} correct -> k={target_k} wrong"
    return ""

# ---------------------------------------------------------------------------
# Cross-k context (full per-k prediction context for exported claims)
# ---------------------------------------------------------------------------

def cross_k_context(records_dir, dataset, condition, ks=MATRIX_K_VALUES):
    """
    Build claim_id -> {k: {predicted_label, confidence, correct, retrieved_doc_ids,
    classifier_input_text, token counts}} for one condition across all available matrix
    depths. A correctness flip establishes degradation as retrieval depth increases. The
    exported document ids and exact classifier input allow manual inspection of whether the
    change is associated with newly retrieved documents, reduced token allocation to earlier
    documents, or both. Document ids are normalised to strings.
    """
    #initialising the per-claim per-k context map
    per_claim = {}
    for k in ks:
        p = os.path.join(records_dir, f"records_{dataset}_k{k}_thr0_5.json")
        if not os.path.exists(p):
            continue
        recs = records_by_condition(load_records(p)).get(condition, [])
        for r in recs:
            cid = str(r.get("claim_id"))
            per_claim.setdefault(cid, {})[k] = {
                "predicted_label": r.get("predicted_label"),
                "confidence": round(r.get("confidence", 0.0), 4),
                "correct": (not is_error(r)),
                "retrieved_doc_ids": sorted(record_retrieved_ids(r)),
                "classifier_input_text": r.get("classifier_input_text"),
                "input_token_count_before_truncation":
                    r.get("input_token_count_before_truncation"),
                "input_token_count_after_truncation":
                    r.get("input_token_count_after_truncation"),
                "was_truncated": r.get("was_truncated"),
            }
    return per_claim

def correct_by_k_from_context(ctx):
    """Reduce a cross-k context map to {k: correct_bool} for transition detection."""
    #extracting just the correctness at each depth
    return {k: v["correct"] for k, v in ctx.items()}

# ---------------------------------------------------------------------------
# Confidence sensitivity of the high-confidence-error diagnostic
# ---------------------------------------------------------------------------

def high_confidence_error_rate(errs, threshold):
    """Percentage of the given error records made with confidence >= threshold."""
    #guarding against an empty error list
    if not errs:
        return None
    n_hc = sum(1 for r in errs if r.get("confidence", 0.0) >= threshold)
    return round(100 * n_hc / len(errs), 1)

def high_confidence_error_sensitivity(errs, thresholds=SENSITIVITY_THRESHOLDS):
    """
    Report the high-confidence-error rate (as a percentage of errors) at each threshold, so a
    reader can see whether the automatic diagnostic is sensitive to the exact 0.6/0.7/0.8
    cut-point. This does not change any manually assigned category; it only characterises the
    automatic signal.
    """
    #computing the rate at each threshold, keyed by a stable string form
    return {f"{t:.2f}": high_confidence_error_rate(errs, t) for t in thresholds}

# ---------------------------------------------------------------------------
# Small statistics helpers (Wilson confidence intervals)
# ---------------------------------------------------------------------------

def wilson_interval(count, total, z=1.96):
    """
    Wilson score 95% confidence interval for a binomial proportion, returned as percentages.
    Preferred over the normal approximation at the small annotation sample sizes here.
    """
    #guarding against an empty denominator
    if total == 0:
        return (None, None)
    phat = count / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (round(100 * lo, 1), round(100 * hi, 1))

# ---------------------------------------------------------------------------
# Annotation keys (stable, include depth k)
# ---------------------------------------------------------------------------

def annotation_key_of(row):
    """
    Return the stable annotation key for a row. Prefer the explicit annotation_key field
    (condition::claim_id::k{k}); fall back to condition::claim_id for older exports. Including
    k avoids ambiguity if multiple depths are ever annotated.
    """
    #using the full stable key when the export provided it
    if row.get("annotation_key"):
        return row["annotation_key"]
    return f"{row.get('condition')}::{row.get('claim_id')}"

# ---------------------------------------------------------------------------
# Mode: export a condition-balanced random sample for manual annotation (SciFact)
# ---------------------------------------------------------------------------

def allocate_caps(max_errors, n_conditions):
    """
    Split the annotation budget across conditions, allocating any remainder from an odd budget
    to the earliest conditions rather than silently dropping it. Returns a list of per-condition
    caps summing to max_errors.
    """
    #computing an even base and distributing the remainder deterministically
    base = max_errors // n_conditions
    rem = max_errors % n_conditions
    return [base + (1 if i < rem else 0) for i in range(n_conditions)]


def build_docs_struct(record, gold_ids):
    """Serialise a record's retrieved documents into a structured, gold-flagged list."""
    #turning each retrieved doc into a structured entry with a gold flag
    docs_struct = []
    for rank, d in enumerate(record.get("retrieved_docs", []), start=1):
        docs_struct.append({
            "rank": rank,
            "doc_id": d.get("doc_id"),
            "score": d.get("score"),
            "stance_score": d.get("stance_score"),
            "neutral_score": d.get("neutral_score"),
            "is_gold": (str(d.get("doc_id")) in gold_ids),
            "text": d.get("text", ""),
        })
    return docs_struct


def export_for_annotation(records_path, records_dir, dataset, out_csv, out_json,
                          gold, k, max_errors=70):
    """
    Build a reproducible manual-annotation sample drawn from BOTH the dense and dense+rerank
    conditions. Sampling is RANDOM within each condition (condition-balanced), which preserves
    each condition's natural error distribution across true classes, so the annotated category
    percentages are unbiased estimates of that condition's error mix. Every retrieved document
    is exported with gold flags and stance scores; each row also carries full cross-k prediction
    context, the paired result under the other condition, and the exact classifier input when
    the pipeline saved it.
    """
    #loading the records at the reported depth and splitting by condition
    records = load_records(records_path)

    #validating the gold lookup against these exact records before any gold-grounded signal
    validate_gold_against_records(records, gold, dataset)
    #confirming the confidence field is a normalised max-softmax value
    validate_confidence_definition(records)

    by_cond = records_by_condition(records)
    #reporting the stance-score range so the 0.5 heuristic is only trusted if scores are in [0,1]
    check_stance_score_range(by_cond.get("dense_reranked_roberta", []))

    #splitting the budget between the two annotated conditions (remainder handled explicitly)
    caps = allocate_caps(max_errors, len(ANNOTATED_CONDITIONS))
    cap_by_condition = dict(zip(ANNOTATED_CONDITIONS, caps))

    #seeding the sampler for reproducibility
    rng = random.Random(SAMPLE_SEED)

    #precomputing full cross-k prediction context per condition for the overload picture
    cross_k = {cond: cross_k_context(records_dir, dataset, cond)
               for cond in ANNOTATED_CONDITIONS}

    #indexing each condition's records by claim id, to attach the paired other-condition result
    by_cond_by_cid = {
        cond: {str(r.get("claim_id")): r for r in by_cond.get(cond, [])}
        for cond in ANNOTATED_CONDITIONS
    }

    #detecting whether the pipeline saved the exact classifier input (input-capture patch).
    #requiring ALL annotated-condition records to carry the field, not merely one, so the
    #strong "verified visible" annotation form is only enabled when every row supports it.
    annotated_records = [r for cond in ANNOTATED_CONDITIONS for r in by_cond.get(cond, [])]
    n_with_input = sum(1 for r in annotated_records if "classifier_input_text" in r)
    classifier_input_available = (len(annotated_records) > 0
                                  and n_with_input == len(annotated_records))

    annotation_rows = []
    class_representation = {}
    sensitivity_by_condition = {}
    for cond in ANNOTATED_CONDITIONS:
        other = [c for c in ANNOTATED_CONDITIONS if c != cond][0]

        #collecting this condition's errors and random-sampling without class forcing
        errors = [r for r in by_cond.get(cond, []) if is_error(r)]
        rng.shuffle(errors)
        selected = errors[:min(cap_by_condition[cond], len(errors))]

        #recording the natural class representation of the sample for transparency
        rep = {}
        for r in selected:
            rep[r.get("true_label")] = rep.get(r.get("true_label"), 0) + 1
        class_representation[cond] = rep
        #recording the high-confidence-error sensitivity over ALL of this condition's errors
        sensitivity_by_condition[cond] = high_confidence_error_sensitivity(errors)
        print(f"Condition '{cond}': {len(errors)} errors, randomly selecting {len(selected)} "
              f"(seed {SAMPLE_SEED}); class mix {rep}.")

        #building one annotation row per selected error, with full context
        for r in selected:
            cid = str(r.get("claim_id"))
            gold_ids = gold.get(cid, set())
            docs_struct = build_docs_struct(r, gold_ids)

            #reading cross-k context and checking whether a smaller-k correct prediction
            #became wrong at the currently annotated target depth
            ctx = cross_k.get(cond, {}).get(cid, {})
            correct_by_k = correct_by_k_from_context(ctx)

            #looking up the SAME claim under the other condition, for the paired comparison
            other_rec = by_cond_by_cid.get(other, {}).get(cid)

            row = {
                #stable key including depth k, used for kappa and duplicate detection
                "annotation_key": f"{cond}::{cid}::k{k}",
                "condition": cond,
                "claim_id": cid,
                "claim": r.get("claim", ""),
                "true_label": r.get("true_label"),
                "predicted_label": r.get("predicted_label"),
                "confidence": round(r.get("confidence", 0.0), 4),
                "num_docs": len(r.get("retrieved_docs", [])),
                "gold_doc_retrieved": gold_doc_retrieved(r, gold),
                #cross-k picture: compact correctness plus full per-k prediction context
                "correct_by_k": json.dumps(correct_by_k),
                "harmful_transition_into_target_k": harmful_transition_into_k(correct_by_k, k),
                "cross_k_context": json.dumps(ctx, ensure_ascii=False),
                #automatic diagnostic signals, shown to support (not replace) human judgement
                "sig_high_confidence_error": signal_high_confidence_error(r),
                "sig_gold_evidence_missing": signal_gold_evidence_missing(r, gold),
                "sig_high_conf_error_with_gold_doc":
                    signal_high_confidence_error_with_gold_doc(r, gold),
                "sig_strong_stance_document": signal_strong_stance_document(r),
                #exact classifier input, exported when the pipeline saved it (else null)
                "classifier_input_text": r.get("classifier_input_text"),
                "input_token_count_before_truncation":
                    r.get("input_token_count_before_truncation"),
                "input_token_count_after_truncation":
                    r.get("input_token_count_after_truncation"),
                "was_truncated": r.get("was_truncated"),
                #all retrieved documents given to the classifier, as a structured field
                "retrieved_documents": json.dumps(docs_struct, ensure_ascii=False),
                #paired result for the SAME claim under the other condition
                "other_condition": other,
                "other_condition_prediction":
                    (other_rec.get("predicted_label") if other_rec else None),
                "other_condition_confidence":
                    (round(other_rec.get("confidence", 0.0), 4) if other_rec else None),
                "other_condition_correct":
                    ((not is_error(other_rec)) if other_rec else None),
                "other_condition_retrieved_documents":
                    (json.dumps(build_docs_struct(other_rec, gold.get(cid, set())),
                                ensure_ascii=False) if other_rec else None),
                #manual fields to be completed by the annotator
                "primary_category": "",
                "annotation_notes": "",
            }
            annotation_rows.append(row)

    #guarding against an empty selection before writing
    if not annotation_rows:
        print("No errors selected - nothing to export. Check the records path and conditions.")
        return

    #warning clearly if the exact classifier input was not captured by the pipeline
    if not classifier_input_available:
        print("\nNOTE: these records do NOT contain classifier_input_text. They were likely "
            "generated by an older pipeline version. The exact final classifier input cannot "
            "be verified, so category 4 (confident_wrong_prediction) uses the weaker form: "
            "sufficient label-consistent evidence appeared in the retrieved context and was "
            "judged likely to have been available to the classifier. Regenerate the Step 5/6 "
            "records using the current pipeline to enable the stronger 'verified visible in "
            "classifier_input_text' interpretation.")

    #wrapping the rows with reproducibility metadata
    output_obj = {
        "metadata": {
            "dataset": dataset,
            "k": k,
            "records_path": records_path,
            "sample_seed": SAMPLE_SEED,
            "sampling_method": "random, balanced by condition (natural class mix preserved)",
            "requested_max_errors": max_errors,
            "per_condition_caps": cap_by_condition,
            "annotated_conditions": ANNOTATED_CONDITIONS,
            "confident_threshold_primary": CONFIDENT_THRESHOLD,
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "high_confidence_error_sensitivity": sensitivity_by_condition,
            "class_representation": class_representation,
            "classifier_input_available": classifier_input_available,
            "n_condition_records_with_classifier_input": n_with_input,
            "n_condition_records_total": len(annotated_records),
            "n_exported": len(annotation_rows),
        },
        "rows": annotation_rows,
    }

    #writing the json (metadata + rows) and the csv (rows only, for spreadsheet editing)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(annotation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(annotation_rows)

    print(f"\nExported {len(annotation_rows)} errors "
          f"({', '.join(ANNOTATED_CONDITIONS)}) for manual annotation.")
    print(f"  {out_json}\n  {out_csv}")
    print("Fill in 'primary_category' using annotation_guide.md, with one of:")
    for c in VALID_CATEGORIES:
        print(f"    {c}")

# ---------------------------------------------------------------------------
# Mode: analyse the completed manual annotation (SciFact, both conditions)
# ---------------------------------------------------------------------------

def read_annotation_rows(annotation_csv):
    """Read annotation rows from the completed csv."""
    #reading rows as utf-8
    rows = []
    with open(annotation_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def validate_annotation(rows):
    """
    Validate category strings, required identifiers, duplicate annotation keys, and logical
    consistency with the predefined taxonomy.

    In particular:
      - confident_wrong_prediction requires confidence >= CONFIDENT_THRESHOLD.
      - evidence_overload requires a recorded smaller-k correct -> target-k wrong transition.
    """
    problems = {
        "blank": 0,
        "invalid_category": [],
        "missing_condition": 0,
        "missing_claim_id": 0,
        "duplicate_keys": [],
        "logical_inconsistencies": [],
    }

    seen = {}

    for r in rows:
        cat = r.get("primary_category", "").strip()
        key = annotation_key_of(r)

        if cat == "":
            problems["blank"] += 1
        #elif cat not in VALID_CATEGORIES:
        elif cat not in VALID_CATEGORIES and cat not in ACCEPTED_NON_CATEGORY:
            problems["invalid_category"].append(cat)

        if not r.get("condition"):
            problems["missing_condition"] += 1

        if not r.get("claim_id"):
            problems["missing_claim_id"] += 1

        seen[key] = seen.get(key, 0) + 1

        #checking logical consistency only for recognised categories
        if cat == CATEGORY_CONFIDENT_WRONG:
            try:
                confidence = float(r.get("confidence", ""))
            except (TypeError, ValueError):
                confidence = None

            if confidence is None or confidence < CONFIDENT_THRESHOLD:
                problems["logical_inconsistencies"].append(
                    f"{key}: confident_wrong_prediction assigned with "
                    f"confidence={r.get('confidence')}; requires >= "
                    f"{CONFIDENT_THRESHOLD}."
                )

        if cat == CATEGORY_OVERLOAD:
            transition = r.get(
                "harmful_transition_into_target_k", ""
            ).strip()

            if not transition:
                problems["logical_inconsistencies"].append(
                    f"{key}: evidence_overload assigned without a smaller-k "
                    f"correct -> target-k wrong transition."
                )

    problems["duplicate_keys"] = sorted(
        key for key, count in seen.items() if count > 1
    )

    print("Annotation validation:")
    print(f"  blank categories:          {problems['blank']}")
    print(
        "  invalid category strings: "
        f"{sorted(set(problems['invalid_category']))}"
    )
    print(f"  missing condition:         {problems['missing_condition']}")
    print(f"  missing claim id:          {problems['missing_claim_id']}")
    print(f"  duplicate annotation keys: {problems['duplicate_keys']}")
    print(
        "  logical inconsistencies:  "
        f"{len(problems['logical_inconsistencies'])}"
    )

    for issue in problems["logical_inconsistencies"][:10]:
        print(f"    - {issue}")

    if len(problems["logical_inconsistencies"]) > 10:
        remaining = len(problems["logical_inconsistencies"]) - 10
        print(f"    ... and {remaining} more")

    return problems

def annotation_is_clean(problems):
    """True when no annotation-data or taxonomy-consistency problems remain."""
    return (
        problems["blank"] == 0
        and not problems["invalid_category"]
        and problems["missing_condition"] == 0
        and problems["missing_claim_id"] == 0
        and not problems["duplicate_keys"]
        and not problems["logical_inconsistencies"]
    )

def analyse_annotation(annotation_csv, out_json, allow_partial=False):
    """
    Read the completed manual annotation and produce the per-condition category breakdown and
    the dense-vs-reranked comparison for categories 1 and 2, using the MANUAL labels. Refuses
    to run on an incomplete or invalid annotation unless allow_partial is set, because computing
    percentages over only the easy (already-labelled) rows would bias the reported distribution.
    Final thesis results must NOT use allow_partial.
    """
    #reading and validating the completed annotation
    rows = read_annotation_rows(annotation_csv)
    problems = validate_annotation(rows)

    #stopping on any data-quality problem unless partial analysis was explicitly requested
    if not annotation_is_clean(problems):
        if not allow_partial:
            raise ValueError(
                "Annotation is incomplete or invalid (see the validation report above: blank, "
                "invalid, missing, or duplicated rows). Correct ALL rows before analysis. "
                "Percentages computed over only the currently-valid rows would be biased toward "
                "the easier cases. Re-run with --allow_partial ONLY for a provisional look; final "
                "thesis results must not use partial mode.")
        print("\nWARNING: --allow_partial is set. Results below are PROVISIONAL, computed over a "
              "subset of rows, and MUST NOT be reported as final thesis results.\n")

    #keeping only rows with a valid manual category
    labelled = [r for r in rows if r.get("primary_category", "").strip() in VALID_CATEGORIES]
    excluded = [r for r in rows if r.get("primary_category", "").strip() == EXCLUDED_BELOW_THRESHOLD]
    print(f"  excluded_below_threshold (reported separately, not in 4-category denominator): {len(excluded)}")
    print(f"\nLoaded {len(rows)} rows; {len(labelled)} carry a valid category.")
    if not labelled:
        print("No valid categories found - fill in 'primary_category' first.")
        return

    #building a per-condition category breakdown with Wilson 95% intervals
    per_condition = {}
    for cond in ANNOTATED_CONDITIONS:
        cond_rows = [r for r in labelled if r.get("condition") == cond]
        counts = {c: 0 for c in VALID_CATEGORIES}
        for r in cond_rows:
            counts[r["primary_category"].strip()] += 1
        total = len(cond_rows)
        per_condition[cond] = {
            "n_annotated": total,
            "counts": counts,
            "pct": {c: (round(100 * counts[c] / total, 1) if total else None)
                    for c in VALID_CATEGORIES},
            "ci95": {c: wilson_interval(counts[c], total) for c in VALID_CATEGORIES},
        }

    #printing the per-condition breakdown with intervals
    for cond in ANNOTATED_CONDITIONS:
        info = per_condition[cond]
        print(f"\n{cond} (n={info['n_annotated']}):")
        for c in VALID_CATEGORIES:
            pct = info["pct"][c]
            pct_str = f"{pct:.1f}%" if pct is not None else "n/a"
            lo, hi = info["ci95"][c]
            ci_str = f"[{lo}, {hi}]%" if lo is not None else "n/a"
            print(f"  {c:28s} {info['counts'][c]:3d}  ({pct_str})  95% CI {ci_str}")

    #comparing categories 1 and 2 between dense and reranked descriptively, with Wilson
    #intervals. No significance test is run because the condition samples may overlap in claims,
    #and each exported row also contains the corresponding other-condition result.
    comparison = {}
    dense = per_condition.get("dense_roberta", {})
    rer = per_condition.get("dense_reranked_roberta", {})
    for cat in [CATEGORY_IRRELEVANT, CATEGORY_CONTRADICTORY]:
        comparison[cat] = {
            "dense_pct": dense.get("pct", {}).get(cat),
            "dense_count": dense.get("counts", {}).get(cat, 0),
            "dense_ci95": dense.get("ci95", {}).get(cat),
            "reranked_pct": rer.get("pct", {}).get(cat),
            "reranked_count": rer.get("counts", {}).get(cat, 0),
            "reranked_ci95": rer.get("ci95", {}).get(cat),
        }

    print("\nReranking effect on categories 1 and 2 (manual labels, descriptive with 95% CIs):")
    for cat in [CATEGORY_IRRELEVANT, CATEGORY_CONTRADICTORY]:
        cmp = comparison[cat]
        d = cmp["dense_pct"]
        rr = cmp["reranked_pct"]
        d_str = f"{d:.1f}%" if d is not None else "n/a"
        r_str = f"{rr:.1f}%" if rr is not None else "n/a"
        print(f"  {cat:28s} dense {d_str}  ->  reranked {r_str}")

    #assembling and saving the analysis result
    result = {
        "used_allow_partial": bool(allow_partial and not annotation_is_clean(problems)),
        "validation": {
            "blank": problems["blank"],
            "invalid_categories": sorted(set(problems["invalid_category"])),
            "excluded_below_threshold_count": len(excluded),
            "missing_condition": problems["missing_condition"],
            "missing_claim_id": problems["missing_claim_id"],
            "duplicate_keys": problems["duplicate_keys"],
            "logical_inconsistencies": problems["logical_inconsistencies"],
        },
        "per_condition_manual_breakdown": per_condition,
        "reranking_effect_categories_1_2": comparison,
        "statistics_note": (
                            "95% intervals are Wilson score intervals. The dense-versus-reranked "
                            "comparison is descriptive, with no significance test, because the "
                            "condition samples may overlap in claims and the annotation sample is "
                            "primarily intended for qualitative failure analysis. With approximately "
                            "35 annotations per condition, the intervals should be interpreted as "
                            "indicative rather than definitive."
                        ),
        "note": ("Percentages are of each condition's annotated errors. Reranking can change "
                 "the total error count, so read counts and the Step 5/6 error rates alongside "
                 "these: a lower proportion among fewer errors is not the same as lower "
                 "absolute incidence. Paired other-condition fields are exported per row to let "
                 "dense-wrong->reranked-correct (and the reverse) be inspected directly."),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved analysis to {out_json}")
    return result

# ---------------------------------------------------------------------------
# Mode: automatic diagnostic signals across conditions (SciFact-Open, no annotation)
# ---------------------------------------------------------------------------

def quantitative_rates(records_path, records_dir, dataset, gold, out_json):
    """
    Compute automatic diagnostic signals across all four conditions, without manual
    annotation. These are proxy signals for possible failure mechanisms, explicitly NOT
    taxonomy rates. Categories undefined for a condition are reported as null, not zero. The
    high-confidence-error diagnostic is reported at 0.6/0.7/0.8 so its sensitivity to the
    cut-point is visible.
    """
    #loading the records and validating them against the gold lookup first
    records = load_records(records_path)
    validate_gold_against_records(records, gold, dataset)
    validate_confidence_definition(records)

    by_cond = records_by_condition(records)
    check_stance_score_range(by_cond.get("dense_reranked_roberta", []))
    gold_available = gold_has_any_doc_ids(gold)

    #computing the per-condition signal rates
    per_condition = {}
    for cond in CONDITIONS:
        recs = by_cond.get(cond, [])
        errs = [r for r in recs if is_error(r)]
        n = len(recs)
        n_err = len(errs)
        if n == 0:
            continue

        #gold-grounded signals are undefined for no_retrieval (no documents) and for any
        #dataset that supplies no gold document ids at all (e.g. SciFact-Open)
        if cond == "no_retrieval" or not gold_available:
            gold_missing_pct = None
            hce_with_gold_pct = None
        else:
            #counting only over errors where the signal is defined
            gm = [signal_gold_evidence_missing(r, gold) for r in errs]
            gm_def = [x for x in gm if x is not None]
            gold_missing_pct = (round(100 * sum(gm_def) / len(gm_def), 1)
                                if gm_def else None)
            hg = [signal_high_confidence_error_with_gold_doc(r, gold) for r in errs]
            hg_def = [x for x in hg if x is not None]
            hce_with_gold_pct = (round(100 * sum(hg_def) / len(hg_def), 1)
                                 if hg_def else None)

        per_condition[cond] = {
            "n_claims": n,
            "n_errors": n_err,
            "error_rate_pct": round(100 * n_err / n, 1),
            #primary threshold rate, kept for continuity with the rest of the analysis
            "high_confidence_error_pct_of_errors":
                high_confidence_error_rate(errs, CONFIDENT_THRESHOLD),
            #sensitivity of that diagnostic across 0.6/0.7/0.8
            "high_confidence_error_pct_by_threshold":
                high_confidence_error_sensitivity(errs),
            "gold_evidence_missing_pct_of_defined_errors": gold_missing_pct,
            "high_conf_error_with_gold_doc_pct_of_defined_errors": hce_with_gold_pct,
        }

    #detecting evidence overload across k, with a fair eligible denominator
    overload = {}
    for cond in ["bm25_roberta", "dense_roberta", "dense_reranked_roberta"]:
        by_k = {}
        for k in MATRIX_K_VALUES:
            p = os.path.join(records_dir, f"records_{dataset}_k{k}_thr0_5.json")
            if os.path.exists(p):
                by_k[k] = records_by_condition(load_records(p)).get(cond, [])
        if len(by_k) >= 2:
            oc, eligible = detect_overload_claims(by_k)
            overload[cond] = {
                "overload_claims": len(oc),
                "eligible_claims": len(eligible),
                "overload_pct_of_eligible": (round(100 * len(oc) / len(eligible), 1)
                                             if eligible else None),
            }

    #assembling the result
    result = {
        "dataset": dataset,
        "gold_doc_ids_available": gold_available,
        "confident_threshold_primary": CONFIDENT_THRESHOLD,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "note": ("Automatic diagnostic signals for possible failure mechanisms, NOT "
                 "manually-verified taxonomy labels. Gold-evidence-missing is an evidence "
                 "recall signal, not proof of topical irrelevance. Do not report these as "
                 "percentages of the four failure categories."),
        "per_condition_signals": per_condition,
        "evidence_overload_across_k": overload,
    }

    #printing a short summary
    print(f"\nAutomatic diagnostic signals ({dataset}) - proxy signals, not taxonomy labels:")
    for cond, s in per_condition.items():
        print(f"  {cond:26s} errors {s['error_rate_pct']:.1f}%  "
              f"high-conf-err@0.7 {s['high_confidence_error_pct_of_errors']}%  "
              f"gold-missing {s['gold_evidence_missing_pct_of_defined_errors']}")
        print(f"    high-conf-err sensitivity {s['high_confidence_error_pct_by_threshold']}")
    if overload:
        print("  evidence-overload (correct-then-wrong across k, of eligible claims):")
        for cond, o in overload.items():
            print(f"    {cond:26s} {o['overload_claims']}/{o['eligible_claims']} "
                  f"({o['overload_pct_of_eligible']}%)")

    #saving the diagnostic signals
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved diagnostic signals to {out_json}")
    return result

# ---------------------------------------------------------------------------
# Mode: intra-annotator agreement (Cohen's kappa) between two annotation passes
# ---------------------------------------------------------------------------

def compute_kappa(first_csv, second_csv, out_json):
    """
    Compute intra-annotator agreement (percentage agreement and Cohen's kappa) between two
    annotation passes over the same errors. Rows are matched by the stable annotation key
    (condition::claim_id::k), which is robust to the export being regenerated. Duplicate keys
    within a pass are fatal, because last-write-wins would silently distort the agreement.
    """
    #reading a pass into a stable-key -> category map, refusing duplicate keys
    def read_pass(path):
        m = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat = row.get("primary_category", "").strip()
                if cat not in VALID_CATEGORIES:
                    continue
                key = annotation_key_of(row)
                if key in m:
                    raise ValueError(
                        f"Duplicate annotation key '{key}' in {path}. Remove duplicates before "
                        f"computing kappa; last-write-wins would otherwise distort agreement.")
                m[key] = cat
        return m

    first = read_pass(first_csv)
    second = read_pass(second_csv)

    #keeping only keys validly labelled in both passes
    shared = [k for k in first if k in second]
    if not shared:
        print("No overlapping labelled errors between the two passes.")
        return None

    #computing observed agreement
    agree = sum(1 for k in shared if first[k] == second[k])
    n = len(shared)
    p_observed = agree / n

    #computing expected agreement for Cohen's kappa
    p_expected = 0.0
    for c in VALID_CATEGORIES:
        p1 = sum(1 for k in shared if first[k] == c) / n
        p2 = sum(1 for k in shared if second[k] == c) / n
        p_expected += p1 * p2
    kappa = (p_observed - p_expected) / (1 - p_expected) if (1 - p_expected) > 0 else None

    #assembling and printing the reliability result
    result = {
        "n_double_annotated": n,
        "percentage_agreement": round(100 * p_observed, 1),
        "cohens_kappa": round(kappa, 3) if kappa is not None else None,
        "matching_key": "annotation_key (condition::claim_id::k)",
    }
    print(f"Intra-annotator reliability on {n} double-annotated errors:")
    print(f"  percentage agreement: {result['percentage_agreement']}%")
    print(f"  Cohen's kappa:        {result['cohens_kappa']}")

    #saving the reliability result
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_records_path(records_dir, dataset, k, explicit=None):
    """
    Resolve the records path. Prefer an explicit --records_path. Otherwise use ONLY the Step 6
    matrix naming records_<dataset>_k<k>_thr0_5.json. The generic Step 5 fallback has been
    removed: falling back to a fixed-depth Step 5 file while still writing "k": <requested_k>
    into the metadata would mislabel the retrieval depth.
    """
    #using an explicit path when provided, verifying it exists
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--records_path not found: {explicit}")
        return explicit
    #otherwise requiring the Step 6 matrix file for the requested depth
    p = os.path.join(records_dir, f"records_{dataset}_k{k}_thr0_5.json")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(
        f"No Step 6 matrix records file found at:\n  {p}\n"
        f"(dataset={dataset}, k={k}). Pass --records_path explicitly, or check --records_dir. "
        f"The generic Step 5 fallback was removed so the saved 'k' always matches the actual "
        f"retrieval depth.")


def main():
    #setting up the command line arguments
    parser = argparse.ArgumentParser(description="Step 7 failure taxonomy analysis")
    parser.add_argument("--mode", required=True,
                        choices=["export", "analyse", "rates", "kappa"])
    parser.add_argument("--dataset", default="scifact",
                        choices=["scifact", "scifact_open"])
    parser.add_argument("--records_dir", default="results/step6_matrix")
    parser.add_argument("--records_path", default=None,
                        help="Explicit records file; required if the Step 6 matrix naming is "
                             "not used. Overrides --records_dir/--k for locating the file.")
    parser.add_argument("--k", type=int, default=3, choices=list(MATRIX_K_VALUES), 
                        help="Primary retrieval depth for the annotation sample. "
                            "Must be one of the Step 6 matrix depths: 1, 3, 5, or 10.",)
    parser.add_argument("--out_dir", default="results/step7_failure")
    parser.add_argument("--max_errors", type=int, default=70)
    parser.add_argument("--allow_partial", action="store_true",
                        help="Analyse an incomplete/invalid annotation (PROVISIONAL only; never "
                             "for final thesis results).")
    parser.add_argument("--first_csv", default=None)
    parser.add_argument("--second_csv", default=None)
    args = parser.parse_args()

    #restricting manual export/analyse to SciFact, matching the scoping decision
    if args.mode in ["export", "analyse"] and args.dataset != "scifact":
        parser.error("Manual export/analyse modes are restricted to SciFact.")

    #ensuring the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "export":
        #loading gold evidence and records, then exporting the balanced random sample
        gold = load_gold_evidence(args.dataset)
        records_path = resolve_records_path(args.records_dir, args.dataset, args.k,
                                            explicit=args.records_path)
        export_for_annotation(
            records_path=records_path,
            records_dir=args.records_dir,
            dataset=args.dataset,
            out_csv=os.path.join(args.out_dir, f"annotation_{args.dataset}.csv"),
            out_json=os.path.join(args.out_dir, f"annotation_{args.dataset}.json"),
            gold=gold,
            k=args.k,
            max_errors=args.max_errors,
        )
    elif args.mode == "analyse":
        analyse_annotation(
            annotation_csv=os.path.join(args.out_dir, f"annotation_{args.dataset}.csv"),
            out_json=os.path.join(args.out_dir, f"analysis_{args.dataset}.json"),
            allow_partial=args.allow_partial,
        )
    elif args.mode == "rates":
        #loading gold evidence and records, then computing diagnostic signals
        gold = load_gold_evidence(args.dataset)
        records_path = resolve_records_path(args.records_dir, args.dataset, args.k,
                                            explicit=args.records_path)
        quantitative_rates(
            records_path=records_path,
            records_dir=args.records_dir,
            dataset=args.dataset,
            gold=gold,
            out_json=os.path.join(args.out_dir, f"rates_{args.dataset}.json"),
        )
    elif args.mode == "kappa":
        #computing intra-annotator agreement between two passes
        if not args.first_csv or not args.second_csv:
            parser.error("--first_csv and --second_csv are required for --mode kappa")
        compute_kappa(
            first_csv=args.first_csv,
            second_csv=args.second_csv,
            out_json=os.path.join(args.out_dir, f"kappa_{args.dataset}.json"),
        )


if __name__ == "__main__":
    main()