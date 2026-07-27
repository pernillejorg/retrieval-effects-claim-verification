"""
Step 10: real-world case study on seafood and sustainability claims.

This script runs a set of real, manually labelled social-media claims about seafood and
sustainability through the same pipeline the rest of the project uses, against the SciFact-Open
corpus. It is the domain-generalisation test: everything earlier in the project is biomedical,
so this step asks whether the findings transfer to a genuinely different subject domain. Poor
retrieval here is expected and, if observed, is part of the phenomenon being studied rather
than an implementation bug, because a biomedical corpus
mostly does not contain evidence for seafood claims.

WHAT IT DOES (and, importantly, what it reuses rather than reinvents):

The retrieval, reranking and classification are the EXACT same components as Step 5's
pipeline.py: the dense (mpnet) retriever, the soft stance reranker, and the two RoBERTa
classifiers (Model 1 claim-only for no-retrieval, Model 2 claim+evidence for the retrieval
conditions). This script imports those functions directly, so the case study cannot silently
diverge from the pipeline the rest of the thesis evaluates. The only things that differ from
Step 5 are the DATA (real seafood claims from a CSV instead of a benchmark split) and that the
retrieval conditions are swept across k = 1, 3, 5, 10.

THREE CONDITIONS ARE RUN, per claim:
  - no_retrieval (Model 1, claim only): the baseline that Step 9 showed can beat retrieval.
  - dense to Model 2, at each k in {1, 3, 5, 10}: plain dense retrieval, no reranking.
  - dense + soft rerank -> Model 2, at each k in {1, 3, 5, 10}: the best-performing pipeline 
    selected from the earlier in-domain experiments.

Running BOTH plain dense and dense+rerank at matched depths is what lets the case study answer
its own question ("does stance reranking filter irrelevant evidence out of domain?"). Without the
plain-dense condition, an irrelevant document in the reranked output could not be attributed to
retrieval versus reranking. This mirrors Steps 5 and 6, which also ran dense and dense+rerank
side by side. Note that at k equal to the rerank pool size, reranking can only REORDER the pool,
not filter documents out of the final set; the effective pool size is recorded per k so this is
explicit in the output.

THE FIVE PRE-COMMITTED, DIRECTIONAL HYPOTHESES (from collection_guide.md), which the write-up
tests qualitatively:
  H1. Retrieval-related failures are expected to involve topically irrelevant or merely related
      evidence MORE for far-fit claims than for near-fit claims (Step 7).
  H2. The stance reranker is expected to still promote confidently mis-stanced documents (Step 4),
      so it is not expected to rescue out-of-domain retrieval and may reorder it unhelpfully.
  H3. On claims whose true label is CONTRADICT, wrong predictions are expected to arrive with
      relatively high max-softmax probability, mirroring the inversion found in Step 8.
  H4. Retrieval is hypothesised to be no better than, and potentially worse than, 
      the no-retrieval baseline on these out-of-domain claims (Step 9), so the
      no-retrieval baseline is run alongside rather than assumed to be worse.
  H5. Confident wrong predictions are expected to persist (Steps 7 and 8), so errors are expected
      to arrive mostly at high max-softmax probability rather than low.

"Confidence" throughout means the maximum softmax probability over the three classes, the same
definition used in Step 8. No separate confidence-scoring model is trained.

Because SciFact-Open has no ground-truth evidence for these claims, there is no recall or
gold-document metric here. The output is per-claim records: the prediction, the max-softmax
probability, the retrieved documents (so relevance can be judged by hand), and whether the
prediction matched the manually assigned reference label. Accuracy is an orienting count over about thirty
deliberately chosen claims, NOT an estimate of real social-media performance. The analysis in
step10_results.md is qualitative and organised around the five hypotheses, using the pre-committed
annotation protocol in results_annotation_guide.md.

Usage structure:
  python realworld/seafood_claims.py \
    --claims_csv realworld/seafood_claims.csv \
    --model1_path models/saved_models/baseline_scifact \
    --model2_path models/saved_models/evidence_scifact \
    --out_dir results/step10_realworld \
    --k_values 1 3 5 10
"""

#importing os for paths and directory creation
import os

#importing sys so the project modules import correctly when run from the repo root
import sys

#adding the project root to the path so `models` and `data` import like they do elsewhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing csv to read the hand-labelled claim set
import csv

#importing json to save the per-claim records and the run summary
import json

#importing argparse for the command line interface
import argparse

#importing hashlib to hash the claims CSV, so the run records exactly which claim file it used
import hashlib

#importing datetime to timestamp the run for provenance
from datetime import datetime, timezone

#importing torch for device selection
import torch

#importing the tokenizer and model classes, same as the pipeline uses
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#reusing the EXACT pipeline components from Step 5, so the case study runs the same code path as
#the rest of the thesis rather than a re-implementation that could quietly differ. Both the plain
#dense and the dense+rerank runners are imported, so the two retrieval conditions can be compared
#at matched depths (this is what lets us isolate the reranker's effect out of domain).
from models.pipeline import (
    run_no_retrieval_pipeline,
    run_dense_pipeline,
    run_dense_reranked_pipeline,
    LABEL_TO_ID,
)

#reusing the retriever and reranker classes the pipeline itself builds
from models.retrieval import DenseRetriever
from models.reranker import StanceReranker

#reusing the SciFact-Open loader so the corpus is identical to Steps 3, 5, 6, 8 and 9
from data.utils import load_scifact_open

# ---------------------------------------------------------------------------
# Small provenance helper
# ---------------------------------------------------------------------------

def sha256_file(path):
    """Return the SHA-256 hex digest of a file, so the run records exactly which claim CSV it
    used. Hand-labelled CSVs can change after a run; the hash makes that detectable."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Loading the hand-labelled claims
# ---------------------------------------------------------------------------

#the labels the classifier and the true labels both use
VALID_LABELS = set(LABEL_TO_ID.keys())
VALID_FITS = {"near", "far"}

def load_claims(claims_csv):
    """
    Read the seafood claim set and shape it into the same dict format the pipeline expects for a
    dataset split: a list of {id, claim, label, evidence_doc_ids}. The extra case-study fields
    (original_quote, source_platform, corpus_fit, label_basis, label_note) are carried alongside
    in a separate lookup so they can be attached back to each record for the qualitative write-up
    without changing the pipeline's input shape.
    """
    if not os.path.exists(claims_csv):
        raise FileNotFoundError(f"Claims CSV not found: {claims_csv}")

    claims_data = []
    meta_by_id = {}
    with open(claims_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"claim_id", "claim_text", "true_label", "corpus_fit"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise KeyError(f"Claims CSV is missing required columns: {sorted(missing_cols)}")

        for row in reader:
            cid = row["claim_id"].strip()
            #failing loudly on empty required fields, so the message points at the actual CSV row
            #rather than surfacing as a confusing error deep inside the pipeline
            if not cid:
                raise ValueError("A CSV row has an empty claim_id.")
            claim_text = row["claim_text"].strip()
            if not claim_text:
                raise ValueError(f"Claim {cid} has an empty claim_text.")

            label = row["true_label"].strip()
            if label not in VALID_LABELS:
                raise ValueError(
                    f"Claim {cid} has true_label '{label}', expected one of {sorted(VALID_LABELS)}.")

            corpus_fit = row.get("corpus_fit", "").strip()
            if corpus_fit not in VALID_FITS:
                raise ValueError(
                    f"Claim {cid} has corpus_fit '{corpus_fit}', expected one of {sorted(VALID_FITS)}.")

            claims_data.append({
                "id": cid,
                "claim": claim_text,
                "label": label,
                #these claims have no annotated evidence in the corpus, by design
                "evidence_doc_ids": [],
            })
            #keeping the provenance and labelling fields for the write-up
            meta_by_id[cid] = {
                "original_quote": row.get("original_quote", "").strip(),
                "source_platform": row.get("source_platform", "").strip(),
                "approx_date": row.get("approx_date", "").strip(),
                "label_basis": row.get("label_basis", "").strip(),
                "corpus_fit": corpus_fit,
                "label_note": row.get("label_note", "").strip(),
            }

    #a unique-id check, the same discipline the other steps use on their inputs
    ids = [c["id"] for c in claims_data]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise ValueError(f"Duplicate claim ids in the CSV: {dupes}")

    return claims_data, meta_by_id

# ---------------------------------------------------------------------------
# Attaching case-study metadata to the pipeline's records
# ---------------------------------------------------------------------------

def enrich_records(records, meta_by_id, condition, k=None):
    """
    Add the case-study fields (source, corpus_fit, the exact quote, and so on) to each record the
    pipeline produced, plus the retrieval depth this run used and an explicit condition tag. This
    keeps the pipeline's record format untouched while giving the qualitative analysis everything
    it needs in one place. The explicit `condition` tag is set here (rather than relying on the
    pipeline's own tag) so plain-dense and reranked records are unambiguous to filter later.
    """
    enriched = []
    for rec in records:
        meta = meta_by_id.get(rec["claim_id"], {})
        row = dict(rec)
        row.update({
            "condition": condition,
            "k": k,
            "original_quote": meta.get("original_quote", ""),
            "source_platform": meta.get("source_platform", ""),
            "approx_date": meta.get("approx_date", ""),
            "label_basis": meta.get("label_basis", ""),
            "corpus_fit": meta.get("corpus_fit", ""),
            "label_note": meta.get("label_note", ""),
        })
        enriched.append(row)
    return enriched

# ---------------------------------------------------------------------------
# A light qualitative summary to orient the write-up
# ---------------------------------------------------------------------------

def _accuracy(records):
    n = len(records)
    correct = sum(1 for r in records if r["correct"])
    return {"n": n, "correct": correct,
            "accuracy_pct": round(100 * correct / n, 1) if n else None}

def _by_corpus_fit(records):
    return {fit: _accuracy([r for r in records if r.get("corpus_fit") == fit])
            for fit in ("near", "far")}

def _confidence_by_true_label(records):
    """For each true label, report accuracy and the mean max-softmax probability, including the
    mean on WRONG predictions (high probability while wrong is the concerning inversion). Split by
    TRUE label, matching the Step 8 definition."""
    out = {}
    for label in sorted(VALID_LABELS):
        subset = [r for r in records if r["true_label"] == label]
        if not subset:
            continue
        confs = [r["confidence"] for r in subset]
        wrong_confs = [r["confidence"] for r in subset if not r["correct"]]
        out[label] = {
            "n": len(subset),
            "accuracy_pct": round(100 * sum(1 for r in subset if r["correct"]) / len(subset), 1),
            "mean_max_softmax": round(sum(confs) / len(confs), 4),
            "n_wrong": len(wrong_confs),
            "mean_max_softmax_when_wrong": (round(sum(wrong_confs) / len(wrong_confs), 4)
                                            if wrong_confs else None),
        }
    return out

def _condition_block(records):
    return {
        "overall": _accuracy(records),
        "by_corpus_fit": _by_corpus_fit(records),
        "confidence_by_true_label": _confidence_by_true_label(records),
    }

def summarise(no_ret_records, dense_by_k, reranked_by_k, k_values, effective_pool_by_k):
    """
    Produce a small summary that orients the manual analysis. It deliberately does NOT try to
    compute failure categories automatically: the taxonomy needs human judgement of whether the
    retrieved documents were relevant, which is the whole point of the case study. What it does
    report is accuracy per condition, accuracy split by the pre-registered corpus_fit (near vs
    far), and max-softmax probability split by true label (so the CONTRADICT inversion can be
    checked). Accuracy is an orienting count over about thirty claims, not a performance estimate.
    """
    summary = {
        "n_claims": len(no_ret_records),
        "no_retrieval": _condition_block(no_ret_records),
        "dense_by_k": {k: _condition_block(dense_by_k[k]) for k in k_values},
        "dense_reranked_by_k": {k: _condition_block(reranked_by_k[k]) for k in k_values},
        "effective_rerank_pool_by_k": {k: effective_pool_by_k[k] for k in k_values},
    }

    #the centrepiece cross-check: does any retrieval depth beat retrieving nothing, out of domain?
    #Step 9 predicts retrieval is counterproductive (H4). This reports the HIGHEST OBSERVED
    #retrieval accuracy across the tested depths, it is a description of what happened on these
    #30 claims, NOT a selected or validated optimal depth.
    no_ret_acc = summary["no_retrieval"]["overall"]["accuracy_pct"]

    def highest_observed(block_by_k):
        vals = [block_by_k[k]["overall"]["accuracy_pct"] for k in k_values
                if block_by_k[k]["overall"]["accuracy_pct"] is not None]
        return max(vals) if vals else None

    highest_dense = highest_observed(summary["dense_by_k"])
    highest_rerank = highest_observed(summary["dense_reranked_by_k"])
    highest_any = max([v for v in (highest_dense, highest_rerank) if v is not None], default=None)

    #the PRIMARY H4 test is at the anchor depth k = 3 (the SciFact-Open optimum used throughout
    #the project), so no-retrieval is compared against dense and dense+rerank AT k = 3. The
    #highest-observed figures across all depths are kept only as a clearly-labelled exploratory
    #description, since a maximum across four depths is a post-run best case, not a hypothesis test.
    #k=3 is guaranteed present by the argument validation in main(), so the anchor is fixed
    anchor_k = 3
    dense_at_anchor = (summary["dense_by_k"][anchor_k]["overall"]["accuracy_pct"]
                       if anchor_k in summary["dense_by_k"] else None)
    rerank_at_anchor = (summary["dense_reranked_by_k"][anchor_k]["overall"]["accuracy_pct"]
                        if anchor_k in summary["dense_reranked_by_k"] else None)

    def _beats(x):
        return (x > no_ret_acc) if (x is not None and no_ret_acc is not None) else None

    summary["retrieval_vs_no_retrieval"] = {
        "anchor_k": anchor_k,
        "no_retrieval_accuracy_pct": no_ret_acc,
        #primary comparison, at the anchor depth
        "dense_accuracy_at_anchor_k": dense_at_anchor,
        "reranked_accuracy_at_anchor_k": rerank_at_anchor,
        "dense_beats_no_retrieval_at_anchor_k": _beats(dense_at_anchor),
        "reranked_beats_no_retrieval_at_anchor_k": _beats(rerank_at_anchor),
        #exploratory only: highest across all tested depths (NOT a validated optimum)
        "highest_observed_dense_accuracy_across_tested_k": highest_dense,
        "highest_observed_reranked_accuracy_across_tested_k": highest_rerank,
        "any_retrieval_beats_no_retrieval_exploratory": _beats(highest_any),
        "note": ("H4 (from Step 9): retrieval is hypothesised to be no better than, and possibly "
                 "worse than, no retrieval on these out-of-domain claims. The PRIMARY test is at "
                 "the anchor depth k = 3. The 'highest observed' figures across all depths are "
                 "exploratory descriptions of what happened on these ~30 claims, NOT selected or "
                 "validated optimal depths, and accuracy here is descriptive, not a significance "
                 "test."),
    }
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 10 real-world seafood/sustainability case study")
    parser.add_argument("--claims_csv", default="realworld/seafood_claims.csv",
                        help="The hand-labelled seafood claim set.")
    parser.add_argument("--model1_path", required=True,
                        help="Model 1 (claim-only), used for the no-retrieval condition.")
    parser.add_argument("--model2_path", required=True,
                        help="Model 2 (claim+evidence), used for the retrieval conditions.")
    parser.add_argument("--out_dir", default="results/step10_realworld")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Retrieval depths to sweep for the dense and dense+rerank conditions.")
    parser.add_argument("--rerank_pool_size", type=int, default=10)
    parser.add_argument("--neutral_threshold", type=float, default=0.5)
    args = parser.parse_args()

    #validating the sweep arguments up front, so a bad invocation fails immediately rather than
    #part way through a 40-minute run
    if any(k <= 0 for k in args.k_values):
        parser.error("--k_values must contain positive integers.")
    if len(args.k_values) != len(set(args.k_values)):
        parser.error("--k_values must not contain duplicates.")
    if args.rerank_pool_size <= 0:
        parser.error("--rerank_pool_size must be positive.")
    if 3 not in args.k_values:
        parser.error("--k_values must include 3, because k=3 is the pre-committed anchor depth "
                     "for the Step 10 analysis.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    #loading the claim set
    claims_data, meta_by_id = load_claims(args.claims_csv)
    print(f"Loaded {len(claims_data)} seafood/sustainability claims")

    #loading the SciFact-Open corpus, identical to the corpus used in Steps 3, 5, 6, 8 and 9
    print("Loading SciFact-Open corpus (this is the same corpus as the earlier steps)...")
    _open_claims, corpus = load_scifact_open(corpus_file="full")
    print(f"Corpus: {len(corpus)} documents")

    #loading both classifiers, exactly as the pipeline does
    print("Loading Model 1 (claim-only) and Model 2 (claim+evidence)...")
    model1_tokenizer = AutoTokenizer.from_pretrained(args.model1_path)
    model1 = AutoModelForSequenceClassification.from_pretrained(args.model1_path).to(device)
    model2_tokenizer = AutoTokenizer.from_pretrained(args.model2_path)
    model2 = AutoModelForSequenceClassification.from_pretrained(args.model2_path).to(device)

    #building the dense retriever and stance reranker ONCE (the retriever encodes the corpus,
    #which is the slow part), then reusing them across every k and both retrieval conditions
    print("Building dense retriever (one-time corpus encoding, ~40 min on GPU)...")
    dense_retriever = DenseRetriever(corpus)
    print("Building stance reranker...")
    stance_reranker = StanceReranker(device=device)

    #condition 1: no retrieval, Model 1. this does not depend on k, so it runs once
    print("\n=== No retrieval (Model 1) ===")
    _m, no_ret_records = run_no_retrieval_pipeline(
        claims_data, model1, model1_tokenizer, device, "scifact_open")
    no_ret_records = enrich_records(no_ret_records, meta_by_id, condition="no_retrieval", k=None)

    #condition 2: plain dense -> Model 2, swept across depths (the diagnostic baseline that lets us
    #isolate the reranker's effect). condition 3: dense + soft rerank -> Model 2, same depths.
    dense_by_k = {}
    reranked_by_k = {}
    effective_pool_by_k = {}
    for k in args.k_values:
        #the reranker needs a pool at least as large as k; record the effective pool so that the
        #"at k = pool size, reranking only reorders" point is explicit in the output
        effective_pool = max(args.rerank_pool_size, k)
        effective_pool_by_k[k] = effective_pool

        print(f"\n=== Dense -> Model 2, k={k} ===")
        _m, d_recs = run_dense_pipeline(
            claims_data, dense_retriever, model2, model2_tokenizer,
            device, "scifact_open", top_k=k)
        dense_by_k[k] = enrich_records(d_recs, meta_by_id, condition="dense", k=k)

        print(f"=== Dense + soft rerank -> Model 2, k={k} (pool={effective_pool}) ===")
        _m, r_recs = run_dense_reranked_pipeline(
            claims_data, dense_retriever, stance_reranker, model2, model2_tokenizer,
            device, "scifact_open", top_k=k,
            rerank_pool_size=effective_pool,
            neutral_threshold=args.neutral_threshold)
        reranked_by_k[k] = enrich_records(r_recs, meta_by_id, condition="dense_reranked", k=k)

    #saving the per-claim records: one combined file, since the write-up reads across conditions
    all_records = list(no_ret_records)
    for k in args.k_values:
        all_records.extend(dense_by_k[k])
        all_records.extend(reranked_by_k[k])
    records_path = os.path.join(args.out_dir, "step10_records.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} per-claim records to {records_path}")

    #saving the orienting summary, with provenance so the run is reproducible
    summary = summarise(no_ret_records, dense_by_k, reranked_by_k, args.k_values, effective_pool_by_k)
    summary["provenance"] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "model1_path": args.model1_path,
        "model2_path": args.model2_path,
        "corpus": "scifact_open",
        "corpus_doc_count": len(corpus),
        "claims_csv": args.claims_csv,
        "claims_csv_sha256": sha256_file(args.claims_csv),
        "k_values": args.k_values,
        "rerank_pool_size_base": args.rerank_pool_size,
        "effective_rerank_pool_by_k": effective_pool_by_k,
        "neutral_threshold": args.neutral_threshold,
        "conditions": ["no_retrieval (Model 1)",
                       "dense -> Model 2 (swept over k)",
                       "dense+soft_rerank -> Model 2 (swept over k)"],
        "confidence_definition": "max softmax probability over the 3 classes (same as Step 8)",
    }
    summary_path = os.path.join(args.out_dir, "step10_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {summary_path}")

    #a short printed read-out to sanity-check before the manual analysis
    print("\n" + "=" * 70)
    print("Quick read-out (full analysis is manual, in step10_results.md)")
    print("=" * 70)
    nr = summary["no_retrieval"]["overall"]
    print(f"No retrieval accuracy: {nr['correct']}/{nr['n']} ({nr['accuracy_pct']}%)")
    for k in args.k_values:
        do = summary["dense_by_k"][k]["overall"]
        ro = summary["dense_reranked_by_k"][k]["overall"]
        print(f"  k={k:<2}  dense: {do['correct']}/{do['n']} ({do['accuracy_pct']}%)   "
              f"dense+rerank: {ro['correct']}/{ro['n']} ({ro['accuracy_pct']}%)   "
              f"[pool={effective_pool_by_k[k]}]")
    rv = summary["retrieval_vs_no_retrieval"]
    print(f"\nH4 comparison at anchor k={rv['anchor_k']}: "
          f"dense beats no retrieval? {rv['dense_beats_no_retrieval_at_anchor_k']}; "
          f"dense+rerank beats no retrieval? {rv['reranked_beats_no_retrieval_at_anchor_k']} "
          f"(dense {rv['dense_accuracy_at_anchor_k']}%, "
          f"dense+rerank {rv['reranked_accuracy_at_anchor_k']}%, "
          f"no retrieval {rv['no_retrieval_accuracy_pct']}%)")
    print("Exploratory across all tested depths: any retrieval beats no retrieval? "
          f"{rv['any_retrieval_beats_no_retrieval_exploratory']} "
          f"(highest dense {rv['highest_observed_dense_accuracy_across_tested_k']}%, "
          f"highest rerank {rv['highest_observed_reranked_accuracy_across_tested_k']}%)")
    print("\nCONTRADICT max-softmax when wrong (H3): no-retrieval condition:")
    cb = summary["no_retrieval"]["confidence_by_true_label"].get("CONTRADICT")
    if cb:
        print(f"  CONTRADICT: {cb['n']} claims, accuracy {cb['accuracy_pct']}%, "
              f"mean max-softmax when wrong {cb['mean_max_softmax_when_wrong']}")


if __name__ == "__main__":
    main()