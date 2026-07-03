"""
utils.py -- shared data loading and normalisation for both datasets

Provides a single, consistent format for all downstream components:
  claims  -> list of dicts: {id, claim, label, evidence_doc_ids}
  corpus  -> dict: {doc_id: abstract_text}

Labels are normalised to a shared 3-class scheme:
  SUPPORT      -- claim is supported by evidence
  CONTRADICT   -- claim is contradicted by evidence
  NEI          -- not enough info / no stance taken

SciFact:
  gold_label values: SUPPORT, CONTRADICT, NOT_ENOUGH_INFO -> mapped directly
  corpus keyed by integer doc ids (stored as strings here for consistency)

SciFact-Open:
  Test-only collection (no train/val split). Evidence is a dict keyed by doc_id,
  each with a SUPPORT/CONTRADICT label (no NEI in raw data). We collapse to one
  claim-level label (any SUPPORT -> SUPPORT, else CONTRADICT) and map claims with
  no evidence to NEI. Read from cached .jsonl files, not HuggingFace.

Usage:
  from data.utils import load_scifact, load_scifact_open
  claims, corpus = load_scifact(split="train")
  claims, corpus = load_scifact_open()
"""

import os
import random
from datasets import load_dataset

SCIFACT_CACHE = os.path.join(os.path.dirname(__file__), "scifact", "cache")

# Unified label names used everywhere downstream
LABEL_SUPPORT    = "SUPPORT"
LABEL_CONTRADICT = "CONTRADICT"
LABEL_NEI        = "NEI"

# SciFact raw label -> unified label
SCIFACT_LABEL_MAP = {
    "SUPPORT":          LABEL_SUPPORT,
    "CONTRADICT":       LABEL_CONTRADICT,
    "NOT_ENOUGH_INFO":  LABEL_NEI,
    #When some claims have empty string label, then treating as NEI
    "":                 LABEL_NEI,  
}

# ---------------------------------------------------------------------------
# SciFact
# ---------------------------------------------------------------------------

def load_scifact(split="train"):
    """
    Returns:
        claims: list of dicts with keys:
            id              (str)   unique claim identifier
            claim           (str)   the claim text
            label           (str)   SUPPORT | CONTRADICT | NEI
            evidence_doc_ids(list)  doc ids of annotated evidence abstracts
                                    (empty list for NEI claims)
        corpus: dict {doc_id (str): abstract_text (str)}
    """
    assert split in ("train", "validation"), f"SciFact has splits: train, validation. Got: {split}"

    dataset = load_dataset("allenai/scifact", "claims", cache_dir=SCIFACT_CACHE)
    corpus_ds = load_dataset("allenai/scifact", "corpus", cache_dir=SCIFACT_CACHE)

    # Build corpus dict: doc_id (str) -> abstract text
    corpus = {}
    for row in corpus_ds["train"]:
        doc_id = str(row["doc_id"])
        # Corpus rows have title + abstract; join for full context
        abstract = row.get("abstract", "")
        if isinstance(abstract, list):
            abstract = " ".join(abstract)
        title = row.get("title", "")
        corpus[doc_id] = f"{title} {abstract}".strip()

    # Build claims list
    claims = []
    for row in dataset[split]:
        evidence_doc_ids = []
        if row.get("cited_doc_ids"):
            evidence_doc_ids = [str(d) for d in row["cited_doc_ids"]]
        claims.append({
            "id":               str(row["id"]),
            "claim":            row["claim"],
            "label":            SCIFACT_LABEL_MAP.get(row["evidence_label"], LABEL_NEI),
            "evidence_doc_ids": evidence_doc_ids,
        })

    return claims, corpus


# ---------------------------------------------------------------------------
# SciFact-Open
# ---------------------------------------------------------------------------

def load_scifact_open():
    """
    SciFact-Open (Wadden et al., 2022) -- TEST-ONLY collection, no train/val split.
    Distributed as .jsonl files (see data/scifact_open/download.py). We read the
    cached files rather than HuggingFace, since SciFact-Open is not on the Hub.

    Label handling (documented in the thesis):
      - SciFact-Open evidence is a dict keyed by doc_id, each with a per-document
        SUPPORT/CONTRADICT label. There is NO NEI label in the raw data.
      - We collapse per-document labels to one claim-level label:
          * any SUPPORT  -> SUPPORT
          * else any CONTRADICT -> CONTRADICT
      - Claims with NO evidence at all are mapped to NEI (they are unverifiable
        against the corpus, which is conceptually SciFact's NEI case). This keeps
        all 279 claims and aligns SciFact-Open with SciFact's 3-class scheme.

    Returns:
        claims: list of dicts with keys:
            id              (str)
            claim           (str)
            label           (str)   SUPPORT | CONTRADICT | NEI
            evidence_doc_ids(list)  doc ids with evidence (empty list for NEI)
        corpus: dict {doc_id (str): "title abstract"}
    """
    import json

    cache_dir = os.path.join(os.path.dirname(__file__), "scifact_open", "cache")
    claims_path = os.path.join(cache_dir, "claims.jsonl")
    corpus_path = os.path.join(cache_dir, "corpus.jsonl")

    #helper: read a .jsonl file into a list of dicts
    def _read_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    #building the corpus dict: doc_id (str) -> "title abstract"
    corpus = {}
    for row in _read_jsonl(corpus_path):
        doc_id = str(row["doc_id"])
        abstract = row.get("abstract", "")
        #abstracts may be stored as a list of sentences; join them if so
        if isinstance(abstract, list):
            abstract = " ".join(abstract)
        title = row.get("title", "")
        corpus[doc_id] = f"{title} {abstract}".strip()

    #counters for the conflict report (claims with both SUPPORT and CONTRADICT evidence)
    conflict_count = 0

    #building the claims list with collapsed labels
    claims = []
    for row in _read_jsonl(claims_path):
        evidence = row.get("evidence", {}) or {}

        #collecting all per-document labels and the doc ids that carry evidence
        doc_labels = [ev["label"] for ev in evidence.values() if isinstance(ev, dict)]
        evidence_doc_ids = [str(doc_id) for doc_id in evidence.keys()]

        #collapsing per-document labels into one claim-level label
        if not doc_labels:
            #no evidence at all -> unverifiable -> NEI
            label = LABEL_NEI
        else:
            #noting conflicts for the thesis write-up (does not change the rule)
            if LABEL_SUPPORT in doc_labels and LABEL_CONTRADICT in doc_labels:
                conflict_count += 1
            #any SUPPORT wins, else CONTRADICT
            if LABEL_SUPPORT in doc_labels:
                label = LABEL_SUPPORT
            else:
                label = LABEL_CONTRADICT

        claims.append({
            "id":               str(row["id"]),
            "claim":            row["claim"],
            "label":            label,
            "evidence_doc_ids": evidence_doc_ids,
        })

    #reporting the conflict count so it can be cited honestly in the thesis
    print(f"[load_scifact_open] {len(claims)} claims loaded "
          f"({conflict_count} had conflicting SUPPORT/CONTRADICT evidence, "
          f"resolved to SUPPORT).")

    return claims, corpus

# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    print("=== SciFact ===")
    claims, corpus = load_scifact(split="train")
    print(f"  claims : {len(claims)}")
    print(f"  corpus : {len(corpus)} abstracts")
    labels = Counter(c["label"] for c in claims)
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    print(f"  sample claim: {claims[0]['claim']}")
    print(f"  sample label: {claims[0]['label']}")

    print()
    print("=== SciFact-Open (test-only) ===")
    claims, corpus = load_scifact_open()
    print(f"  claims : {len(claims)}")
    print(f"  corpus : {len(corpus)} abstracts")
    labels = Counter(c["label"] for c in claims)
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    print(f"  sample claim: {claims[0]['claim'][:100]}...")
