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

SciClaimHunt:
  Type values: positive -> SUPPORT, negative -> CONTRADICT
  No NEI class exists in SciClaimHunt; all claims are binary
  Evidence is a pre-extracted passage string, not a doc id reference
  We construct a synthetic corpus from the evidence strings, keyed by row index
  The dataset has one split (train); we split 80/10/10 for train/val/test

Usage:
  from data.utils import load_scifact, load_sciclaimhunt
  claims, corpus = load_scifact(split="train")
  claims, corpus = load_sciclaimhunt(split="train")
"""

import os
import random
from datasets import load_dataset

SCIFACT_CACHE = os.path.join(os.path.dirname(__file__), "scifact", "cache")
SCICLAIMHUNT_CACHE = os.path.join(os.path.dirname(__file__), "sciclaimhunt", "cache")

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

# SciClaimHunt raw label -> unified label
SCICLAIMHUNT_LABEL_MAP = {
    "positive": LABEL_SUPPORT,
    "negative": LABEL_CONTRADICT,
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
# SciClaimHunt
# ---------------------------------------------------------------------------

def load_sciclaimhunt(split="train", seed=42):
    """
    SciClaimHunt has one split (train, 110k rows).
    We manually split 80/10/10 -> train/val/test using a fixed seed.

    Returns:
        claims: list of dicts with keys:
            id              (str)   row index as string
            claim           (str)   the claim text
            label           (str)   SUPPORT | CONTRADICT
            evidence_doc_ids(list)  single synthetic doc id ['sch_{id}']
        corpus: dict {'sch_{id}': evidence_text}
            Keyed by 'sch_{row_index}'; value is the Evidence passage string
    """
    assert split in ("train", "val", "test"), f"SciClaimHunt splits: train, val, test. Got: {split}"

    raw = load_dataset("AnshulS/dataset_for_scicllaimhunt", cache_dir=SCICLAIMHUNT_CACHE)
    all_rows = list(raw["train"])

    # Reproducible 80/10/10 split
    rng = random.Random(seed)
    indices = list(range(len(all_rows)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    split_indices = {
        "train": indices[:n_train],
        "val":   indices[n_train:n_train + n_val],
        "test":  indices[n_train + n_val:],
    }

    selected = [all_rows[i] for i in split_indices[split]]

    corpus = {}
    claims = []
    for row in selected:
        row_id = str(row["Unnamed: 0"])
        doc_id = f"sch_{row_id}"
        corpus[doc_id] = row["Evidence"]
        claims.append({
            "id":               row_id,
            "claim":            row["Claim"],
            "label":            SCICLAIMHUNT_LABEL_MAP[row["Type"]],
            "evidence_doc_ids": [doc_id],
        })

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
    print("=== SciClaimHunt (train split) ===")
    claims, corpus = load_sciclaimhunt(split="train")
    print(f"  claims : {len(claims)}")
    print(f"  corpus : {len(corpus)} evidence passages")
    labels = Counter(c["label"] for c in claims)
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    print(f"  sample claim: {claims[0]['claim'][:100]}...")
