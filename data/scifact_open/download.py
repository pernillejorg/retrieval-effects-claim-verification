"""
download.py -- SciFact-Open dataset verification/loader.

SciFact-Open (Wadden et al., 2022) is NOT on HuggingFace. The data files are
obtained via the authors' script (github.com/dwadden/scifact-open, script/get_data.sh)
and placed in this folder's cache/. See the project README for the one-time
download commands.

This script verifies the cached files are present and prints a summary of the
real schema. Run once after downloading: python data/scifact_open/download.py
"""

import os
import json
from collections import Counter

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

#the files expected after running the authors' get_data.sh
EXPECTED_FILES = ["claims.jsonl", "corpus.jsonl", "corpus_candidates.jsonl"]


def _read_jsonl(path):
    #reading a .jsonl file (one JSON object per line) into a list of dicts
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verify_scifact_open():
    print("Verifying SciFact-Open cache...")

    #checking every expected file is present before going further
    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(CACHE_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing SciFact-Open files in {CACHE_DIR}: {missing}\n"
            "Download them first with the authors' script "
            "(github.com/dwadden/scifact-open, bash script/get_data.sh), "
            "then copy the .jsonl files into cache/. See project README."
        )

    #loading and summarising the claims file
    claims = _read_jsonl(os.path.join(CACHE_DIR, "claims.jsonl"))
    print(f"  claims.jsonl           : {len(claims)} claims")

    #peeking at one claim to see the REAL schema (keys may differ from assumptions)
    print("\nSample claim (claims[0]):")
    sample = claims[0]
    print(f"  keys available : {list(sample.keys())}")
    print(f"  id             : {sample.get('id')}")
    print(f"  claim          : {sample.get('claim')}")
    print(f"  evidence (raw) : {sample.get('evidence')}")

    #counting the per-document evidence labels across all claims
    #(SciFact-Open evidence is expected to be a dict keyed by doc_id, each with a label)
    label_counter = Counter()
    claims_with_evidence = 0
    for c in claims:
        evidence = c.get("evidence", {})
        if evidence:
            claims_with_evidence += 1
            for doc_id, ev in evidence.items():
                #ev may be a dict with a "label" key -- guard in case the schema differs
                if isinstance(ev, dict):
                    label_counter[ev.get("label")] += 1

    print(f"\nclaims WITH evidence   : {claims_with_evidence}")
    print(f"claims WITHOUT evidence: {len(claims) - claims_with_evidence}")
    print("\nPer-document evidence label distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    #reporting corpus sizes by counting lines (avoids loading the huge file into memory)
    corpus_path = os.path.join(CACHE_DIR, "corpus.jsonl")
    candidates_path = os.path.join(CACHE_DIR, "corpus_candidates.jsonl")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_count = sum(1 for _ in f)
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_count = sum(1 for _ in f)
    print(f"\ncorpus.jsonl           : {corpus_count} abstracts (full corpus)")
    print(f"corpus_candidates.jsonl: {candidates_count} abstracts (retrieved subset)")

    #peeking at one corpus document to see how abstracts are structured
    with open(corpus_path, "r", encoding="utf-8") as f:
        first_doc = json.loads(f.readline())
    print("\nSample corpus doc keys:", list(first_doc.keys()))

    print("\nSciFact-Open verification complete. Cached at:", CACHE_DIR)
    return claims

if __name__ == "__main__":
    verify_scifact_open()