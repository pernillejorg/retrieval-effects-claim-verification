"""
Counting how many whole retrieved documents fit inside the 512-token window.

The Part A design concatenates retrieved documents whole, in rank order, until the
limit is reached. This script reproduces that packing on the saved records and
reports how many documents actually fit per claim, so the saturation claim rests on
a count rather than on dividing the median document length into the budget.

Usage:
    python analysis/tools/count_docs_that_fit.py
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "roberta-base"
MAX_LENGTH = 512
#reserving the four special tokens RoBERTa adds around a sentence pair
SPECIAL_TOKENS = 4


def count_fitting(records_path, condition, tokenizer):
    """Returning, per claim, how many whole documents fit after the claim."""
    with open(records_path) as f:
        records = json.load(f)

    counts = []
    for record in records:
        if record.get("condition") != condition:
            continue

        claim_length = len(tokenizer.encode(record["claim"], add_special_tokens=False))
        budget = MAX_LENGTH - claim_length - SPECIAL_TOKENS

        used, fitted = 0, 0
        for document in record.get("retrieved_docs", []):
            length = len(tokenizer.encode(document["text"], add_special_tokens=False))
            #stopping at the first document that does not fit whole, as Part A does
            if used + length > budget:
                break
            used += length
            fitted += 1
        counts.append(fitted)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_dir", default="results/step5_pipeline")
    parser.add_argument("--condition", default="dense_roberta")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for dataset in ["scifact", "scifact_open"]:
        path = os.path.join(
            args.records_dir, f"step5_records_{dataset}_k3_thr0_5.json"
        )
        if not os.path.exists(path):
            print(f"{dataset}: no records at {path}")
            continue

        counts = count_fitting(path, args.condition, tokenizer)
        distribution = Counter(counts)
        print(
            f"{dataset}: n={len(counts)}  mean={np.mean(counts):.2f}  "
            f"median={int(np.median(counts))}  "
            f"distribution={dict(sorted(distribution.items()))}"
        )


if __name__ == "__main__":
    main()