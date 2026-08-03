"""
Measuring the token length of retrieved documents.

Relevance to the dissertation paper:
Reports the median and interquartile range of retrieved document length, measured
with the same RoBERTa tokenizer the classifier uses. The numbers this produces are
what the dissertation cites when explaining context saturation: with a 512-token
input and the claim taking part of it, only about one and a half abstracts fit, so
under whole-document concatenation the third document at k = 3 never reaches the
classifier.

Measuring in tokens rather than characters matters here, because the 512 limit is a
token limit; a character count would not tell you how many documents actually fit.

Usage:
    python analysis/tools/measure_doc_lengths.py \
        --records_dir results/step5_pipeline \
        --condition dense_roberta
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

#matching the classifier so the counts reflect the tokenizer that enforces the limit
TOKENIZER_NAME = "roberta-base"


def measure(records_path, condition, tokenizer):
    """Returning the token length of every retrieved document in one condition."""
    with open(records_path) as f:
        records = json.load(f)

    lengths = []
    for record in records:
        if record.get("condition") != condition:
            continue
        for document in record.get("retrieved_docs", []):
            #counting content tokens only, since the special tokens belong to the
            #assembled input rather than to the document itself
            token_ids = tokenizer.encode(document["text"], add_special_tokens=False)
            lengths.append(len(token_ids))
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_dir", default="results/step5_pipeline")
    parser.add_argument("--condition", default="dense_roberta")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", default="0_5")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for dataset in ["scifact", "scifact_open"]:
        filename = f"step5_records_{dataset}_k{args.k}_thr{args.threshold}.json"
        path = os.path.join(args.records_dir, filename)

        if not os.path.exists(path):
            print(f"{dataset}: no records at {path}")
            continue

        lengths = measure(path, args.condition, tokenizer)
        if not lengths:
            print(f"{dataset}: no documents found for condition {args.condition}")
            continue

        print(
            f"{dataset}: n={len(lengths)}  "
            f"median={int(np.median(lengths))}  "
            f"IQR={int(np.percentile(lengths, 25))}-{int(np.percentile(lengths, 75))}  "
            f"max={max(lengths)}"
        )


if __name__ == "__main__":
    main()
