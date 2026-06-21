"""
download.py -- SciClaimHunt dataset loader
Loads AnshulS/dataset_for_scicllaimhunt from Hugging Face and caches locally.
Source paper: arXiv:2502.10003 (Kumar et al., 2025)
Dataset: AnshulS/dataset_for_scicllaimhunt (CC-BY-4.0)

Note on structure vs SciFact:
  - Labels are binary: 'positive' (supported) / 'negative' (refuted)
    SciFact uses 3-way: SUPPORT / CONTRADICT / NOT_ENOUGH_INFO
    NOT_ENOUGH_INFO claims have no equivalent here; this is handled in utils.py
  - Evidence column contains extracted passages; research_paper_full contains full text
  - Only one split (train, 110k rows); we manually split into train/val/test in utils.py
  - 3.18 GB total; first run will take a while to download

Run once before any experiments: python data/sciclaimhunt/download.py
"""

import os
from collections import Counter
from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def download_sciclaimhunt():
    print("Downloading SciClaimHunt from HuggingFace (AnshulS/dataset_for_scicllaimhunt)...")
    print("Note: ~3.18 GB, first run may take a while.")
    dataset = load_dataset("AnshulS/dataset_for_scicllaimhunt", cache_dir=CACHE_DIR)
    print(f"  train split: {len(dataset['train'])} rows")

    print("\nColumn names:", dataset["train"].column_names)

    print("\nSample row (train[0]):")
    sample = dataset["train"][0]
    print(f"  Type    : {sample['Type']}")
    print(f"  Claim   : {sample['Claim'][:120]}...")
    print(f"  Evidence: {sample['Evidence'][:120]}...")

    print("\nLabel distribution (full train split):")
    labels = [ex["Type"] for ex in dataset["train"]]
    for label, count in Counter(labels).most_common():
        pct = 100 * count / len(labels)
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\nSciClaimHunt download complete. Cached to:", CACHE_DIR)
    return dataset


if __name__ == "__main__":
    download_sciclaimhunt()
