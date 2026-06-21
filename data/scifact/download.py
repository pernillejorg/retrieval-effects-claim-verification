"""
download.py -- SciFact dataset loader
Loads allenai/scifact from Hugging Face and caches it locally.
Run once before any experiments: python data/scifact/download.py
"""

import os
from collections import Counter
from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def download_scifact():
    print("Downloading SciFact claims (train + validation)...")
    dataset = load_dataset("allenai/scifact", cache_dir=CACHE_DIR)
    print(f"  train     : {len(dataset['train'])} claims")
    print(f"  validation: {len(dataset['validation'])} claims")

    print("\nDownloading SciFact corpus (abstracts)...")
    corpus = load_dataset("allenai/scifact", "corpus", cache_dir=CACHE_DIR)
    print(f"  corpus    : {len(corpus['train'])} abstracts")

    print("\nSample claim (train[0]):")
    sample = dataset["train"][0]
    print(f"  id      : {sample['id']}")
    print(f"  claim   : {sample['claim']}")
    print(f"  label   : {sample['gold_label']}")
    print(f"  evidence: {sample['evidence']}")

    print("\nLabel distribution (train):")
    labels = [ex["gold_label"] for ex in dataset["train"]]
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")

    print("\nSciFact download complete. Cached to:", CACHE_DIR)
    return dataset, corpus


if __name__ == "__main__":
    download_scifact()
