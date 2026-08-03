"""
Emitting the pgfplots coordinates used by the dissertation figures.

Relevance to the dissertation paper:
The two figures in the Results section are drawn in TikZ rather than exported from a
plotting library, so their coordinates sit in the LaTeX source. This script prints
those coordinate lines directly from the saved result files, so the figures can be
regenerated rather than retyped whenever a number changes, and so the numbers in the
paper are demonstrably derived from the results rather than transcribed by hand.

Two figures are covered:
    ksweep     macro F1 against retrieval depth, mean and SD over three seeds
    bylabel    confidence AUROC by true label at k = 3

Usage:
    python analysis/tools/emit_figure_coords.py --figure ksweep
    python analysis/tools/emit_figure_coords.py --figure bylabel
"""

import argparse
import csv
import json
import os

#keeping the plot order the paper uses, so the emitted lines can be pasted in sequence
CONDITIONS = [
    "no_retrieval",
    "bm25_roberta",
    "dense_roberta",
    "dense_reranked_roberta",
]
DEPTHS = [1, 3, 5, 10]
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]


def emit_ksweep(summary_path):
    """Printing depth-sweep coordinates with error bars, one line per condition."""
    with open(summary_path) as f:
        summary = json.load(f)

    for dataset in ["scifact", "scifact_open"]:
        print(f"% ---- {dataset} ----")
        for condition in CONDITIONS:
            points = []
            for k in DEPTHS:
                cell = summary[dataset][condition][str(k)]
                #the +- form is what pgfplots reads as an explicit y error bar
                points.append(f"({k},{cell['mean']:.4f})+-(0,{cell['sd']:.4f})")
            print(f"% {condition}")
            print("coordinates {" + " ".join(points) + "};")
        print()


def emit_bylabel(csv_dir):
    """Printing per-label AUROC coordinates, one line per condition."""
    for dataset in ["scifact", "scifact_open"]:
        path = os.path.join(csv_dir, f"confidence_by_label_{dataset}_k3.csv")
        if not os.path.exists(path):
            print(f"% {dataset}: no file at {path}")
            continue

        #indexing by condition and label so the output order matches the figure
        scores = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["condition"], row["true_label"])
                scores[key] = float(row["auroc_confidence_detects_correct"])

        print(f"% ---- {dataset} ----")
        for condition in CONDITIONS:
            points = [
                f"({label},{scores[(condition, label)]:.4f})"
                for label in LABELS
                if (condition, label) in scores
            ]
            print(f"% {condition}")
            print("coordinates {" + " ".join(points) + "};")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["ksweep", "bylabel"], required=True)
    parser.add_argument(
        "--summary",
        default="results/step6_matrix/step6_multiseed_matrix/matrix_multiseed_summary.json",
    )
    parser.add_argument("--csv_dir", default="results/step8_confidence")
    args = parser.parse_args()

    if args.figure == "ksweep":
        emit_ksweep(args.summary)
    else:
        emit_bylabel(args.csv_dir)


if __name__ == "__main__":
    main()
