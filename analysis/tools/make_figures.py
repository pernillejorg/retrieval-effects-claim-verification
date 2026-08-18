"""
Building the two Results figures directly from the saved result files.

Relevance to the dissertation paper:
Both figures are written as PGF, so matplotlib renders every label through LaTeX
itself and the output stays vector and in the document's own font. Nothing is typed
by hand: the depth sweep reads matrix_multiseed_summary.json and the per-label
figure reads the two confidence_by_label CSVs, so a change in the results changes
the figures without anyone editing coordinates.

The LaTeX preamble needs two lines for the output to compile:
    \\usepackage{pgf}
    \\providecommand{\\mathdefault}[1]{#1}

Usage:
    python analysis/tools/make_figures.py
    python analysis/tools/make_figures.py --out_dir figs
"""

import argparse
import csv
import json
import os

import matplotlib

#selecting the PGF backend before pyplot is imported, so text is typeset by LaTeX
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.3,
})

import matplotlib.pyplot as plt  # noqa: E402

#using consistent colours, line styles, and markers so the conditions remain
#doing this for simplicity to distinguish on screen and in greyscale print
CONDITIONS = [
    ("no_retrieval", "No retrieval", "#4D4D4D", "--", "none"),
    ("bm25_roberta", "BM25", "#D19A3E", "-",  "s"),
    ("dense_roberta", "Dense", "#4C78A8", "-",  "o"),
    ("dense_reranked_roberta", "Dense + rerank", "#B05A67", ":", "^"),
]
DEPTHS = [1, 3, 5, 10]
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
CORPORA = [
    ("scifact", "SciFact (5{,}183 documents)"),
    ("scifact_open", "SciFact-Open (500{,}000 documents)"),
]

#IEEE single column is 3.5 inches, so sizing at source avoids scaling the fonts later
COLUMN_WIDTH_IN = 3.4


def make_ksweep(summary_path, out_path):
    """Drawing macro F1 against retrieval depth, with three-seed error bars."""
    with open(summary_path) as f:
        summary = json.load(f)

    figure, axes = plt.subplots(
        #2, 1, figsize=(COLUMN_WIDTH_IN, 3.8), sharex=True
        2, 1, figsize=(COLUMN_WIDTH_IN, 4.4), sharex=True
    )

    #offsetting the three retrieval series slightly on x so their error bars sit
    #side by side rather than overlapping where the conditions converge
    offsets = {"bm25_roberta": -0.18, "dense_roberta": 0.0,
               "dense_reranked_roberta": 0.18}

    for axis, (dataset, title) in zip(axes, CORPORA):
        for key, label, colour, style, marker in CONDITIONS:
            means = [summary[dataset][key][str(k)]["mean"] for k in DEPTHS]
            deviations = [summary[dataset][key][str(k)]["sd"] for k in DEPTHS]

            if key == "no_retrieval":
                #drawing the constant baseline as a band, since repeating the same
                #error bar at four depths says nothing the band does not
                mean, deviation = means[0], deviations[0]
                axis.axhspan(
                    mean - deviation, mean + deviation,
                    color="0.80", zorder=0,
                )
                axis.axhline(
                    mean, color=colour, linestyle=style, label=label,
                )
                continue

            shifted = [k + offsets[key] for k in DEPTHS]
            axis.errorbar(
                shifted, means, yerr=deviations,
                label=label, color=colour, linestyle=style,
                marker=marker, markersize=4.0, capsize=2.0, elinewidth=0.7,
            )

        axis.set_title(title)
        axis.set_ylabel("Macro F1")
        axis.set_ylim(0.30, 0.65)
        #labelling every 0.1 so a reader can recover values between the curves
        axis.set_yticks([0.3, 0.4, 0.5, 0.6])
        axis.grid(True, color="0.88")
        axis.set_axisbelow(True)

    axes[1].set_xlabel("Retrieval depth $k$")
    axes[1].set_xticks(DEPTHS)
    #placing the legend below both panels so neither plot area is obscured
    '''
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.41),
        ncol=2, frameon=True, edgecolor="0.7",
    )

    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    '''
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.32),
        ncol=2, frameon=True, edgecolor="0.7",
    )

    #figure.tight_layout(pad=0.2)
    figure.tight_layout(pad=0.2, rect=[0, 0.03, 1, 1])
    figure.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"wrote {out_path}")


def read_by_label(csv_path):
    """Returning AUROC keyed by condition and true label."""
    scores = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            scores[(row["condition"], row["true_label"])] = float(
                row["auroc_confidence_detects_correct"]
            )
    return scores


def make_bylabel(csv_dir, out_path):
    """Drawing confidence AUROC by true label, with a chance line at 0.5."""
    figure, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 3.2))

    #reusing the CONDITIONS colours thats above so a condition looks the same in both figures
    positions = range(len(LABELS))
    width = 0.2

    for axis, (dataset, title) in zip(axes, CORPORA):
        scores = read_by_label(
            os.path.join(csv_dir, f"confidence_by_label_{dataset}_k3.csv")
        )
        for index, (key, label, colour, _, _) in enumerate(CONDITIONS):
            values = [scores[(key, name)] for name in LABELS]
            offsets = [p + (index - 1.5) * width for p in positions]
            axis.bar(
                offsets, values, width * 0.9,
                label=label, facecolor=colour, edgecolor="black", linewidth=0.5,
            )
        #marking chance, since a bar below this line is the inversion the paper reports
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title.split(" (")[0])
        axis.set_ylabel("AUROC")
        axis.set_ylim(0, 1.05)
        #labelling quarter points so the gradient across conditions is readable
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticks(list(positions))
        axis.set_xticklabels(LABELS)
        axis.grid(True, axis="y", color="0.88")
        axis.set_axisbelow(True)

    #placing chance in the gap between the first two groups, clear of every bar
    axes[0].text(
        0.5, 0.53, "chance",
        fontsize=6, style="italic", ha="center",
    )
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, frameon=True, edgecolor="0.7",
    )

    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="results/step6_matrix/step6_multiseed_matrix/matrix_multiseed_summary.json",
    )
    parser.add_argument("--csv_dir", default="results/step8_confidence")
    parser.add_argument("--out_dir", default="figs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    make_ksweep(args.summary, os.path.join(args.out_dir, "ksweep.pgf"))
    make_bylabel(args.csv_dir, os.path.join(args.out_dir, "bylabel.pgf"))


if __name__ == "__main__":
    main()
