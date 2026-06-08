#!/usr/bin/env python3
"""Convert TRRUST raw TSV to graph edge CSV (source, target, weight).

TRRUST v2: TF -> Target regulatory relationships, manually curated from literature.
We treat each TF->Target pair as a directed edge with weight=1.0.
Only edges where both genes appear in the perturbation gene set will be used.
"""

import csv
import sys
from pathlib import Path

TRRUST_FILE = Path(__file__).resolve().parent.parent / "data_gab" / "trrust_rawdata.human.tsv"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data_train" / "trrust_human_regulation.csv"


def convert():
    edges = set()  # deduplicate
    with open(TRRUST_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tf = row["TF"].strip().upper()
            target = row["Target"].strip().upper()
            if tf and target and tf != target:
                edges.add((tf, target))

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        for src, tgt in sorted(edges):
            writer.writerow([src, tgt, 1.0])

    print(f"Converted {len(edges)} unique TRRUST regulatory edges to {OUTPUT_FILE}")


if __name__ == "__main__":
    convert()
