#!/usr/bin/env python3
"""Build an IC-weighted gene-gene GO similarity graph.

Input is a gene-to-GO annotation table. Required columns are gene id and GO id;
an optional namespace/domain column can be used to restrict to BP/MF/CC.
The output schema matches the scLong graph CSV: source,target,importance.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True, help="CSV/TSV gene-to-GO annotation table.")
    p.add_argument("--output", required=True, help="Output source,target,importance CSV.")
    p.add_argument("--gene-column", default="gene_id")
    p.add_argument("--go-column", default="go_id")
    p.add_argument("--namespace-column", default=None)
    p.add_argument("--namespace", default=None, help="Optional namespace/domain filter, e.g. biological_process.")
    p.add_argument("--delimiter", default=None, help="Input delimiter. Defaults to auto-detect from suffix.")
    p.add_argument("--min-similarity", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--include-self", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def input_delimiter(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def main() -> None:
    args = parse_args()
    in_path = Path(args.annotations)
    out_path = Path(args.output)
    gene_to_terms: dict[str, set[str]] = defaultdict(set)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=input_delimiter(in_path, args.delimiter))
        if reader.fieldnames is None:
            raise ValueError("Annotation file has no header.")
        missing = [c for c in (args.gene_column, args.go_column) if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"Missing required columns: {missing}. Available: {reader.fieldnames}")
        if args.namespace_column and args.namespace_column not in reader.fieldnames:
            raise KeyError(f"Missing namespace column: {args.namespace_column}")

        for row in reader:
            if args.namespace_column and args.namespace:
                if row.get(args.namespace_column) != args.namespace:
                    continue
            gene = row[args.gene_column].strip().upper()
            go_id = row[args.go_column].strip()
            if gene and go_id:
                gene_to_terms[gene].add(go_id)

    genes = sorted(gene_to_terms)
    if not genes:
        raise ValueError("No gene-to-GO annotations were loaded.")

    term_counts = Counter(term for terms in gene_to_terms.values() for term in terms)
    n_genes = len(genes)
    term_ic = {term: math.log((n_genes + 1.0) / (count + 1.0)) for term, count in term_counts.items()}
    term_to_genes: dict[str, list[str]] = defaultdict(list)
    for gene, terms in gene_to_terms.items():
        for term in terms:
            term_to_genes[term].append(gene)

    pair_shared_ic: dict[tuple[str, str], float] = defaultdict(float)
    for term, term_genes in term_to_genes.items():
        ic = term_ic[term]
        sorted_genes = sorted(term_genes)
        for src in sorted_genes:
            for tgt in sorted_genes:
                pair_shared_ic[(src, tgt)] += ic

    gene_total_ic = {
        gene: sum(term_ic[term] for term in terms)
        for gene, terms in gene_to_terms.items()
    }
    per_target: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (src, tgt), shared_ic in pair_shared_ic.items():
        union_ic = gene_total_ic[src] + gene_total_ic[tgt] - shared_ic
        if union_ic <= 0:
            continue
        sim = shared_ic / union_ic
        if src == tgt and not args.include_self:
            continue
        if sim < args.min_similarity:
            continue
        per_target[tgt].append((sim, src))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "importance"])
        for tgt in genes:
            edges = sorted(per_target.get(tgt, []), key=lambda item: (-item[0], item[1]))
            for sim, src in edges[: args.top_k + int(args.include_self)]:
                writer.writerow([src, tgt, f"{sim:.10g}"])

    print(f"Wrote IC-weighted GO graph for {len(genes)} genes to {out_path}")


if __name__ == "__main__":
    main()
