#!/usr/bin/env python3
"""Simple overlap checks between adata genes and TRRUST genes."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple overlap report for target_gene/var_names against TRRUST")
    parser.add_argument(
        "--adata",
        type=str,
        default="/home/zhangshibo24s/cell_flow/data_train",
        help="Path to .h5ad or a directory containing *hvg.h5ad files",
    )
    parser.add_argument(
        "--target-gene-key",
        type=str,
        default="target_gene",
        help="obs column containing perturbation gene labels",
    )
    parser.add_argument(
        "--grn-path",
        type=str,
        default="/home/zhangshibo24s/cell_flow/data/trrust_rawdata.human_add.tsv",
        help="Path to TRRUST TSV file (expects TF and Target columns)",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=2,
        help="Maximum hops for influence matrix construction",
    )
    return parser.parse_args()


def load_adata(path_str: str) -> ad.AnnData:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"adata path not found: {path}")

    if path.is_dir():
        h5ad_files = sorted([p for p in path.iterdir() if p.name.endswith("hvg.h5ad")])
        if len(h5ad_files) == 0:
            raise ValueError(f"No *hvg.h5ad files found in directory: {path}")
        if len(h5ad_files) >= 3:
            selected = h5ad_files[:3]
        else:
            selected = h5ad_files
        print(f"[INFO] Loading {len(selected)} files from directory {path}")
        adatas = [ad.read_h5ad(str(p)) for p in selected]
        try:
            return ad.concat(adatas, join="outer", label="batch", keys=[p.stem for p in selected])
        except Exception:
            return ad.concat(adatas)

    print(f"[INFO] Loading single file: {path}")
    return ad.read_h5ad(str(path))


def read_trrust_genes(grn_path: str) -> tuple[pd.DataFrame, set[str], int]:
    df = pd.read_csv(grn_path, sep="\t", header=0)
    if "TF" not in df.columns or "Target" not in df.columns:
        raise ValueError("TRRUST file must contain 'TF' and 'Target' columns")

    table = df[["TF", "Target"]].copy()
    table["TF"] = table["TF"].astype(str).str.strip().str.upper()
    table["Target"] = table["Target"].astype(str).str.strip().str.upper()
    table = table[(table["TF"] != "") & (table["Target"] != "")]

    prior_genes = set(table["TF"]).union(set(table["Target"]))
    return table, prior_genes, len(table)


def main() -> None:
    args = parse_args()

    adata = load_adata(args.adata)
    if args.target_gene_key not in adata.obs:
        raise KeyError(f"obs column not found: {args.target_gene_key}")

    trrust_table, prior_genes, total_edges = read_trrust_genes(args.grn_path)
    adata_var_genes = {str(g).upper() for g in adata.var_names}
    adata_target_genes = {
        str(g).strip().upper()
        for g in adata.obs[args.target_gene_key].dropna().astype(str).values
        if str(g).strip() != ""
    }

    tf_genes = set(trrust_table["TF"])
    target_genes = set(trrust_table["Target"])

    overlap_targetgene_tf = adata_target_genes.intersection(tf_genes)
    overlap_targetgene_target = adata_target_genes.intersection(target_genes)
    overlap_var_tf = adata_var_genes.intersection(tf_genes)
    overlap_var_target = adata_var_genes.intersection(target_genes)
    

    print("\n=== Simple Overlap Report ===")
    print(f"adata shape: n_cells={adata.n_obs}, n_genes={adata.n_vars}")
    print(f"TRRUST edges: {total_edges}")

    print("\n1) adata.obs[target_gene] vs TRRUST TF")
    print(f"adata unique target_gene: {len(adata_target_genes)}")
    print(f"TRRUST unique TF: {len(tf_genes)}")
    print(
        "overlap: "
        f"{len(overlap_targetgene_tf)} ({len(overlap_targetgene_tf) / max(len(adata_target_genes), 1):.4f})"
    )
    print(
        f"overlap (Target U Target): {len(overlap_targetgene_target)}"
        f" ({len(overlap_targetgene_target) / max(len(adata_target_genes), 1):.4f})"
    )

    print("\n2) adata.var_names vs TRRUST TF/Target")
    print(f"TRRUST unique TF: {len(tf_genes)}")
    print(f"TRRUST unique Target: {len(target_genes)}")
    print(
        "var_names overlap TF: "
        f"{len(overlap_var_tf)} ({len(overlap_var_tf) / max(len(adata_var_genes), 1):.4f})"
    )
    print(
        "var_names overlap Target: "
        f"{len(overlap_var_target)} ({len(overlap_var_target) / max(len(adata_var_genes), 1):.4f})"
    )


if __name__ == "__main__":
    main()
