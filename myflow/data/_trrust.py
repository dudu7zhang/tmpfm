"""TRRUST regulatory network integration.

TRRUST v2 provides TF→Target regulatory relationships curated from literature.
Unlike GO (functional similarity) and STRING (protein interaction) which connect
perturbation genes to each other, TRRUST connects perturbation genes (TFs) to
their downstream target genes in the expression space.

This module provides:
1. Loading TRRUST data and building TF→target gene mappings
2. Per-condition target gene masks for gene mask bias (方案 A)
3. Per-gene target features for cross-attention bias (方案 B)
"""

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

__all__ = ["TRRUSTManager", "build_trrust_target_masks"]


class TRRUSTManager:
    """Load TRRUST and build TF→target gene index mappings."""

    def __init__(self, trrust_file: str | Path, gene_symbols: list[str]):
        self.gene_symbols = gene_symbols
        self.gene_to_idx = {g.upper(): i for i, g in enumerate(gene_symbols)}
        self.n_genes = len(gene_symbols)

        # TF → set of target gene indices
        self.tf_to_targets: dict[str, set[int]] = {}
        self._load(trrust_file)

    def _load(self, trrust_file: str | Path):
        with open(trrust_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both raw TRRUST (TF/Target) and converted CSV (source/target)
                tf = row.get("TF", row.get("source", "")).strip().upper()
                target = row.get("Target", row.get("target", "")).strip().upper()
                if tf == target:
                    continue
                # Only target needs to be an expression gene; TF can be perturbation-only
                if target not in self.gene_to_idx:
                    continue
                self.tf_to_targets.setdefault(tf, set()).add(self.gene_to_idx[target])

        n_tfs = len(self.tf_to_targets)
        n_edges = sum(len(v) for v in self.tf_to_targets.values())
        print(f"TRRUST: {n_tfs} TFs → {n_edges} target edges mapped to {self.n_genes} expression genes")

    def get_target_mask(self, pert_gene: str) -> np.ndarray:
        """Binary mask (n_genes,) of TRRUST targets for one perturbation gene."""
        mask = np.zeros(self.n_genes, dtype=np.float32)
        targets = self.tf_to_targets.get(pert_gene.upper())
        if targets:
            mask[list(targets)] = 1.0
        return mask

    def get_condition_target_mask(self, pert_genes: list[str]) -> np.ndarray:
        """Aggregate binary mask (n_genes,) for a set of perturbation genes."""
        mask = np.zeros(self.n_genes, dtype=np.float32)
        for g in pert_genes:
            targets = self.tf_to_targets.get(g.upper())
            if targets:
                mask[list(targets)] = 1.0
        return mask


def build_trrust_target_masks(
    trrust_file: str | Path,
    gene_symbols: list[str],
    condition_list: list[str],
    pert_gene_fn,
) -> dict[str, np.ndarray]:
    """Precompute per-condition TRRUST target masks.

    Args:
        trrust_file: Path to TRRUST CSV (source, target, weight).
        gene_symbols: All expression gene symbols in order.
        condition_list: List of unique condition names.
        pert_gene_fn: Callable mapping condition_name → list of perturbation gene symbols.

    Returns:
        Dict mapping condition_name → binary target mask (n_genes,).
    """
    mgr = TRRUSTManager(trrust_file, gene_symbols)
    masks: dict[str, np.ndarray] = {}
    for cond_name in condition_list:
        pert_genes = pert_gene_fn(cond_name)
        mask = mgr.get_condition_target_mask(pert_genes)
        if mask.sum() > 0:
            masks[cond_name] = mask

    n_with_targets = len(masks)
    print(f"TRRUST target masks: {n_with_targets}/{len(condition_list)} conditions have known targets")
    return masks
