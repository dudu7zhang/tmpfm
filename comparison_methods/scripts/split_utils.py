"""Norman split logic matching scDFM's additive/holdout protocols.
Mirrors build_scdfm_norman_split from train_cellflow_norman_scdfm.py."""

import numpy as np


def _parse_condition_genes(condition: str) -> list[str]:
    genes = []
    for gene in str(condition).split("+"):
        gene = gene.strip()
        if not gene or gene.lower() == "ctrl":
            continue
        genes.append(gene)
    return genes


def _is_single_condition(condition: str) -> bool:
    return len(_parse_condition_genes(condition)) == 1


def _is_double_condition(condition: str) -> bool:
    return len(_parse_condition_genes(condition)) == 2


def build_scdfm_norman_split(
    conditions: list[str],
    split_method: str,
    fold: int,
    test_fraction: float,
    holdout_genes_count: int = 12,
    seed_base: int = 42,
) -> tuple[set[str], set[str], dict]:
    """Build Norman splits following scDFM's additive/holdout protocols.

    additive: test is a seeded 30% subset of double perturbations; all single
    perturbations stay in train.

    holdout/unseen: hold out single genes; test contains those single perturbations
    and every double perturbation involving any held-out gene.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    non_control = sorted(c for c in conditions if _parse_condition_genes(c))
    single_conditions = sorted(c for c in non_control if _is_single_condition(c))
    double_conditions = sorted(c for c in non_control if _is_double_condition(c))
    other_conditions = sorted(set(non_control) - set(single_conditions) - set(double_conditions))

    rng = np.random.default_rng(seed_base + fold)

    if split_method == "additive":
        shuffled = rng.permutation(double_conditions)
        n_test = max(1, int(len(shuffled) * test_fraction))
        test_conditions = set(str(c) for c in shuffled[:n_test])
        train_conditions = set(non_control) - test_conditions
        split_info = {
            "split_method": split_method,
            "fold": int(fold),
            "seed_base": int(seed_base),
            "split_seed": int(seed_base + fold),
            "single_conditions_total": len(single_conditions),
            "double_conditions_total": len(double_conditions),
            "other_conditions_total": len(other_conditions),
        }
    elif split_method in {"holdout", "unseen"}:
        double_genes = sorted({gene for cond in double_conditions for gene in _parse_condition_genes(cond)})
        if holdout_genes_count <= 0:
            raise ValueError(f"holdout_genes_count must be positive, got {holdout_genes_count}")
        if holdout_genes_count >= len(double_genes):
            raise ValueError(
                f"holdout_genes_count ({holdout_genes_count}) must be smaller than available double genes ({len(double_genes)})."
            )
        holdout_genes = set(str(g) for g in rng.permutation(double_genes)[:holdout_genes_count])
        test_conditions = {
            c for c in non_control if any(g in holdout_genes for g in _parse_condition_genes(c))
        }
        train_conditions = set(non_control) - test_conditions
        split_info = {
            "split_method": split_method,
            "fold": int(fold),
            "seed_base": int(seed_base),
            "split_seed": int(seed_base + fold),
            "holdout_genes_count": int(holdout_genes_count),
            "holdout_genes": sorted(holdout_genes),
            "single_conditions_total": len(single_conditions),
            "double_conditions_total": len(double_conditions),
            "other_conditions_total": len(other_conditions),
        }
    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    split_info.update({
        "train_conditions_count": len(train_conditions),
        "test_conditions_count": len(test_conditions),
        "train_single_conditions_count": sum(_is_single_condition(c) for c in train_conditions),
        "train_double_conditions_count": sum(_is_double_condition(c) for c in train_conditions),
        "test_single_conditions_count": sum(_is_single_condition(c) for c in test_conditions),
        "test_double_conditions_count": sum(_is_double_condition(c) for c in test_conditions),
    })
    return train_conditions, test_conditions, split_info
