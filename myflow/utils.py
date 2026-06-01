from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import numpy as np
import decoupler as dc
import pandas as pd

ScaleCost_t = float | Literal["mean", "max_cost", "median"]

__all__ = ["match_linear", "default_prng_key", "build_adj_matrix", "build_condition_gene_masks"]


def _read_trrust_table(grn_path: str):
    import pandas as pd

    df = pd.read_csv(grn_path, sep="\t", header=0)
    tf_col = "TF"
    target_col = "Target"
    out = df[[tf_col, target_col]].copy()
    out["TF"] = out["TF"].astype(str).str.strip()
    out["Target"] = out["Target"].astype(str).str.strip()
    return out[(out["TF"] != "") & (out["Target"] != "")]


def build_adj_matrix(adata, trrust_path: str, max_hops: int = 2) -> np.ndarray:
    """Build adjacency matrix from TRRUST GRN file.

    Parameters
    ----------
    adata
        AnnData object with gene names in var_names.
    trrust_path
        Path to TRRUST tab-separated file with TF and Target columns.
    max_hops
        Number of hops for transitive closure (default 2).

    Returns
    -------
    Adjacency matrix of shape (n_genes, n_genes).
    """
    grn_df = _read_trrust_table(trrust_path)
    genes = [str(g) for g in adata.var_names]
    gene_to_idx = {g.upper(): i for i, g in enumerate(genes)}
    n_genes = len(genes)

    adj = np.zeros((n_genes, n_genes), dtype=np.float32)
    for _, row in grn_df.iterrows():
        tf = row["TF"].upper()
        target = row["Target"].upper()
        if tf in gene_to_idx and target in gene_to_idx:
            i, j = gene_to_idx[tf], gene_to_idx[target]
            adj[i, j] = 1.0

    # Transitive closure up to max_hops
    adj_power = adj.copy()
    result = adj.copy()
    for hop in range(1, max_hops):
        adj_power = adj_power @ adj
        result = np.clip(result + adj_power, 0, 1)

    return result


def build_condition_gene_masks(
    adata,
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]],
    matrix: np.ndarray,
) -> np.ndarray:
    """Build per-condition masks using adjacency matrix and condition covariates.

    For each condition index, the first covariate value is interpreted as TF name.
    The mask selects genes reachable from that TF in the adjacency matrix.

    Parameters
    ----------
    adata
        AnnData object.
    perturbation_idx_to_covariates
        Mapping from condition index to covariate tuple.
    matrix
        Adjacency matrix of shape (n_genes, n_genes).

    Returns
    -------
    Masks array of shape (n_conditions, n_genes).
    """
    genes = [str(g).upper() for g in adata.var_names]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    num_conditions = (max(perturbation_idx_to_covariates.keys()) + 1) if perturbation_idx_to_covariates else 0
    num_genes = len(genes)
    masks = np.zeros((num_conditions, num_genes), dtype=np.float32)

    for idx, covariates in perturbation_idx_to_covariates.items():
        tf_name = str(covariates[0]).upper()
        if tf_name in gene_to_idx:
            tf_idx = gene_to_idx[tf_name]
            masks[idx] = matrix[tf_idx]

    return masks


def match_linear(
    source_batch: jnp.ndarray,
    target_batch: jnp.ndarray,
    cost_fn: costs.CostFn | None = costs.SqEuclidean(),
    epsilon: float | None = 1.0,
    scale_cost: ScaleCost_t = "mean",
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    threshold: float | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Compute solution to a linear OT problem.

    Parameters
    ----------
    source_batch
        Source point cloud of shape ``[n, d]``.
    target_batch
        Target point cloud of shape ``[m, d]``.
    cost_fn
        Cost function to use for the linear OT problem.
    epsilon
        Regularization parameter.
    scale_cost
        Scaling of the cost matrix.
    tau_a
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is
        in the source distribution. If :math:`1`, the problem is balanced in the source distribution.
    tau_b
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is in the target
        distribution. If :math:`1`, the problem is balanced in the target distribution.
    threshold
        Convergence criterion for the Sinkhorn algorithm.
    kwargs
        Additional arguments for :class:`ott.solvers.linear.sinkhorn.Sinkhorn`.

    Returns
    -------
    Optimal transport matrix between ``'source_batch'`` and ``'target_batch'``.
    """
    if threshold is None:
        threshold = 1e-3 if (tau_a == 1.0 and tau_b == 1.0) else 1e-2
    geom = pointcloud.PointCloud(
        source_batch,
        target_batch,
        cost_fn=cost_fn,
        epsilon=epsilon,
        scale_cost=scale_cost,
    )
    problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    solver = sinkhorn.Sinkhorn(threshold=threshold, **kwargs)
    out = solver(problem)
    return out.matrix


def default_prng_key(rng: jax.Array | None) -> jax.Array:
    """Get the default PRNG key.

    Parameters
    ----------
    rng: PRNG key.

    Returns
    -------
      If ``rng = None``, returns the default PRNG key. Otherwise, it returns
      the unmodified ``rng`` key.
    """
    return jax.random.key(0) if rng is None else rng
