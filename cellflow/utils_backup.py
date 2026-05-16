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

__all__ = ["match_linear", "default_prng_key", "build_grn_table", "build_condition_gene_masks"]


def _read_trrust_table(grn_path: str):
    import pandas as pd

    df = pd.read_csv(grn_path, sep="\t", header=0)
    tf_col = "TF"
    target_col = "Target"
    out = df[[tf_col, target_col]].copy()
    out["TF"] = out["TF"].astype(str).str.strip()
    out["Target"] = out["Target"].astype(str).str.strip()
    return out[(out["TF"] != "") & (out["Target"] != "")]


# def _fetch_collectri_from_decoupler(organism: str = "human"):
#     return dc.op.collectri(organism=organism)


def build_grn_table(
    adata,
    trrust_path: str,
    # collectri_organism: str = "human",
):
    trrust = [_read_trrust_table(trrust_path)] # TF Target
    # collectri = _fetch_collectri_from_decoupler(organism=collectri_organism) # source target
    
    # tf_col = "source"
    # target_col = "target"

    # cdf = collectri[[tf_col, target_col]].copy()
    # cdf.columns = ["TF", "Target"]
    # cdf = cdf.dropna()
    # cdf["TF"] = cdf["TF"].astype(str).str.strip()
    # cdf["Target"] = cdf["Target"].astype(str).str.strip()
    # trrust.append(cdf[(cdf["TF"] != "") & (cdf["Target"] != "")])

    # grn_df = pd.concat(trrust, ignore_index=True).drop_duplicates()
    # grn_df["TF_u"] = grn_df["TF"].str.upper()
    # grn_df["Target_u"] = grn_df["Target"].str.upper()

    # genes = [str(g) for g in adata.var_names]
    # perturb_genes = [str(g).upper() for g in adata.obs["target_gene"].unique()]
    # genes_upper = [g.upper() for g in genes]
    # genes_upper_set = set(genes_upper)
    # perturb_genes_set = set(perturb_genes)

    # # Keep only edges whose target gene exists in adata.var_names.
    # grn_df = grn_df[grn_df["Target_u"].isin(genes_upper_set)]
    # grn_df = grn_df[grn_df["TF_u"].isin(perturb_genes_set)]
    return trrust

    

def build_condition_gene_masks(
    adata,
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]],
    trrust_path: str,
    # collectri_organism: str = "human",
) -> np.ndarray:
    """Build per-condition masks using TF->target edges and condition TF labels.

    For each condition index, the first covariate value is interpreted as TF name.
    If TF exists in GRN and its targets are present in ``adata.var_names``, those target
    indices are set to 1 for that condition mask row.
    """

    grn_df = build_grn_table(
        adata,
        trrust_path=trrust_path,
        # collectri_organism=collectri_organism,
    )

    # 对于adata中的每个.obs["target_gene"]，找到它在grn_df中作为TF的行，并将对应的Target_u转换为大写后与adata.var_names进行匹配，得到对应的索引。
    # 最终返回一个二维数组，行数等于条件数量，列数等于adata.var_names的数量，其中每行对应一个条件，每列对应一个基因，如果该基因是该条件的TF的靶基因，则该位置为1，否则为0。
    gene_to_index = {str(g).upper(): i for i, g in enumerate(adata.var_names)}
    # Row index is aligned to perturbation_idx (can be non-contiguous), not just dict length.
    num_conditions = (max(perturbation_idx_to_covariates.keys()) + 1) if perturbation_idx_to_covariates else 0
    num_genes = len(adata.var_names)
    masks = np.zeros((num_conditions, num_genes), dtype=np.float32)
    for idx, covariates in perturbation_idx_to_covariates.items():
        tf_name = str(covariates[0]).upper()
        target_genes = grn_df[grn_df["TF_u"] == tf_name]["Target_u"].unique()
        for target in target_genes:
            if target in gene_to_index:
                target_idx = gene_to_index[target]
                masks[idx, target_idx] = 1.0
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
