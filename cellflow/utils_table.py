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
    
    tf_col = "source"
    target_col = "target"

    cdf = collectri[[tf_col, target_col]].copy()
    cdf.columns = ["TF", "Target"]
    cdf = cdf.dropna()
    cdf["TF"] = cdf["TF"].astype(str).str.strip()
    cdf["Target"] = cdf["Target"].astype(str).str.strip()
    trrust.append(cdf[(cdf["TF"] != "") & (cdf["Target"] != "")])

    grn_df = pd.concat(trrust, ignore_index=True).drop_duplicates()
    grn_df["TF_u"] = grn_df["TF"].str.upper()
    grn_df["Target_u"] = grn_df["Target"].str.upper()

    genes = [str(g) for g in adata.var_names]
    perturb_genes = [str(g).upper() for g in adata.obs["target_gene"].unique()]
    genes_upper = [g.upper() for g in genes]
    genes_upper_set = set(genes_upper)
    perturb_genes_set = set(perturb_genes)

    # Keep only edges whose target gene exists in adata.var_names.
    grn_df = grn_df[grn_df["Target_u"].isin(genes_upper_set)]
    grn_df = grn_df[grn_df["TF_u"].isin(perturb_genes_set)]
    return grn_df


if __name__ == "__main__":
    # trrust_path = "/home/zhangshibo24s/cell_flow/data/trrust_rawdata.human_add.tsv"
    # collectri_organism = "human"
    # trrust_df = _read_trrust_table(trrust_path)
    # collectri = _fetch_collectri_from_decoupler(organism=collectri_organism)
    # # print(collectri)
    # tf_col = "source"
    # target_col = "target"

    # cdf = collectri[[tf_col, target_col]].copy()
    # cdf.columns = ["TF", "Target"]
    # cdf = cdf.dropna()
    # cdf["TF"] = cdf["TF"].astype(str).str.strip()
    # cdf["Target"] = cdf["Target"].astype(str).str.strip()

    # # 2️⃣ 合并
    # merged = pd.concat(
    #     [trrust_df[["TF", "Target"]], cdf[["TF", "Target"]]],
    #     ignore_index=True
    # )

    # # 合并前总行数
    # before_dedup = len(merged)

    # # 3️⃣ 去重
    # merged = merged.drop_duplicates().reset_index(drop=True)

    # # 合并后行数
    # after_dedup = len(merged)

    # # 4️⃣ 再验证重复行数（更直观）
    # duplicate_rows = before_dedup - after_dedup
    # print(f"重复行数（concat 后重复）: {duplicate_rows}")

    # # 保存
    # merged.to_csv(trrust_path, sep="\t", index=False)
    
    # df = pd.read_csv("/home/zhangshibo24s/cell_flow/data/trrust_rawdata.human_add.tsv", sep="\t", header=0)
    # print(len(df))
    # df = pd.read_csv("/home/zhangshibo24s/cell_flow/data/trrust_rawdata.human.tsv", sep="\t", header=0)
    # print(len(df))