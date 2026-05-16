# import h5py

# filename = '9606.protein.network.embeddings.v12.0.h5'

# with h5py.File(filename, 'r') as f:
#     meta_keys = f['metadata'].attrs.keys()
#     for key in meta_keys:
#         print(key, f['metadata'].attrs[key])

#     embedding = f['embeddings'][:]
#     proteins = f['proteins'][:]
	
#     # protein names are stored as bytes, convert them to strings
#     proteins = [p.decode('utf-8') for p in proteins]

import scanpy as sc

adata = sc.read_h5ad("/home/zhangshibo24s/cell_flow/data_train/hepg2_hvg.h5ad")

# 读入 TRRUST 数据
import pandas as pd
trrust_df = pd.read_csv(
"/home/zhangshibo24s/cell_flow/data/trrust_rawdata.human_add.tsv",sep="\t")
print(trrust_df.head())
# exit()
# 过滤 TRRUST 数据，保留仅包含在 adata.obs["target_gene"] 中的基因
trrust_df = trrust_df[
    trrust_df["TF"].isin(adata.obs["target_gene"])
]
# print(f"过滤后的 TRRUST 数据行数: {len(trrust_df)}")
# exit()
# 选取其中一个 TF（示例），计算该扰动基因相对于 non-targeting 的差异表达基因
target_gene = trrust_df.iloc[2]["TF"]

import numpy as np
import anndata

def get_deg_sets(adata_obj, group="target"):
    degs = adata_obj.uns['rank_genes_groups']
    genes = np.array(degs['names'][group])
    logfc = np.array(degs['logfoldchanges'][group])
    pvals_adj = np.array(degs['pvals_adj'][group])
    sig_mask = pvals_adj < 0.05
    return genes[sig_mask], logfc[sig_mask]

def compute_des_single(real_genes, pred_genes, pred_logfc):
    """
    返回 (recall, precision)
    recall = |real ∩ pred_topk| / |real|
    precision = |real ∩ pred_topk| / |pred|
    当 |pred| > |real| 时，从 pred 中取前 |real| 个绝对 logFC 最大的基因作为 topk
    """
    real_set = set(real_genes)
    pred_set = set(pred_genes)
    n_true = len(real_set)
    n_pred = len(pred_set)

    if n_true == 0 or n_pred == 0:
        return 0.0, 0.0

    if n_pred <= n_true:
        inter = len(real_set & pred_set)
        recall = inter / n_true
        precision = inter / n_pred
        return recall, precision

    idx = np.argsort(-np.abs(pred_logfc))[:n_true]
    pred_topk = set(np.array(pred_genes)[idx])
    inter = len(real_set & pred_topk)
    recall = inter / n_true
    precision = inter / n_pred
    return recall, precision


# 从 adata 中分出 control (non-targeting) 和 被扰动的细胞
ctrl = adata[adata.obs["target_gene"] == "non-targeting"].copy()
pert = adata[adata.obs["target_gene"] == target_gene].copy()

# 合并并计算差异表达（ctrl vs target）
combined = anndata.concat([ctrl, pert], label='condition', keys=['ctrl', 'target'])
sc.tl.rank_genes_groups(
    combined,
    groupby='condition',
    reference='ctrl',
    method='wilcoxon'
)

# 提取显著 DEGs
de_genes, de_logfc = get_deg_sets(combined, group="target")

# TRRUST 中该 TF 的 target 列表
trrust_targets = trrust_df[trrust_df["TF"] == target_gene]["Target"].tolist()

# 计算 DES（以 TRRUST targets 作为真实靶基因集合，观测 DEGs 作为预测）
recall, precision = compute_des_single(trrust_targets, de_genes, de_logfc)
intersection = set(de_genes).intersection(set(trrust_targets))
print(f"差异表达基因数量: {len(de_genes)}")
print(f"TRRUST Target 基因数量: {len(trrust_targets)}")
print(f"交集数量: {len(intersection)}")
print(f"DES recall: {recall:.4f}, precision: {precision:.4f}")