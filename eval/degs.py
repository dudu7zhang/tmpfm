import sys
import os

import pandas as pd
import scanpy as sc
import numpy as np
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import anndata
import seaborn as sns
import matplotlib.font_manager
from matplotlib import rcParams
import scanpy as sc

font_list = []
fpaths = matplotlib.font_manager.findSystemFonts()
for i in fpaths:
    try:
        f = matplotlib.font_manager.get_font(i)
        font_list.append(f.family_name)
    except RuntimeError:
        pass

font_list = set(font_list)
# plot_font = 'Helvetica' if 'Helvetica' in font_list else 'FreeSans'
# rcParams['font.family'] = plot_font
rcParams.update({'font.size': 10})
rcParams.update({'figure.dpi': 300})
rcParams.update({'figure.figsize': (3,3)})
rcParams.update({'savefig.dpi': 500})
warnings.filterwarnings('ignore')



import scanpy as sc
import numpy as np
import pandas as pd

def get_deg_sets(adata, group="target"):
    """
    从 rank_genes_groups 结果中提取目标 group 的
    - DE 基因名数组
    - logfoldchanges 数组（与基因一一对应）
    """
    degs = adata.uns['rank_genes_groups']

    genes = np.array(degs['names'][group])
    logfc = np.array(degs['logfoldchanges'][group])

    # FDR < 0.05
    pvals_adj = np.array(degs['pvals_adj'][group])
    sig_mask = pvals_adj < 0.05

    return genes[sig_mask], logfc[sig_mask]


def compute_des_single(real_genes, pred_genes, pred_logfc):
    """
    根据公式计算一个 perturbation 的 DES：
    real_genes: 真实 DE 基因集合
    pred_genes: 预测 DE 基因集合
    pred_logfc: 预测 DE 基因的 log fold change（用于选 top |real|）
    """

    real_set = set(real_genes)
    # print(len(real_set))
    pred_set = set(pred_genes)
    # print(len(pred_set))
    n_true = len(real_set)
    n_pred = len(pred_set)

    if n_true == 0:
        return 0.0

    # 情况 1：预测 DE 基因数量 <= 真实数量
    if n_pred <= n_true:
        inter = real_set.intersection(pred_set)
        return len(inter) / n_true

    # 情况 2：预测 DE 基因数量 > 真实数量 → 取前 n_true 个 logFC 最大的基因
    idx = np.argsort(-np.abs(pred_logfc))[:n_true]
    pred_topk_genes = set(np.array(pred_genes)[idx])

    inter = real_set.intersection(pred_topk_genes)
    
    #------#
    # real_set = real_set[:10]
    # pred_set = pred_set[:10]
    # inter = real_set.intersection(pred_set)
    # print(inter)
    # print(real_set)
    # print(inter/real_set)
    return len(inter)/n_true, len(inter)/n_pred


def compute_des(ctrl, target, pred):
    """
    ctrl: control AnnData
    target: GT perturbed AnnData
    pred: model predicted perturbed AnnData

    返回 DES 分数
    """

    # 合并真实数据
    combined_real = ctrl.concatenate(target, batch_key='condition',
                                     batch_categories=['ctrl', 'target'])
    sc.tl.rank_genes_groups(
        combined_real,
        groupby='condition',
        reference='ctrl',
        method='t-test'
    )
    real_genes, real_logfc = get_deg_sets(combined_real, group="target")


    # 合并预测数据
    if 'gene_symbol' in pred.var.columns:
        pred.var.index = pred.var['gene_symbol']
        pred.var_names = pred.var['gene_symbol'].values

    combined_pred = ctrl.concatenate(pred, batch_key='condition',
                                     batch_categories=['ctrl', 'target'])
    sc.tl.rank_genes_groups(
        combined_pred,
        groupby='condition',
        reference='ctrl',
        method='t-test'
    )
    pred_genes, pred_logfc = get_deg_sets(combined_pred, group="target")


    # 计算 DES
    des_score = compute_des_single(
        real_genes=real_genes,
        pred_genes=pred_genes,
        pred_logfc=pred_logfc
    )
    return des_score


def compute_des_new(ctrl, target, pred):
    """
    计算 DES
    """
    # 1. 计算真实数据的 DEGs (Wilcoxon)
    # 使用 anndata.concat 替代 deprecated 的 concatenate
    combined_real = anndata.concat([ctrl, target], label='condition', keys=['ctrl', 'target'])
    sc.tl.rank_genes_groups(
        combined_real, 
        groupby='condition', 
        reference='ctrl', 
        method='t-test' # 改为 Wilcoxon
    )
    real_genes, _ = get_deg_sets(combined_real, group="target")

    # 2. 计算预测数据的 DEGs (Wilcoxon)
    combined_pred = anndata.concat([ctrl, pred], label='condition', keys=['ctrl', 'target'])
    sc.tl.rank_genes_groups(
        combined_pred, 
        groupby='condition', 
        reference='ctrl', 
        method='t-test' # 改为 Wilcoxon
    )
    pred_genes, pred_logfc = get_deg_sets(combined_pred, group="target")

    # 3. 计算得分
    return compute_des_single(real_genes, pred_genes, pred_logfc)


ctrl = sc.read_h5ad('/home/zhangshibo24s/cell_flow/data/jurkat_ctrl.h5ad')
target = sc.read_h5ad('/home/zhangshibo24s/cell_flow/data/jurkat_validation.h5ad')
# print(target)
#state_20000, ours_w
pred = sc.read_h5ad('/home/zhangshibo24s/cell_flow/outputs/predictions_20260426_203108/predictions_20260426_203108.h5ad')
# print(pred)
des_recall, des_acc = compute_des(ctrl, target, pred)
print("DES score =", des_recall, des_acc)



# 前n个degs的mse

