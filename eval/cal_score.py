import scanpy as sc
import argparse
import os
import numpy as np
import pandas as pd
import anndata
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy
from data.loaddata import load_data_sam
import time
import warnings
warnings.filterwarnings('ignore')

def cal_metric(pred_mean, real_mean):
    """
    基础指标：衡量整体分布 (L2, MSE, MAE等)
    """
    # r2 = r2_score(real_mean, pred_mean)
    # pearsonr, _ = scipy.stats.pearsonr(real_mean, pred_mean)
    mse = mean_squared_error(real_mean, pred_mean)
    mae = mean_absolute_error(real_mean, pred_mean)
    l2 = np.linalg.norm(real_mean - pred_mean) # L2 距离
    
    return mse, mae, l2

def cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=20):
    """
    进阶Delta指标：衡量模型是否真正学到了"扰动引起的变化"
    """
    # 真实扰动变化和预测扰动变化
    delta_real = real_mean - ctrl_mean
    delta_pred = pred_mean - ctrl_mean
    
    # 1. Pearson Δ (所有基因变化量的相关性)
    pearson_delta, _ = scipy.stats.pearsonr(delta_real, delta_pred)
    
    # 选取真实扰动下，绝对变化最大的 Top-K 基因
    top_n_idx = np.argsort(np.abs(delta_real))[-top_k:]
    
    # 2. Pearson Δ20 (对于最显著变化的基因的预测相关性)
    if len(top_n_idx) > 1:
        pearson_delta_top_k, _ = scipy.stats.pearsonr(delta_real[top_n_idx], delta_pred[top_n_idx])
    else:
        pearson_delta_top_k = 0.0

    # 3. DS (Direction Score)：衡量变化方向（上调还是下调）预测对的比例 (Top K基因)
    sign_real = np.sign(delta_real[top_n_idx])
    sign_pred = np.sign(delta_pred[top_n_idx])
    ds_score = np.mean([1 if r == p else 0 for r, p in zip(sign_real, sign_pred)])
    
    return pearson_delta, pearson_delta_top_k, ds_score

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
    pred_set = set(pred_genes)
    n_true = len(real_set)
    n_pred = len(pred_set)

    if n_true == 0:
        return 0.0, 0.0

    # 情况 1：预测 DE 基因数量 <= 真实数量
    if n_pred <= n_true:
        inter = real_set.intersection(pred_set)
        return len(inter) / n_true, (len(inter) / n_pred if n_pred > 0 else 0)

    # 情况 2：预测 DE 基因数量 > 真实数量 → 取前 n_true 个 logFC 最大的基因
    idx = np.argsort(-np.abs(pred_logfc))[:n_true]
    pred_topk_genes = set(np.array(pred_genes)[idx])

    inter = real_set.intersection(pred_topk_genes)
    return len(inter)/n_true, len(inter)/n_pred

def compute_des(ctrl, target, pred):
    """
    ctrl: control AnnData
    target: GT perturbed AnnData
    pred: model predicted perturbed AnnData
    返回 DES 分数 以及 DE-Spearman rho
    """
    # 兼容处理索引以免重叠
    combined_real = ctrl.concatenate(target, batch_key='condition',
                                     batch_categories=['ctrl', 'target'])
    sc.tl.rank_genes_groups(combined_real, groupby='condition', reference='ctrl', method='t-test')
    real_genes, real_logfc = get_deg_sets(combined_real, group="target")

    if 'gene_symbol' in pred.var.columns:
        pred.var.index = pred.var['gene_symbol']
        pred.var_names = pred.var['gene_symbol'].values

    combined_pred = ctrl.concatenate(pred, batch_key='condition',
                                     batch_categories=['ctrl', 'target'])
    sc.tl.rank_genes_groups(combined_pred, groupby='condition', reference='ctrl', method='t-test')
    pred_genes, pred_logfc = get_deg_sets(combined_pred, group="target")

    # ----- 补充 DE-Spearman rho 计算 -----
    de_spearman = 0.0
    if len(real_genes) > 1:
        # 获取真实显著性基因对应的 pred_logfc 集合
        # 注意: pred 里面的 rank_genes_groups 包含了所有基因，只是 get_deg_sets 做了过滤
        # 为了 DE-Spearman，我们需要在预测集中找到那些真实DE基因的 logfc
        degs_pred_all = combined_pred.uns['rank_genes_groups']
        all_pred_genes = np.array(degs_pred_all['names']["target"])
        all_pred_logfc = np.array(degs_pred_all['logfoldchanges']["target"])
        
        # 构建 gene 到 pred_logfc 的映射，提取对应真实基因的值
        pred_logfc_map = dict(zip(all_pred_genes, all_pred_logfc))
        
        real_matched_logfc = []
        pred_matched_logfc = []
        for rg, r_fc in zip(real_genes, real_logfc):
            if rg in pred_logfc_map:
                real_matched_logfc.append(r_fc)
                pred_matched_logfc.append(pred_logfc_map[rg])
        
        if len(real_matched_logfc) > 1:
            de_spearman, _ = scipy.stats.spearmanr(real_matched_logfc, pred_matched_logfc)
    # -------------------------------------

    des_recall, des_acc = compute_des_single(real_genes, pred_genes, pred_logfc)
    return des_recall, des_acc, de_spearman

if __name__ == "__main__":
    # 读取 Control 对照组数据 (用来算 Delta)
    # 取矩阵的第一维进行mean，注意由于有的数据可能是稀疏矩阵，如果是稀疏矩阵请用 .A.mean(0) 或转化为稠密再算。如果报错，请确保其是稠密数组
    ctrl_data = sc.read_h5ad("/home/zhangshibo24s/cell_flow/data/k562_ctrl.h5ad")
    ctrl_mean = np.array(ctrl_data.X.mean(axis=0)).flatten()

    # 读取 Real Target 真实扰动后的数据
    real_result = sc.read_h5ad("/home/zhangshibo24s/cell_flow/data/k562_validation.h5ad")
    real_mean = np.array(real_result.X.mean(axis=0)).flatten()

    # pred_result = []
    # state_result = sc.read_h5ad("/home/zhangshibo24s/cell_flow/outputs/predictions_20260426_192442/predictions_20260426_192442.h5ad")
    # state_mean = np.array(state_result.X.mean(axis=0)).flatten()
    path = "/home/zhangshibo24s/cell_flow/outputs/predictions_20260429_005010/predictions_20260429_005010.h5ad"
    print(path)
    ours_result = sc.read_h5ad(path)
    ours_mean = np.array(ours_result.X.mean(axis=0)).flatten()

    # Baseline 评估 (可选)
    # r2, pearsonr, mse, mae, l2 = cal_metric(state_mean, real_mean)
    # pearson_del, pearson_del_top20, ds = cal_delta_metric(ctrl_mean, real_mean, state_mean)
    # print("=" * 50)
    # print("Baseline / State Metrics:")
    # print(f"Basic => R2: {r2:.4f}, Pearson: {pearsonr:.4f}, MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
    # print(f"Delta => Pearson Δ: {pearson_del:.4f}, Pearson Δ20: {pearson_del_top20:.4f}, DS: {ds:.4f}")
    
    print("=" * 50)
    print("Ours Metrics:")
    mse, mae, l2 = cal_metric(ours_mean, real_mean)
    pearson_del, pearson_del_top20, ds = cal_delta_metric(ctrl_mean, real_mean, ours_mean)
    print(f"Basic => MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
    print(f"Delta => Pearson Δ: {pearson_del:.4f}, Pearson Δ20: {pearson_del_top20:.4f}, DS: {ds:.4f}")
    
    # 计算 DES 和 DE-Spearman rho
    des_recall, des_acc, de_spearman = compute_des(ctrl_data, real_result, ours_result)
    print(f"DES   => Recall: {des_recall:.4f}, Accuracy: {des_acc:.4f}, DE-Spearman rho: {de_spearman:.4f}")
    print("=" * 50)
    