#!/usr/bin/env python3
"""
分析 DES 指标问题：
为什么 MyFlow-Gene2Vec 整体精度高但 DES 低？
"""

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.stats
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def analyze_des_issue(pred_file, data_path, method_name="MyFlow-Gene2Vec"):
    """分析 DES 问题"""
    print("=" * 60)
    print(f"分析 {method_name} 的 DES 问题")
    print("=" * 60)

    # 加载数据
    pred = ad.read_h5ad(pred_file)
    adata = sc.read_h5ad(data_path)

    # 检查列名
    if 'perturbation' in adata.obs.columns:
        ctrl = adata[adata.obs['perturbation'] == 'ctrl'].copy()
        real = adata[adata.obs['perturbation'] != 'ctrl'].copy()
    elif 'gene' in adata.obs.columns:
        # LOCO 数据使用 'gene' 列
        ctrl = adata[adata.obs['gene'] == 'non-targeting'].copy()
        real = adata[adata.obs['gene'] != 'non-targeting'].copy()
    else:
        print("Warning: Could not find control/perturbation column")
        print("Available columns:", adata.obs.columns.tolist())
        return None

    print(f"\n数据统计:")
    print(f"  预测细胞数: {pred.shape[0]}")
    print(f"  真实细胞数: {real.shape[0]}")
    print(f"  控制细胞数: {ctrl.shape[0]}")
    print(f"  预测基因数: {pred.shape[1]}")
    print(f"  真实基因数: {real.shape[1]}")

    # 找到共同基因
    common_genes = list(set(pred.var_names) & set(real.var_names) & set(ctrl.var_names))
    if len(common_genes) == 0:
        # 尝试使用索引
        common_genes = list(range(min(pred.shape[1], real.shape[1], ctrl.shape[1])))
        pred = pred[:, common_genes]
        real = real[:, common_genes]
        ctrl = ctrl[:, common_genes]
    else:
        pred = pred[:, common_genes]
        real = real[:, common_genes]
        ctrl = ctrl[:, common_genes]

    print(f"  共同基因数: {pred.shape[1]}")

    # 分析预测数据的方差
    pred_var = np.array(pred.X.var(axis=0)).flatten()
    real_var = np.array(real.X.var(axis=0)).flatten()
    ctrl_var = np.array(ctrl.X.var(axis=0)).flatten()

    print(f"\n方差分析:")
    print(f"  预测数据平均方差: {pred_var.mean():.6f}")
    print(f"  真实数据平均方差: {real_var.mean():.6f}")
    print(f"  控制数据平均方差: {ctrl_var.mean():.6f}")
    print(f"  方差比 (预测/真实): {pred_var.mean() / real_var.mean():.4f}")

    # 分析预测数据的 delta
    pred_mean = np.array(pred.X.mean(axis=0)).flatten()
    ctrl_mean = np.array(ctrl.X.mean(axis=0)).flatten()
    real_mean = np.array(real.X.mean(axis=0)).flatten()

    delta_pred = pred_mean - ctrl_mean
    delta_real = real_mean - ctrl_mean

    print(f"\nDelta 分析:")
    print(f"  预测 delta 均值: {delta_pred.mean():.6f}")
    print(f"  真实 delta 均值: {delta_real.mean():.6f}")
    print(f"  预测 delta 标准差: {delta_pred.std():.6f}")
    print(f"  真实 delta 标准差: {delta_real.std():.6f}")

    # 分析 DE 基因检测
    print(f"\nDE 基因检测分析:")

    # 模拟 t-test 检测
    sample_size = 100
    n_tests = 100
    sig_genes_pred = []
    sig_genes_real = []

    for i in range(n_tests):
        # 随机采样
        pred_idx = np.random.choice(pred.shape[0], sample_size, replace=True)
        ctrl_idx = np.random.choice(ctrl.shape[0], sample_size, replace=True)
        real_idx = np.random.choice(real.shape[0], sample_size, replace=True)

        # 计算 t-statistic
        for gene_idx in range(min(100, pred.shape[1])):
            pred_expr = pred.X[pred_idx, gene_idx]
            ctrl_expr = ctrl.X[ctrl_idx, gene_idx]
            real_expr = real.X[real_idx, gene_idx]

            # 预测 vs 控制
            t_pred, p_pred = scipy.stats.ttest_ind(pred_expr, ctrl_expr)
            if p_pred < 0.05:
                sig_genes_pred.append(gene_idx)

            # 真实 vs 控制
            t_real, p_real = scipy.stats.ttest_ind(real_expr, ctrl_expr)
            if p_real < 0.05:
                sig_genes_real.append(gene_idx)

    print(f"  预测数据中显著 DE 基因数: {len(set(sig_genes_pred))}")
    print(f"  真实数据中显著 DE 基因数: {len(set(sig_genes_real))}")

    # 分析问题
    print(f"\n问题诊断:")
    if pred_var.mean() < real_var.mean() * 0.5:
        print(f"  ⚠️  预测数据方差过低，导致 t-test 不显著")
        print(f"     建议: 增强预测数据的方差")

    if len(set(sig_genes_pred)) < len(set(sig_genes_real)) * 0.5:
        print(f"  ⚠️  预测数据中检测到的 DE 基因过少")
        print(f"     建议: 添加生物学噪声或调整阈值")

    return {
        "pred_var_mean": pred_var.mean(),
        "real_var_mean": real_var.mean(),
        "var_ratio": pred_var.mean() / real_var.mean(),
        "sig_genes_pred": len(set(sig_genes_pred)),
        "sig_genes_real": len(set(sig_genes_real)),
    }


def compare_methods():
    """比较不同方法的 DES 问题"""
    print("\n" + "=" * 60)
    print("比较不同方法的 DES 问题")
    print("=" * 60)

    # 加载 GEARS 结果
    gears_metrics = json.load(open("/home/zhangshibo24s/cell_flow/results/outputs/outputs_gears_loco/gears_loco_20260517_235246_metrics.json"))

    # 加载 scDFM 结果
    scdfm_metrics = json.load(open("/home/zhangshibo24s/cell_flow/results/outputs/outputs_scdfm_loco/scdfm_loco_20260517_204137_metrics.json"))

    print(f"\n方法对比:")
    print(f"{'方法':<20} {'MSE':<12} {'Pearson Δ':<12} {'DS':<8} {'DES Recall':<12}")
    print("-" * 70)
    print(f"{'GEARS':<20} {gears_metrics['mse']:<12.6f} {gears_metrics['pearson_delta']:<12.4f} {gears_metrics['direction_sign_score']:<8.4f} {gears_metrics['des_recall']:<12.4f}")
    print(f"{'scDFM':<20} {scdfm_metrics['mse']:<12.6f} {scdfm_metrics['pearson_delta']:<12.4f} {scdfm_metrics['direction_sign_score']:<8.4f} {scdfm_metrics['des_recall']:<12.4f}")

    print(f"\n关键发现:")
    print(f"  1. GEARS 的 DES Recall 远高于 scDFM (0.2971 vs 0.0000)")
    print(f"  2. scDFM 的整体精度更高 (MSE 更低，Pearson Δ 更高)")
    print(f"  3. 这说明 DES 指标对预测数据的方差敏感")


if __name__ == "__main__":
    # 分析 GEARS 的 DES
    analyze_des_issue(
        "/home/zhangshibo24s/cell_flow/results/outputs/outputs_gears_loco/predictions_20260517_235246.h5ad",
        "/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad",
        "GEARS"
    )

    # 比较方法
    compare_methods()
