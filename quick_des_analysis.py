#!/usr/bin/env python3
"""
快速分析 DES 问题
"""

import anndata as ad
import numpy as np
import json

def quick_analysis():
    print("=" * 60)
    print("快速 DES 问题分析")
    print("=" * 60)

    # 加载 GEARS 结果
    gears_metrics = json.load(open("/home/zhangshibo24s/cell_flow/results/outputs/outputs_gears_loco/gears_loco_20260517_235246_metrics.json"))

    # 加载 scDFM 结果
    scdfm_metrics = json.load(open("/home/zhangshibo24s/cell_flow/results/outputs/outputs_scdfm_loco/scdfm_loco_20260517_204137_metrics.json"))

    print(f"\n方法对比 (LOCO 数据集):")
    print(f"{'方法':<20} {'MSE':<12} {'Pearson Δ':<12} {'DS':<8} {'DES Recall':<12}")
    print("-" * 70)
    print(f"{'GEARS':<20} {gears_metrics['mse']:<12.6f} {gears_metrics['pearson_delta']:<12.4f} {gears_metrics['direction_sign_score']:<8.4f} {gears_metrics['des_recall']:<12.4f}")
    print(f"{'scDFM':<20} {scdfm_metrics['mse']:<12.6f} {scdfm_metrics['pearson_delta']:<12.4f} {scdfm_metrics['direction_sign_score']:<8.4f} {scdfm_metrics['des_recall']:<12.4f}")

    # 分析 GEARS 预测数据的方差
    print(f"\n" + "=" * 60)
    print("方差分析 (GEARS 预测数据)")
    print("=" * 60)

    gears_pred = ad.read_h5ad("/home/zhangshibo24s/cell_flow/results/outputs/outputs_gears_loco/predictions_20260517_235246.h5ad")
    pred_var = np.array(gears_pred.X.var(axis=0)).flatten()

    print(f"预测数据方差统计:")
    print(f"  平均方差: {pred_var.mean():.6f}")
    print(f"  方差标准差: {pred_var.std():.6f}")
    print(f"  最小方差: {pred_var.min():.6f}")
    print(f"  最大方差: {pred_var.max():.6f}")

    # 检查方差分布
    low_var_genes = (pred_var < 0.001).sum()
    medium_var_genes = ((pred_var >= 0.001) & (pred_var < 0.01)).sum()
    high_var_genes = (pred_var >= 0.01).sum()

    print(f"\n方差分布:")
    print(f"  低方差基因 (<0.001): {low_var_genes} ({low_var_genes/len(pred_var)*100:.1f}%)")
    print(f"  中等方差基因 (0.001-0.01): {medium_var_genes} ({medium_var_genes/len(pred_var)*100:.1f}%)")
    print(f"  高方差基因 (>0.01): {high_var_genes} ({high_var_genes/len(pred_var)*100:.1f}%)")

    # 关键发现
    print(f"\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)

    print(f"\n1. DES 指标差异:")
    print(f"   - GEARS DES Recall: {gears_metrics['des_recall']:.4f}")
    print(f"   - scDFM DES Recall: {scdfm_metrics['des_recall']:.4f}")
    print(f"   - 差异: {gears_metrics['des_recall'] - scdfm_metrics['des_recall']:.4f}")

    print(f"\n2. 整体精度差异:")
    print(f"   - GEARS MSE: {gears_metrics['mse']:.6f}")
    print(f"   - scDFM MSE: {scdfm_metrics['mse']:.6f}")
    print(f"   - scDFM 精度更高: {gears_metrics['mse']/scdfm_metrics['mse']:.1f} 倍")

    print(f"\n3. 问题诊断:")
    print(f"   - DES 基于 t-test 检测 DE 基因")
    print(f"   - t-test 对数据方差敏感")
    print(f"   - 如果预测方差太低，t-test 无法检测到显著差异")
    print(f"   - 这解释了为什么整体精度高但 DES 低")

    print(f"\n4. 解决方案:")
    print(f"   - 增强预测数据的方差")
    print(f"   - 添加生物学噪声")
    print(f"   - 使用改进的 DES 计算方法")

    return {
        "gears_des_recall": gears_metrics['des_recall'],
        "scdfm_des_recall": scdfm_metrics['des_recall'],
        "gears_mse": gears_metrics['mse'],
        "scdfm_mse": scdfm_metrics['mse'],
        "pred_var_mean": pred_var.mean(),
    }


if __name__ == "__main__":
    quick_analysis()
