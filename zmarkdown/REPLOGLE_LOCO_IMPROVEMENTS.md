# Replogle LOCO MyFlow 调试改进记录

日期: 2025-06-04

## 问题诊断过程

### 1. 初始问题：结果极差
- DEG R² = -0.19, Pearson Δ = 0.05, PCC = 0.09
- 对比方法: CellFlow Pearson Δ=0.50, TxPert Pearson Δ=0.62
- 发现: 跨条件预测相关度0.991，模型对所有perturbation输出几乎相同的预测

### 2. GNN图过大（→ 24K节点 → 68节点）
`_pert_genes` 使用全部gene2vec字典（24,447个基因）而非实际数据中的扰动基因（68个）。
- 24K节点 × 420K边 的GNN消息传递导致过度平滑
- 修复: 改用 `adata.obs["target_gene"].unique()` 只包含实际扰动基因
- 文件: `scripts/train_myflow_loco_new.py` line 911-916

### 3. cell_type条件干扰（→ 关闭）
LOCO是跨细胞系迁移任务。cell_type作为显式condition会让模型学到 `(gene, cell_type)` 组合映射，测试时遇到新组合 `(gene, hepg2)` 无法泛化。
- 修复: `use_cell_type_condition` 默认从 True 改为 False
- 保留 `split_covariates=["cell_type"]` 确保source/target配对不跨细胞系
- 文件: `scripts/train_myflow_loco_new.py` line 771

### 4. condition_embedding_dim过大（512 → 32）
512维condition embedding配合16维GNN输出导致信息瓶颈(512→16→512)，condition embedding全都塌缩成相似向量。
- 与Norman对齐: 使用32（ConditionalVelocityField默认值）
- 其他配置也与Norman对齐: `cosine_loss_weight=0.0`, `condition_combined_loss_weight=0.0`, `cond_output_dropout=0.0`

### 5. ODE预测爆炸（→ probability_path加噪声）
同一seed、相同配置下，训练结果随机性好坏。根因:
- `probability_path` 默认 `constant_noise(0.0)`：velocity field只需在精确的插值线上正确，线外可以任意尖锐
- 测试时source cell稍偏离训练轨迹就进入wild区域，ODE积分发散（max值到43721）
- 修复: 加 `probability_path={"constant_noise": 0.1}`，强制模型在插值线邻域内学习平滑velocity field
- 外加 `adamw` + `weight_decay=1e-5` 作为L2正则化
- 文件: `scripts/train_myflow_loco_new.py` line 1233

## 最终参数配置

| 参数 | 值 | 说明 |
|---|---|---|
| `pert-gnn-enabled` | True | 扰动基因GNN |
| `pert-gnn-hidden-dim` | 16 | GNN隐藏维度 |
| `condition-embedding-dim` | 32 | 条件嵌入维度（与Norman对齐） |
| `use-cell-type-condition` | False | 不用cell_type作condition |
| `endpoint-mse-weight` | 1.0 | 终点MSE监督权重 |
| `condition-combined-loss-weight` | 0.0 | 关闭组合分布损失 |
| `cosine-loss-weight` | 0.0 | 关闭余弦损失 |
| `cond-output-dropout` | 0.0 | 关闭dropout |
| `batch-size` | 256 | |
| `learning-rate` | 5e-4 | |
| `gradient-accumulation-steps` | 1 | |
| `match-every-n` | 20 | OT匹配频率 |
| `cross-attn-layers` | 1 | |
| `gene-attn-dim` | 16 | |
| `gene-self-attn-layers` | 0 | |
| `cross-attn-heads` | 4 | |
| `num-iterations` | 30000 | |
| optimizer | adamw (wd=1e-5) | |
| probability_path | constant_noise(0.1) | **关键：ODE稳定性** |
| GNN节点数 | 68（数据中的实际扰动基因） | |
| GO边数 | 110 | |

## 运行命令

```bash
GPU_MYFLOW=6 bash run_replogle_loco_5methods.sh
```

## 修改的文件

1. `scripts/train_myflow_loco_new.py`
   - `_pert_genes`: 24K → 68（只用数据中实际出现的扰动基因）
   - `use_cell_type_condition`: default True → False
   - `adam` → `adamw` (weight_decay=1e-5)
   - `probability_path`: 加 `{"constant_noise": 0.1}`
   - 预测clip: `(0, None)` → `(0, 10)`

2. `run_replogle_loco_5methods.sh`
   - 参数全部与上述配置对齐
   - GPU默认改用6（避免GPU 0-3残留内存）
