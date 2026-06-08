# 研究点1：基于先验知识引导的条件流匹配扰动预测

本文档从当前实验入口脚本和其 import 的核心代码出发，梳理第一个研究点的真实实现。对应入口包括 `run_norman_additive_5methods.sh`、`run_replogle_loco_5methods.sh`、`scripts/train_myflow_norman_additive.py` 和 `scripts/train_myflow_loco_new.py`；核心模型实现位于 `myflow/model/_myflow.py`、`myflow/networks/_velocity_field.py`、`myflow/networks/_set_encoders.py`、`myflow/data/_datamanager.py` 和 `myflow/solvers/_otfm.py`。

## 1. 研究目标

第一个研究点面向虚拟细胞扰动预测任务，目标是在条件流匹配框架下预测细胞从对照状态到扰动状态的转录组分布变化。与仅将扰动基因作为 one-hot 或 gene2vec 查表特征的方法不同，本研究将 GO 功能关联和 STRING 蛋白互作网络显式注入扰动条件编码，并通过基因级注意力机制将扰动条件与当前表达状态融合，从而提升未见组合扰动和跨细胞系泛化能力。

当前方法可命名为 PriorFlow，即 Prior-knowledge Guided Conditional Flow Matching。其核心不是把模型改成单纯的终点均值预测器，而是在 MyFlow/OT-CFM 的分布生成路径上加入扰动基因图先验、条件到基因的交叉注意力和扰动条件感知的速度场掩码。

## 2. 实验入口与评估设置

### Norman additive

入口脚本为 `run_norman_additive_5methods.sh`，调用 `scripts/train_myflow_norman_additive.py`。该实验使用 Norman 2019 K562 组合扰动数据，采用 scDFM-style additive split：测试集为一部分双基因扰动组合，训练集中保留所有单基因扰动以及未被划入测试的组合扰动。因此该设置主要评估模型对未见双基因组合扰动的组合泛化能力。

MyFlow 运行配置启用：

- `--pert-gnn-enabled`
- `--enhanced-pert-gnn`
- `--condition-embedding-dim 256`
- `--pert-gnn-hidden-dim 128`
- `--pert-gnn-num-layers 4`
- `--pert-gnn-num-heads 4`

对比方法包括 GEARS、CellFlow、scDFM、TxPert 和 CPA。

### Replogle LOCO

入口脚本为 `run_replogle_loco_5methods.sh`，调用 `scripts/train_myflow_loco_new.py`。该实验使用 Replogle 数据集，采用 Leave-One-Cell-Line-Out 设置，默认留出 `hepg2` 细胞系。训练集包含其他细胞系的扰动响应，同时保留 hepg2 的 non-targeting 对照和一部分 hepg2 扰动；测试集为 hepg2 中另一部分扰动。测试扰动基因在其他细胞系中出现过，但其 hepg2 响应被完全留出，因此该设置主要评估跨细胞系迁移泛化。

MyFlow 运行配置启用：

- `--pert-gnn-enabled`
- `--enhanced-pert-gnn`
- `--condition-embedding-dim 512`
- `--pert-gnn-hidden-dim 128`
- `--pert-gnn-num-layers 4`
- `--pert-gnn-num-heads 4`
- `--cross-attn-layers 1`
- `--gene-attn-dim 64`
- `--cross-attn-heads 4`
- `--match-every-n 20`

对比方法同样包括 GEARS、CellFlow、scDFM、TxPert 和 CPA。

## 3. 数据与扰动 token 构建

两个训练脚本均将扰动基因统一到 gene symbol 空间，避免 Ensembl ID、gene symbol 和表达矩阵基因名之间的错配。

Norman 脚本中，`guide_merged` 表示单扰动或双扰动条件。脚本通过 `_parse_condition_genes` 将如 `A+B` 的组合解析为最多两个扰动基因 token，并构建 `pert_gene_1`、`pert_gene_2` 两列。每个 token 使用 `data_gab/gene2vec_dict.pt` 中的 gene2vec 作为基础表示；缺失基因使用零向量并以 `missing::GENE` 标识。对照或 padding 使用 `ctrl` 零向量。

Replogle 脚本中，`obs["gene"]` 被作为 `target_gene`，`non-targeting` 表示对照。脚本要求使用 gene symbol，不允许退回到 `gene_id`。每个扰动条件由单个 `target_gene` token 表示。

两个脚本都会在 `adata.uns["perturb_gene_symbol_to_idx"]` 中写入扰动基因到整数索引的映射。随后 `myflow/data/_datamanager.py` 在构建条件张量时会额外生成 `gene_perturbation_indices`。这一步很关键：普通 gene2vec token 仍可作为输入，但模型中的扰动 GNN 分支会优先使用这些整数索引，在可学习扰动节点嵌入表上进行图消息传递。

## 4. 先验图构建

扰动图节点只包含当前数据集中实际出现的扰动基因，而不是全基因组或完整 gene2vec 字典。这样可以避免在大规模无关节点上过度平滑，使扰动条件编码更聚焦。

边来自两个先验来源：

- GO 功能图：默认读取 TxPert 公开包中的 `go_top_50.csv`，边字段包括 `source`、`target` 和 `importance`。
- STRING PPI：默认读取 TxPert 公开包中的 `v11.5.parquet`，边字段包括 `regulator`、`target` 和 `weight`。

脚本先筛选出两端都属于扰动基因集合的 GO 边。若 GO 边数量小于 100，则补充 STRING PPI 子图。边权按目标节点入度归一化：

$$
\tilde{w}_{ij}=\frac{w_{ij}}{\sum_k w_{kj}+\epsilon}.
$$

最终图以 `edge_src`、`edge_tgt`、`edge_w` 传入 `MyFlow.prepare_model(..., perturbation_gnn_kwargs=...)`。

## 5. 模型调用链

训练脚本调用：

```python
cf = MyFlow(adata, solver="otfm")
cf.prepare_data(...)
cf.prepare_model(..., perturbation_gnn_kwargs=..., ...)
cf.train(...)
```

`myflow/model/_myflow.py` 中的 `prepare_model` 会构建 `ConditionalVelocityField`，并将 `perturbation_gnn_kwargs` 注入到向量场对象的 `x_gnn_config`。随后 `ConditionalVelocityField.setup` 根据 `num_pert_genes`、`edge_src`、`edge_tgt`、`edge_w`、`enhanced_gnn` 等参数创建扰动侧 GNN。

当 batch 中包含 `gene_perturbation_indices` 时，`ConditionalVelocityField.__call__` 会用扰动 GNN 的输出替换普通的 `gene_perturbation` 表示：

```python
cond["gene_perturbation"] = self.perturbation_gnn(
    cond["gene_perturbation_indices"],
    self.pert_embeddings,
    deterministic=not train,
)
```

因此，条件编码的真实路径是：

扰动基因 symbol → DataManager 整数索引 → 可学习扰动节点嵌入 → GO/STRING GNN → ConditionEncoder attention pooling → 条件嵌入。

## 6. 扰动侧图编码器

基础 `PerturbationGNN` 位于 `myflow/networks/_set_encoders.py`，采用固定归一化边权进行消息传递。每层计算邻居加权聚合、MLP 变换、dropout 和残差 LayerNorm，然后根据当前扰动条件的 gene index gather token 表示。

增强版 `EnhancedPerturbationGNN` 是当前脚本启用的版本。它包括：

- 可学习扰动节点嵌入；
- 多头 GATv2 风格图注意力；
- 边权作为 attention bias；
- virtual node 用于全局信息交换；
- 多层残差归一化和 FFN。

对于 Norman 双扰动，GNN 会分别返回两个扰动基因 token；对于 Replogle 单扰动，返回一个 token。padding 位置使用 `-1`，模型会通过 valid mask 将其置零。

## 7. 条件编码与表达状态融合

`ConditionEncoder` 使用 set encoder 处理单扰动或多扰动 token。默认 pooling 是 `attention_token`，即加入一个可学习 class token，通过多头注意力聚合多个扰动基因，输出固定维度的条件嵌入。

`ConditionalVelocityField` 不再把表达向量简单送入 MLP，而是将每个基因表达值拆成两部分：

- 表达量投影：`gene_val_proj(x_t[:, :, None])`
- 可学习基因身份嵌入：`gene_id_emb`

二者相加得到基因级表达 token。可选的 gene self-attention 用于表达基因之间的信息交换。随后模型以条件嵌入作为 query，以基因表达 token 作为 key/value，进行 condition-to-gene cross-attention。该设计让扰动条件主动检索当前表达谱中与其相关的基因上下文，而不是简单拼接条件向量和表达向量。

cross-attention 输出经过 `fusion_proj` 后与正弦时间编码一起输入 decoder，预测流匹配速度场 $v_\theta(x_t,t|c)$。

## 8. 扰动条件感知速度场掩码

`ConditionalVelocityField` 中还包含 `gene_mask_head`。它从条件嵌入预测每个基因的 mask：

$$
m_c=\sigma(W_m z_c+b_m),
$$

并做均值归一化，最后乘到 velocity 上：

$$
v_\theta^{final}=v_\theta \odot \frac{m_c}{\mathrm{mean}(m_c)+\epsilon}.
$$

该设计体现了扰动响应的稀疏性先验：一个扰动通常只影响部分基因程序。均值归一化避免模型通过整体缩小 velocity 来逃避训练损失，使 mask 更像“基因重要性重分配”而不是简单的全局收缩。

## 9. 训练目标

训练器使用 `myflow/solvers/_otfm.py` 中的 `OTFlowMatching`。基础目标为条件流匹配损失：

$$
\mathcal{L}_{FM}=\mathbb{E}\|v_\theta(x_t,t,c)-u_t(x_t|x_0,x_1)\|^2.
$$

训练中可选加入多种终端和扰动效应约束，包括：

- endpoint MSE：约束 $\hat{x}_1=x_t+(1-t)v_\theta$ 接近真实扰动细胞；
- condition mean delta loss：按扰动条件对齐平均扰动效应；
- top-delta loss：强化真实变化幅度最大的基因；
- SNR endpoint loss：对稳定且高信噪比的扰动响应基因提高权重；
- cosine delta loss：约束预测扰动方向；
- combined distribution loss：用 energy distance 或 Sinkhorn 对齐终端分布；
- delta head loss：可选辅助分支，从条件嵌入直接预测 per-gene delta。

当前五方法脚本中，Norman 和 Replogle 的部分损失权重通过环境变量控制；实际默认重点是保留 OT-CFM 分布生成主路径，并视任务需要加入 endpoint/SNR/cosine 等弱监督。

## 10. 评价指标

训练脚本在预测后按扰动条件进行评估，避免把不同扰动条件混在一起导致正负效应抵消。主要指标包括：

- MSE、MAE、L2；
- Pearson delta、top-k Pearson delta；
- fold-change delta 相关；
- direction sign agreement；
- DEG 子集上的 R2、EV、PCC；
- DEG overlap 的 precision、recall、F1、Jaccard；
- DE Spearman。

Norman additive 重点验证未见组合扰动；Replogle LOCO 重点验证跨细胞系泛化。对比方法覆盖图模型、流匹配模型、扩散模型、Transformer/GNN 方法和自编码器方法，能够支撑第一个研究点的实验论证。

## 11. 可写入开题报告的核心表述

第一个研究点可以概括为：构建一个基于生物先验知识增强的条件流匹配框架，将 GO 功能关系和 STRING PPI 网络编码为扰动基因图，通过可学习扰动节点嵌入和增强图注意力网络获得扰动条件表示；再通过基因身份嵌入、条件到基因的交叉注意力和扰动条件感知速度场掩码，将扰动机制与当前细胞表达状态耦合；最后在 OT-CFM 框架下学习从对照分布到扰动分布的连续向量场，并在 Norman additive 和 Replogle LOCO 两类泛化设置中评估未见组合扰动和跨细胞系预测能力。
