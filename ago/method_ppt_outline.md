# 基于 GO 增强的扰动响应预测方法

## 第 1 页：研究背景与主要研究问题

### 标题
单细胞扰动响应预测：从已观测扰动推断未知细胞状态变化

### 页面要点
- 单细胞扰动实验可以揭示基因调控、疾病机制和药物作用机制。
- 真实实验通常成本高、规模大、组合扰动空间庞大，难以穷尽所有扰动条件。
- 核心研究问题：给定未扰动细胞状态和扰动信息，预测细胞在该扰动下的基因表达响应。
- 关键挑战包括：
  - 未见过的单基因或组合扰动预测；
  - 不同细胞类型、不同数据集之间的泛化；
  - 扰动机制与基因功能知识的有效整合；
  - 预测结果不仅要拟合均值，还要保持真实表达分布特征。

### 图示建议
左侧画“未扰动细胞 + 扰动条件”，右侧画“预测扰动后表达谱”，中间用模型模块连接。

---

## 第 2 页：现有解决方法

### 标题
现有扰动响应预测方法

### 页面要点
- 基于深度生成模型的方法：
  - 使用 VAE、扩散模型、flow matching 等建模扰动前后细胞状态分布变化；
  - 优点是可以生成细胞级表达谱，适合建模复杂分布。
- 基于图神经网络的方法：
  - 利用基因调控网络、PPI 网络或扰动基因关系建模基因间依赖；
  - 典型思想是通过图结构传播扰动影响。
- 基于 Transformer / attention 的方法：
  - 利用 attention 机制建模基因、扰动、细胞状态之间的高阶交互；
  - 适合处理组合扰动和多 token 条件输入。
- 基于最优传输或流匹配的方法：
  - 学习从 control 分布到 target perturbation 分布的连续变换；
  - 适合从分布层面刻画扰动过程。

### 可提及的对比方法
- GEARS：强调基因图结构和组合扰动泛化。
- scDFM：使用 flow matching / diffusion 思路建模扰动响应。
- PerturbDiff、Squidiff：基于扩散或生成式建模的扰动预测方法。
- CellFlow：基于条件最优传输流匹配的开源扰动响应预测框架。

---

## 第 3 页：现有方法仍存在的问题

### 标题
现有方法的局限性

### 页面要点
- 生物功能知识利用不足：
  - 很多方法主要依赖表达数据或预训练基因表示；
  - GO 等功能注释知识没有被同时用于表达侧和扰动侧。
- 表达状态与扰动机制交互建模不足：
  - 扰动效果依赖当前细胞状态；
  - 仅简单拼接 cell embedding 和 perturbation embedding，可能无法充分捕捉两者之间的条件依赖。
- 对组合扰动和未见扰动的泛化仍然困难：
  - 组合扰动不是单扰动效果的简单线性叠加；
  - 对未见基因、未见组合或跨细胞类型场景，模型容易过拟合训练扰动。
- 训练目标对分布级一致性的约束不足：
  - 仅优化点级误差或 velocity matching，可能导致预测均值接近但表达分布偏差较大；
  - 对 terminal perturbed distribution 的约束不足会影响下游 DEG、DES 等指标。
  GEARS（Nature Biotechnology 2024）：通过基因图建模基因调控关系，以提升组合扰动情形下的泛化预测能力。
scDFM（ICLR 2026）：基于 flow matching / diffusion 框架学习细胞从未扰动到扰动状态的连续演化过程，用于预测扰动响应。
PerturbDiff / Squidiff：基于潜空间扩散生成模型进行扰动响应建模，但对未见扰动的泛化依赖训练分布覆盖。
CellFlow（bioRxiv 2025）：基于条件最优传输与 flow matching 建模细胞状态在扰动前后的连续映射关系。

### 图示建议
画四个问题框：知识缺失、交互不足、泛化困难、分布偏差。

---

## 第 4 页：本文方法的总体思路

### 标题
GO 增强的条件流匹配扰动响应预测框架

### 页面要点
- 本方法 MyFlow 在 CellFlow / OT-flow matching 框架基础上，引入 GO 知识增强和分布级联合训练目标。
- 总体输入包括：
  - control 细胞表达谱；
  - 扰动基因 token；
  - GO gene-to-gene graph / gene functional context。
- 核心思想：
  - 在表达侧引入 GO 图融合，使模型感知基因功能邻域；
  - 在扰动侧引入 GO 图融合，使扰动 token 包含功能相关基因上下文；
  - 使用 cross-attention 建模扰动条件与表达图表示之间的交互；
  - 使用 combined loss 约束预测终点分布，提高分布层面一致性。

### 一句话贡献
将 GO 功能知识同时注入细胞表达表示和扰动表示，并通过 cross-attention 与 combined loss 提升扰动响应预测的泛化能力和分布一致性。

---

## 第 5 页：我的解决方法解决了什么问题

### 标题
方法设计与问题对应关系

### 页面要点
| 现有问题 | 本方法对应设计 | 预期改进 |
| --- | --- | --- |
| GO 功能知识利用不足 | 表达侧 GO graph fusion + 扰动侧 GO token fusion | 引入生物功能先验，增强可解释性与泛化 |
| 表达状态与扰动机制交互不足 | 条件 embedding 查询表达侧 GO 图节点的 cross-attention | 建模“当前细胞状态下该扰动如何起作用” |
| 组合扰动泛化困难 | 每个扰动基因 token 独立融合 GO 上下文，再由 condition encoder 聚合 | 更好处理双基因扰动和未见组合 |
| 预测分布偏差 | flow matching loss + Sinkhorn / Energy combined distribution loss | 提高预测终点分布与真实扰动分布的一致性 |

### 强调点
- 不是简单加入一个 GO 特征，而是在表达侧和扰动侧分别建模 GO 知识。
- 不是简单拼接表达和扰动，而是通过 attention 让扰动条件选择相关的基因功能上下文。
- 不只优化局部速度场，也额外约束终点分布。

---

## 第 6 页：模型架构

### 标题
模型架构：GO 双侧增强 + Cross-attention + Combined loss

### 页面要点
- 输入模块：
  - control expression：未扰动细胞表达向量；
  - perturbation tokens：单基因或双基因扰动 token；
  - GO graph：由 gene2vec 与 GO gene-to-gene graph 构成的功能图。
- 表达侧 GO 融合：
  - 将表达值映射为基因节点表示；
  - 加入 gene2vec 表示；
  - 通过 GO 图进行消息传播；
  - 使用扰动条件 embedding 作为 query，对表达图节点进行 cross-attention；
  - 通过 gate 将 GO 图表示与原始表达 encoder 表示融合。
- 扰动侧 GO 融合：
  - 对每个扰动基因 token，基于 GO 图节点进行 attention readout；
  - 得到扰动相关的 GO neighborhood context；
  - 使用 gate 融合原始扰动 token 和 GO context；
  - 双扰动保持两个 token，后续由 condition encoder 聚合。
- 条件流匹配预测：
  - 编码时间 t、当前状态 x_t 和扰动条件 embedding；
  - 神经速度场预测 v_t；
  - 通过 ODE / flow integration 得到预测扰动后表达谱。
- 训练目标：
  - 基础 flow matching loss；
  - condition encoder regularization；
  - terminal distribution combined loss：Sinkhorn divergence + Energy distance。

### 图示建议
从左到右画：
Control expression / Perturbation tokens / GO graph → 双分支 GO fusion → Cross-attention → Conditional velocity field → Predicted perturbed expression。
底部单独画 Combined loss，连接 predicted terminal state 和 observed perturbed expression。

---

## 第 7 页：数据集与任务设置

### 标题
实验数据集与评估场景

### 页面要点
- Norman 2019 K562 组合扰动数据集：
  - 包含单基因扰动和双基因组合扰动；
  - 适合评估组合扰动预测能力；
  - 使用 additive split：测试部分双基因组合，训练中保留相关单基因扰动；
  - 使用 holdout split：留出部分基因，测试涉及这些基因的单扰动和组合扰动。
- Replogle 数据集：
  - 大规模 CRISPR 扰动数据；
  - 用于评估跨条件或跨细胞类型泛化；
  - LOCO 设置：留出特定细胞类型或条件用于测试。
- GO / gene functional graph：
  - 使用 gene2vec 作为基因初始表示；
  - 使用 GO gene-to-gene graph 表示基因功能关联；
  - 同时服务于表达侧图融合和扰动侧 token 融合。

### 评估目标
- 未见组合扰动预测；
- 未见基因或 held-out gene 泛化；
- 跨细胞类型 / LOCO 泛化；
- 表达分布一致性和差异表达相关指标。

---

## 第 8 页：预期实验与消融设计

### 标题
实验设计：验证每个模块的贡献

### 页面要点
- 主实验：
  - 与 GEARS、scDFM、PerturbDiff、Squidiff、CellFlow baseline 对比；
  - 在 Norman additive、Norman holdout、Replogle LOCO 上评估。
- 消融实验：
  - 去掉表达侧 GO fusion；
  - 去掉扰动侧 GO fusion；
  - 去掉 combined loss；
  - GO graph attention 与 neighborhood-only graph attention 对比；
  - clean baseline：关闭 GO graph fusion，并设置 combined loss 权重为 0。
- 评估指标：
  - 表达预测误差；
  - Pearson / Spearman correlation；
  - DEG overlap / DES；
  - 分布距离或下游扰动效应一致性。

### 预期结论
如果完整模型在未见组合、held-out gene 和 LOCO 设置下稳定优于 baseline，说明 GO 双侧增强、cross-attention 交互建模和 combined loss 对泛化与分布一致性均有贡献。

---

## 第 9 页：总结

### 标题
总结

### 页面要点
- 本研究关注单细胞扰动响应预测中的泛化和分布一致性问题。
- 现有方法在功能知识利用、表达-扰动交互建模、组合扰动泛化和分布级约束方面仍有不足。
- 本方法提出 GO 双侧增强框架：
  - 表达侧加入 GO graph fusion；
  - 扰动侧加入 GO token fusion；
  - 使用 cross-attention 建模表达状态与扰动机制交互；
  - 使用 combined loss 强化终点扰动分布约束。
- 该框架有望提升未见扰动、组合扰动和跨细胞类型场景下的预测性能。
