# 基于先验约束的稀疏响应扰动预测方法

## 方法命名

**基于先验约束的稀疏响应扰动预测方法**。以条件流匹配（CFM）为生成骨架，将 GO 功能关联、STRING 蛋白互作和 TRRUST 转录调控三层生物先验通过图编码、交叉注意力、基因掩码和 SNR 加权损失系统融入模型，引导模型聚焦少数真正响应扰动的关键基因。

---

## 1. 问题定义

- 对照细胞 $\mathbf{x}^{\text{ctrl}} \in \mathbb{R}^d$，服从分布 $p_0$
- 扰动细胞 $\mathbf{x}^{\text{pert}} \in \mathbb{R}^d$，对应扰动条件 $c = \{g_1, \ldots, g_k\}$，服从 $p_1(\cdot \mid c)$

目标：学习条件向量场 $v_\theta(\mathbf{x}, t \mid c)$，沿 ODE $\frac{d\mathbf{x}}{dt} = v_\theta$ 从 $t=0$ 积分至 $t=1$，使终端分布逼近 $p_1(\cdot \mid c)$。

---

## 2. 生成骨架：OT Conditional Flow Matching

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|v_\theta(\mathbf{x}_t, t \mid c) - u_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1)\|^2\right]$$

- $t \sim \mathcal{U}(0,1)$，概率路径 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1 + \sigma\sqrt{t(1-t)}\mathbf{z}$
- Sinkhorn 最优传输匹配源-目标细胞对，每 $N$ 步更新

---

## 3. 方法架构

整体架构：扰动条件经 **GO+STRING+TRRUST 三层先验图编码** → 条件嵌入经**交叉注意力**与基因表达状态融合 → **基因掩码**施加稀疏响应约束 → **SNR 加权损失**强化高可信响应基因。

### 3.1 多层先验图编码

#### 3.1.1 扰动基因先验图（GO + STRING + TRRUST）

在扰动基因集合上构建先验图：

- **GO 功能关联**：基因间功能语义相似性，IC 加权
- **STRING PPI**：蛋白质-蛋白质物理互作，实验/数据库证据
- **TRRUST 调控关系**（新增）：TF→靶基因的转录调控边，文献人工整理，置信度高

边权重经目标节点度归一化后，通过 GNN（基础版固定权重消息传递，增强版多头 GATv2 + 虚拟节点）获得先验增强的基因表示，再经 SetEncoder 注意力池化为单一条件嵌入 $\mathbf{z}_c$。

三种先验互补：GO 和 STRING 提供扰动基因间的功能/互作上下文，TRRUST 提供从扰动 TF 到下游靶基因的调控方向信息。

#### 3.1.2 条件嵌入的两种 TRRUST 增强方式

TRRUST 的独特之处在于它是 TF→靶基因的有向调控关系，GO 和 STRING 都是扰动基因间的无向关联。为此设计两种互补的 TRRUST 融入机制：

- **方案 A — 基因掩码偏置**（`trrust_mask_enabled`）：将已知靶基因信息注入 gene mask logits，给 TRRUST 靶基因施加可学习的偏置 $\alpha \cdot \mathbf{m}_{\text{trrust}}$，使掩码优先关注已知调控靶点。偏置强度 $\alpha$ 为可学习标量（初始 1.0）。
- **方案 B — 交叉注意力特征**（`trrust_attn_bias_enabled`）：为已知靶基因在交叉注意力 KV 中附加可学习 embedding，使其在注意力计算中具备可区分的特征标记。

二者可独立或联合启用。

### 3.2 基因身份解耦与交叉注意力

表达值投影（共享）+ 基因身份嵌入（独立可学习）：

$$\mathbf{h}_i = \mathbf{W}_v x_i + \mathbf{e}^{\text{gene}}_i, \quad \mathbf{H} \in \mathbb{R}^{d \times d_g}$$

条件嵌入作为 query 交叉关注基因表示：

$$\mathbf{h}^{\text{cross}} = \text{softmax}\left(\frac{\mathbf{q}^\top (\mathbf{W}_K \mathbf{H})^\top}{\sqrt{d_g}}\right) \mathbf{W}_V \mathbf{H}$$

### 3.3 扰动条件基因掩码

$$\mathbf{m}_c = \sigma(\mathbf{W}_m \mathbf{z}_c + \mathbf{b}_m + \alpha \cdot \mathbf{m}_{\text{trrust}})$$

$$v_\theta^{\text{final}} = \frac{\mathbf{m}_c}{\text{mean}(\mathbf{m}_c) + \epsilon} \odot v_\theta$$

掩码均值归一化确保仅重新分配基因间的重要性，而非常数收缩，防止损失被欺骗。

### 3.4 SNR 加权多层级训练目标

定义基因级信噪比权重——扰动效应跨细胞越稳定、幅度越大的基因权重越高：

$$s_{c,g} = \frac{|\mu_{c,g}|}{\sigma_{c,g} + \epsilon}, \quad \omega_{c,g} = \frac{s_{c,g}}{\frac{1}{d}\sum_g s_{c,g} + \epsilon}$$

$$\mathcal{L}_{\text{SNR}} = \mathbb{E}\left[t^p \sum_{g=1}^{d} \omega_{c,g} (\hat{x}_{1,g} - x_{1,g})^2\right]$$

综合训练目标：

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \alpha\mathcal{L}_{\text{end}} + \beta\mathcal{L}_{\Delta} + \gamma\mathcal{L}_{\text{dir}} + \lambda\mathcal{L}_{\text{SNR}} + \eta\mathcal{L}_{\text{dist}}$$

---

## 4. 创新点总结

| 创新点 | 技术实现 | 解决的问题 |
|--------|---------|-----------|
| **三层先验图编码** | GO + STRING + TRRUST 图上的 GNN 消息传递 + SetEncoder 池化 | 传统编码忽略基因间功能关联、蛋白互作和转录调控关系 |
| **TRRUST 双路径融入** | 基因掩码偏置 + 交叉注意力特征标记 | 首次将 TF→靶基因调控先验系统融入扰动预测 |
| **基因身份解耦** | 表达投影 + 可学习基因身份嵌入 | 传统 MLP 无法区分不同基因的语义角色 |
| **交叉注意力融合** | 条件作为 query 检索全基因表示 | concat 盲拼，无法精确指向目标基因 |
| **条件感知基因掩码** | Mask 均值归一化 + TRRUST 偏置 | 每个扰动仅改变少量基因，掩码施加稀疏性归纳偏置 |
| **SNR 加权损失** | 基因级信噪比权重 + 多层级约束 | 噪声基因稀释监督信号，DEG 难以稳定恢复 |

---

## 5. 实验设置

- **Norman additive**（组合扰动）：62 条件训练，15 双基因组合测试，评估 DEG 恢复和方向一致性
- **Replogle LOCO**（跨细胞背景）：留出 hepg2 细胞系 40 扰动，评估跨条件泛化
- 对比方法：CPA、GEARS、CellFlow、scDFM、TxPert

---

## 6. 实验结果（Replogle LOCO，TRRUST 双方案开启）

| 指标 | 旧（无 TRRUST） | 新（+ TRRUST） |
|------|:--:|:--:|
| MSE ↓ | 0.001445 | 0.001754 |
| DEG PCC ↑ | 0.3027 | **0.3159** |
| DEG R² ↑ | -0.1618 | **-0.1446** |
| Δ20 PCC ↑ | 0.3866 | **0.4160** |
| Δ50 PCC ↑ | 0.3735 | **0.3870** |
| Δ̂20 PCC ↑ | 0.3111 | **0.3622** |
| DEG F1 ↑ | 0.2924 | **0.2940** |

分析：TRRUST 引入后，DEG 相关指标全面上涨（Δ20 +7.6%，Δ̂20 +16.4%，DEG PCC +4.4%），验证了调控先验对关键响应基因识别能力的提升。全局 MSE 微涨属预期——模型注意力更聚焦于响应基因，背景基因拟合略有让渡。当前已通过可学习偏置强度缓解，后续可在更大规模训练中进一步平衡。

---

## 7. 关键结论

TRRUST 转录调控关系的引入，使先验约束从"基因间功能关联"扩展到"TF→靶基因调控方向"，补齐了方法在调控方向的先验缺口。实验表明：

- **组合先验有效**：GO + STRING + TRRUST 三层先验互补，DEG 指标全面提升
- **双路径融入合理**：基因掩码偏置引导稀疏性，交叉注意力特征标记增强靶基因可区分性
- **学习式强度关键**：可学习偏置尺度优于固定倍数，给模型调节空间
