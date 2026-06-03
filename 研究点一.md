# BioPrior-FM: Biological Prior-informed Flow Matching for Single-Cell Perturbation Response Prediction

## 方法命名

**BioPrior-FM**（Biological Prior-informed Flow Matching），全称：基于生物先验知识的流匹配扰动响应预测方法。

方法核心思路：将基因本体论（Gene Ontology, GO）功能相似性图和蛋白质互作网络（STRING PPI）作为先验知识，通过共享的gene2vec嵌入空间和图神经网络编码，显式建模扰动基因与全部输出基因之间的功能关联，从而引导Conditional Flow Matching的向量场学习，实现对单细胞扰动响应的高精度预测。

---

## 1. 问题定义

给定单细胞基因表达数据集，包含：

- **对照细胞**（control）：未受扰动的细胞，表达谱记为 $\mathbf{x}^{\text{ctrl}} \in \mathbb{R}^d$，其中 $d$ 为基因数
- **扰动细胞**（perturbed）：受到特定基因扰动（如敲除/敲低）的细胞，表达谱记为 $\mathbf{x}^{\text{pert}} \in \mathbb{R}^d$
- **扰动条件** $c$：由一组扰动基因构成的集合 $c = \{g_1, g_2, \ldots, g_k\}$，每个基因 $g_j$ 通过预训练的gene2vec嵌入 $\mathbf{e}_{g_j} \in \mathbb{R}^{d_e}$ 表示

**目标**：学习一个条件向量场 $v_\theta(\mathbf{x}, t \mid c)$，使得从对照分布 $p_0$ 沿ODE $d\mathbf{x}/dt = v_\theta(\mathbf{x}, t \mid c)$ 积分到 $t=1$ 时，生成的样本 $\mathbf{x}_1$ 的分布逼近真实的扰动分布 $p_1(\cdot \mid c)$。

---

## 2. 背景：Conditional Flow Matching (CFM)

Flow Matching (Lipman et al., 2022) 通过学习一个时变向量场 $v_t(\mathbf{x})$ 来变换概率分布。给定 $t \in [0,1]$ 时的条件概率路径 $p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$ 和对应的条件向量场 $u_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$，CFM的训练目标为：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_t}\left[\|v_\theta(\mathbf{x}_t, t \mid c) - u_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1)\|^2\right]$$

其中 $t \sim \mathcal{U}(0, 1)$，$(\mathbf{x}_0, \mathbf{x}_1)$ 为源-目标细胞对（分别来自对照和扰动分布），$\mathbf{x}_t$ 通过概率路径从 $\mathbf{x}_0$ 插值到 $\mathbf{x}_1$。

我们采用 **OT-CFM**（Optimal Transport CFM, Tong et al., 2023; Pooladian et al., 2023），通过Sinkhorn算法求解源-目标细胞间的最优传输匹配矩阵 $\mathbf{M} \in \mathbb{R}^{n \times n}$，然后按匹配概率采样训练对：

$$\mathbf{M} = \arg\min_{\mathbf{M} \in \Pi(\mathbf{p}, \mathbf{q})} \langle \mathbf{M}, \mathbf{C} \rangle - \varepsilon H(\mathbf{M})$$

其中 $\mathbf{C}_{ij} = \|\mathbf{x}_0^{(i)} - \mathbf{x}_1^{(j)}\|^2$ 为代价矩阵，$\varepsilon$ 为熵正则系数。此匹配每 $N$ 步重新计算一次，使训练过程中的源-目标配对保持最优。

概率路径选用 **Brownian Bridge**：

$$\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1 + \sigma \sqrt{t(1 - t)} \cdot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

对应条件向量场：

$$u_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1) = \frac{\mathbf{x}_1 - \mathbf{x}_t}{1 - t} - \frac{\sigma^2(1 - 2t)}{2t(1 - t)} \cdot \mathbf{z}$$

---

## 3. BioPrior-FM 整体架构

BioPrior-FM的核心创新在于将生物先验知识（GO功能图谱 + PPI互作网络）系统地融入Flow Matching范式，具体通过以下模块实现：

### 3.1 条件编码器（Condition Encoder）

扰动条件 $c$ 被编码为一个集合。每个扰动基因 $g_j$ 具有gene2vec嵌入 $\mathbf{e}_{g_j} \in \mathbb{R}^{d_e}$，形成扰动token序列 $\mathbf{T}_c \in \mathbb{R}^{k \times d_e}$。

条件编码器采用**Set Encoder**结构，将变长的基因集合编码为固定维度的条件嵌入 $\mathbf{z}_c \in \mathbb{R}^{d_z}$：

$$\mathbf{z}_c, \log\boldsymbol{\sigma}_c^2 = \text{SetEncoder}(\mathbf{T}_c)$$

Set Encoder内部流程：
1. **输入投影**：每个扰动token通过MLP映射到隐空间
2. **Self-Attention交互**：利用多头自注意力建模扰动基因间的功能协同
3. **注意力池化**：通过可学习的 `[CLS]` token（`attention_token`模式）或种子注意力（`attention_seed`模式）聚合多个扰动token为单一的全局条件嵌入
4. **输出生成**：池化后的表示经过MLP产生条件嵌入 $\mathbf{z}_c$

编码器支持两种模式：
- **确定模式**：直接输出 $\mathbf{z}_c$，正则化为L2范数惩罚 $\frac{1}{2}\|\mathbf{z}_c\|^2$
- **随机模式**：输出高斯分布参数，$\mathbf{z}_c \sim \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\sigma}_c^2)$，正则化为KL散度

### 3.2 GO先验响应编码器（GO Response Prior Encoder）

这是BioPrior-FM的**核心贡献模块**。其关键洞察是：扰动基因的功能效应通过GO功能相似性图谱传递到所有输出基因，这种传递可以通过共享图编码显式建模。

**构建共享基因嵌入表**：将全部输出基因（如2000个HVG）和全部扰动基因（如68个）合并到同一张gene2vec表中，所有基因共享同一个GO功能图谱。

$$\mathbf{G} \in \mathbb{R}^{N_{\text{all}} \times d_e}, \quad N_{\text{all}} = N_{\text{out}} + N_{\text{pert}}$$

**图消息传递**：在GO图上执行GAT-style注意力消息传递（Graph Attention Network）：

$$\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_e})}{\sum_{j' \in \mathcal{N}(i)} \exp(\mathbf{q}_i^\top \mathbf{k}_{j'} / \sqrt{d_e})}$$

$$\mathbf{g}_i^{(l+1)} = \text{LayerNorm}\left(\mathbf{g}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \cdot \mathbf{g}_j^{(l)}\right)$$

其中 $\mathbf{q}_i = \mathbf{W}_Q \mathbf{g}_i$, $\mathbf{k}_j = \mathbf{W}_K \mathbf{g}_j$，$\mathcal{N}(i)$ 为基因 $i$ 在GO图中的邻居集（取top-$K$重要性边）。共进行 $L$ 层消息传递。

**拆分嵌入**：消息传递后，将更新后的节点嵌入按索引拆分为两部分：

$$\mathbf{Z}_{\text{out}} = \mathbf{G}^{(L)}[\mathcal{I}_{\text{out}}] \in \mathbb{R}^{N_{\text{out}} \times d_e}, \quad \mathbf{Z}_{\text{pert}} = \mathbf{G}^{(L)}[\mathcal{I}_{\text{pert}}] \in \mathbb{R}^{N_{\text{pert}} \times d_e}$$

**对偶特征先验**：对于Batch中的每个细胞，通过pairwise交互建模扰动基因与每个输出基因的关系：

$$\text{pair}(\mathbf{z}_i, \mathbf{z}_p) = [\mathbf{z}_i \;\|\; \mathbf{z}_p \;\|\; \mathbf{z}_i \odot \mathbf{z}_p \;\|\; |\mathbf{z}_i - \mathbf{z}_p|] \in \mathbb{R}^{4d_e}$$

$$\boldsymbol{\rho} = \text{MLP}_{\text{out}}(\text{SiLU}(\text{MLP}_{\text{in}}(\text{pair}(\mathbf{Z}_{\text{out}}, \mathbf{z}_p)))) \in \mathbb{R}^{N_{\text{out}} \times d_\rho}$$

其中 $\mathbf{z}_p$ 为当前batch中扰动基因嵌入的均值池化结果，$\boldsymbol{\rho} \in \mathbb{R}^{N_{\text{out}} \times d_\rho}$ 为每个输出基因的扰动响应先验特征。

### 3.3 PPI先验编码器（Perturbation Graph Prior Encoder）

除了GO功能相似性，BioPrior-FM还集成了STRING PPI网络作为扰动基因之间的功能互作先验。

**扰动基因子图构建**：从STRING PPI数据库中提取扰动基因之间的互作边，仅保留扰动基因集合内部的连接，构建扰动专属子图。

**图编码**：在扰动子图上执行加权消息传递：

$$\mathbf{g}_i^{(l+1)} = \text{LayerNorm}\left(\mathbf{g}_i^{(l)} + \sum_{j \in \mathcal{N}_{\text{PPI}}(i)} w_{ji} \cdot \mathbf{g}_j^{(l)}\right)$$

其中 $w_{ji}$ 为STRING组合分数归一化后的边权重。

编码后的扰动基因嵌入同样通过对偶特征与输出基因计算响应先验 $\boldsymbol{\rho}_{\text{PPI}}$。

### 3.4 扰动token图上下文融合（Graph Perturbation Token Fusion）

为丰富条件编码器的输入，该模块将每个扰动token与其在GO图中的k-hop功能邻居进行注意力融合：

$$\mathbf{q} = \mathbf{W}_Q^{\text{fuse}} \mathbf{T}_c, \quad \mathbf{K}_{\text{nb}} = \mathbf{W}_K^{\text{fuse}} \mathbf{G}[\mathcal{N}_k(\text{gene})]$$

$$\alpha^{\text{fuse}} = \text{softmax}\left(\frac{\mathbf{q} \mathbf{K}_{\text{nb}}^\top}{\sqrt{d}}\right) \cdot w_{\text{nb}}$$

$$\mathbf{T}_c^{\text{fused}} = \gamma \odot \mathbf{T}_c + (1 - \gamma) \odot (\alpha^{\text{fuse}} \cdot \mathbf{G}[\mathcal{N}_k(\text{gene})])$$

其中 $\gamma = \sigma(\mathbf{W}_\gamma[\mathbf{T}_c \;\|\; \text{graph\_context}])$ 为可学习的门控系数。

### 3.5 条件向量场（Conditional Velocity Field）

向量场 $v_\theta(\mathbf{x}, t \mid c)$ 由三个编码器和一个解码器组成：

**时间编码**：

$$\mathbf{h}_t = \text{MLP}_t\left(\text{Sinusoidal}(t; f_{\text{max}}, \omega_{\text{max}})\right) \in \mathbb{R}^{d_t}$$

$$\text{Sinusoidal}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \ldots, \sin(\omega_F t), \cos(\omega_F t)], \quad \omega_f = 2\pi \cdot f_{\text{max}}^{f/F}$$

**Delta-Gated状态编码**：利用扰动响应的高度稀疏性——绝大多数基因在扰动后表达不变，该机制通过软注意力聚焦偏离batch均值的基因：

$$\boldsymbol{\delta} = \mathbf{x}_t - \bar{\mathbf{x}}_t \quad \text{（batch内均值）}$$

$$\mathbf{w} = \text{softmax}\left(|\boldsymbol{\delta}| / (\tau + \epsilon)\right)$$

$$\tilde{\mathbf{x}}_t = \mathbf{x}_t \odot (1 + \alpha \cdot \mathbf{w})$$

$$\mathbf{h}_x = \text{MLP}_x(\tilde{\mathbf{x}}_t) \in \mathbb{R}^{d_x}$$

其中 $\tau$ 和 $\alpha$ 为可学习参数。该机制放大偏离均值的基因信号，抑制未变化基因的噪声。

**条件融合**：三种嵌入通过以下方式之一融合（conditioning模式）：

- **拼接模式**：$\mathbf{h} = [\mathbf{h}_t \;\|\; \mathbf{h}_x \;\|\; \mathbf{z}_c]$
- **FiLM模式**：$\mathbf{h} = \boldsymbol{\gamma}(\mathbf{h}_t, \mathbf{z}_c) \odot \mathbf{h}_x + \boldsymbol{\beta}(\mathbf{h}_t, \mathbf{z}_c)$
- **ResNet模式**：$\mathbf{h} = \mathbf{h}_x + \text{MLP}_{\text{res}}([\mathbf{h}_t \;\|\; \mathbf{z}_c])$

**扰动条件基因掩码**：学习每个扰动条件下哪些基因最可能受影响，强制稀疏性：

$$\mathbf{m}_c = \sigma\left(\mathbf{W}_m \mathbf{z}_c + \mathbf{b}_m\right) \in (0, 1)^d$$

$$v_\theta(\mathbf{x}, t \mid c) = \mathbf{m}_c \odot \text{MLP}_{\text{dec}}(\mathbf{h})$$

该门控机制确保每个扰动只调控其功能相关的基因，而非对所有基因都有非零向量场分量。

### 3.6 多层级训练目标

BioPrior-FM采用复合损失函数，从多个层面监督向量场学习：

**(1) Flow Matching基础损失**：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|v_\theta(\mathbf{x}_t, t \mid c) - u_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1)\|^2\right]$$

**(2) 终端分布匹配损失**（Sinkhorn散度 + Energy距离）：

通过向量场的一步预测 $\hat{\mathbf{x}}_1 = \mathbf{x}_t + (1 - t) \cdot v_\theta$（使用stop-gradient + $t^p$时间门控），在 $t$ 接近1时约束预测分布与真实分布的匹配：

$$\mathcal{L}_{\text{dist}} = w_{\text{sinkhorn}} \cdot S_\varepsilon(\hat{\mathbf{X}}_1, \mathbf{X}_1) + w_{\text{energy}} \cdot \mathcal{E}(\hat{\mathbf{X}}_1, \mathbf{X}_1)$$

其中Energy距离定义为：
$$\mathcal{E}(\mathbf{X}, \mathbf{Y}) = 2\mathbb{E}\|\mathbf{X} - \mathbf{Y}\| - \mathbb{E}\|\mathbf{X} - \mathbf{X}'\| - \mathbb{E}\|\mathbf{Y} - \mathbf{Y}'\|$$

时间门控 $g(t) = t^p$（$p=2$）使该损失随 $t \to 1$ 平滑增强，避免早期噪声干扰轨迹学习。

**(3) 终端MSE监督**：

$$\mathcal{L}_{\text{endpoint}} = \mathbb{E}\left[t^p \cdot \omega(\boldsymbol{\delta}_{\text{true}}) \cdot \|\mathbf{x}_t + (1 - t)v_\theta - \mathbf{x}_1\|^2\right]$$

其中 $\omega(\boldsymbol{\delta}_{\text{true}})$ 为高delta基因权重：

$$\omega(\boldsymbol{\delta}) = \min\left(1 + w_{\text{high}} \cdot \frac{|\boldsymbol{\delta}|}{\text{mean}(|\boldsymbol{\delta}|)}, \omega_{\text{max}}\right)$$

**(4) 条件均值Delta监督**：在同条件细胞内计算均值差异的监督：

$$\mathcal{L}_{\text{mean-}\Delta} = \mathbb{E}_{c}\left[\|\mathbb{E}[\hat{\mathbf{x}}_1 - \mathbf{x}_0 \mid c] - \mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid c]\|^2\right]$$

**(5) Cosine方向监督**：约束预测的扰动方向与真实方向对齐：

$$\mathcal{L}_{\text{cos}} = \mathbb{E}\left[1 - \frac{(\hat{\mathbf{x}}_1 - \mathbf{x}_0)^\top(\mathbf{x}_1 - \mathbf{x}_0)}{\|\hat{\mathbf{x}}_1 - \mathbf{x}_0\| \cdot \|\mathbf{x}_1 - \mathbf{x}_0\|}\right]$$

**(6) 条件编码正则化**（确定模式）：

$$\mathcal{L}_{\text{reg}} = \frac{\lambda}{2}\|\mathbf{z}_c\|^2$$

**总损失函数**：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FM}} + \alpha_1 \mathcal{L}_{\text{dist}} + \alpha_2 \mathcal{L}_{\text{endpoint}} + \alpha_3 \mathcal{L}_{\text{mean-}\Delta} + \alpha_4 \mathcal{L}_{\text{cos}} + \mathcal{L}_{\text{reg}}$$

其中 $[\alpha_1, \alpha_2, \alpha_3, \alpha_4]$ 为各辅助损失的权重超参数。

### 3.7 推理（预测）

训练完成后，给定对照细胞 $\mathbf{x}_0$ 和目标扰动条件 $c$，通过求解ODE获得预测表达谱：

$$\hat{\mathbf{x}}_1 = \mathbf{x}_0 + \int_{0}^{1} v_\theta(\mathbf{x}_\tau, \tau \mid c)\ d\tau$$

使用diffrax库的Tsit5自适应步长求解器，设置相对容差rtol=$10^{-5}$，绝对容差atol=$10^{-5}$。

---

## 4. 方法创新点总结

| 创新点 | 技术方案 | 作用 |
|--------|---------|------|
| **GO先验编码** | 共享gene2vec表 + GAT图消息传递 + 对偶特征建模 | 显式建模扰动基因→全部输出基因的功能传递路径 |
| **PPI先验编码** | STRING扰动子图 + 加权消息传递 | 捕获扰动基因间的互作协同效应 |
| **Delta-Gated编码** | 可学习温度的softmax门控放大偏离基因 | 利用扰动响应稀疏性，聚焦真正变化的基因 |
| **基因掩码稀疏化** | 条件嵌入→sigmoid门控→逐基因屏蔽 | 每个扰动只调控功能相关的基因子集 |
| **多层级复合损失** | FM + 终端Sinkhorn/Energy + 终端MSE + Delta均值 + Cosine方向 | 从分布/细胞/基因/方向四个层面联合约束 |

---

## 5. 与现有方法的对比

| | scGen | CPA | CellOT | **BioPrior-FM** |
|---|---|---|---|---|
| 建模方式 | VAE + latent空间加和 | 组合自编码器 | OT映射 | 条件流匹配（Conditional FM） |
| 扰动编码 | one-hot | one-hot + 组合 | one-hot | gene2vec嵌入 + GO/PPI图编码 |
| 先验知识 | 无 | 无 | 无 | **GO功能图谱 + STRING PPI网络** |
| 基因级别建模 | 全基因统一 | 全基因统一 | 全基因统一 | **基因掩码稀疏化** |
| OT匹配 | 无 | 无 | Sinkhorn | Sinkhorn（每N步重匹配） |
| 损失函数 | ELBO | 重构损失 | OT距离 | **FM + 5项辅助损失** |
