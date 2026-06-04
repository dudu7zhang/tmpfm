# PriorFlow: Prior-knowledge Guided Conditional Flow Matching for Single-Cell Perturbation Response Prediction

## 方法命名

**PriorFlow** (Prior-knowledge guided Flow matching for perturbation response prediction)。

核心思想：将基因功能先验知识（GO功能图谱 + STRING蛋白质互作网络）通过**可学习图神经网络**显式编码进扰动条件表示，再以**交叉注意力机制**引导条件信息与全基因表达谱的深度融合，在 Conditional Flow Matching 框架下实现高精度的单细胞扰动响应预测。

---

## 1. 问题定义

给定单细胞转录组数据集：

- **对照细胞** $\mathbf{x}^{\text{ctrl}} \in \mathbb{R}^d$（$d$ 为基因数），服从分布 $p_0$
- **扰动细胞** $\mathbf{x}^{\text{pert}} \in \mathbb{R}^d$，对应扰动条件 $c = \{g_1, \ldots, g_k\}$（$k$ 个扰动基因），服从条件分布 $p_1(\cdot \mid c)$

**目标**：学习条件向量场 $v_\theta(\mathbf{x}, t \mid c)$，使得从 $t=0$ 到 $t=1$ 沿 ODE

$$\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}, t \mid c)$$

积分得到的终端分布逼近真实扰动分布 $p_1(\cdot \mid c)$。

---

## 2. 背景：Optimal Transport Conditional Flow Matching

Flow Matching (Lipman et al., 2022) 通过匹配条件概率路径上的向量场来学习分布变换。给定条件概率路径 $p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$ 及对应的真实条件向量场 $u_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$，CFM 目标为：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|v_\theta(\mathbf{x}_t, t \mid c) - u_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1)\|^2\right]$$

其中 $t \sim \mathcal{U}(0,1)$，$\mathbf{x}_t$ 沿概率路径从 $\mathbf{x}_0$ 插值至 $\mathbf{x}_1$。

PriorFlow 采用 **OT-CFM** (Tong et al., 2023; Pooladian et al., 2023)，通过 Sinkhorn 最优传输匹配源-目标细胞：

$$\mathbf{M}^* = \arg\min_{\mathbf{M} \in \Pi(\mathbf{p},\mathbf{q})} \langle \mathbf{M}, \mathbf{C} \rangle - \varepsilon H(\mathbf{M})$$

其中 $\mathbf{C}_{ij} = \|\mathbf{x}_0^{(i)} - \mathbf{x}_1^{(j)}\|^2$，匹配每 $N$ 步更新一次。概率路径选用 Brownian Bridge：

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1 + \sigma\sqrt{t(1-t)}\,\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

---

## 3. PriorFlow 架构

PriorFlow 的架构由三个核心组件构成：(1) 基于先验知识图的扰动条件编码器，(2) 可学习基因身份与交叉注意力融合，(3) 扰动条件基因掩码。

```
 扰动基因ID                        x_t (B,d) 基因表达
     │                                  │
     ▼                                  ├─ Dense(16)(标量表达值)
 Embed(可学习) → GNN(GO+STRING)          ├─ gene_id_emb[d] (身份)
     │                                  │
     ▼                                  └─ h_genes = val + id  (B,d,16)
 SetEncoder(注意力池化)                       │
     │                                       │
     ▼                                       │
   z_c (B, Dz) ──── cross_attn query ───────┘
     │                              │
     │                              ▼
     │                         h_cross (B,16)
     │                              │
     │                         Dense(512) → x_encoded
     │                              │
     │   t → Sinusoidal → MLP → h_t │
     │              └── concat ─────┘
     │                      │
     │                   Decoder MLP
     │                      │
     │                   Δx (velocity)
     │                      │
     └─→ Dense(d) → sigmoid → ⊙
                           masked Δx
```

### 3.1 先验知识图扰动编码器

这是 PriorFlow 的**核心创新**。传统方法（one-hot、gene2vec 查表）将每个扰动基因视为独立实体，忽略了基因间丰富的功能关联。PriorFlow 显式利用 GO 功能图谱和 STRING PPI 网络构建扰动基因专属知识图谱，将功能关系编码为可学习的增强表示。

**图构建**：给定 $N$ 个扰动基因 $\mathcal{G} = \{g_1, \ldots, g_N\}$，从 GO 和 STRING 数据库中提取基因间的功能关联边：

$$\mathcal{E} = \{(i, j, w_{ij}) \mid \text{基因 } i \text{ 与 } j \text{ 存在 GO/STRING 关联}\}$$

构建有向有权图，边权重经目标节点度归一化：

$$\tilde{w}_{ij} = \frac{w_{ij}}{\sum_{k} w_{kj} + \epsilon}$$

**可学习图嵌入**（随机初始化，训练中学习）：

$$\mathbf{E} \in \mathbb{R}^{N \times d_g}, \quad d_g = 16$$

每行 $\mathbf{e}_i$ 是扰动基因 $i$ 的初始嵌入。与 TxPert 不同之处在于，TxPert 仅使用 PPI 网络，而 PriorFlow 同时融合 GO 和 STRING 两种互补的先验知识源。

**图消息传递**（$L$ 层残差连接）：

$$\mathbf{H}^{(0)} = \mathbf{E}$$

$$\mathbf{H}^{(l+1)} = \text{LayerNorm}\left(\mathbf{H}^{(l)} + \text{MLP}^{(l)}\left(\sum_{(i,j) \in \mathcal{E}} \mathbf{h}_i^{(l)} \cdot \tilde{w}_{ij}\right)\right)$$

消息传递后，根据当前条件中扰动基因的整数 ID 索引 gather per-token 表示：

$$\mathbf{T}_c = \text{Gather}(\mathbf{H}^{(L)}, \text{idx}_c) \in \mathbb{R}^{k \times d_g}$$

**集合池化**：通过可学习 `[CLS]` token 对 $k$ 个扰动基因 token 做注意力池化得到单一条件向量：

$$\mathbf{z}_c = \text{AttentionPool}(\mathbf{W}_Q \cdot \text{CLS}, \mathbf{W}_K \mathbf{T}_c, \mathbf{W}_V \mathbf{T}_c) \in \mathbb{R}^{D_z}$$

其中 $D_z$ 为条件嵌入维度（默认 256）。对于多基因组合扰动（如 $k=2$），SetEncoder 的注意力池化自动学习组合效应的权重分配。

### 3.2 身份解耦基因编码与交叉注意力

传统方法将表达谱 $\mathbf{x}_t \in \mathbb{R}^d$ 直接送入 MLP，每个基因被视为无身份标识的标量，模型无法区分"基因 A 表达值为 2.0"与"基因 B 表达值为 2.0"之间的本质差异。PriorFlow 创新性地**解耦表达量级与基因身份**：

**表达量投影**（所有基因共享）：

$$\mathbf{v}_i = \mathbf{W}_v \cdot x_i + \mathbf{b}_v \in \mathbb{R}^{d_g}$$

**基因身份嵌入**（每个基因独立可学习）：

$$\mathbf{E}^{\text{gene}} \in \mathbb{R}^{d \times d_g}$$

**融合表示**：

$$\mathbf{h}_i = \mathbf{v}_i + \mathbf{e}^{\text{gene}}_i, \quad \mathbf{H} = [\mathbf{h}_1, \ldots, \mathbf{h}_d] \in \mathbb{R}^{d \times d_g}$$

**交叉注意力条件融合**：与简单 concat 的本质区别在于结构化条件注入。$\mathbf{z}_c$ 作为 query 主动检索基因特征 $\mathbf{H}$，让条件显式选择最相关的基因子集：

$$\mathbf{q} = \mathbf{W}_Q^{\text{cross}} \mathbf{z}_c \in \mathbb{R}^{d_g}$$

$$\mathbf{h}^{\text{cross}} = \text{softmax}\left(\frac{\mathbf{q}^\top (\mathbf{W}_K^{\text{cross}} \mathbf{H})^\top}{\sqrt{d_g}}\right) \mathbf{W}_V^{\text{cross}} \mathbf{H} \in \mathbb{R}^{d_g}$$

$$\mathbf{h}_x = \text{SiLU}(\mathbf{W}_{\text{fusion}} \cdot \mathbf{h}^{\text{cross}}) \in \mathbb{R}^{D_h}$$

值得注意的是，$\mathbf{z}_c$ **不直接进入 decoder concat**，仅通过交叉注意力和基因掩码两条路径施加影响。这避免了 concat 模式下条件信息的过度支配，使架构更加简洁高效。

### 3.3 时间编码与解码器

时间信息通过正弦编码注入：

$$\mathbf{h}_t = \text{MLP}_t(\text{Sinusoidal}(t)) \in \mathbb{R}^{D_h}$$

$$\text{Sinusoidal}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \ldots, \sin(\omega_F t), \cos(\omega_F t)], \quad \omega_f = 2\pi \cdot f_{\max}^{f/F}$$

解码器将时间信息与条件感知的细胞状态拼接后解码：

$$\mathbf{h} = [\mathbf{h}_t \;\|\; \mathbf{h}_x], \quad v_\theta = \text{MLP}_{\text{dec}}(\mathbf{h}) \in \mathbb{R}^d$$

### 3.4 扰动条件基因掩码

每个特定基因扰动仅影响其功能相关的少量基因——大多数基因在扰动前后表达不变。PriorFlow 通过条件感知稀疏门控将这一先验纳入模型：

$$\mathbf{m}_c = \sigma(\mathbf{W}_m \mathbf{z}_c + \mathbf{b}_m) \in (0, 1)^d$$

$$v_\theta^{\text{final}} = \mathbf{m}_c \odot v_\theta$$

掩码乘在向量场（velocity，即 $d\mathbf{x}/dt$）上。对于与当前扰动无关的基因，掩码值趋近于 0，使得沿 ODE 轨迹上该基因的变化速率 $dx_i/dt \approx 0$，表达值保持与对照接近。模型因此被约束为仅预测少量真正受扰动影响的基因的**变化**，而非对全基因组做无差别预测。该掩码直接由条件嵌入 $\mathbf{z}_c$ 生成，端到端可学习。

### 3.5 多层级训练目标

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|v_\theta - u_t\|^2\right]$$

$$\mathcal{L}_{\text{endpoint}} = \mathbb{E}\left[t^2 \cdot \omega(\boldsymbol{\delta}) \cdot \|\hat{\mathbf{x}}_1 - \mathbf{x}_1\|^2\right], \quad \hat{\mathbf{x}}_1 = \mathbf{x}_t + (1-t) \cdot v_\theta$$

$$\omega(\boldsymbol{\delta}) = \min\left(1 + w_{\text{high}} \cdot \frac{|\boldsymbol{\delta}|}{\text{mean}(|\boldsymbol{\delta}|) + \epsilon}, \; \omega_{\max}\right)$$

$$\mathcal{L}_{\text{mean-}\Delta} = \mathbb{E}_{c}\left[\|\mathbb{E}[\hat{\mathbf{x}}_1 - \mathbf{x}_0 \mid c] - \mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid c]\|^2\right]$$

$$\mathcal{L}_{\text{cos}} = \mathbb{E}\left[1 - \frac{(\hat{\mathbf{x}}_1 - \mathbf{x}_0)^\top(\mathbf{x}_1 - \mathbf{x}_0)}{\|\hat{\mathbf{x}}_1 - \mathbf{x}_0\| \cdot \|\mathbf{x}_1 - \mathbf{x}_0\| + \epsilon}\right]$$

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \alpha_1 \mathcal{L}_{\text{endpoint}} + \alpha_2 \mathcal{L}_{\text{mean-}\Delta} + \alpha_3 \mathcal{L}_{\text{cos}}$$

四项损失分别从**轨迹、终端、条件均值、方向**四个层面对齐预测与真实，形成全面的监督信号。

---

## 4. 创新点总结

| 创新点 | 技术实现 | 解决的痛点 |
|--------|---------|-----------|
| **先验知识图扰动编码** | GO+STRING 图上可学习 Embed + GNN 消息传递 | 传统编码忽略基因功能关联，无法建模扰动基因间的协同与拮抗 |
| **身份解耦基因表示** | 表达量投影 + 可学习基因身份嵌入 | 传统 MLP 将基因视为无身份标量，无法区分不同基因的语义角色 |
| **交叉注意力条件融合** | $\mathbf{z}_c$ 作为 query 检索 $\mathbf{H}$ | 传统 concat 盲拼，条件信息无法精确指向目标基因 |
| **条件感知基因掩码** | $\mathbf{m}_c = \sigma(\mathbf{W}_m \mathbf{z}_c) \odot v_\theta$ | 每个扰动仅改变少量基因，掩码约束无关基因的 $dx/dt \approx 0$，施加扰动响应稀疏性偏置 |
| **多层级监督信号** | FM + 终端MSE + Delta均值 + Cosine | 单一 FM 损失缺乏对终端预测和方向对齐的直接约束 |

---

## 5. 与现有方法对比

| | scGen | CPA | CellOT | TxPert | **PriorFlow** |
|---|---|---|---|---|---|
| 基础框架 | VAE | 组合AE | OT映射 | GNN+MLP | **条件流匹配** |
| 扰动编码 | one-hot | one-hot | one-hot | PPI Embed | **GO+STRING Embed** |
| 基因-条件融合 | latent加和 | 组合解码 | 条件输入 | concat | **交叉注意力** |
| 输出稀疏性 | 无 | 无 | 无 | 无 | **基因掩码** |
| OT匹配 | 无 | 无 | Sinkhorn | 无 | **Sinkhorn** |
| 先验知识 | 无 | 无 | 无 | PPI | **GO+PPI** |

---

## 6. 推理

$$\hat{\mathbf{x}}_1 = \mathbf{x}_0 + \int_0^1 v_\theta(\mathbf{x}_\tau, \tau \mid c)\ d\tau$$

采用 Tsit5 自适应步长 ODE 求解器 (diffrax)，容差 $10^{-5}$。
