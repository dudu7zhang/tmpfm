notion

## 两个新增的先验知识
solver_kwargs.setdefault("gene_mask", grn_mask)
solver_kwargs.setdefault("condition_gene_masks", cond_gene_masks)


# 单独开一页举个例子看分别代表什么

它包含了 8 个关键字段，我帮你按用途分成 3 组来理解：
## 第一组：细胞归属标记（指明哪些细胞属于哪个条件）
这部分用于过滤细胞，把 adata 的行和特定的条件对应起来。如果 return_mask=False（纯生成条件），可能为 None。

### split_covariates_mask (np.ndarray 列, 长度=细胞数)
含义：标记每个细胞属于哪个 Control 子群（源分布）。
值：整数（例如 0, 1, 2...），表示不同的 split 索引。如果不属于任何 control 组（即被扰动细胞），值为 -1。
作用：训练/评测时，用它把未受扰动的控制组细胞（Control cells）分桶。

### perturbation_covariates_mask (np.ndarray 列, 长度=细胞数)
含义：标记每个细胞属于哪个具体的扰动条件（目标分布）。
值：整数（例如 0, 1, 2...），表示不同的扰动索引。如果是 control 组细胞，值为 -1。
作用：训练时，用它把受到扰动的细胞（Perturbed cells）进行分类，以便计算特定维度的 Loss。
第二组：索引翻译表（把整数 IDX 还原成原本的人类可读条件）
第一组只给了 0, 1, 2 这样的数字，第二组负责翻译这些数字究竟代表什么。

### split_idx_to_covariates (dict)
含义：Control 子群索引 
具体条件的映射。
例子：{0: ("CellTypeA", "Batch1"), 1: ("CellTypeB", "Batch1")}
作用：让模型知道 split_covariates_mask == 0 的到底是那一批源细胞。

### perturbation_idx_to_covariates (dict)
含义：扰动索引 
具体扰动组合的映射。
例子：{0: ("DrugA", 10.0), 1: ("DrugA", 50.0), 2: ("DrugB", 10.0)}
作用：让模型知道 perturbation_covariates_mask == 0 的目标细胞对应的究竟是哪个药物和浓度。

### perturbation_idx_to_id (dict)
含义：扰动索引 
自定义扰动 ID （如果有 condition_id_key 的话）。
作用：如果用户的 covariate 表里设定了唯一的 ID 列（比如 "cond_001"），这里就能查回原 ID。
第三组：送入神经网络参与计算的数值特征
这里是真正参与前向传播的部分。

### condition_data (dict[str, np.ndarray])
含义：每个扰动组的特征嵌入矩阵 (Embedding Tensors)。
结构：{ covariate_group_name : Tensor }
形状：对于特定的 group，张量形状通常是 [目标条件总数, max_combination_length, 特征维度]。
作用：输入给神经网络的条件端，它是经过 one-hot 编码或者从 adata.uns 字典查表取出来的连续向量值。

### control_to_perturbation (dict[int, np.ndarray])
含义：Control 组到底应该和哪些 Perturbation 组进行 OT (Optimal Transport) 匹配。
结构：{ split_idx : [pert_idx_1, pert_idx_2, ...] }。
例子：{0: [1, 2], 1: [3]}，表示第 0 个 Control 组可以作为起点，去预测第 1 和 2 两个扰动结果。
作用：指导网络生成训练正负样本对或设定预测的目标集。

### max_combination_length (int)
含义：一个细胞最多同时拥有多少个主扰动。
作用：因为有的细胞加了单药，有的加了双药联合扰动，为了能在同一个 batch 里并行计算，必须把扰动序列通过 null_value 补齐到同样长度（Padding）。这里记录的就是对齐长度。

### 简而言之：
ReturnData 提供了"源从哪来 (split_covariates_mask)"、"目标是谁 (perturbation_covariates_mask)"，"目标长什么样 (condition_data)"以及""两者谁去对接谁 (control_to_perturbation)"的完整模型数据表示


## 在prepare_model里面

condition_gene_masks是做好的对于每个扰动基因target_gene的一个mask

## 3.25的任务，用上这个mask。再想创新点