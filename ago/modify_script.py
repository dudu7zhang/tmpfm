import re

with open('/home/zhangshibo24s/cell_flow/train_myflow_Replogle_Nadig.py', 'r') as f:
    content = f.read()

# 1. Update argparse defaults
content = content.replace(
    'p.add_argument("--overwrite", action="store_true")',
    'p.add_argument("--overwrite", action="store_true")\n    p.add_argument("--holdout-cell-line", default="k562", help="Cell line to hold out.")'
).replace(
    'default="/home/zhangshibo24s/cell_flow/data_train",',
    'default="/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad",'
)

# 2. Update data loading
old_data_load = '''    if adata_path.is_dir():
        # h5ad_files = sorted([p for p in adata_path.iterdir() if p.name.endswith("hvg.h5ad")])
        # if len(h5ad_files) < 0:
        #     raise ValueError(
        #         f"Need at least 4 *hvg.h5ad files in directory for 3-train/1-test split, found {len(h5ad_files)}"
        #     )
        # train_files = h5ad_files[:3]
        # print("Using training files:", [p.name for p in train_files])
        # 剩rpe1文件作为测试集
        # Load and concatenate training files
        train_files = [
            "/home/zhangshibo24s/cell_flow/data_train/hepg2_hvg.h5ad",
            "/home/zhangshibo24s/cell_flow/data_train/jurkat_hvg.h5ad",
            # "/home/zhangshibo24s/cell_flow/data_train/k562_hvg.h5ad",
            "/home/zhangshibo24s/cell_flow/data_train/rpe1_hvg.h5ad"
        ]
        adatas = [ad.read_h5ad(str(p)) for p in train_files]
        print("Concatenating training AnnData objects...")
        try:
            adata = ad.concat(adatas, join="outer", label="batch", keys=[p.stem for p in train_files])
        except Exception:
            adata = ad.concat(adatas)
 
        print("Training adata.obs columns:", list(adata.obs.columns))
    else:
        print("Loading data:", adata_path)
        adata = ad.read_h5ad(str(adata_path))
        print("adata.obs columns:", list(adata.obs.columns))'''

new_data_load = '''    print("Loading merged dataset:", adata_path)
    adata = ad.read_h5ad(str(adata_path))
    # 为保证和原来代码的兼容性，进行列名平替
    if 'gene' in adata.obs:
        adata.obs['target_gene'] = adata.obs['gene']
    if 'cell_line' in adata.obs:
        adata.obs['cell_type'] = adata.obs['cell_line']
        
    print("adata.obs columns:", list(adata.obs.columns))
    
    # === 使用已经提前筛好的 Highly Variable Genes ===
    if "highly_variable" in adata.var:
        print(f"Filtering by highly variable genes. Original vars: {adata.n_vars}")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"After HVG filtering vars: {adata.n_vars}")
    else:
        print("Warning: highly_variable column not found in dataset!")'''

content = content.replace(old_data_load, new_data_load)

# 3. Update covariates and LOCO split logic
old_split = '''    perturbation_covariates = {"gene_perturbation": ["target_gene"]}
    perturbation_reps = {"gene_perturbation": rep_key}
    if args.control_key not in adata.obs:
        adata.obs[args.control_key] = adata.obs["target_gene"].astype(str) == "non-targeting"

    print(f"Total cells before split: {adata.n_obs}")
    n_total = adata.n_obs
    rng_val = np.random.default_rng(args.seed)
    val_indices = rng_val.choice(n_total, int(n_total * 0.05), replace=False)
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[val_indices] = True

    adata_val = adata[val_mask].copy()
    adata = adata[~val_mask].copy()
    print(f"Using {adata.n_obs} cells for training and {adata_val.n_obs} cells for validation.")'''

new_split = '''    # === 增加 cell_type 作为额外的 condition ===
    perturbation_covariates = {
        "gene_perturbation": ["target_gene"],
        "cell_type": ["cell_type"]
    }
    perturbation_reps = {"gene_perturbation": rep_key}
    if args.control_key not in adata.obs:
        adata.obs[args.control_key] = adata.obs["target_gene"].astype(str) == "non-targeting"

    print(f"Total cells before split: {adata.n_obs}")
    
    # ================= Leave-One-Cell-Line-Out (LOCO) Split Logic =================
    holdout = args.holdout_cell_line
    assert holdout in adata.obs['cell_type'].unique(), f"Holdout cell line {holdout} not found in adata.obs['cell_type']"
    
    other_mask = adata.obs['cell_type'] != holdout
    holdout_mask = adata.obs['cell_type'] == holdout
    
    # 提取 holdout 细胞系独有的所有扰动
    perts = adata[holdout_mask].obs['target_gene'].unique().tolist()
    pert_targets = [p for p in perts if p != 'non-targeting']
    
    rng = np.random.default_rng(args.seed)
    shuffled_perts = rng.permutation(pert_targets)
    n_train_perts = int(0.3 * len(shuffled_perts))
    
    # 30% 到训练集，70% 到测试集（用来评测）
    train_perts = set(shuffled_perts[:n_train_perts])
    test_perts = set(shuffled_perts[n_train_perts:])
    
    # 训练集: 其它3个细胞系全部 + holdout的30%扰动 + holdout的non-targeting(让模型知道基态)
    train_mask = other_mask | (holdout_mask & adata.obs['target_gene'].isin(train_perts)) | (holdout_mask & (adata.obs['target_gene'] == 'non-targeting'))
    # 零样本测试集: holdout的另外70%扰动
    test_mask = holdout_mask & adata.obs['target_gene'].isin(test_perts)
    
    adata_test_holdout = adata[test_mask].copy() 
    adata_train_full = adata[train_mask].copy()
    
    # 标准的训练-验证集划分 (从训练集中抽 5% 给 validation 观察曲线)
    n_train_total = adata_train_full.n_obs
    val_indices = rng.choice(n_train_total, int(n_train_total * 0.05), replace=False)
    val_mask_arr = np.zeros(n_train_total, dtype=bool)
    val_mask_arr[val_indices] = True
    
    adata_val = adata_train_full[val_mask_arr].copy()
    adata = adata_train_full[~val_mask_arr].copy()
    
    print(f"Leave-One-Cell-Line-Out Split:")
    print(f"  Holdout cell line: {holdout}")
    print(f"  Holdout perturbations in Train: {len(train_perts)} (30%)")
    print(f"  Holdout perturbations in Test : {len(test_perts)} (70%)")
    print(f"  Using {adata.n_obs} cells for training, {adata_val.n_obs} for validation.")
    print(f"  Zero-shot testing set contains {adata_test_holdout.n_obs} cells.")
    # =============================================================================='''

content = content.replace(old_split, new_split)

# 4. Update predicting logic
old_pred = '''    print("Starting prediction...")
    test_adata_path = ROOT / "data" / "k562_ctrl.h5ad"
    test_adata = sc.read_h5ad(str(test_adata_path))
    test_adata = align_adata_to_selected_ensembl(
        adata=test_adata,
        symbol_to_ensembl=symbol_to_ensembl,
    )
    test_adata.obs[args.control_key] = True
    test_adata.uns[rep_key] = emb_dict
    groups = test_adata.obs.groupby("target_gene").groups'''

new_pred = '''    print("Starting prediction on the 70% zero-shot holdout tests...")
    # 提取 holdout 细胞系的 control 作为测试集的 baseline 输入
    # (即上面划分时放进 adata_train_full 的 non-targeting 细胞)
    test_adata = adata_train_full[(adata_train_full.obs['cell_type']==holdout) & (adata_train_full.obs[args.control_key]==True)].copy()
    
    # 提取测试集中存在的全部未见过扰动
    groups = adata_test_holdout.obs.groupby("target_gene").groups'''

content = content.replace(old_pred, new_pred)

# 5. Fix prediction covariate to include cell_type
old_pred_cov = '''        covariate_data = pd.DataFrame({
            "target_gene": [gene],
            args.control_key: [False]
        })'''

new_pred_cov = '''        covariate_data = pd.DataFrame({
            "target_gene": [gene],
            "cell_type": [holdout],
            args.control_key: [False]
        })'''

content = content.replace(old_pred_cov, new_pred_cov)

with open('/home/zhangshibo24s/cell_flow/train_myflow_loco.py', 'w') as f:
    f.write(content)
print('Successfully generated train_myflow_loco.py')
