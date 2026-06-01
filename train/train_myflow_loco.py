#!/usr/bin/env python3
"""
Notes:
- This script expects to be run from the repository root (`/home/zhangshibo24s/cell_project`).
- It adds the local `myflow/src` package to `sys.path` so you don't need to install the package.
- Provide `--perturbation-covariates` as a JSON string mapping covariate-name -> list-of-obs-columns.
- Optionally provide `--perturbation-reps` as a JSON dict mapping covariate-name -> key-in-adata.uns holding embeddings.
"""

import argparse
import os
from pathlib import Path
import re

# Set GPU/JAX environment before importing torch/jax/myflow.
ROOT = Path(__file__).resolve().parent
# Ensure only one GPU is visible by default (can override from shell env).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# Avoid large up-front JAX memory preallocation that can look like multi-GPU usage.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import anndata as ad
import mygene
import pandas as pd
import torch
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats
from datetime import datetime
from myflow.model._myflow import MyFlow
from myflow.training import Metrics

ENSG_PATTERN = re.compile(r"^ENSG\d+$", re.IGNORECASE)

# ==================== Evaluation Metrics from cal_score.py ====================
def cal_metric(pred_mean, real_mean):
    mse = mean_squared_error(real_mean, pred_mean)
    mae = mean_absolute_error(real_mean, pred_mean)
    l2 = np.linalg.norm(real_mean - pred_mean)
    return mse, mae, l2

def cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=20):
    delta_real = real_mean - ctrl_mean
    delta_pred = pred_mean - ctrl_mean
    pearson_delta, _ = scipy.stats.pearsonr(delta_real, delta_pred)
    
    top_n_idx = np.argsort(np.abs(delta_real))[-top_k:]
    if len(top_n_idx) > 1:
        pearson_delta_top_k, _ = scipy.stats.pearsonr(delta_real[top_n_idx], delta_pred[top_n_idx])
    else:
        pearson_delta_top_k = 0.0

    sign_real = np.sign(delta_real[top_n_idx])
    sign_pred = np.sign(delta_pred[top_n_idx])
    ds_score = np.mean([1 if r == p else 0 for r, p in zip(sign_real, sign_pred)])
    return pearson_delta, pearson_delta_top_k, ds_score

def get_deg_sets(adata, group="target"):
    degs = adata.uns['rank_genes_groups']
    genes = np.array(degs['names'][group])
    logfc = np.array(degs['logfoldchanges'][group])
    pvals_adj = np.array(degs['pvals_adj'][group])
    sig_mask = pvals_adj < 0.05
    return genes[sig_mask], logfc[sig_mask]

def compute_des_single(real_genes, pred_genes, pred_logfc):
    real_set = set(real_genes)
    pred_set = set(pred_genes)
    n_true = len(real_set)
    n_pred = len(pred_set)

    if n_true == 0:
        return 0.0, 0.0

    if n_pred <= n_true:
        inter = real_set.intersection(pred_set)
        return len(inter) / n_true, (len(inter) / n_pred if n_pred > 0 else 0)

    idx = np.argsort(-np.abs(pred_logfc))[:n_true]
    pred_topk_genes = set(np.array(pred_genes)[idx])
    inter = real_set.intersection(pred_topk_genes)
    return len(inter)/n_true, len(inter)/n_pred

def compute_des(ctrl, target, pred):
    combined_real = ctrl.concatenate(target, batch_key='condition', batch_categories=['ctrl', 'target'])
    sc.tl.rank_genes_groups(combined_real, groupby='condition', reference='ctrl', method='t-test')
    real_genes, real_logfc = get_deg_sets(combined_real, group="target")

    if 'gene_symbol' in pred.var.columns:
        pred.var.index = pred.var['gene_symbol']
        pred.var_names = pred.var['gene_symbol'].values

    combined_pred = ctrl.concatenate(pred, batch_key='condition', batch_categories=['ctrl', 'target'])
    sc.tl.rank_genes_groups(combined_pred, groupby='condition', reference='ctrl', method='t-test')
    pred_genes, pred_logfc = get_deg_sets(combined_pred, group="target")

    de_spearman = 0.0
    if len(real_genes) > 1:
        degs_pred_all = combined_pred.uns['rank_genes_groups']
        all_pred_genes = np.array(degs_pred_all['names']["target"])
        all_pred_logfc = np.array(degs_pred_all['logfoldchanges']["target"])
        pred_logfc_map = dict(zip(all_pred_genes, all_pred_logfc))
        
        real_matched_logfc = []
        pred_matched_logfc = []
        for rg, r_fc in zip(real_genes, real_logfc):
            if rg in pred_logfc_map:
                real_matched_logfc.append(r_fc)
                pred_matched_logfc.append(pred_logfc_map[rg])
        
        if len(real_matched_logfc) > 1:
            de_spearman, _ = scipy.stats.spearmanr(real_matched_logfc, pred_matched_logfc)

    des_recall, des_acc = compute_des_single(real_genes, pred_genes, pred_logfc)
    return des_recall, des_acc, de_spearman
# ==============================================================================

def _extract_ensembl_id(entry) -> str | None:
    if entry is None:
        return None
    if isinstance(entry, list):
        for item in entry:
            if isinstance(item, dict) and "gene" in item:
                val = str(item["gene"]).strip().upper()
                if val:
                    return val
            elif isinstance(item, str):
                val = item.strip().upper()
                if val:
                    return val
        return None
    if isinstance(entry, dict):
        gene = entry.get("gene")
        if gene is not None:
            return str(gene).strip().upper()
        return None
    return str(entry).strip().upper()


def build_symbol_to_ensembl(symbols: list[str]) -> dict[str, str]:
    symbols = [str(s).strip() for s in symbols]
    unique_symbols = list(dict.fromkeys(symbols))
    symbol_to_ensembl: dict[str, str] = {}

    already_ensg = [s for s in unique_symbols if ENSG_PATTERN.match(s)]
    for s in already_ensg:
        symbol_to_ensembl[s] = s.upper()

    unresolved = [s for s in unique_symbols if s not in symbol_to_ensembl]
    if unresolved:
        mg = mygene.MyGeneInfo()
        import time
        query = []
        for attempt in range(3):
            try:
                query = mg.querymany(
                    unresolved,
                    scopes="symbol,alias",
                    fields="ensembl.gene",
                    species="human",
                    as_dataframe=False,
                    returnall=False,
                    verbose=False,
                )
                break  # If successful, break out of retry loop
            except Exception as e:
                print(f"Network error querying MyGene (attempt {attempt+1}): {e}")
                if attempt == 2:
                    raise e
                time.sleep(2)
        for row in query:
            q = str(row.get("query", "")).strip()
            ensembl_id = _extract_ensembl_id(row.get("ensembl"))
            if q and ensembl_id:
                symbol_to_ensembl[q] = ensembl_id
    return symbol_to_ensembl


def align_adata_to_selected_ensembl(
    adata: ad.AnnData,
    symbol_to_ensembl: dict[str, str],
) -> ad.AnnData:
    original_symbols = [str(g).strip() for g in adata.var_names]
    mapped_ids = [symbol_to_ensembl.get(s, s).upper() for s in original_symbols]

    keep_idx = []
    seen: set[str] = set()
    for i, gid in enumerate(mapped_ids):
        # We don't drop non-ENSG anymore, to guarantee all provided input genes are kept (except duplicates)
        if gid in seen:
            continue
        seen.add(gid)
        keep_idx.append(i)

    if not keep_idx:
        raise ValueError("No valid genes left.")

    adata = adata[:, keep_idx].copy()
    kept_ids = [mapped_ids[i] for i in keep_idx]
    kept_symbols = [original_symbols[i] for i in keep_idx]
    adata.var["gene_symbol"] = kept_symbols
    adata.var_names = kept_ids

    return adata


def build_matched_gene2vec(
    selected_gene_ids_file: Path,
    selected_gene2vec_file: Path,
    ordered_ids: list[str],
    save_dir: Path,
) -> tuple[Path, Path]:
    with open(selected_gene_ids_file, "r", encoding="utf-8") as f:
        all_ids = [line.strip().upper() for line in f if line.strip()]
    id_to_idx = {g: i for i, g in enumerate(all_ids)}
    full_vec = np.load(selected_gene2vec_file)
    dim = full_vec.shape[1]

    matched_vecs = []
    for g in ordered_ids:
        if g in id_to_idx:
            matched_vecs.append(full_vec[id_to_idx[g]].astype(np.float32))
        else:
            # If the gene is not found in gene2vec dictionary, initialize with zeros
            matched_vecs.append(np.zeros(dim, dtype=np.float32))
            
    matched_vec = np.stack(matched_vecs)

    save_dir.mkdir(parents=True, exist_ok=True)
    gene_ids_out = save_dir / "selected_gene_ids_matched.txt"
    gene2vec_out = save_dir / "selected_gene2vec_matched.npy"

    with open(gene_ids_out, "w", encoding="utf-8") as f:
        for g in ordered_ids:
            f.write(f"{g}\n")
    np.save(gene2vec_out, matched_vec)

    return gene_ids_out, gene2vec_out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adata",
        default="/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad",
        required=False,
        help="Path to anndata (.h5ad) or directory containing *hvg.h5ad files"
    )
    p.add_argument(
        "--sample-rep",
        default="X",
        help="Key in adata.obsm to use as sample representation (default: X)",
    )
    p.add_argument("--control-key", default="control", help="obs column marking control samples (default: control)")
    p.add_argument(
        "--perturbation-reps",
        default="{}",
        help="JSON mapping perturbation_name -> adata.uns key containing embeddings (optional)",
    )
    p.add_argument("--num-iterations", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--skip-prediction", action="store_true")
    # p.add_argument("--valid-freq", type=int, default=500)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--holdout-cell-line", default="hepg2", help="Cell line to hold out.")
    # p.add_argument("--condition-combined-loss-weight", type=float, default=0.1)
    # p.add_argument("--preset", choices=["jurkat"], default=None, help="Optional preset for known datasets (loads embeddings automatically)")
    return p.parse_args()


def main():
    args = parse_args()
    adata_path = Path(args.adata)
    if not adata_path.exists():
        raise FileNotFoundError(f"adata path not found: {adata_path}")

    print("Loading merged dataset:", adata_path)
    adata = ad.read_h5ad(str(adata_path))
    # 为保证和原来代码的兼容性，进行列名平替
    if 'gene_id' in adata.obs:
        adata.obs['target_gene'] = adata.obs['gene_id'].astype(str)
    elif 'gene' in adata.obs:
        adata.obs['target_gene'] = adata.obs['gene'].astype(str)
    if 'cell_line' in adata.obs:
        adata.obs['cell_type'] = adata.obs['cell_line'].astype(str)
        
    print("adata.obs columns:", list(adata.obs.columns))
    
    # === 使用已经提前筛好的 Highly Variable Genes ===
    if "highly_variable" in adata.var:
        print(f"Filtering by highly variable genes. Original vars: {adata.n_vars}")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"After HVG filtering vars: {adata.n_vars}")
    else:
        print("Warning: highly_variable column not found in dataset!")
    
    # === 过滤靶点: 只保留有 ESM2 特征的靶点 ===
    # esm_path = ROOT / "data" / "ESM2_pert_features.pt"
    # emb = torch.load(str(esm_path), map_location="cpu")

    # === 过滤靶点: 统一构建 gene2vec 字典 ===
    g2v_path = "/home/zhangshibo24s/cell_flow/data_train/selected_gene2vec_27k.npy"
    g2v_genes_path = "/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt"
    
    # 读入向量 (27874, 200) 和基因名列表 (27874个)
    g2v_array = np.load(g2v_path)
    with open(g2v_genes_path, "r") as f:
        g2v_genes = [line.strip() for line in f.readlines()]
    
    # 将它们打包成字典，格式就和之前 ESM2.pt 完全一致了！
    emb = {gene: torch.tensor(vec, dtype=torch.float32) for gene, vec in zip(g2v_genes, g2v_array)}
    # 后面可以继续用 emb['基因名']，不需要改后面过滤靶点的逻辑
    esm_keys = set(emb.keys())
    
    print(f"Original Obs shape: {adata.n_obs}")
    valid_mask = adata.obs["target_gene"].isin(esm_keys) | (adata.obs["target_gene"] == "non-targeting")
    adata = adata[valid_mask].copy()
    print(f"Filtered out {sum(~valid_mask)} cells whose target_gene lacks ESM2 features.")
    print(f"Current Obs shape: {adata.n_obs}")

    selected_gene_ids_file = ROOT / "data_train" / "selected_genes_27k.txt"
    selected_gene2vec_file = ROOT / "data_train" / "selected_gene2vec_27k.npy"
    gene2go_graph_file = ROOT / "data_train" / "human_ens_gene2go_graph.csv"

    print("Mapping var_names to Ensembl IDs via mygene and aligning to ontology gene list...")
    symbol_to_ensembl = build_symbol_to_ensembl([str(g) for g in adata.var_names])
    adata = align_adata_to_selected_ensembl(
        adata=adata,
        symbol_to_ensembl=symbol_to_ensembl,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_ids_file, matched_gene2vec_file = build_matched_gene2vec(
        selected_gene_ids_file=selected_gene_ids_file,
        selected_gene2vec_file=selected_gene2vec_file,
        ordered_ids=[str(g).upper() for g in adata.var_names],
        save_dir=out_dir,
    )
    print(f"Aligned genes: {adata.n_vars}")

    emb_dict = dict(emb)
    for k, v in list(emb_dict.items()):
        if torch.is_tensor(v):
            emb_dict[k] = v.cpu().numpy()
        else:
            emb_dict[k] = np.asarray(v)
    rep_key = "ESM2_pert_features"
    adata.uns[rep_key] = emb_dict
    # === 增加 cell_type 作为额外的 condition ===
    perturbation_covariates = {
        "gene_perturbation": ["target_gene"],
        "cell_type": ["cell_type"]
    }
    
    cell_lines = ['hepg2', 'jurkat', 'k562', 'rpe1']
    ct_emb_dict = {}
    for i, cl in enumerate(cell_lines):
        # one-hot representation for cell lines
        emb = np.zeros(len(cell_lines), dtype=np.float32)
        emb[i] = 1.0
        ct_emb_dict[cl] = emb
    adata.uns["cell_type_embeddings"] = ct_emb_dict
    
    perturbation_reps = {"gene_perturbation": rep_key, "cell_type": "cell_type_embeddings"}
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
    n_test_perts = int(0.3 * len(shuffled_perts))
    
    # 前 30% 到训练集，后 30% 到测试集
    train_perts = set(shuffled_perts[:n_train_perts])
    test_perts = set(shuffled_perts[-n_test_perts:])
    
    # 训练集: 其它3个细胞系全部 + holdout的前30%扰动 + holdout的non-targeting(让模型知道基态)
    train_mask = other_mask | (holdout_mask & adata.obs['target_gene'].isin(train_perts)) | (holdout_mask & (adata.obs['target_gene'] == 'non-targeting'))
    # 零样本测试集: holdout的后30%扰动
    test_mask = holdout_mask & adata.obs['target_gene'].isin(test_perts)
    
    adata_test_holdout = adata[test_mask].copy() 
    adata_train_full = adata[train_mask].copy()
    
    # 标准的训练-验证集划分 (从训练集中抽 5% 给 validation 观察曲线)
    n_train_total = adata_train_full.n_obs
    val_indices = rng.choice(n_train_total, int(n_train_total * 0.006), replace=False)
    val_mask_arr = np.zeros(n_train_total, dtype=bool)
    val_mask_arr[val_indices] = True
    
    adata_val = adata_train_full[val_mask_arr].copy()
    adata = adata_train_full[~val_mask_arr].copy()
    
    print(f"Leave-One-Cell-Line-Out Split:")
    print(f"  Holdout cell line: {holdout}")
    print(f"  Holdout perturbations in Train: {len(train_perts)} (30%)")
    print(f"  Holdout perturbations in Test : {len(test_perts)} (30%)")
    print(f"  Using {adata.n_obs} cells for training, {adata_val.n_obs} for validation.")
    print(f"  Zero-shot testing set contains {adata_test_holdout.n_obs} cells.")
    # ==============================================================================

    print("Initializing MyFlow (this may import jax/flax/ott)")
    cf = MyFlow(adata, solver=args.solver)
    print("Preparing data for training")
    cf.prepare_data(
        sample_rep=args.sample_rep,
        control_key=args.control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_reps,
    )
    # print("Preparing validation data")
    # cf.prepare_validation_data(
    #     adata_val,
    #     name="val",
    #     n_conditions_on_log_iteration=5,
    #     predict_kwargs={"batch_size": 1024, "n_time_steps": 100},
    # )
    print("Preparing model (default architecture). This may take a few seconds")
    # cf.prepare_model(seed=args.seed)
    cf.prepare_model(
        seed=args.seed,
        condition_encoder_kwargs={
            "x_graph_fusion_kwargs": {
                "enabled": True,
                "dim": int(np.load(matched_gene2vec_file).shape[1]),
                "max_seq_len": int(adata.n_vars),
                "max_edges": 80000,
                "gene2vec_file": str(matched_gene2vec_file),
                "gene_ids_file": str(matched_ids_file),
                "gene2go_graph_file": str(gene2go_graph_file),
            }
            # 知识加入的多少
        },
        solver_kwargs={
            "condition_combined_loss_weight": 0.01,
        },
        # solver_kwargs={
        #     "condition_change_eps": 1e-8,
        #     "condition_mask_smooth": 0.1,
        #     "condition_mask_kl_mix_change": 0.3,
        #     "condition_change_weight": 0.2,
        #     "condition_mask_aux_weight": 0.2,
        #     "condition_fused_mode": "adaptive",
        # },
    )
    print(f"Start training: iterations={args.num_iterations}, batch_size={args.batch_size}")
    # metrics_cb = Metrics(metrics=["r_squared", "mmd"])
    cf.train(
        num_iterations=args.num_iterations, 
        batch_size=args.batch_size,
        valid_freq=0,  # 临时禁用验证
        # callbacks=[metrics_cb],
        # monitor_metrics=["val_r_squared_mean", "val_mmd_mean"]
    )
    print("DEBUG: cf.train() exited successfully! Moving to save model...", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out_path = out_dir / f"model_{timestamp}"
    model_out_path.mkdir(parents=True, exist_ok=True)

    print(f"DEBUG: Calling cf.save() on path {model_out_path}...", flush=True)
    cf.save(str(model_out_path), file_prefix=None, overwrite=args.overwrite)
    print("DEBUG: cf.save() finished successfully!", flush=True)

    print("Saving model to output directory", flush=True)
    print("Training finished. Model saved {}.".format(model_out_path), flush=True)

    if args.skip_prediction:
        print("Skipping prediction stage due to --skip-prediction flag.")
        return

    print("DEBUG: Starting prediction setup...", flush=True)
    print("Starting prediction on the 70% zero-shot holdout tests...", flush=True)
    # 提取 holdout 细胞系的 control 作为测试集的 baseline 输入
    # (即上面划分时放进 adata_train_full 的 non-targeting 细胞)
    test_adata = adata_train_full[(adata_train_full.obs['cell_type']==holdout) & (adata_train_full.obs[args.control_key]==True)].copy()
    
    # 提取测试集中存在的全部未见过扰动
    groups = adata_test_holdout.obs.groupby("target_gene").groups

    all_X = []
    all_obs = []

    for gene, idx in groups.items():
        # 如果测试集里的这个基因对应的扰动细胞比可用的 non-targeting 多，就放回采样去匹配数量
        sample_size = len(idx)
        if sample_size > test_adata.n_obs:
            sampled_idx = np.random.choice(test_adata.n_obs, size=sample_size, replace=True)
        else:
            sampled_idx = np.random.choice(test_adata.n_obs, size=sample_size, replace=False)
        sub_adata = test_adata[sampled_idx].copy()
        
        covariate_data = pd.DataFrame({
            "target_gene": [gene],
            "cell_type": [holdout],
            args.control_key: [False]
        })
        predict_kwargs = {
            "adata": sub_adata,
            "covariate_data": covariate_data,
            "sample_rep": args.sample_rep,
        }
        if args.solver == "otfm":
            predict_kwargs["predict_batch_size"] = args.predict_batch_size
        preds = cf.predict(**predict_kwargs)
        arr = list(preds.values())[0]
        arr = np.asarray(arr)
        all_X.append(arr)
        obs = pd.DataFrame({
            "perturbation": [gene] * arr.shape[0]
        })
        all_obs.append(obs)
    print("Prediction finished")
    X = np.vstack(all_X)
    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=test_adata.var.copy())
    pred_dir = Path(args.output_dir) / f"predictions_{timestamp}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_file = pred_dir / f"predictions_{timestamp}.h5ad"
    adata_pred.write_h5ad(out_file)
    print(f"Saved prediction file: {out_file}")

    print("\n" + "=" * 50)
    print("Evaluating Predictions Against Ground Truth (Global Metrics)...")
    
    # 提取控制组(CTRL)、真实扰动(REAL)和模型预测(PRED) 的全局平均表达谱
    try:
        ctrl_mean = np.array(test_adata.X.mean(axis=0)).flatten()
        real_mean = np.array(adata_test_holdout.X.mean(axis=0)).flatten()
        ours_mean = np.array(adata_pred.X.mean(axis=0)).flatten()

        mse, mae, l2 = cal_metric(ours_mean, real_mean)
        pearson_del, pearson_del_top20, ds = cal_delta_metric(ctrl_mean, real_mean, ours_mean)
        
        print(f"Basic => MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
        print(f"Delta => Pearson Δ: {pearson_del:.4f}, Pearson Δ20: {pearson_del_top20:.4f}, DS: {ds:.4f}")
        
        print("\nCalculating global DES & DE-Spearman (this might take a few seconds)...")
        # 兼容处理索引以免重叠
        ctrl_copy = test_adata.copy()
        real_copy = adata_test_holdout.copy()
        pred_copy = adata_pred.copy()
        
        # 必须确保 var_names 对齐，避免 concatenate 时 var 错乱
        des_recall, des_acc, de_spearman = compute_des(ctrl_copy, real_copy, pred_copy)
        print(f"DES   => Recall: {des_recall:.4f}, Accuracy: {des_acc:.4f}, DE-Spearman rho: {de_spearman:.4f}")
    except Exception as e:
        print(f"Evaluation failed (usually due to sparse matrix formatting or dimension mismatch): {e}")
    print("=" * 50)

if __name__ == "__main__":
    main()
