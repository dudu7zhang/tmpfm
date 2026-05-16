#!/usr/bin/env python3
"""
Notes:
- This script expects to be run from the repository root (`/home/zhangshibo24s/cell_project`).
- It adds the local `cellflow/src` package to `sys.path` so you don't need to install the package.
- Provide `--perturbation-covariates` as a JSON string mapping covariate-name -> list-of-obs-columns.
- Optionally provide `--perturbation-reps` as a JSON dict mapping covariate-name -> key-in-adata.uns holding embeddings.
"""

import argparse
import json
import os
from pathlib import Path
import re
import random
import sys

# Set GPU/JAX environment before importing torch/jax/cellflow.
ROOT = Path(__file__).resolve().parent


def _read_early_cli_option(name: str, default: str) -> str:
    prefix = f"{name}="
    for i, arg in enumerate(sys.argv[1:]):
        if arg == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


# Ensure only one GPU is visible before importing torch/jax/cellflow.
os.environ["CUDA_VISIBLE_DEVICES"] = _read_early_cli_option("--gpu-id", os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
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
from cellflow.model._cellflow import CellFlow
from cellflow.training import Metrics

ENSG_PATTERN = re.compile(r"^ENSG\d+$", re.IGNORECASE)
DEFAULT_SEED = 20240508

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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stratified_subsample_obs(
    adata: ad.AnnData,
    fraction: float,
    rng: np.random.Generator,
    group_key: str,
) -> ad.AnnData:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1 or adata.n_obs == 0:
        return adata.copy()

    selected_positions = []
    for _, positions in adata.obs.groupby(group_key, observed=True).indices.items():
        positions = np.asarray(positions)
        n_keep = max(1, int(round(len(positions) * fraction)))
        selected_positions.extend(rng.choice(positions, size=n_keep, replace=False).tolist())

    selected_positions = np.asarray(selected_positions)
    selected_positions.sort()
    return adata[selected_positions].copy()


def summarize_adata_split(adata: ad.AnnData, control_key: str) -> dict:
    control_mask = adata.obs[control_key].astype(bool).to_numpy()
    pert_obs = adata.obs.loc[~control_mask]
    summary = {
        "cells": int(adata.n_obs),
        "controls": int(control_mask.sum()),
        "targets": int((~control_mask).sum()),
        "target_genes": int(pert_obs["target_gene"].nunique()) if "target_gene" in pert_obs else 0,
        "target_conditions": int(pert_obs[["target_gene", "cell_type"]].drop_duplicates().shape[0])
        if {"target_gene", "cell_type"}.issubset(pert_obs.columns)
        else 0,
        "by_cell_type": {},
    }
    if "cell_type" in adata.obs:
        for cell_type, sub_obs in adata.obs.groupby("cell_type", observed=True):
            sub_control = sub_obs[control_key].astype(bool).to_numpy()
            sub_pert = sub_obs.loc[~sub_control]
            summary["by_cell_type"][str(cell_type)] = {
                "cells": int(len(sub_obs)),
                "controls": int(sub_control.sum()),
                "targets": int((~sub_control).sum()),
                "target_conditions": int(sub_pert[["target_gene", "cell_type"]].drop_duplicates().shape[0])
                if {"target_gene", "cell_type"}.issubset(sub_pert.columns)
                else 0,
                "cell_level_pairs": int(sub_control.sum() * (~sub_control).sum()),
            }
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


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
    p.add_argument("--num-iterations", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--skip-prediction", action="store_true")
    # p.add_argument("--valid-freq", type=int, default=500)
    p.add_argument("--output-dir", default="results/outputs/outputs")
    p.add_argument("--run-name", default=None, help="Optional run name used in saved model/prediction/log filenames.")
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"), help="Visible GPU id for this run.")
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--holdout-cell-line", default="hepg2", help="Cell line to hold out.")
    p.add_argument(
        "--train-cell-fraction",
        type=float,
        default=0.3,
        help="Fraction of training cells to keep after LOCO split, stratified by target_gene.",
    )
    p.add_argument(
        "--test-cell-fraction",
        type=float,
        default=0.3,
        help="Fraction of zero-shot test cells to keep after LOCO split, stratified by target_gene.",
    )
    p.add_argument(
        "--x-graph-fusion-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable x graph fusion. Use --no-x-graph-fusion-enabled for baseline.",
    )
    p.add_argument(
        "--use-cell-type-condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cell_type as an explicit model condition. Use --no-use-cell-type-condition for strict baseline.",
    )
    p.add_argument(
        "--use-cell-type-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split control-target sampling by cell_type. Use --no-use-cell-type-split for strict baseline.",
    )
    p.add_argument("--condition-combined-loss-weight", type=float, default=0.01)
    # p.add_argument("--preset", choices=["jurkat"], default=None, help="Optional preset for known datasets (loads embeddings automatically)")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    print(f"Using fixed random seed: {args.seed}")
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_name or timestamp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / f"experiment_config_{run_label}.json",
        {
            "run_label": run_label,
            "timestamp": timestamp,
            "args": vars(args),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
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
    perturbation_covariates = {"gene_perturbation": ["target_gene"]}
    perturbation_reps = {"gene_perturbation": rep_key}
    if args.use_cell_type_condition:
        perturbation_covariates["cell_type"] = ["cell_type"]
        cell_lines = ['hepg2', 'jurkat', 'k562', 'rpe1']
        ct_emb_dict = {}
        for i, cl in enumerate(cell_lines):
            # one-hot representation for cell lines
            emb = np.zeros(len(cell_lines), dtype=np.float32)
            emb[i] = 1.0
            ct_emb_dict[cl] = emb
        adata.uns["cell_type_embeddings"] = ct_emb_dict
        perturbation_reps["cell_type"] = "cell_type_embeddings"
    if args.control_key not in adata.obs:
        adata.obs[args.control_key] = adata.obs["target_gene"].astype(str) == "non-targeting"

    print(f"Total cells before split: {adata.n_obs}")
    
    # ================= Leave-One-Cell-Line-Out (LOCO) Split Logic =================
    holdout = args.holdout_cell_line
    assert holdout in adata.obs['cell_type'].unique(), f"Holdout cell line {holdout} not found in adata.obs['cell_type']"
    
    other_mask = adata.obs['cell_type'] != holdout
    holdout_mask = adata.obs['cell_type'] == holdout
    
    # 提取 holdout 细胞系独有的所有扰动
    perts = sorted(adata[holdout_mask].obs['target_gene'].astype(str).unique().tolist())
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
    
    adata_train_full = adata[train_mask].copy()
    adata_test_holdout = adata[test_mask].copy()
    train_cells_before_subsample = adata_train_full.n_obs
    test_cells_before_subsample = adata_test_holdout.n_obs
    adata_train_full = stratified_subsample_obs(
        adata_train_full,
        fraction=args.train_cell_fraction,
        rng=rng,
        group_key="target_gene",
    )
    adata_test_holdout = stratified_subsample_obs(
        adata_test_holdout,
        fraction=args.test_cell_fraction,
        rng=rng,
        group_key="target_gene",
    )
    
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
    print(f"  Training cells kept: {adata_train_full.n_obs}/{train_cells_before_subsample} ({args.train_cell_fraction:.2%}) before validation split.")
    print(f"  Test cells kept    : {adata_test_holdout.n_obs}/{test_cells_before_subsample} ({args.test_cell_fraction:.2%}).")
    print(f"  Using {adata.n_obs} cells for training, {adata_val.n_obs} for validation.")
    print(f"  Zero-shot testing set contains {adata_test_holdout.n_obs} cells.")
    write_json(
        out_dir / f"split_summary_{run_label}.json",
        {
            "holdout_cell_line": holdout,
            "holdout_perturbations_total": len(pert_targets),
            "holdout_perturbations_train": len(train_perts),
            "holdout_perturbations_test": len(test_perts),
            "train_cells_before_subsample": int(train_cells_before_subsample),
            "test_cells_before_subsample": int(test_cells_before_subsample),
            "train_cell_fraction": args.train_cell_fraction,
            "test_cell_fraction": args.test_cell_fraction,
            "train_full_before_validation": summarize_adata_split(adata_train_full, args.control_key),
            "train_passed_to_cellflow": summarize_adata_split(adata, args.control_key),
            "validation": summarize_adata_split(adata_val, args.control_key),
            "zero_shot_test": summarize_adata_split(adata_test_holdout, args.control_key),
        },
    )
    # ==============================================================================

    print("Initializing CellFlow (this may import jax/flax/ott)")
    cf = CellFlow(adata, solver=args.solver)
    print("Preparing data for training")
    split_covariates = ["cell_type"] if args.use_cell_type_split else None
    cf.prepare_data(
        sample_rep=args.sample_rep,
        control_key=args.control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_reps,
        split_covariates=split_covariates,
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
                "enabled": args.x_graph_fusion_enabled,
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
            "condition_combined_loss_weight": args.condition_combined_loss_weight,
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
        seed=args.seed,
        valid_freq=0,  # 临时禁用验证
        # callbacks=[metrics_cb],
        # monitor_metrics=["val_r_squared_mean", "val_mmd_mean"]
    )
    print("DEBUG: cf.train() exited successfully! Moving to save model...", flush=True)
    if cf.trainer is not None and getattr(cf.trainer, "training_logs", None):
        logs = cf.trainer.training_logs
        pd.DataFrame({k: pd.Series(v) for k, v in logs.items()}).to_csv(
            out_dir / f"training_logs_{run_label}.csv",
            index_label="step",
        )

    model_out_path = out_dir / f"model_{run_label}"
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
    print("Starting prediction on the zero-shot holdout tests...", flush=True)
    # 提取 holdout 细胞系的 control 作为测试集的 baseline 输入
    # (即上面划分时放进 adata_train_full 的 non-targeting 细胞)
    test_adata = adata_train_full[(adata_train_full.obs['cell_type']==holdout) & (adata_train_full.obs[args.control_key]==True)].copy()
    
    # 提取测试集中存在的全部未见过扰动
    groups = adata_test_holdout.obs.groupby("target_gene").groups

    all_X = []
    all_obs = []

    # Group genes by sample_size to reduce repeated JAX recompilations caused by varying input shapes.
    # This keeps per-gene prediction sizes identical to the current behavior (sample_size == #real test cells for the gene),
    # but amortizes compilation across genes that share the same sample_size.
    size_to_genes: dict[int, list[str]] = {}
    for gene, idx in groups.items():
        sample_size = int(len(idx))
        size_to_genes.setdefault(sample_size, []).append(str(gene))
    bucket_sizes = sorted(size_to_genes.keys())
    print(
        f"Prediction buckets by sample_size: {len(bucket_sizes)} unique sizes "
        f"across {len(groups)} genes. Largest sample_size={max(bucket_sizes) if bucket_sizes else 0}.",
        flush=True,
    )

    for sample_size in bucket_sizes:
        genes = size_to_genes[sample_size]
        print(f"  Bucket sample_size={sample_size}: {len(genes)} genes", flush=True)
        for gene in genes:
            # If this gene's test cells > available non-targeting controls, sample with replacement to match size.
            if sample_size > test_adata.n_obs:
                sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=True)
            else:
                sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=False)
            # Avoid .copy() to reduce repeated large memory copies during prediction.
            sub_adata = test_adata[sampled_idx]

            covariate_data = pd.DataFrame(
                {
                    "target_gene": [gene],
                    "cell_type": [holdout],
                    args.control_key: [False],
                }
            )
            predict_kwargs = {
                "adata": sub_adata,
                "covariate_data": covariate_data,
                "sample_rep": args.sample_rep,
            }
            if args.solver == "otfm":
                predict_kwargs["predict_batch_size"] = args.predict_batch_size
            preds = cf.predict(**predict_kwargs)
            arr = np.asarray(list(preds.values())[0])
            all_X.append(arr)
            obs = pd.DataFrame({"perturbation": [gene] * arr.shape[0]})
            all_obs.append(obs)
    print("Prediction finished")
    X = np.vstack(all_X)
    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=test_adata.var.copy())
    pred_dir = Path(args.output_dir) / f"predictions_{run_label}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_file = pred_dir / f"predictions_{run_label}.h5ad"
    adata_pred.write_h5ad(out_file)
    print(f"Saved prediction file: {out_file}")

    print("\n" + "=" * 50)
    print("Evaluating Predictions Against Ground Truth (Global Metrics)...")
    
    # 提取控制组(CTRL)、真实扰动(REAL)和模型预测(PRED) 的全局平均表达谱
    metrics_summary = {
        "run_label": run_label,
        "prediction_file": str(out_file),
        "success": False,
    }
    try:
        ctrl_mean = np.array(test_adata.X.mean(axis=0)).flatten()
        real_mean = np.array(adata_test_holdout.X.mean(axis=0)).flatten()
        ours_mean = np.array(adata_pred.X.mean(axis=0)).flatten()

        mse, mae, l2 = cal_metric(ours_mean, real_mean)
        pearson_del, pearson_del_top20, ds = cal_delta_metric(ctrl_mean, real_mean, ours_mean)
        
        print(f"Basic => MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
        print(f"Delta => Pearson Δ: {pearson_del:.4f}, Pearson Δ20: {pearson_del_top20:.4f}, DS: {ds:.4f}")
        metrics_summary.update(
            {
                "success": True,
                "mse": float(mse),
                "mae": float(mae),
                "l2": float(l2),
                "pearson_delta": float(pearson_del),
                "pearson_delta_top20": float(pearson_del_top20),
                "direction_sign_score": float(ds),
            }
        )
        
        print("\nCalculating per-condition DES & DE-Spearman...")
        ctrl_copy = test_adata.copy()
        des_per_condition = []
        perturbation_groups = adata_test_holdout.obs.groupby("target_gene").groups
        for gene, idx in perturbation_groups.items():
            real_cond = adata_test_holdout[idx].copy()
            pred_mask = adata_pred.obs["perturbation"] == gene
            pred_cond = adata_pred[pred_mask].copy()
            if real_cond.n_obs == 0 or pred_cond.n_obs == 0:
                continue
            try:
                d_recall, d_acc, d_spearman = compute_des(ctrl_copy, real_cond, pred_cond)
                des_per_condition.append({
                    "condition": str(gene),
                    "des_recall": float(d_recall),
                    "des_accuracy": float(d_acc),
                    "de_spearman": float(d_spearman) if not np.isnan(d_spearman) else None,
                })
            except Exception:
                continue

        if des_per_condition:
            des_df = pd.DataFrame(des_per_condition)
            des_recall_avg = des_df["des_recall"].mean()
            des_acc_avg = des_df["des_accuracy"].mean()
            spearman_valid = des_df["de_spearman"].dropna()
            de_spearman_avg = float(spearman_valid.mean()) if len(spearman_valid) > 0 else float("nan")
            print(f"DES (per-condition avg) => Recall: {des_recall_avg:.4f}, Accuracy: {des_acc_avg:.4f}, DE-Spearman rho: {de_spearman_avg:.4f}")
            metrics_summary.update(
                {
                    "des_recall": float(des_recall_avg),
                    "des_accuracy": float(des_acc_avg),
                    "de_spearman": float(de_spearman_avg),
                    "des_conditions_count": len(des_per_condition),
                }
            )
            des_file = out_dir / f"des_per_condition_{run_label}.csv"
            des_df.to_csv(des_file, index=False)
            print(f"Saved per-condition DES: {des_file}")
        else:
            print("No valid per-condition DES computed.")
    except Exception as e:
        print(f"Evaluation failed (usually due to sparse matrix formatting or dimension mismatch): {e}")
        metrics_summary["error"] = str(e)
    write_json(out_dir / f"metrics_summary_{run_label}.json", metrics_summary)
    print("=" * 50)

if __name__ == "__main__":
    main()
