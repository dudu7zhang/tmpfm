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


def stratified_cap_obs(
    adata: ad.AnnData,
    max_cells: int | None,
    rng: np.random.Generator,
    group_key: str,
) -> ad.AnnData:
    if max_cells is None or max_cells <= 0 or adata.n_obs <= max_cells:
        return adata.copy()

    fraction = max_cells / adata.n_obs
    return stratified_subsample_obs(adata, fraction=fraction, rng=rng, group_key=group_key)


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
        default="/home/zhangshibo24s/cell_flow/data_train/zebrafish_processed.h5ad",
        required=False,
        help="Path to anndata (.h5ad) or directory containing *hvg.h5ad files"
    )
    p.add_argument(
        "--sample-rep",
        default="X",
        help="Key in adata.obsm to use as sample representation (default: X)",
    )
    p.add_argument("--control-key", default="is_control", help="obs column marking control samples.")
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
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--run-name", default=None, help="Optional run name used in saved model/prediction/log filenames.")
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"), help="Visible GPU id for this run.")
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target-key", default="gene_target", help="obs column with perturbation target.")
    p.add_argument("--time-key", default="timepoint", help="obs column used as zebrafish temporal context.")
    p.add_argument("--test-condition-fraction", type=float, default=0.3)
    p.add_argument("--val-fraction", type=float, default=0.006)
    p.add_argument(
        "--train-cell-fraction",
        type=float,
        default=0.1,
        help="Fraction of training cells to keep after LOCO split, stratified by target_gene.",
    )
    p.add_argument(
        "--test-cell-fraction",
        type=float,
        default=0.1,
        help="Fraction of zero-shot test cells to keep after LOCO split, stratified by target_gene.",
    )
    p.add_argument(
        "--max-train-cells",
        type=int,
        default=0,
        help="Optional hard cap for training cells after fraction subsampling and before validation. 0 disables the cap.",
    )
    p.add_argument(
        "--max-test-cells",
        type=int,
        default=0,
        help="Optional hard cap for zero-shot test cells after fraction subsampling. 0 disables the cap.",
    )
    p.add_argument(
        "--x-graph-fusion-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable x graph fusion. Disabled by default for zebrafish unless compatible graph/gene2vec assets are provided.",
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

    print("Loading zebrafish dataset:", adata_path)
    adata = ad.read_h5ad(str(adata_path))
    if args.target_key not in adata.obs:
        raise KeyError(f"target key {args.target_key!r} not found in adata.obs")
    if args.time_key not in adata.obs:
        raise KeyError(f"time key {args.time_key!r} not found in adata.obs")
    if args.control_key not in adata.obs:
        raise KeyError(f"control key {args.control_key!r} not found in adata.obs")

    adata.obs["target_gene"] = adata.obs[args.target_key].astype(str)
    adata.obs["cell_type"] = adata.obs[args.time_key].astype(float)
    adata.obs[args.control_key] = adata.obs[args.control_key].astype(bool)

    print("adata.obs columns:", list(adata.obs.columns))
    
    if "highly_variable" in adata.var:
        print(f"Filtering by highly variable genes. Original vars: {adata.n_vars}")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"After HVG filtering vars: {adata.n_vars}")
    else:
        print("Warning: highly_variable column not found in dataset!")

    print(f"Current Obs shape: {adata.n_obs}")
    print(f"Current Var shape: {adata.n_vars}")
    if args.x_graph_fusion_enabled:
        raise ValueError(
            "x_graph_fusion is disabled for zebrafish by default because the current graph/gene2vec assets are human-specific."
        )

    perturbation_covariates = {"gene_perturbation": ["target_gene"]}
    perturbation_reps = {}
    if args.use_cell_type_condition:
        perturbation_covariates["cell_type"] = ["cell_type"]

    print(f"Total cells before split: {adata.n_obs}")

    # Split by perturbation-time condition. Controls remain available for every train/test timepoint.
    rng = np.random.default_rng(args.seed)
    control_mask = adata.obs[args.control_key].astype(bool)
    pert_conditions = (
        adata.obs.loc[~control_mask, ["target_gene", "cell_type"]]
        .drop_duplicates()
        .sort_values(["cell_type", "target_gene"])
    )
    condition_values = [tuple(row) for row in pert_conditions.to_numpy()]
    shuffled_conditions = rng.permutation(np.asarray(condition_values, dtype=object))
    n_test_conditions = max(1, int(round(args.test_condition_fraction * len(shuffled_conditions))))
    test_conditions = {tuple(x) for x in shuffled_conditions[:n_test_conditions]}
    train_conditions = set(condition_values) - test_conditions
    train_target_genes = sorted({str(gene) for gene, _ in train_conditions})
    test_target_genes = sorted({str(gene) for gene, _ in test_conditions})

    condition_tuples = list(zip(adata.obs["target_gene"], adata.obs["cell_type"], strict=False))
    is_test_condition = np.array([tuple(x) in test_conditions for x in condition_tuples], dtype=bool)
    is_train_condition = np.array([tuple(x) in train_conditions for x in condition_tuples], dtype=bool)

    train_mask = control_mask.to_numpy() | ((~control_mask.to_numpy()) & is_train_condition)
    test_mask = (~control_mask.to_numpy()) & is_test_condition
    
    adata_train_full = adata[train_mask].copy()
    adata_test_holdout = adata[test_mask].copy()
    train_cells_before_subsample = adata_train_full.n_obs
    test_cells_before_subsample = adata_test_holdout.n_obs
    adata_train_full = stratified_subsample_obs(
        adata_train_full,
        fraction=args.train_cell_fraction,
        rng=rng,
        group_key="condition",
    )
    adata_test_holdout = stratified_subsample_obs(
        adata_test_holdout,
        fraction=args.test_cell_fraction,
        rng=rng,
        group_key="condition",
    )
    train_cells_before_cap = adata_train_full.n_obs
    test_cells_before_cap = adata_test_holdout.n_obs
    adata_train_full = stratified_cap_obs(
        adata_train_full,
        max_cells=args.max_train_cells,
        rng=rng,
        group_key="condition",
    )
    adata_test_holdout = stratified_cap_obs(
        adata_test_holdout,
        max_cells=args.max_test_cells,
        rng=rng,
        group_key="condition",
    )
    
    n_train_total = adata_train_full.n_obs
    val_indices = rng.choice(n_train_total, int(n_train_total * args.val_fraction), replace=False)
    val_mask_arr = np.zeros(n_train_total, dtype=bool)
    val_mask_arr[val_indices] = True
    
    adata_val = adata_train_full[val_mask_arr].copy()
    adata = adata_train_full[~val_mask_arr].copy()
    
    print("Zebrafish perturbation-time split:")
    print(f"  Perturbation-time conditions in Train: {len(train_conditions)}")
    print(f"  Perturbation-time conditions in Test : {len(test_conditions)} ({args.test_condition_fraction:.2%})")
    print(f"  Train target genes: {len(train_target_genes)}")
    print(f"  Test target genes : {len(test_target_genes)}")
    print(f"  Test target genes preview: {test_target_genes[:20]}")
    print(f"  Training cells kept: {train_cells_before_cap}/{train_cells_before_subsample} ({args.train_cell_fraction:.2%}) before cap.")
    if args.max_train_cells > 0:
        print(f"  Training cells after cap: {adata_train_full.n_obs}/{args.max_train_cells}")
    print(f"  Test cells kept    : {test_cells_before_cap}/{test_cells_before_subsample} ({args.test_cell_fraction:.2%}) before cap.")
    if args.max_test_cells > 0:
        print(f"  Test cells after cap    : {adata_test_holdout.n_obs}/{args.max_test_cells}")
    print(f"  Using {adata.n_obs} cells for training, {adata_val.n_obs} for validation.")
    print(f"  Zero-shot testing set contains {adata_test_holdout.n_obs} cells.")
    write_json(
        out_dir / f"split_summary_{run_label}.json",
        {
            "test_condition_fraction": args.test_condition_fraction,
            "perturbation_time_conditions_total": len(condition_values),
            "perturbation_time_conditions_train": len(train_conditions),
            "perturbation_time_conditions_test": len(test_conditions),
            "train_cells_before_subsample": int(train_cells_before_subsample),
            "test_cells_before_subsample": int(test_cells_before_subsample),
            "train_cells_before_cap": int(train_cells_before_cap),
            "test_cells_before_cap": int(test_cells_before_cap),
            "train_cell_fraction": args.train_cell_fraction,
            "test_cell_fraction": args.test_cell_fraction,
            "max_train_cells": int(args.max_train_cells),
            "max_test_cells": int(args.max_test_cells),
            "train_target_genes": train_target_genes,
            "test_target_genes": test_target_genes,
            "train_conditions": [{"target_gene": str(gene), "cell_type": float(cell_type)} for gene, cell_type in sorted(train_conditions)],
            "test_conditions": [{"target_gene": str(gene), "cell_type": float(cell_type)} for gene, cell_type in sorted(test_conditions)],
            "train_full_before_validation": summarize_adata_split(adata_train_full, args.control_key),
            "train_passed_to_cellflow": summarize_adata_split(adata, args.control_key),
            "validation": summarize_adata_split(adata_val, args.control_key),
            "zero_shot_test": summarize_adata_split(adata_test_holdout, args.control_key),
        },
    )
    # ==============================================================================

    # Ensure X is CSR (CellFlow's DataManager only handles CSR sparse matrices correctly)
    import scipy.sparse as sp
    if sp.issparse(adata.X) and not isinstance(adata.X, sp.csr_matrix):
        print(f"Converting X from {type(adata.X).__name__} to csr_matrix")
        adata.X = sp.csr_matrix(adata.X)

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
    cf.prepare_model(
        seed=args.seed,
        condition_encoder_kwargs={"x_graph_fusion_kwargs": {"enabled": False}},
        solver_kwargs={"condition_combined_loss_weight": args.condition_combined_loss_weight},
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
    print("Starting prediction on the zero-shot perturbation-time tests...", flush=True)
    groups = adata_test_holdout.obs.groupby(["target_gene", "cell_type"], observed=True).groups

    all_X = []
    all_obs = []

    prediction_controls = []
    for (gene, cell_type), idx in groups.items():
        test_adata = adata_train_full[
            (adata_train_full.obs["cell_type"] == cell_type)
            & (adata_train_full.obs[args.control_key].astype(bool))
        ].copy()
        if test_adata.n_obs == 0:
            print(f"Skipping prediction for {(gene, cell_type)} because no matching control cells were found.")
            continue
        prediction_controls.append(test_adata)
        sample_size = len(idx)
        if sample_size > test_adata.n_obs:
            sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=True)
        else:
            sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=False)
        sub_adata = test_adata[sampled_idx].copy()
        
        covariate_data = pd.DataFrame({
            "target_gene": [gene],
            "cell_type": [cell_type],
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
            "perturbation": [gene] * arr.shape[0],
            "target_gene": [gene] * arr.shape[0],
            "cell_type": [cell_type] * arr.shape[0],
        })
        all_obs.append(obs)
    print("Prediction finished")
    if not all_X:
        raise RuntimeError("No predictions were generated; check control availability for test timepoints.")
    X = np.vstack(all_X)
    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=adata.var.copy())
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
        ctrl_eval = ad.concat(prediction_controls, join="outer") if len(prediction_controls) > 1 else prediction_controls[0]
        ctrl_mean = np.array(ctrl_eval.X.mean(axis=0)).flatten()
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

        condition_metric_rows = []
        for (gene, cell_type), _ in groups.items():
            real_mask = (
                (adata_test_holdout.obs["target_gene"] == gene)
                & (adata_test_holdout.obs["cell_type"] == cell_type)
            ).to_numpy()
            pred_mask = (
                (adata_pred.obs["target_gene"] == gene)
                & (adata_pred.obs["cell_type"] == cell_type)
            ).to_numpy()
            ctrl_mask = (
                (adata_train_full.obs["cell_type"] == cell_type)
                & (adata_train_full.obs[args.control_key].astype(bool))
            ).to_numpy()
            if real_mask.sum() == 0 or pred_mask.sum() == 0 or ctrl_mask.sum() == 0:
                continue

            real_group_mean = np.array(adata_test_holdout[real_mask].X.mean(axis=0)).flatten()
            pred_group_mean = np.array(adata_pred[pred_mask].X.mean(axis=0)).flatten()
            ctrl_group_mean = np.array(adata_train_full[ctrl_mask].X.mean(axis=0)).flatten()
            group_mse, group_mae, group_l2 = cal_metric(pred_group_mean, real_group_mean)
            group_pearson_delta, group_pearson_delta_top20, group_ds = cal_delta_metric(
                ctrl_group_mean,
                real_group_mean,
                pred_group_mean,
            )
            condition_metric_rows.append(
                {
                    "target_gene": str(gene),
                    "cell_type": float(cell_type),
                    "real_cells": int(real_mask.sum()),
                    "pred_cells": int(pred_mask.sum()),
                    "control_cells": int(ctrl_mask.sum()),
                    "mse": float(group_mse),
                    "mae": float(group_mae),
                    "l2": float(group_l2),
                    "pearson_delta": float(group_pearson_delta),
                    "pearson_delta_top20": float(group_pearson_delta_top20),
                    "direction_sign_score": float(group_ds),
                }
            )

        condition_metrics_file = out_dir / f"condition_metrics_{run_label}.csv"
        condition_metrics_df = pd.DataFrame(condition_metric_rows)
        if not condition_metrics_df.empty:
            condition_metrics_df = condition_metrics_df.sort_values(["cell_type", "target_gene"])
        condition_metrics_df.to_csv(condition_metrics_file, index=False)
        print(f"Saved per-condition metrics: {condition_metrics_file}")
        metrics_summary["condition_metrics_file"] = str(condition_metrics_file)
        metrics_summary["condition_metrics_count"] = len(condition_metric_rows)
        
        print("\nCalculating global DES & DE-Spearman (this might take a few seconds)...")
        # 兼容处理索引以免重叠
        ctrl_copy = ctrl_eval.copy()
        real_copy = adata_test_holdout.copy()
        pred_copy = adata_pred.copy()
        
        # 必须确保 var_names 对齐，避免 concatenate 时 var 错乱
        des_recall, des_acc, de_spearman = compute_des(ctrl_copy, real_copy, pred_copy)
        print(f"DES   => Recall: {des_recall:.4f}, Accuracy: {des_acc:.4f}, DE-Spearman rho: {de_spearman:.4f}")
        metrics_summary.update(
            {
                "des_recall": float(des_recall),
                "des_accuracy": float(des_acc),
                "de_spearman": float(de_spearman),
            }
        )
    except Exception as e:
        print(f"Evaluation failed (usually due to sparse matrix formatting or dimension mismatch): {e}")
        metrics_summary["error"] = str(e)
    write_json(out_dir / f"metrics_summary_{run_label}.json", metrics_summary)
    print("=" * 50)

if __name__ == "__main__":
    main()
