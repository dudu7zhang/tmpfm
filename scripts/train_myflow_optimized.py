#!/usr/bin/env python3
"""
Optimized MyFlow training script with novel graph encoding.

Novel features:
1. Graph Dropout: Randomly drops edges during training for regularization
2. Multi-scale Residual Graph Propagation: 2-layer propagation with residual connections
3. Sparse Attention: Only attend to top-k most relevant genes
4. Optimized MLP: Smaller but still larger than GEARS (512/1024 vs 64)
5. Reduced gradient accumulation: 5 steps instead of 20

Usage:
    python train_myflow_optimized.py --adata /path/to/data.h5ad --run-name optimized
"""

import argparse
import json
import os
from pathlib import Path
import re
import random
import sys

ROOT = Path(__file__).resolve().parent.parent


def _read_early_cli_option(name: str, default: str) -> str:
    prefix = f"{name}="
    for i, arg in enumerate(sys.argv[1:]):
        if arg == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


os.environ["CUDA_VISIBLE_DEVICES"] = _read_early_cli_option("--gpu-id", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import anndata as ad
import mygene
import optax
import pandas as pd
import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats
from datetime import datetime
from myflow.model._myflow import MyFlow

ENSG_PATTERN = re.compile(r"^ENSG\d+$", re.IGNORECASE)
DEFAULT_SEED = 20240508


# ==================== Evaluation Metrics ====================
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
    return len(inter) / n_true, len(inter) / n_pred


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
                break
            except Exception as e:
                print(f"Network error querying MyGene (attempt {attempt + 1}): {e}")
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
        "conditions": int(pert_obs["condition"].nunique()) if "condition" in pert_obs.columns else 0,
    }
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _parse_condition_genes(condition: str) -> list[str]:
    """Return non-control gene symbols from a Norman guide_merged condition."""
    genes = []
    for gene in str(condition).split("+"):
        gene = gene.strip()
        if not gene or gene.lower() == "ctrl":
            continue
        genes.append(gene)
    return genes


def _is_single_condition(condition: str) -> bool:
    genes = _parse_condition_genes(condition)
    return len(genes) == 1


def _is_double_condition(condition: str) -> bool:
    genes = _parse_condition_genes(condition)
    return len(genes) == 2


def build_scdfm_norman_additive_split(
    conditions: list[str],
    fold: int,
    test_fraction: float,
    seed_base: int = 42,
) -> tuple[set[str], set[str], dict]:
    """Build Norman split following scDFM additive protocol."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    non_control = sorted(c for c in conditions if _parse_condition_genes(c))
    single_conditions = sorted(c for c in non_control if _is_single_condition(c))
    double_conditions = sorted(c for c in non_control if _is_double_condition(c))
    other_conditions = sorted(set(non_control) - set(single_conditions) - set(double_conditions))

    rng = np.random.default_rng(seed_base + fold)

    shuffled = rng.permutation(double_conditions)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_conditions = set(str(c) for c in shuffled[:n_test])
    train_conditions = set(non_control) - test_conditions
    split_info = {
        "split_method": "additive",
        "fold": int(fold),
        "seed_base": int(seed_base),
        "split_seed": int(seed_base + fold),
        "single_conditions_total": len(single_conditions),
        "double_conditions_total": len(double_conditions),
        "other_conditions_total": len(other_conditions),
        "additive_note": "All single perturbations are kept in training; only double perturbations are split into test.",
    }

    if not test_conditions:
        raise ValueError("No test conditions generated for additive split")
    if not train_conditions:
        raise ValueError("No train conditions generated for additive split")

    split_info.update(
        {
            "train_conditions_count": len(train_conditions),
            "test_conditions_count": len(test_conditions),
            "train_single_conditions_count": sum(_is_single_condition(c) for c in train_conditions),
            "train_double_conditions_count": sum(_is_double_condition(c) for c in train_conditions),
            "test_single_conditions_count": sum(_is_single_condition(c) for c in test_conditions),
            "test_double_conditions_count": sum(_is_double_condition(c) for c in test_conditions),
        }
    )
    return train_conditions, test_conditions, split_info


def _build_norman_gene_tokens(
    conditions: list[str],
    gene2vec_dict: dict[str, np.ndarray],
    embedding_dim: int,
    max_genes: int = 2,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[str, str]]:
    """Build per-gene condition tokens for Norman combinatorial perturbations."""
    unique_symbols = sorted({g for cond in conditions for g in _parse_condition_genes(cond)})
    symbol_to_ensembl = build_symbol_to_ensembl(unique_symbols) if unique_symbols else {}

    token_to_vec: dict[str, np.ndarray] = {
        "ctrl": np.zeros(embedding_dim, dtype=np.float32),
    }
    symbol_to_token: dict[str, str] = {}
    missing_symbols: list[str] = []

    for symbol in unique_symbols:
        ensembl = symbol_to_ensembl.get(symbol, "").upper()
        if ensembl and ensembl in gene2vec_dict:
            symbol_to_token[symbol] = ensembl
            token_to_vec[ensembl] = gene2vec_dict[ensembl].astype(np.float32)
        else:
            token = f"missing::{symbol.upper()}"
            symbol_to_token[symbol] = token
            token_to_vec[token] = np.zeros(embedding_dim, dtype=np.float32)
            missing_symbols.append(symbol)

    if missing_symbols:
        preview = ", ".join(missing_symbols[:20])
        suffix = "..." if len(missing_symbols) > 20 else ""
        print(
            f"  Warning: {len(missing_symbols)} perturbation genes could not be mapped to gene2vec "
            f"after symbol->Ensembl conversion: {preview}{suffix}. Using zero vectors for them."
        )

    condition_to_tokens: dict[str, list[str]] = {}
    for condition in conditions:
        symbols = sorted(_parse_condition_genes(condition))
        if len(symbols) > max_genes:
            print(f"  Warning: condition '{condition}' has >{max_genes} genes; truncating extras.")
            symbols = symbols[:max_genes]
        tokens = [symbol_to_token[s] for s in symbols]
        tokens.extend(["ctrl"] * (max_genes - len(tokens)))
        condition_to_tokens[str(condition)] = tokens

    return condition_to_tokens, token_to_vec, symbol_to_ensembl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adata",
        default="/home/zhangshibo24s/cell_flow/data_train/norman_2019_adata.h5ad",
        required=False,
    )
    p.add_argument("--sample-rep", default="X")
    p.add_argument("--control-key", default="is_control")
    p.add_argument("--target-key", default="guide_identity", help="obs column with perturbation target ID")
    p.add_argument("--condition-key", default="guide_merged", help="obs column with perturbation condition name")
    p.add_argument("--control-value", default="ctrl", help="Value in condition-key that marks control cells")
    p.add_argument("--num-iterations", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--skip-prediction", action="store_true")
    p.add_argument("--output-dir", default="results/outputs/outputs_optimized")
    p.add_argument("--run-name", default="optimized")
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fold", type=int, default=0, help="scDFM-style random re-split fold. Uses split seed 42 + fold.")
    p.add_argument("--split-seed-base", type=int, default=42, help="Base seed for scDFM-style split folds.")
    p.add_argument("--test-condition-fraction", type=float, default=0.3)
    p.add_argument("--val-fraction", type=float, default=0.006)
    p.add_argument("--train-cell-fraction", type=float, default=0.3)
    p.add_argument("--test-cell-fraction", type=float, default=0.3)
    p.add_argument("--max-train-cells", type=int, default=0, help="Hard cap for training cells. 0 disables.")
    p.add_argument("--max-test-cells", type=int, default=0, help="Hard cap for test cells. 0 disables.")
    p.add_argument(
        "--x-graph-fusion-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable x graph fusion (gene2vec + GO graph).",
    )
    p.add_argument(
        "--use-cell-type-condition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use cell_type as model condition. Default False (K562 only).",
    )
    p.add_argument("--condition-combined-loss-weight", type=float, default=0.01)

    # Optimized parameters
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--decoder-dims", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--time-encoder-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--graph-dim", type=int, default=512)
    p.add_argument("--graph-dropout", type=float, default=0.2)
    p.add_argument("--graph-num-layers", type=int, default=2)
    p.add_argument("--graph-max-edges", type=int, default=50000)
    p.add_argument("--graph-top-k-attn", type=int, default=50)
    p.add_argument("--gradient-accumulation-steps", type=int, default=5)

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

    print("Loading dataset:", adata_path)
    adata = ad.read_h5ad(str(adata_path))
    print(f"Raw data: {adata.n_obs} cells x {adata.n_vars} genes")

    # Column mapping
    if args.target_key not in adata.obs:
        raise KeyError(f"target key '{args.target_key}' not found in adata.obs")
    if args.condition_key not in adata.obs:
        raise KeyError(f"condition key '{args.condition_key}' not found in adata.obs")

    adata.obs["target_gene"] = adata.obs[args.target_key].astype(str)
    adata.obs["condition"] = adata.obs[args.condition_key].astype(str)
    adata.obs[args.control_key] = (adata.obs["condition"] == args.control_value)
    print(f"Control cells: {adata.obs[args.control_key].sum()}")
    print(f"Perturbation cells: {(~adata.obs[args.control_key]).sum()}")
    print(f"Unique perturbation conditions: {adata.obs.loc[~adata.obs[args.control_key], 'condition'].nunique()}")

    # HVG filtering
    if "highly_variable" in adata.var:
        print(f"Filtering by highly variable genes. Original vars: {adata.n_vars}")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"After HVG filtering vars: {adata.n_vars}")
    else:
        print("Warning: highly_variable column not found in dataset!")

    print(f"Current Obs shape: {adata.n_obs}")
    print(f"Current Var shape: {adata.n_vars}")

    # ==================== Build gene2vec perturbation embeddings ====================
    g2v_path = "/home/zhangshibo24s/cell_flow/data_train/selected_gene2vec_27k.npy"
    g2v_genes_path = "/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt"

    g2v_array = np.load(g2v_path)
    with open(g2v_genes_path, "r") as f:
        g2v_genes = [line.strip() for line in f.readlines()]
    gene2vec_dict = {gene.upper(): vec.astype(np.float32) for gene, vec in zip(g2v_genes, g2v_array)}
    embedding_dim = g2v_array.shape[1]
    print(f"Loaded gene2vec: {len(g2v_genes)} genes, dim={embedding_dim}")

    unique_conditions = adata.obs["condition"].drop_duplicates().astype(str).tolist()
    condition_to_tokens, pert_emb, pert_symbol_to_ensembl = _build_norman_gene_tokens(
        conditions=unique_conditions,
        gene2vec_dict=gene2vec_dict,
        embedding_dim=embedding_dim,
        max_genes=2,
    )
    print(
        f"Built perturbation gene-token embeddings for {len(pert_emb)} tokens "
        f"across {len(condition_to_tokens)} conditions"
    )

    adata.obs["pert_gene_1"] = adata.obs["condition"].map(lambda c: condition_to_tokens[str(c)][0]).astype(str)
    adata.obs["pert_gene_2"] = adata.obs["condition"].map(lambda c: condition_to_tokens[str(c)][1]).astype(str)

    rep_key = "gene2vec_pert_gene_tokens"
    adata.uns[rep_key] = pert_emb
    adata.uns["norman_perturbation_symbol_to_ensembl"] = pert_symbol_to_ensembl
    adata.uns["norman_condition_to_gene_tokens"] = condition_to_tokens

    # Filter to cells whose condition was parsed successfully.
    valid_targets = set(condition_to_tokens.keys())
    valid_mask = adata.obs["condition"].isin(valid_targets)
    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        print(f"Filtering out {n_filtered} cells without parsed perturbation condition embeddings")
        adata = adata[valid_mask].copy()
    print(f"Cells after embedding filter: {adata.n_obs}")

    # Gene alignment
    selected_gene_ids_file = ROOT / "data_train" / "selected_genes_27k.txt"
    selected_gene2vec_file = ROOT / "data_train" / "selected_gene2vec_27k.npy"
    gene2go_graph_file = ROOT / "data_train" / "human_ens_gene2go_graph.csv"

    print("Mapping var_names to Ensembl IDs via mygene...")
    symbol_to_ensembl = build_symbol_to_ensembl([str(g) for g in adata.var_names])
    adata = align_adata_to_selected_ensembl(adata=adata, symbol_to_ensembl=symbol_to_ensembl)

    matched_ids_file, matched_gene2vec_file = build_matched_gene2vec(
        selected_gene_ids_file=selected_gene_ids_file,
        selected_gene2vec_file=selected_gene2vec_file,
        ordered_ids=[str(g).upper() for g in adata.var_names],
        save_dir=out_dir,
    )
    print(f"Aligned genes: {adata.n_vars}")

    # Perturbation covariates
    perturbation_covariates = {"gene_perturbation": ["pert_gene_1", "pert_gene_2"]}
    perturbation_reps = {"gene_perturbation": rep_key}

    if args.use_cell_type_condition:
        adata.obs["cell_type"] = "K562"
        perturbation_covariates["cell_type"] = ["cell_type"]
        ct_emb = {"K562": np.array([1.0], dtype=np.float32)}
        adata.uns["cell_type_embeddings"] = ct_emb
        perturbation_reps["cell_type"] = "cell_type_embeddings"

    # Ensure control key is boolean
    adata.obs[args.control_key] = adata.obs[args.control_key].astype(bool)

    print(f"Total cells before split: {adata.n_obs}")

    # ==================== scDFM-style Norman perturbation split ====================
    rng = np.random.default_rng(args.seed)
    control_mask = adata.obs[args.control_key].astype(bool)

    pert_conditions = (
        adata.obs.loc[~control_mask, "condition"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    train_conditions, test_conditions, split_info = build_scdfm_norman_additive_split(
        conditions=pert_conditions,
        fold=args.fold,
        test_fraction=args.test_condition_fraction,
        seed_base=args.split_seed_base,
    )

    is_test_condition = adata.obs["condition"].isin(test_conditions).to_numpy()
    is_train_condition = adata.obs["condition"].isin(train_conditions).to_numpy()

    train_mask = control_mask.to_numpy() | ((~control_mask.to_numpy()) & is_train_condition)
    test_mask = (~control_mask.to_numpy()) & is_test_condition

    adata_train_full = adata[train_mask].copy()
    adata_test_additive = adata[test_mask].copy()
    train_cells_before_subsample = adata_train_full.n_obs
    test_cells_before_subsample = adata_test_additive.n_obs

    # Subsample by condition
    adata_train_full = stratified_subsample_obs(
        adata_train_full,
        fraction=args.train_cell_fraction,
        rng=rng,
        group_key="condition",
    )
    adata_test_additive = stratified_subsample_obs(
        adata_test_additive,
        fraction=args.test_cell_fraction,
        rng=rng,
        group_key="condition",
    )

    train_cells_before_cap = adata_train_full.n_obs
    test_cells_before_cap = adata_test_additive.n_obs

    # Hard cap
    adata_train_full = stratified_cap_obs(
        adata_train_full,
        max_cells=args.max_train_cells,
        rng=rng,
        group_key="condition",
    )
    adata_test_additive = stratified_cap_obs(
        adata_test_additive,
        max_cells=args.max_test_cells,
        rng=rng,
        group_key="condition",
    )

    # Validation split from training set
    n_train_total = adata_train_full.n_obs
    val_indices = rng.choice(n_train_total, int(n_train_total * args.val_fraction), replace=False)
    val_mask_arr = np.zeros(n_train_total, dtype=bool)
    val_mask_arr[val_indices] = True

    adata_val = adata_train_full[val_mask_arr].copy()
    adata = adata_train_full[~val_mask_arr].copy()

    print("Norman 2019 scDFM-style perturbation split:")
    print(f"  Split method: additive, fold={args.fold}, split_seed={args.split_seed_base + args.fold}")
    print(f"  Perturbation conditions in Train: {len(train_conditions)}")
    print(f"  Perturbation conditions in Test : {len(test_conditions)}")
    print(f"  Train singles/doubles: {split_info['train_single_conditions_count']}/{split_info['train_double_conditions_count']}")
    print(f"  Test singles/doubles : {split_info['test_single_conditions_count']}/{split_info['test_double_conditions_count']}")
    print(f"  Test conditions preview: {sorted(test_conditions)[:10]}")
    print(f"  Training cells kept: {train_cells_before_cap}/{train_cells_before_subsample} ({args.train_cell_fraction:.2%}) before cap.")
    if args.max_train_cells > 0:
        print(f"  Training cells after cap: {adata_train_full.n_obs}/{args.max_train_cells}")
    print(f"  Test cells kept    : {test_cells_before_cap}/{test_cells_before_subsample} ({args.test_cell_fraction:.2%}) before cap.")
    if args.max_test_cells > 0:
        print(f"  Test cells after cap    : {adata_test_additive.n_obs}/{args.max_test_cells}")
    print(f"  Using {adata.n_obs} cells for training, {adata_val.n_obs} for validation.")
    print(f"  Zero-shot testing set contains {adata_test_additive.n_obs} cells.")
    write_json(
        out_dir / f"split_summary_{run_label}.json",
        {
            "test_condition_fraction": args.test_condition_fraction,
            "split_info": split_info,
            "perturbation_conditions_total": len(pert_conditions),
            "perturbation_conditions_train": len(train_conditions),
            "perturbation_conditions_test": len(test_conditions),
            "train_cells_before_subsample": int(train_cells_before_subsample),
            "test_cells_before_subsample": int(test_cells_before_subsample),
            "train_cells_before_cap": int(train_cells_before_cap),
            "test_cells_before_cap": int(test_cells_before_cap),
            "train_cell_fraction": args.train_cell_fraction,
            "test_cell_fraction": args.test_cell_fraction,
            "max_train_cells": int(args.max_train_cells),
            "max_test_cells": int(args.max_test_cells),
            "train_conditions": sorted(train_conditions),
            "test_conditions": sorted(test_conditions),
            "train_summary": summarize_adata_split(adata, args.control_key),
            "validation_summary": summarize_adata_split(adata_val, args.control_key),
            "test_summary": summarize_adata_split(adata_test_additive, args.control_key),
        },
    )
    # ==============================================================================

    # Ensure X is CSR (MyFlow's DataManager only handles CSR sparse matrices correctly)
    if sp.issparse(adata.X) and not isinstance(adata.X, sp.csr_matrix):
        print(f"Converting X from {type(adata.X).__name__} to csr_matrix")
        adata.X = sp.csr_matrix(adata.X)

    print("Initializing MyFlow (this may import jax/flax/ott)")
    cf = MyFlow(adata, solver=args.solver)
    print("Preparing data for training")
    split_covariates = ["cell_type"] if args.use_cell_type_condition else None
    cf.prepare_data(
        sample_rep=args.sample_rep,
        control_key=args.control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_reps,
        split_covariates=split_covariates,
    )
    print("Preparing optimized model architecture...")
    cf.prepare_model(
        seed=args.seed,
        condition_encoder_kwargs={
            "x_graph_fusion_kwargs": {
                "enabled": args.x_graph_fusion_enabled,
                "dim": args.graph_dim,
                "max_seq_len": int(adata.n_vars),
                "max_edges": args.graph_max_edges,
                "gene2vec_file": str(matched_gene2vec_file),
                "gene_ids_file": str(matched_ids_file),
                "gene2go_graph_file": str(gene2go_graph_file),
                "graph_dropout": args.graph_dropout,
                "num_layers": args.graph_num_layers,
                "top_k_attn": args.graph_top_k_attn,
            }
        },
        solver_kwargs={
            "condition_combined_loss_weight": args.condition_combined_loss_weight,
        },
        # Optimized architecture
        hidden_dims=tuple(args.hidden_dims),
        decoder_dims=tuple(args.decoder_dims),
        time_encoder_dims=tuple(args.time_encoder_dims),
        optimizer=optax.MultiSteps(optax.adam(5e-5), args.gradient_accumulation_steps),
    )
    print(f"Start training: iterations={args.num_iterations}, batch_size={args.batch_size}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Decoder dims: {args.decoder_dims}")
    print(f"  Graph dim: {args.graph_dim}")
    print(f"  Graph dropout: {args.graph_dropout}")
    print(f"  Graph layers: {args.graph_num_layers}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    cf.train(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        seed=args.seed,
        valid_freq=0,
    )
    print("Training completed. Saving model...", flush=True)
    if cf.trainer is not None and getattr(cf.trainer, "training_logs", None):
        logs = cf.trainer.training_logs
        pd.DataFrame({k: pd.Series(v) for k, v in logs.items()}).to_csv(
            out_dir / f"training_logs_{run_label}.csv",
            index_label="step",
        )

    model_out_path = out_dir / f"model_{run_label}"
    model_out_path.mkdir(parents=True, exist_ok=True)
    cf.save(str(model_out_path), file_prefix=None, overwrite=args.overwrite)
    print(f"Model saved to {model_out_path}", flush=True)

    if args.skip_prediction:
        print("Skipping prediction stage due to --skip-prediction flag.")
        return

    print("Starting prediction on the zero-shot additive tests...", flush=True)
    groups = adata_test_additive.obs.groupby("condition").groups

    all_X = []
    all_obs = []

    prediction_controls = []
    control_adata = adata_train_full[adata_train_full.obs[args.control_key].astype(bool)].copy()
    if control_adata.n_obs == 0:
        raise RuntimeError("No control cells found in training set for prediction.")
    prediction_controls.append(control_adata)

    for condition, idx in groups.items():
        sample_size = len(idx)
        if sample_size > control_adata.n_obs:
            sampled_idx = rng.choice(control_adata.n_obs, size=sample_size, replace=True)
        else:
            sampled_idx = rng.choice(control_adata.n_obs, size=sample_size, replace=False)
        sub_adata = control_adata[sampled_idx].copy()

        # Find the guide_identity for this condition
        guide_id = adata_test_additive.obs.loc[
            adata_test_additive.obs["condition"] == condition, "target_gene"
        ].iloc[0]
        gene_tokens = condition_to_tokens[str(condition)]

        covariate_data = pd.DataFrame({
            "target_gene": [guide_id],
            "condition": [condition],
            "pert_gene_1": [gene_tokens[0]],
            "pert_gene_2": [gene_tokens[1]],
            args.control_key: [False],
        })
        if args.use_cell_type_condition:
            covariate_data["cell_type"] = "K562"

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
        obs_df = pd.DataFrame({
            "perturbation": [condition] * arr.shape[0],
            "target_gene": [guide_id] * arr.shape[0],
        })
        all_obs.append(obs_df)
    print("Prediction finished")
    if not all_X:
        raise RuntimeError("No predictions were generated.")
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

    metrics_summary = {
        "run_label": run_label,
        "prediction_file": str(out_file),
        "success": False,
    }
    try:
        ctrl_eval = ad.concat(prediction_controls, join="outer") if len(prediction_controls) > 1 else prediction_controls[0]
        ctrl_mean = np.array(ctrl_eval.X.mean(axis=0)).flatten()
        real_mean = np.array(adata_test_additive.X.mean(axis=0)).flatten()
        ours_mean = np.array(adata_pred.X.mean(axis=0)).flatten()

        mse, mae, l2 = cal_metric(ours_mean, real_mean)
        pearson_del, pearson_del_top20, ds = cal_delta_metric(ctrl_mean, real_mean, ours_mean)

        print(f"Basic => MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
        print(f"Delta => Pearson delta: {pearson_del:.4f}, Pearson delta20: {pearson_del_top20:.4f}, DS: {ds:.4f}")
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
        for condition, idx in groups.items():
            real_mask = (adata_test_additive.obs["condition"] == condition).to_numpy()
            pred_mask = (adata_pred.obs["perturbation"] == condition).to_numpy()
            if real_mask.sum() == 0 or pred_mask.sum() == 0:
                continue

            real_group_mean = np.array(adata_test_additive[real_mask].X.mean(axis=0)).flatten()
            pred_group_mean = np.array(adata_pred[pred_mask].X.mean(axis=0)).flatten()
            ctrl_group_mean = ctrl_mean  # global control mean

            group_mse, group_mae, group_l2 = cal_metric(pred_group_mean, real_group_mean)
            group_pearson_delta, group_pearson_delta_top20, group_ds = cal_delta_metric(
                ctrl_group_mean, real_group_mean, pred_group_mean,
            )
            condition_metric_rows.append(
                {
                    "condition": str(condition),
                    "real_cells": int(real_mask.sum()),
                    "pred_cells": int(pred_mask.sum()),
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
            condition_metrics_df = condition_metrics_df.sort_values("condition")
        condition_metrics_df.to_csv(condition_metrics_file, index=False)
        print(f"Saved per-condition metrics: {condition_metrics_file}")
        metrics_summary["condition_metrics_file"] = str(condition_metrics_file)
        metrics_summary["condition_metrics_count"] = len(condition_metric_rows)

        print("\nCalculating per-condition DES & DE-Spearman...")
        ctrl_copy = ctrl_eval.copy()
        des_per_condition = []
        for condition, idx in groups.items():
            real_mask = (adata_test_additive.obs["condition"] == condition).to_numpy()
            pred_mask = (adata_pred.obs["perturbation"] == condition).to_numpy()
            if real_mask.sum() == 0 or pred_mask.sum() == 0:
                continue
            real_cond = adata_test_additive[real_mask].copy()
            pred_cond = adata_pred[pred_mask].copy()
            try:
                d_recall, d_acc, d_spearman = compute_des(ctrl_copy, real_cond, pred_cond)
                des_per_condition.append({
                    "condition": str(condition),
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

            # Merge DES into condition_metrics CSV
            if not condition_metrics_df.empty:
                des_merge = des_df.rename(columns={
                    "des_recall": "des_recall",
                    "des_accuracy": "des_accuracy",
                    "de_spearman": "de_spearman",
                })
                condition_metrics_df = condition_metrics_df.merge(
                    des_merge[["condition", "des_recall", "des_accuracy", "de_spearman"]],
                    on="condition", how="left",
                )
                condition_metrics_df.to_csv(condition_metrics_file, index=False)
                print(f"Updated per-condition metrics with DES: {condition_metrics_file}")
        else:
            print("No valid per-condition DES computed.")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        metrics_summary["error"] = str(e)
    write_json(out_dir / f"metrics_summary_{run_label}.json", metrics_summary)
    print("=" * 50)


if __name__ == "__main__":
    main()
