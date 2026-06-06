#!/usr/bin/env python3
"""
Train MyFlow on the Norman et al. 2019 combinatorial perturbation dataset
with a dedicated scDFM-style additive split.

Additive setting:
- Test set is a random subset of double perturbation conditions.
- All single perturbations remain in training.
- Keep all control cells in training.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_early_cli_option(name: str, default: str) -> str:
    prefix = f"{name}="
    for i, arg in enumerate(sys.argv[1:]):
        if arg == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


os.environ["CUDA_VISIBLE_DEVICES"] = _read_early_cli_option("--gpu-id", os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import anndata as ad
import jax.numpy as jnp
import pandas as pd
import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import scipy.stats
from datetime import datetime
import optax
from sklearn.metrics import r2_score
from myflow.model._myflow import MyFlow

sys.path.insert(0, str(ROOT / "comparison_methods" / "scripts"))
from split_utils import build_scdfm_norman_split
from eval_utils import evaluate_predictions

DEFAULT_SEED = 20240508


# ==================== Evaluation Metrics ====================

def identify_degs(ctrl_mean, target_mean):
    """Identify DEGs using robust z-score on delta magnitudes."""
    delta = target_mean - ctrl_mean
    median_delta = np.median(delta)
    mad = np.median(np.abs(delta - median_delta)) * 1.4826
    if mad < 1e-10:
        mad = np.std(delta)
    if mad < 1e-10:
        return np.array([], dtype=int)
    z_scores = np.abs(delta - median_delta) / (mad + 1e-10)
    return np.where(z_scores > 2.0)[0]


def cal_deg_metrics(ctrl_mean, real_mean, pred_mean, deg_indices):
    """Compute R², EV, PCC on DEG genes only (delta space)."""
    if len(deg_indices) == 0:
        return {"r2_deg": float("nan"), "ev_deg": float("nan"), "pcc_deg": float("nan")}
    delta_real = real_mean[deg_indices] - ctrl_mean[deg_indices]
    delta_pred = pred_mean[deg_indices] - ctrl_mean[deg_indices]
    r2 = r2_score(delta_real, delta_pred)
    residual = delta_real - delta_pred
    ev = 1.0 - np.var(residual) / (np.var(delta_real) + 1e-10)
    if len(delta_real) < 2:
        pcc = 0.0
    else:
        pcc, _ = scipy.stats.pearsonr(delta_real, delta_pred)
        if np.isnan(pcc):
            pcc = 0.0
    return {"r2_deg": float(r2), "ev_deg": float(ev), "pcc_deg": float(pcc)}


# ==============================================================================




def load_gene2vec_dict(path: Path) -> dict[str, np.ndarray]:
    raw = torch.load(str(path), map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Expected gene2vec dict at {path}, got {type(raw)!r}")
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        arr = arr.astype(np.float32, copy=False).reshape(-1)
        if arr.size:
            out[str(key).strip().upper()] = arr
    if not out:
        raise ValueError(f"No usable gene2vec entries found in {path}")
    dims = {v.shape[0] for v in out.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent gene2vec dimensions in {path}: {sorted(dims)[:5]}")
    return out


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def _build_norman_gene_tokens(
    conditions: list[str],
    gene2vec_dict: dict[str, np.ndarray],
    embedding_dim: int,
    max_genes: int = 2,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    """Build per-gene condition tokens for Norman combinatorial perturbations.

    Each condition is represented by up to two gene-symbol tokens.
    """
    unique_symbols = sorted({g.upper() for cond in conditions for g in _parse_condition_genes(cond)})

    token_to_vec: dict[str, np.ndarray] = {
        "ctrl": np.zeros(embedding_dim, dtype=np.float32),
    }
    symbol_to_token: dict[str, str] = {}
    missing_symbols: list[str] = []

    for symbol in unique_symbols:
        if symbol in gene2vec_dict:
            symbol_to_token[symbol] = symbol
            token_to_vec[symbol] = gene2vec_dict[symbol].astype(np.float32)
        else:
            token = f"missing::{symbol}"
            symbol_to_token[symbol] = token
            token_to_vec[token] = np.zeros(embedding_dim, dtype=np.float32)
            missing_symbols.append(symbol)

    if missing_symbols:
        preview = ", ".join(missing_symbols[:20])
        suffix = "..." if len(missing_symbols) > 20 else ""
        print(
            f"  Warning: {len(missing_symbols)} perturbation gene symbols are missing from gene2vec: "
            f"{preview}{suffix}. Using zero vectors for them."
        )

    condition_to_tokens: dict[str, list[str]] = {}
    for condition in conditions:
        symbols = sorted(g.upper() for g in _parse_condition_genes(condition))
        if len(symbols) > max_genes:
            print(f"  Warning: condition '{condition}' has >{max_genes} genes; truncating extras.")
            symbols = symbols[:max_genes]
        tokens = [symbol_to_token[s] for s in symbols]
        tokens.extend(["ctrl"] * (max_genes - len(tokens)))
        condition_to_tokens[str(condition)] = tokens

    return condition_to_tokens, token_to_vec


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
    p.add_argument("--num-iterations", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--skip-prediction", action="store_true")
    p.add_argument("--output-dir", default="results/outputs/outputs_myflow_norman_additive")
    p.add_argument("--run-name", default="myflow_norman_additive")
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--conditioning", choices=["film", "concatenation"], default="concatenation")
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
        "--use-cell-type-condition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use cell_type as model condition. Default False (K562 only).",
    )
    p.add_argument("--condition-combined-loss-weight", type=float, default=0.0)
    p.add_argument("--endpoint-mse-weight", type=float, default=1.0, help="Weight for direct endpoint MSE loss (no stop_gradient).")
    p.add_argument("--condition-mean-delta-weight", type=float, default=0.0, help="Weight for condition-level mean delta supervision.")
    p.add_argument("--high-delta-endpoint-weight", type=float, default=0.0, help="Extra endpoint/mean-loss weight on genes with large true condition delta.")
    p.add_argument("--high-delta-max-weight", type=float, default=4.0, help="Maximum per-gene multiplier used by high-delta endpoint/mean losses.")
    p.add_argument("--top-delta-loss-weight", type=float, default=0.0, help="Extra condition-mean delta loss on top true-response genes. Default 0 disables.")
    p.add_argument("--top-delta-endpoint-weight", type=float, default=0.0, help="Extra per-gene multiplier for endpoint/mean losses on top true-response genes. Default 0 disables.")
    p.add_argument("--top-delta-fraction", type=float, default=0.05, help="Fraction of genes treated as top-delta genes for top-delta losses.")
    p.add_argument("--top-delta-min-genes", type=int, default=20, help="Minimum number of top-delta genes per condition.")
    p.add_argument("--terminal-loss-time-power", type=float, default=2.0, help="Power for terminal-loss time gate t^p; larger values focus endpoint-style losses closer to t=1.")
    p.add_argument("--cosine-loss-weight", type=float, default=0.0, help="Weight for cosine similarity loss on delta direction.")
    p.add_argument("--flow-noise", type=float, default=0.0, help="Gaussian noise std in flow matching path.")
    p.add_argument("--snr-endpoint-weight", type=float, default=0.0, help="Weight for SNR-weighted endpoint MSE (per-gene signal-to-noise ratio, higher weight on DEGs). Default 0 disables.")
    p.add_argument("--cov-loss-weight", type=float, default=0.0, help="Weight for covariance-preserving loss (gene-gene covariance structure matching). Default 0 disables.")
    #1e-3
    p.add_argument("--learning-rate", type=float, default=5e-4, help="Base learning rate for Adam optimizer.")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--decoder-dims", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--time-encoder-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--condition-embedding-dim", type=int, default=256, help="Dimension of condition embedding.")
    p.add_argument("--cross-attn-layers", type=int, default=1, help="Number of cross-attention layers (default: 1).")
    p.add_argument("--gene-attn-dim", type=int, default=16, help="Dimension of gene attention embeddings (default: 16).")
    p.add_argument("--gene-self-attn-layers", type=int, default=0, help="Number of gene self-attention layers (default: 0).")
    p.add_argument("--cross-attn-heads", type=int, default=4, help="Number of cross-attention heads (default: 4).")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--match-every-n", type=int, default=20, help="Run OT Sinkhorn matching every N steps.")
    p.add_argument(
        "--gene2vec-dict",
        default=str(ROOT / "data_gab" / "gene2vec_dict.pt"),
        help="Symbol-keyed gene2vec dictionary for perturbation tokens.",
    )
    p.add_argument("--pert-gnn-enabled", action="store_true", help="Enable perturbation-side GNN prior.")
    p.add_argument("--pert-gnn-hidden-dim", type=int, default=16)
    p.add_argument("--pert-gnn-num-layers", type=int, default=2)
    p.add_argument("--pert-gnn-num-heads", type=int, default=4, help="Number of attention heads for enhanced GNN.")
    p.add_argument("--enhanced-pert-gnn", action="store_true", help="Use EnhancedPerturbationGNN (multi-head attention + virtual node).")
    p.add_argument("--delta-head-enabled", action="store_true", help="Enable direct delta prediction head (condition → per-gene delta).")
    p.add_argument("--delta-head-hidden", type=int, default=256, help="Hidden dim of the delta head MLP.")
    p.add_argument("--delta-head-weight", type=float, default=0.0, help="Weight for delta head MSE loss. Default 0 disables.")
    p.add_argument(
        "--pert-gnn-go-file",
        default=str(ROOT / "comparison_methods" / "TxPert-main" / "data" / "graphs" / "go" / "go_top_50.csv"),
    )
    p.add_argument(
        "--pert-gnn-ppi-file",
        default=str(ROOT / "comparison_methods" / "TxPert-main" / "data" / "graphs" / "string" / "v11.5.parquet"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    print(f"Using fixed random seed: {args.seed}")
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_name or timestamp
    out_dir = Path(args.output_dir)
    _run_id = os.environ.get("MYFLOW_RUN_ID")
    if _run_id:
        out_dir = out_dir.with_name(f"{out_dir.name}_{_run_id}")
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

    print("Loading Norman 2019 dataset:", adata_path)
    adata = ad.read_h5ad(str(adata_path))
    print(f"Raw data: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"obs columns: {list(adata.obs.columns)}")

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

    # ==================== Perturbation gene token embeddings (gene2vec) ====================
    gene2vec_dict = load_gene2vec_dict(Path(args.gene2vec_dict))
    embedding_dim = next(iter(gene2vec_dict.values())).shape[0]
    print(f"Loaded symbol gene2vec: {len(gene2vec_dict)} genes, dim={embedding_dim}")

    unique_conditions = adata.obs["condition"].drop_duplicates().astype(str).tolist()
    condition_to_tokens, pert_emb = _build_norman_gene_tokens(
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

    valid_targets = set(condition_to_tokens.keys())
    valid_mask = adata.obs["condition"].isin(valid_targets)
    if (~valid_mask).sum() > 0:
        print(f"Filtering out {(~valid_mask).sum()} cells without parsed perturbation condition embeddings")
        adata = adata[valid_mask].copy()
    print(f"Cells after embedding filter: {adata.n_obs}")

    perturbation_covariates = {"gene_perturbation": ["pert_gene_1", "pert_gene_2"]}
    perturbation_reps = {"gene_perturbation": rep_key}

    # Build gene_name → int index mapping for perturbation genes (used by DataManager to emit indices)
    _pert_genes = sorted(set(k for k in pert_emb if k != "ctrl" and not k.startswith("missing::")))
    _pert_gene_to_idx = {g: i for i, g in enumerate(_pert_genes)}
    adata.uns["perturb_gene_symbol_to_idx"] = _pert_gene_to_idx
    print(f"Perturbation gene index mapping: {len(_pert_gene_to_idx)} genes")

    # ==================== Build perturbation graph (GO + STRING, with caching) ====================
    def _build_perturbation_graph(pert_genes, go_file, ppi_file):
        """Build GO+STRING graph edges over perturbation genes. Returns (edge_src, edge_tgt, edge_w)."""
        pert_genes = sorted(set(pert_genes))
        if not pert_genes:
            return None, None, None

        gene_to_idx = {g: i for i, g in enumerate(pert_genes)}
        n_nodes = len(pert_genes)
        edge_src, edge_tgt, edge_w = [], [], []

        # GO edges
        go_path = Path(go_file)
        if go_path.exists():
            with open(go_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s, t = str(row.get("source", "")).upper(), str(row.get("target", "")).upper()
                    if s in gene_to_idx and t in gene_to_idx:
                        edge_src.append(gene_to_idx[s])
                        edge_tgt.append(gene_to_idx[t])
                        edge_w.append(float(row.get("importance", 1.0)))

        # STRING PPI supplement if GO edges are sparse
        if len(edge_src) < 100:
            ppi_path = Path(ppi_file)
            if ppi_path.exists():
                print("  GO edges < 100, loading STRING PPI supplement...")
                import pandas as _pd
                df = _pd.read_parquet(ppi_path)
                df["regulator"] = df["regulator"].astype(str).str.upper()
                df["target"] = df["target"].astype(str).str.upper()
                mask = df["regulator"].isin(gene_to_idx) & df["target"].isin(gene_to_idx)
                for _, row in df.loc[mask].iterrows():
                    edge_src.append(gene_to_idx[row["regulator"]])
                    edge_tgt.append(gene_to_idx[row["target"]])
                    edge_w.append(float(row.get("weight", 1.0)))

        if not edge_src:
            return None, None, None

        src_arr = np.array(edge_src, dtype=np.int32)
        tgt_arr = np.array(edge_tgt, dtype=np.int32)
        w_arr = np.array(edge_w, dtype=np.float32)
        deg = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(deg, tgt_arr, w_arr)
        w_norm = w_arr / (deg[tgt_arr] + 1e-8)
        return jnp.array(src_arr), jnp.array(tgt_arr), jnp.array(w_norm)

    _gnn_src, _gnn_tgt, _gnn_w = _build_perturbation_graph(
        _pert_genes, args.pert_gnn_go_file, args.pert_gnn_ppi_file,
    )

    perturbation_gnn_kwargs = {}
    if args.pert_gnn_enabled and _gnn_src is not None and _gnn_src.shape[0] > 0:
        perturbation_gnn_kwargs = {
            "enabled": True,
            "hidden_dim": args.pert_gnn_hidden_dim,
            "num_layers": args.pert_gnn_num_layers,
            "num_pert_genes": len(_pert_gene_to_idx),
            "edge_src": _gnn_src,
            "edge_tgt": _gnn_tgt,
            "edge_w": _gnn_w,
            "enhanced_gnn": args.enhanced_pert_gnn,
            "gnn_hidden_dim": args.pert_gnn_hidden_dim,
            "gnn_num_layers": args.pert_gnn_num_layers,
            "gnn_num_heads": args.pert_gnn_num_heads,
        }
        print(f"Perturbation GNN graph: {len(_pert_gene_to_idx)} nodes, {_gnn_src.shape[0]} edges")
    elif args.pert_gnn_enabled:
        print("WARNING: Perturbation GNN enabled but no edges found — disabling.")

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
    train_conditions, test_conditions, split_info = build_scdfm_norman_split(
        conditions=pert_conditions,
        split_method="additive",
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
    print("Preparing model (default architecture). This may take a few seconds")
    cf.prepare_model(
        seed=args.seed,
        condition_embedding_dim=args.condition_embedding_dim,
        hidden_dims=args.hidden_dims,
        decoder_dims=args.decoder_dims,
        time_encoder_dims=args.time_encoder_dims,
        cross_attn_layers=args.cross_attn_layers,
        gene_attn_dim=args.gene_attn_dim,
        gene_self_attn_layers=args.gene_self_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        optimizer=optax.MultiSteps(optax.adam(args.learning_rate), args.gradient_accumulation_steps),
        conditioning=args.conditioning,
        probability_path={"constant_noise": args.flow_noise},
        perturbation_gnn_kwargs=perturbation_gnn_kwargs,
        delta_head_enabled=args.delta_head_enabled,
        delta_head_hidden=args.delta_head_hidden,
        solver_kwargs={
            "condition_combined_loss_weight": args.condition_combined_loss_weight,
            "match_every_n": args.match_every_n,
            "endpoint_mse_weight": args.endpoint_mse_weight,
            "condition_mean_delta_weight": args.condition_mean_delta_weight,
            "high_delta_endpoint_weight": args.high_delta_endpoint_weight,
            "high_delta_max_weight": args.high_delta_max_weight,
            "top_delta_loss_weight": args.top_delta_loss_weight,
            "top_delta_endpoint_weight": args.top_delta_endpoint_weight,
            "top_delta_fraction": args.top_delta_fraction,
            "top_delta_min_genes": args.top_delta_min_genes,
            "terminal_loss_time_power": args.terminal_loss_time_power,
            "cosine_loss_weight": args.cosine_loss_weight,
            "delta_head_weight": args.delta_head_weight,
            "snr_endpoint_weight": args.snr_endpoint_weight,
            "cov_loss_weight": args.cov_loss_weight,
        },
    )
    print("===== Hyperparameter Summary =====")
    print(f"  solver: otfm")
    print(f"  seed: {args.seed}")
    print(f"  fold: {args.fold}")
    print(f"  num_iterations: {args.num_iterations}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  match_every_n: {args.match_every_n}")
    print(f"")
    print(f"  test_condition_fraction: {args.test_condition_fraction}")
    print(f"  val_fraction: {args.val_fraction}")
    print(f"  train_cell_fraction: {args.train_cell_fraction}")
    print(f"  test_cell_fraction: {args.test_cell_fraction}")
    print(f"")
    print(f"  conditioning: {args.conditioning}")
    print(f"  hidden_dims: {args.hidden_dims}")
    print(f"  decoder_dims: {args.decoder_dims}")
    print(f"  time_encoder_dims: {args.time_encoder_dims}")
    print(f"  cross_attn_layers: {args.cross_attn_layers}")
    print(f"  gene_attn_dim: {args.gene_attn_dim}")
    print(f"  gene_self_attn_layers: {args.gene_self_attn_layers}")
    print(f"  cross_attn_heads: {args.cross_attn_heads}")
    print(f"  condition_embedding_dim: {args.condition_embedding_dim}")
    print(f"  cond_output_dropout: 0.9 (default)")
    print(f"  condition_mode: deterministic (default)")
    print(f"  pooling: attention_token (default)")
    print(f"  layer_norm_before_concatenation: False (default)")
    print(f"  probability_path: constant_noise(0.0) (default)")
    print(f"")
    print(f"  endpoint_mse_weight: {args.endpoint_mse_weight}")
    print(f"  condition_combined_loss_weight: {args.condition_combined_loss_weight}")
    print(f"  condition_mean_delta_weight: {args.condition_mean_delta_weight}")
    print(f"  cosine_loss_weight: {args.cosine_loss_weight}")
    print(f"  high_delta_endpoint_weight: {args.high_delta_endpoint_weight}")
    print(f"  high_delta_max_weight: {args.high_delta_max_weight}")
    print(f"  top_delta_loss_weight: {args.top_delta_loss_weight}")
    print(f"  top_delta_endpoint_weight: {args.top_delta_endpoint_weight}")
    print(f"  top_delta_fraction: {args.top_delta_fraction}")
    print(f"  top_delta_min_genes: {args.top_delta_min_genes}")
    print(f"  snr_endpoint_weight: {args.snr_endpoint_weight}")
    print(f"  cov_loss_weight: {args.cov_loss_weight}")
    print(f"  terminal_loss_time_power: {args.terminal_loss_time_power}")
    print(f"")
    print(f"  pert_gnn_enabled: {args.pert_gnn_enabled}")
    if args.pert_gnn_enabled:
        print(f"  pert_gnn_hidden_dim: {args.pert_gnn_hidden_dim}")
        print(f"  pert_gnn_num_layers: {args.pert_gnn_num_layers}")
        print(f"  enhanced_pert_gnn: {args.enhanced_pert_gnn}")
        if args.enhanced_pert_gnn:
            print(f"  pert_gnn_num_heads: {args.pert_gnn_num_heads}")
    print(f"  use_cell_type_condition: {args.use_cell_type_condition}")
    print(f"===================================")
    print(f"Start training: iterations={args.num_iterations}, batch_size={args.batch_size}")
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
    X = np.clip(X, 0, None)  # clamp negative predicted expression to 0
    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=adata.var.copy())
    out_file = out_dir / f"predictions_{run_label}.h5ad"
    adata_pred.write_h5ad(out_file)
    print(f"Saved prediction file: {out_file}")

    print("\n" + "=" * 50)
    ctrl_eval = ad.concat(prediction_controls, join="outer") if len(prediction_controls) > 1 else prediction_controls[0]
    evaluate_predictions(
        ctrl_eval, adata_test_additive, adata_pred,
        str(out_dir / run_label),
        real_condition_key="condition",
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
