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
import json
import os
from pathlib import Path
import re
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
import mygene
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

ENSG_PATTERN = re.compile(r"^ENSG\d+$", re.IGNORECASE)
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


def _build_norman_gene_tokens(
    conditions: list[str],
    gene2vec_dict: dict[str, np.ndarray],
    embedding_dim: int,
    max_genes: int = 2,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[str, str]]:
    """Build per-gene condition tokens for Norman combinatorial perturbations.

    Each condition is represented by up to two gene tokens. Gene symbols are
    converted to Ensembl IDs before gene2vec lookup, because the bundled
    selected_gene2vec_27k dictionary is keyed by Ensembl IDs.
    """
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
    p.add_argument("--output-dir", default="results/outputs/outputs_norman_scdfm_additive")
    p.add_argument("--run-name", default="norman_scdfm_additive")
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
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
        "--use-cell-type-condition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use cell_type as model condition. Default False (K562 only).",
    )
    p.add_argument("--condition-combined-loss-weight", type=float, default=0.0)
    p.add_argument("--endpoint-mse-weight", type=float, default=1.0, help="Weight for direct endpoint MSE loss (no stop_gradient).")
    p.add_argument("--learning-rate", type=float, default=5e-4, help="Base learning rate for Adam optimizer.")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--decoder-dims", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--time-encoder-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--match-every-n", type=int, default=20, help="Run OT Sinkhorn matching every N steps.")
    p.add_argument("--go-response-top-k", type=int, default=20, help="Top GO-similar incoming neighbors per gene.")
    p.add_argument("--go-response-rho-dim", type=int, default=128, help="GO response prior feature dimension.")
    p.add_argument("--go-response-weight-power", type=float, default=1.5, help="Sharpen GO similarity weights before normalization.")
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
        hidden_dims=args.hidden_dims,
        decoder_dims=args.decoder_dims,
        time_encoder_dims=args.time_encoder_dims,
        optimizer=optax.MultiSteps(optax.adam(args.learning_rate), args.gradient_accumulation_steps),
        condition_encoder_kwargs={
            "go_response_kwargs": {
                "enabled": True,
                "dim": int(np.load(matched_gene2vec_file).shape[1]),
                "rho_dim": args.go_response_rho_dim,
                "max_seq_len": int(adata.n_vars),
                "gene2vec_file": str(matched_gene2vec_file),
                "gene_ids_file": str(matched_ids_file),
                "gene2go_graph_file": str(gene2go_graph_file),
                "top_k": args.go_response_top_k,
                "weight_power": args.go_response_weight_power,
            }
        },
        conditioning="film",
        solver_kwargs={
            "condition_combined_loss_weight": args.condition_combined_loss_weight,
            "match_every_n": args.match_every_n,
            "endpoint_mse_weight": args.endpoint_mse_weight,
        },
    )
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
    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=adata.var.copy())
    pred_dir = Path(args.output_dir) / f"predictions_{run_label}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_file = pred_dir / f"predictions_{run_label}.h5ad"
    adata_pred.write_h5ad(out_file)
    print(f"Saved prediction file: {out_file}")

    print("\n" + "=" * 50)
    ctrl_eval = ad.concat(prediction_controls, join="outer") if len(prediction_controls) > 1 else prediction_controls[0]
    evaluate_predictions(
        ctrl_eval, adata_test_additive, adata_pred,
        str(out_dir / f"myflow_norman_additive_{run_label}"),
        real_condition_key="condition",
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
