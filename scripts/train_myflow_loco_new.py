#!/usr/bin/env python3
"""
Notes:
- This script expects to be run from the repository root (`/home/zhangshibo24s/cell_project`).
- It adds the local `myflow/src` package to `sys.path` so you don't need to install the package.
- Provide `--perturbation-covariates` as a JSON string mapping covariate-name -> list-of-obs-columns.
- Optionally provide `--perturbation-reps` as a JSON dict mapping covariate-name -> key-in-adata.uns holding embeddings.
"""

import argparse
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import random
import sys

# Set GPU/JAX environment before importing torch/jax/myflow.
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


# Ensure only one GPU is visible before importing torch/jax/myflow.
os.environ["CUDA_VISIBLE_DEVICES"] = _read_early_cli_option("--gpu-id", os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
# Avoid large up-front JAX memory preallocation that can look like multi-GPU usage.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import anndata as ad
import pandas as pd
import torch
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats
from datetime import datetime
import optax
from myflow.model._myflow import MyFlow
from myflow.training import Metrics

DEFAULT_SEED = 20240508

# ==================== Evaluation Metrics from cal_score.py ====================
def cal_metric(pred_mean, real_mean):
    mse = mean_squared_error(real_mean, pred_mean)
    mae = mean_absolute_error(real_mean, pred_mean)
    l2 = np.linalg.norm(real_mean - pred_mean)
    return mse, mae, l2

def cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=20, ds_top_k=None, sign_eps=1e-8):
    delta_real = real_mean - ctrl_mean
    delta_pred = pred_mean - ctrl_mean
    pearson_delta, _ = scipy.stats.pearsonr(delta_real, delta_pred)

    # Fold-change (relative delta): δ / ctrl, for high-expr vs low-expr gene fairness
    fc_real = delta_real / (ctrl_mean + 1e-8)
    fc_pred = delta_pred / (ctrl_mean + 1e-8)
    pearson_delta_hat, _ = scipy.stats.pearsonr(fc_real, fc_pred)

    top_n_idx = np.argsort(np.abs(delta_real))[-top_k:]
    if len(top_n_idx) > 1:
        pearson_delta_top_k, _ = scipy.stats.pearsonr(delta_real[top_n_idx], delta_pred[top_n_idx])
        pearson_delta_hat_top_k, _ = scipy.stats.pearsonr(fc_real[top_n_idx], fc_pred[top_n_idx])
    else:
        pearson_delta_top_k = 0.0
        pearson_delta_hat_top_k = 0.0

    # DS is all-gene sign agreement by default.  A positive ds_top_k is kept
    # only for backward-compatible ad-hoc analyses.
    if ds_top_k is None or ds_top_k <= 0:
        ds_idx = np.arange(delta_real.shape[0])
    else:
        ds_idx = np.argsort(np.abs(delta_real))[-ds_top_k:]
    sign_real = np.where(np.abs(delta_real[ds_idx]) > sign_eps, np.sign(delta_real[ds_idx]), 0)
    sign_pred = np.where(np.abs(delta_pred[ds_idx]) > sign_eps, np.sign(delta_pred[ds_idx]), 0)
    ds_score = np.mean([1 if r == p else 0 for r, p in zip(sign_real, sign_pred)])
    return pearson_delta, pearson_delta_top_k, ds_score, pearson_delta_hat, pearson_delta_hat_top_k

def _align_pred_var_names(ctrl, target, pred):
    """Keep prediction gene IDs in the same namespace as ctrl/target."""
    ctrl_names = [str(v) for v in ctrl.var_names]
    target_names = [str(v) for v in target.var_names]
    pred_names = [str(v) for v in pred.var_names]

    reference = set(ctrl_names) & set(target_names)
    current_overlap = len(reference & set(pred_names))

    symbol_overlap = -1
    if "gene_symbol" in pred.var.columns:
        symbols = [str(v) for v in pred.var["gene_symbol"].values]
        symbol_overlap = len(reference & set(symbols))

    if symbol_overlap > current_overlap:
        pred.var_names = [str(v) for v in pred.var["gene_symbol"].values]
        pred.var.index = pred.var_names

    common = [g for g in ctrl.var_names if g in target.var_names and g in pred.var_names]
    if not common:
        raise ValueError("No common genes between ctrl, target, and prediction for DEG evaluation.")
    return ctrl[:, common].copy(), target[:, common].copy(), pred[:, common].copy()


def identify_degs(ctrl_mean, target_mean):
    """Identify DEGs by comparing ctrl vs target distributions (robust z-score)."""
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


def cal_deg_overlap_metrics(ctrl_mean, real_mean, pred_mean, threshold=2.0):
    """Compute DEG overlap (precision/recall/F1/Jaccard) between pred-DEGs and real-DEGs."""
    delta_real = real_mean - ctrl_mean
    delta_pred = pred_mean - ctrl_mean

    def _identify(delta):
        median_d = np.median(delta)
        mad = np.median(np.abs(delta - median_d)) * 1.4826
        if mad < 1e-10:
            mad = np.std(delta)
        if mad < 1e-10:
            return set()
        z = np.abs(delta - median_d) / (mad + 1e-10)
        return set(np.where(z > threshold)[0])

    real_de = _identify(delta_real)
    pred_de = _identify(delta_pred)
    overlap = len(real_de & pred_de)
    n_real = len(real_de)
    n_pred = len(pred_de)
    union = len(real_de | pred_de)

    precision = overlap / n_pred if n_pred > 0 else 0.0
    recall = overlap / n_real if n_real > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = overlap / union if union > 0 else 0.0

    return {
        "deg_precision": float(precision),
        "deg_recall": float(recall),
        "deg_f1": float(f1),
        "deg_jaccard": float(jaccard),
        "n_real_degs": int(n_real),
        "n_pred_degs": int(n_pred),
        "n_overlap_degs": int(overlap),
    }


def compute_deg_metrics_per_condition(ctrl_adata, real_adata, pred_adata,
                                      real_condition_key="target_gene",
                                      pred_condition_key="perturbation"):
    """Compute R², EV, PCC on DEG genes per condition, plus DEG overlap."""
    real_conditions = real_adata.obs[real_condition_key].unique()
    results = []
    for cond in real_conditions:
        real_mask = real_adata.obs[real_condition_key] == cond
        pred_mask = pred_adata.obs[pred_condition_key] == cond
        if real_mask.sum() == 0 or pred_mask.sum() == 0:
            continue
        real_cond = real_adata[real_mask].copy()
        pred_cond = pred_adata[pred_mask].copy()
        try:
            ctrl_aligned, real_aligned, pred_aligned = _align_pred_var_names(
                ctrl_adata.copy(), real_cond, pred_cond
            )
            ctrl_mean = np.array(ctrl_aligned.X.mean(axis=0)).flatten()
            real_mean = np.array(real_aligned.X.mean(axis=0)).flatten()
            pred_mean = np.array(pred_aligned.X.mean(axis=0)).flatten()
            deg_idx = identify_degs(ctrl_mean, real_mean)
            deg_metrics = cal_deg_metrics(ctrl_mean, real_mean, pred_mean, deg_idx)
            deg_overlap = cal_deg_overlap_metrics(ctrl_mean, real_mean, pred_mean)

            # Per-condition delta metrics (Pearson Δ, Δ20, Δ50, Δ100, Δ1000, DS)
            cond_pearson_d, cond_pearson_d20, cond_ds, cond_pearson_dh, cond_pearson_dh20 = cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=20)
            _, cond_pearson_d50, _, _, cond_pearson_dh50 = cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=50)
            _, cond_pearson_d100, _, _, cond_pearson_dh100 = cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=100)
            _, cond_pearson_d1000, _, _, cond_pearson_dh1000 = cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=1000)

            # DE Spearman: rank correlation of logFC on real DE genes
            de_spearman = float("nan")
            try:
                combined_real = ctrl_aligned.concatenate(
                    real_aligned, batch_key="condition", batch_categories=["ctrl", "target"]
                )
                sc.tl.rank_genes_groups(
                    combined_real, groupby="condition", reference="ctrl", method="t-test"
                )
                real_de_genes = np.array(combined_real.uns["rank_genes_groups"]["names"]["target"])
                real_de_pvals = np.array(combined_real.uns["rank_genes_groups"]["pvals_adj"]["target"])
                sig_mask = real_de_pvals < 0.05
                real_sig_genes = set(real_de_genes[sig_mask])

                if len(real_sig_genes) > 1:
                    # Manual pred logFC: mean(pred) - mean(ctrl).
                    # scanpy's rank_genes_groups gives wrong logFC for predicted data
                    # because near-zero variance in predictions inflates the t-test.
                    pred_logfc_arr = pred_mean - ctrl_mean
                    pred_logfc_map = dict(zip(
                        [str(v) for v in pred_aligned.var_names], pred_logfc_arr
                    ))

                    real_logfc_map = dict(zip(
                        real_de_genes,
                        np.array(combined_real.uns["rank_genes_groups"]["logfoldchanges"]["target"]),
                    ))
                    matched_real, matched_pred = [], []
                    for g in real_sig_genes:
                        if g in pred_logfc_map and g in real_logfc_map:
                            matched_real.append(real_logfc_map[g])
                            matched_pred.append(pred_logfc_map[g])
                    if len(matched_real) > 1:
                        de_spearman, _ = scipy.stats.spearmanr(matched_real, matched_pred)
                        if np.isnan(de_spearman):
                            de_spearman = 0.0
            except Exception:
                pass

            results.append({
                "condition": str(cond),
                "n_degs": int(len(deg_idx)),
                **deg_metrics,
                **deg_overlap,
                "de_spearman": de_spearman,
                "condition_ds": float(cond_ds),
                "condition_pearson_delta": float(cond_pearson_d) if not np.isnan(cond_pearson_d) else 0.0,
                "condition_pearson_delta_hat": float(cond_pearson_dh) if not np.isnan(cond_pearson_dh) else 0.0,
                "condition_pearson_delta_top20": float(cond_pearson_d20) if not np.isnan(cond_pearson_d20) else 0.0,
                "condition_pearson_delta_hat_top20": float(cond_pearson_dh20) if not np.isnan(cond_pearson_dh20) else 0.0,
                "condition_pearson_delta_top50": float(cond_pearson_d50) if not np.isnan(cond_pearson_d50) else 0.0,
                "condition_pearson_delta_hat_top50": float(cond_pearson_dh50) if not np.isnan(cond_pearson_dh50) else 0.0,
                "condition_pearson_delta_top100": float(cond_pearson_d100) if not np.isnan(cond_pearson_d100) else 0.0,
                "condition_pearson_delta_hat_top100": float(cond_pearson_dh100) if not np.isnan(cond_pearson_dh100) else 0.0,
                "condition_pearson_delta_top1000": float(cond_pearson_d1000) if not np.isnan(cond_pearson_d1000) else 0.0,
                "condition_pearson_delta_hat_top1000": float(cond_pearson_dh1000) if not np.isnan(cond_pearson_dh1000) else 0.0,
            })
        except Exception:
            continue
    return results
# ==============================================================================

def _resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No {label} found. Tried: {joined}")


def load_gene2vec_dict(path: Path) -> dict[str, np.ndarray]:
    raw = torch.load(str(path), map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Expected gene2vec dict at {path}, got {type(raw)!r}")
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        arr = arr.astype(np.float32, copy=False).reshape(-1)
        if arr.size:
            out[str(key).strip()] = arr
    if not out:
        raise ValueError(f"No usable gene2vec entries found in {path}")
    dims = {v.shape[0] for v in out.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent gene2vec dimensions in {path}: {sorted(dims)[:5]}")
    return out


def build_matched_gene2vec_from_dict(
    gene2vec: dict[str, np.ndarray],
    ordered_genes: list[str],
    save_dir: Path,
    cache_dir: Path | None = None,
    cache_prefix: str = "matched_symbol",
    label: str = "expression genes",
) -> tuple[Path, Path]:
    ordered = [str(g).strip() for g in ordered_genes]
    digest = hashlib.sha1("\n".join(ordered).encode("utf-8")).hexdigest()[:16]
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_ids = cache_dir / f"{cache_prefix}_gene_ids_{digest}.txt"
        cached_vec = cache_dir / f"{cache_prefix}_gene2vec_{digest}.npy"
        if cached_ids.exists() and cached_vec.exists():
            print(f"Using cached matched symbol gene2vec: {cached_vec}")
            return cached_ids, cached_vec

    dim = next(iter(gene2vec.values())).shape[0]
    lookup = {k.upper(): v for k, v in gene2vec.items()}
    vectors = [lookup.get(g.upper(), np.zeros(dim, dtype=np.float32)) for g in ordered]
    matched_vec = np.stack(vectors).astype(np.float32, copy=False)

    gene_ids_out = (
        cache_dir / f"{cache_prefix}_gene_ids_{digest}.txt"
        if cache_dir is not None
        else save_dir / "symbol_gene_ids_matched.txt"
    )
    gene2vec_out = (
        cache_dir / f"{cache_prefix}_gene2vec_{digest}.npy"
        if cache_dir is not None
        else save_dir / "symbol_gene2vec_matched.npy"
    )
    gene_ids_out.parent.mkdir(parents=True, exist_ok=True)
    with open(gene_ids_out, "w", encoding="utf-8") as f:
        for gene in ordered:
            f.write(f"{gene}\n")
    np.save(gene2vec_out, matched_vec)
    missing = sum(1 for gene in ordered if gene.upper() not in lookup)
    if missing:
        print(f"Warning: {missing}/{len(ordered)} {label} missing from gene2vec dict; using zero vectors.")
    return gene_ids_out, gene2vec_out


def build_symbol_go_graph_from_edge_file(
    source_graph_file: Path,
    genes: list[str],
    save_dir: Path,
    cache_dir: Path | None = None,
    cache_prefix: str = "replogle_symbol_txpert",
) -> Path:
    ordered = [str(g).strip().upper() for g in genes]
    gene_set = set(ordered)
    with open(source_graph_file, "rb") as f:
        source_digest = hashlib.sha1(f.read()).hexdigest()[:16]
    digest = hashlib.sha1(
        ("\n".join(ordered) + f"\n{source_graph_file}\n{source_digest}").encode("utf-8")
    ).hexdigest()[:16]
    out_file = (
        cache_dir / f"{cache_prefix}_go_graph_{digest}.csv"
        if cache_dir is not None
        else save_dir / f"{cache_prefix}_go_graph.csv"
    )
    if out_file.exists():
        print(f"Using cached TxPert GO graph: {out_file}")
        return out_file

    n_edges = 0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(source_graph_file, "r", encoding="utf-8") as src_f, open(
        out_file, "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(src_f)
        writer = csv.DictWriter(out_f, fieldnames=["source", "target", "importance"])
        writer.writeheader()
        for row in reader:
            src = str(row.get("source", "")).strip().upper()
            tgt = str(row.get("target", "")).strip().upper()
            if src in gene_set and tgt in gene_set:
                writer.writerow(
                    {
                        "source": src,
                        "target": tgt,
                        "importance": float(row.get("importance", 1.0)),
                    }
                )
                n_edges += 1
    print(f"Wrote TxPert GO graph subset: {out_file} ({n_edges} edges, {len(gene_set)} genes)")
    return out_file


def build_go_edge_cache(
    gene_ids_file: Path,
    gene2go_graph_file: Path,
    max_seq_len: int,
    top_k: int,
    weight_power: float,
    cache_dir: Path,
    cache_prefix: str,
) -> Path:
    with open(gene_ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip().upper() for line in f if line.strip()]
    ids = ids[:max_seq_len]
    ids_digest = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:16]
    with open(gene2go_graph_file, "rb") as f:
        graph_digest = hashlib.sha1(f.read()).hexdigest()[:16]
    power_tag = str(weight_power).replace(".", "p")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"{cache_prefix}_go_edges_n{max_seq_len}_top{top_k}_pow{power_tag}_{ids_digest}_{graph_digest}.npz"
    )
    if cache_file.exists():
        print(f"Using cached GO edge index: {cache_file}")
        return cache_file

    id_to_idx = {gid: i for i, gid in enumerate(ids)}
    per_target: dict[int, list[tuple[float, int]]] = {}
    keep_k = max(int(top_k), 0) + 1
    with open(gene2go_graph_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = str(row.get("source", "")).upper()
            tgt = str(row.get("target", "")).upper()
            if src not in id_to_idx or tgt not in id_to_idx:
                continue
            weight = float(row.get("importance", 1.0))
            if weight_power != 1.0:
                weight = weight ** weight_power
            heap = per_target.setdefault(id_to_idx[tgt], [])
            item = (weight, id_to_idx[src])
            if len(heap) < keep_k:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    edge_src: list[int] = []
    edge_tgt: list[int] = []
    edge_w: list[float] = []
    for tgt_idx, heap in per_target.items():
        for weight, src_idx in heap:
            edge_src.append(src_idx)
            edge_tgt.append(tgt_idx)
            edge_w.append(weight)

    src_arr = np.asarray(edge_src, dtype=np.int32)
    tgt_arr = np.asarray(edge_tgt, dtype=np.int32)
    w_arr = np.asarray(edge_w, dtype=np.float32)
    if src_arr.size:
        deg = np.zeros((max_seq_len,), dtype=np.float32)
        np.add.at(deg, tgt_arr, w_arr)
        w_arr = w_arr / (deg[tgt_arr] + 1e-8)
    np.savez_compressed(cache_file, edge_src=src_arr, edge_tgt=tgt_arr, edge_w=w_arr)
    with open(cache_file.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "gene_ids_file": str(gene_ids_file),
                "gene2go_graph_file": str(gene2go_graph_file),
                "max_seq_len": int(max_seq_len),
                "top_k": int(top_k),
                "weight_power": float(weight_power),
                "edge_count": int(src_arr.size),
                "gene_ids_sha1": ids_digest,
                "graph_sha1": graph_digest,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote GO edge cache: {cache_file} ({src_arr.size} edges)")
    return cache_file


def build_ppi_edge_cache(
    gene_ids_file: Path,
    ppi_file: Path,
    perturb_genes: set[str],
    max_seq_len: int,
    cache_dir: Path,
    cache_prefix: str,
) -> Path:
    """Build edge cache for perturbation-level PPI graph (STRING)."""
    import pandas as pd

    perturb_genes = {str(g).strip().upper() for g in perturb_genes if str(g).strip()}
    with open(gene_ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip().upper() for line in f if line.strip()]
    ids = ids[:max_seq_len]
    ids_digest = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:16]
    perturb_digest = hashlib.sha1("\n".join(sorted(perturb_genes)).encode("utf-8")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_prefix}_ppi_edges_n{max_seq_len}_{ids_digest}_{perturb_digest}.npz"
    if cache_file.exists():
        print(f"Using cached PPI edge index: {cache_file}")
        return cache_file

    id_to_idx = {gid: i for i, gid in enumerate(ids)}
    perturb_indices = {id_to_idx[g] for g in perturb_genes if g in id_to_idx}

    df = pd.read_parquet(ppi_file, columns=["regulator", "target", "weight"])
    df["regulator"] = df["regulator"].astype(str).str.upper()
    df["target"] = df["target"].astype(str).str.upper()
    perturb_symbols = {g for g in perturb_genes if g in id_to_idx}
    df = df[df["regulator"].isin(perturb_symbols) & df["target"].isin(perturb_symbols)]
    if df.empty:
        src_arr = np.zeros((0,), dtype=np.int32)
        tgt_arr = np.zeros((0,), dtype=np.int32)
        w_arr = np.zeros((0,), dtype=np.float32)
    else:
        src_arr = df["regulator"].map(id_to_idx).to_numpy(dtype=np.int32)
        tgt_arr = df["target"].map(id_to_idx).to_numpy(dtype=np.int32)
        w_arr = df["weight"].astype(np.float32).to_numpy()
    if src_arr.size:
        deg = np.zeros((max_seq_len,), dtype=np.float32)
        np.add.at(deg, tgt_arr, w_arr)
        w_arr = w_arr / (deg[tgt_arr] + 1e-8)
    np.savez_compressed(cache_file, edge_src=src_arr, edge_tgt=tgt_arr, edge_w=w_arr)
    with open(cache_file.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "gene_ids_file": str(gene_ids_file),
                "ppi_file": str(ppi_file),
                "max_seq_len": int(max_seq_len),
                "edge_count": int(src_arr.size),
                "perturbation_genes_requested": int(len(perturb_genes)),
                "perturbation_genes_in_combined_table": int(len(perturb_indices)),
                "gene_ids_sha1": ids_digest,
                "perturbation_genes_sha1": perturb_digest,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote PPI edge cache: {cache_file} ({src_arr.size} edges, {len(perturb_indices)} perturbation genes)")
    return cache_file


def build_unified_edge_cache(
    gene_ids_file: Path,
    gene2go_graph_file: Path,
    ppi_file: Path,
    perturb_genes: set[str],
    max_seq_len: int,
    top_k: int,
    weight_power: float,
    ppi_weight_scale: float,
    cache_dir: Path,
    cache_prefix: str,
) -> Path:
    """Build unified edge cache merging GO + PPI edges into one graph."""
    import pandas as pd

    with open(gene_ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip().upper() for line in f if line.strip()]
    ids = ids[:max_seq_len]
    ids_digest = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:16]
    perturb_digest = hashlib.sha1("\n".join(sorted(perturb_genes)).encode("utf-8")).hexdigest()[:16]
    power_tag = str(weight_power).replace(".", "p")
    ppi_tag = str(ppi_weight_scale).replace(".", "p")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_prefix}_unified_edges_n{max_seq_len}_top{top_k}_pow{power_tag}_ppi{ppi_tag}_{ids_digest}_{perturb_digest}.npz"
    if cache_file.exists():
        print(f"Using cached unified edge index: {cache_file}")
        return cache_file

    id_to_idx = {gid: i for i, gid in enumerate(ids)}
    perturb_indices = {id_to_idx[g] for g in perturb_genes if g in id_to_idx}

    # Collect all edges as (src, tgt, weight) tuples
    all_edges: list[tuple[int, int, float]] = []

    # 1. Load GO edges (same logic as build_go_edge_cache)
    per_target: dict[int, list[tuple[float, int]]] = {}
    keep_k = max(int(top_k), 0) + 1
    with open(gene2go_graph_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = str(row.get("source", "")).upper()
            tgt = str(row.get("target", "")).upper()
            if src not in id_to_idx or tgt not in id_to_idx:
                continue
            weight = float(row.get("importance", 1.0))
            if weight_power != 1.0:
                weight = weight ** weight_power
            heap = per_target.setdefault(id_to_idx[tgt], [])
            item = (weight, id_to_idx[src])
            if len(heap) < keep_k:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    go_count = 0
    for tgt_idx, heap in per_target.items():
        for weight, src_idx in heap:
            all_edges.append((src_idx, tgt_idx, weight))
            go_count += 1

    # 2. Load PPI edges (STRING parquet, perturbation-only subgraph)
    df = pd.read_parquet(ppi_file)
    ppi_count = 0
    for _, row in df.iterrows():
        src_gene = str(row.get("regulator", "")).upper()
        tgt_gene = str(row.get("target", "")).upper()
        if src_gene not in id_to_idx or tgt_gene not in id_to_idx:
            continue
        src_idx = id_to_idx[src_gene]
        tgt_idx = id_to_idx[tgt_gene]
        if src_idx not in perturb_indices or tgt_idx not in perturb_indices:
            continue
        weight = float(row.get("weight", 1.0)) * ppi_weight_scale
        all_edges.append((src_idx, tgt_idx, weight))
        ppi_count += 1

    # Build arrays and normalize by target degree
    edge_src = np.asarray([e[0] for e in all_edges], dtype=np.int32)
    edge_tgt = np.asarray([e[1] for e in all_edges], dtype=np.int32)
    edge_w = np.asarray([e[2] for e in all_edges], dtype=np.float32)
    if edge_src.size:
        deg = np.zeros((max_seq_len,), dtype=np.float32)
        np.add.at(deg, edge_tgt, edge_w)
        edge_w = edge_w / (deg[edge_tgt] + 1e-8)

    np.savez_compressed(cache_file, edge_src=edge_src, edge_tgt=edge_tgt, edge_w=edge_w)
    print(f"Wrote unified edge cache: {cache_file} ({edge_src.size} edges: {go_count} GO + {ppi_count} PPI)")
    return cache_file


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dense_X(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def mean_expr(adata: ad.AnnData) -> np.ndarray:
    return np.asarray(adata.X.mean(axis=0)).ravel().astype(np.float32)


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
        default="/home/zhangshibo24s/cell_flow/data_gab/replogle_gab_merged_hvg.h5ad",
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
    p.add_argument("--num-iterations", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument(
        "--predict-n-cells",
        type=int,
        default=64,
        help=(
            "Fixed number of holdout-cell-line control cells to generate per perturbation. "
            "Using a fixed value avoids repeated JAX recompilation for many condition-specific sample sizes. "
            "Set <=0 to match each condition's real test cell count."
        ),
    )
    p.add_argument(
        "--prediction-max-allowed",
        type=float,
        default=8.0,
        help="Fail prediction if any raw generated expression exceeds this value.",
    )
    p.add_argument(
        "--prediction-frac-gt-max-allowed",
        type=float,
        default=0.0,
        help="Fail prediction if this fraction of raw generated values exceed --prediction-max-allowed.",
    )
    p.add_argument("--skip-prediction", action="store_true")
    # p.add_argument("--valid-freq", type=int, default=500)
    p.add_argument("--output-dir", default="results/outputs/outputs")
    p.add_argument("--run-name", default=None, help="Optional run name used in saved model/prediction/log filenames.")
    p.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"), help="Visible GPU id for this run.")
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--conditioning", choices=["film", "concatenation"], default="concatenation")
    p.add_argument("--pert-gnn-enabled", action="store_true", help="Enable perturbation-side GNN prior.")
    p.add_argument("--pert-gnn-hidden-dim", type=int, default=16)
    p.add_argument("--pert-gnn-num-layers", type=int, default=2)
    p.add_argument("--pert-gnn-num-heads", type=int, default=4, help="Number of attention heads for enhanced GNN.")
    p.add_argument("--enhanced-pert-gnn", action="store_true", help="Use EnhancedPerturbationGNN (multi-head attention + virtual node) instead of basic GCN.")
    p.add_argument("--no-gene-mask", action="store_true", help="Disable the per-gene sigmoid mask on velocity output.")
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
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--holdout-cell-line", default="hepg2", help="Cell line to hold out.")
    p.add_argument(
        "--train-cell-fraction",
        type=float,
        default=1.0,
        help="Fraction of training cells to keep after LOCO split, stratified by target_gene.",
    )
    p.add_argument(
        "--test-cell-fraction",
        type=float,
        default=1.0,
        help="Fraction of heldout-cell-line test cells to keep after LOCO split, stratified by target_gene.",
    )
    p.add_argument(
        "--n-train-perts",
        type=int,
        default=28,
        help="Number of holdout-cell-line perturbations to include in training. Overrides --train-pert-fraction.",
    )
    p.add_argument(
        "--n-test-perts",
        type=int,
        default=40,
        help="Number of holdout-cell-line perturbations to use for testing.",
    )
    p.add_argument(
        "--use-cell-type-condition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use cell_type as an explicit model condition. Use --no-use-cell-type-condition for strict baseline.",
    )
    p.add_argument(
        "--use-cell-type-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split control-target sampling by cell_type. Use --no-use-cell-type-split for strict baseline.",
    )
    p.add_argument("--condition-combined-loss-weight", type=float, default=0.003)
    p.add_argument(
        "--condition-combined-sinkhorn-weight",
        type=float,
        default=0.0,
        help="Sinkhorn component in the optional combined terminal distribution loss. Default 0 skips Sinkhorn entirely.",
    )
    p.add_argument(
        "--condition-combined-energy-weight",
        type=float,
        default=1.0,
        help="Energy-distance component in the optional combined terminal distribution loss.",
    )
    p.add_argument("--condition-combined-epsilon", type=float, default=1e-2)
    p.add_argument("--endpoint-mse-weight", type=float, default=0.1)
    p.add_argument("--condition-mean-delta-weight", type=float, default=0.0, help="Weight for condition-level mean delta supervision.")
    p.add_argument("--high-delta-endpoint-weight", type=float, default=0.0, help="Extra endpoint/mean-loss weight on genes with large true condition delta.")
    p.add_argument("--high-delta-max-weight", type=float, default=4.0, help="Maximum per-gene multiplier used by high-delta endpoint/mean losses.")
    p.add_argument("--top-delta-loss-weight", type=float, default=0.0, help="Extra condition-mean delta loss on top true-response genes. Default 0 disables.")
    p.add_argument("--top-delta-endpoint-weight", type=float, default=0.0, help="Extra per-gene multiplier for endpoint/mean losses on top true-response genes. Default 0 disables.")
    p.add_argument("--top-delta-fraction", type=float, default=0.05, help="Fraction of genes treated as top-delta genes for top-delta losses.")
    p.add_argument("--top-delta-min-genes", type=int, default=20, help="Minimum number of top-delta genes per condition.")
    p.add_argument("--terminal-loss-time-power", type=float, default=2.0, help="Power for terminal-loss time gate t^p; larger values focus endpoint-style losses closer to t=1.")
    p.add_argument("--cosine-loss-weight", type=float, default=0.1, help="Weight for cosine similarity loss on delta (directional accuracy).")
    p.add_argument("--flow-noise", type=float, default=0.1, help="Gaussian noise std in flow matching path. Higher = smoother velocity field, better generalization.")
    p.add_argument("--snr-endpoint-weight", type=float, default=0.0, help="Weight for SNR-weighted endpoint MSE (per-gene signal-to-noise ratio, higher weight on DEGs). Default 0 disables.")
    p.add_argument("--cov-loss-weight", type=float, default=0.0, help="Weight for covariance-preserving loss (gene-gene covariance structure matching). Default 0 disables.")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--decoder-dims", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--time-encoder-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--condition-embedding-dim", type=int, default=32, help="Dimension of condition embedding (default: 512, was 256).")
    p.add_argument("--cross-attn-layers", type=int, default=2, help="Number of cross-attention layers (default: 2 for LOCO).")
    p.add_argument("--gene-attn-dim", type=int, default=64, help="Dimension of gene attention embeddings (default: 64 for LOCO).")
    p.add_argument("--gene-self-attn-layers", type=int, default=1, help="Number of gene self-attention layers (default: 1 for LOCO).")
    p.add_argument("--cross-attn-heads", type=int, default=8, help="Number of cross-attention heads (default: 8 for LOCO).")
    p.add_argument("--cond-output-dropout", type=float, default=0.1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=20)
    p.add_argument("--gradient-clip-norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    p.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for Adam optimizer.")
    p.add_argument(
        "--match-every-n",
        type=int,
        default=5,
        help="Run OT Sinkhorn sample matching every N steps. Set <=0 to disable OT matching.",
    )
    p.add_argument(
        "--gene2vec-dict",
        default=str(ROOT / "data_gab" / "gene2vec_dict.pt"),
        help="Symbol-keyed gene2vec dictionary for perturbation tokens.",
    )
    p.add_argument(
        "--cache-dir",
        default=str(ROOT / "data_train" / "myflow_cache" / "replogle"),
        help="Directory for reusable static assets.",
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
    # Keep perturbation identifiers in gene-symbol space.
    if 'gene' in adata.obs:
        adata.obs['target_gene'] = adata.obs['gene'].astype(str)
    elif 'gene_id' in adata.obs:
        raise ValueError("Replogle symbol-mode training requires adata.obs['gene']; refusing to use adata.obs['gene_id'].")
    else:
        raise ValueError("Missing perturbation gene column: expected adata.obs['gene'].")
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
    
    gene2vec_dict_file = _resolve_existing_path([Path(args.gene2vec_dict)], "symbol gene2vec dict")
    emb = load_gene2vec_dict(gene2vec_dict_file)
    gene2vec_keys_upper = {gene.upper() for gene in emb}

    print(f"Original Obs shape: {adata.n_obs}")
    valid_mask = adata.obs["target_gene"].astype(str).str.upper().isin(gene2vec_keys_upper) | (
        adata.obs["target_gene"].astype(str) == "non-targeting"
    )
    adata = adata[valid_mask].copy()
    print(f"Filtered out {sum(~valid_mask)} cells whose target_gene lacks symbol gene2vec features.")
    print(f"Current Obs shape: {adata.n_obs}")

    adata.var_names = [str(g).strip() for g in adata.var_names]
    adata.var.index = adata.var_names

    # Simple perturbation gene token embeddings (no GO/PPI prior on expression genes)
    rep_key = "gene2vec_symbol_features"
    emb_dict = {k: np.asarray(v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in emb.items()}
    emb_dict["non-targeting"] = np.zeros(next(iter(emb_dict.values())).shape[0], dtype=np.float32)
    adata.uns[rep_key] = emb_dict
    perturbation_covariates = {"gene_perturbation": ["target_gene"]}
    perturbation_reps = {"gene_perturbation": rep_key}

    # Cell type embeddings (one-hot per cell line)
    if args.use_cell_type_condition:
        cell_lines = sorted(adata.obs["cell_type"].drop_duplicates().tolist())
        ct_emb_dict = {}
        for i, cl in enumerate(cell_lines):
            emb_arr = np.zeros(len(cell_lines), dtype=np.float32)
            emb_arr[i] = 1.0
            ct_emb_dict[cl] = emb_arr
        adata.uns["cell_type_embeddings"] = ct_emb_dict
        perturbation_covariates["cell_type"] = ["cell_type"]
        perturbation_reps["cell_type"] = "cell_type_embeddings"

    # Build gene_name → int index mapping for perturbation genes
    # Only include genes that actually appear as perturbations in the data,
    # NOT the full gene2vec dictionary (24K genes).  Building the GNN over
    # 24K nodes causes over-smoothing and makes all gene embeddings identical.
    _pert_genes_raw = sorted(set(adata.obs["target_gene"].astype(str).unique()) - {"non-targeting"})
    _pert_genes = [g for g in _pert_genes_raw if g in emb_dict]
    _pert_gene_to_idx = {g: i for i, g in enumerate(_pert_genes)}
    adata.uns["perturb_gene_symbol_to_idx"] = _pert_gene_to_idx
    print(f"Perturbation gene index mapping: {len(_pert_gene_to_idx)} genes")

    def _build_perturbation_graph(pert_genes, go_file, ppi_file):
        """Build GO+STRING graph edges over perturbation genes. Returns (edge_src, edge_tgt, edge_w)."""
        pert_genes = sorted(set(pert_genes))
        if not pert_genes:
            return None, None, None
        gene_to_idx = {g: i for i, g in enumerate(pert_genes)}
        n_nodes = len(pert_genes)
        edge_src, edge_tgt, edge_w = [], [], []
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
        return np.array(src_arr, dtype=np.int32), np.array(tgt_arr, dtype=np.int32), np.array(w_norm, dtype=np.float32)

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

    if args.control_key not in adata.obs:
        adata.obs[args.control_key] = adata.obs["target_gene"].astype(str) == "non-targeting"

    print(f"Total cells before split: {adata.n_obs}")

    # ================= Leave-One-Cell-Line-Out (LOCO) Split Logic =================
    holdout = args.holdout_cell_line
    assert holdout in adata.obs['cell_type'].unique(), f"Holdout cell line {holdout} not found in adata.obs['cell_type']"
    
    other_mask = adata.obs['cell_type'] != holdout
    holdout_mask = adata.obs['cell_type'] == holdout
    
    # Test perturbations in the held-out cell line must also be observed in
    # training cell lines. This keeps the task as cell-line transfer for seen
    # perturbation genes, while still reserving responses in the held-out line.
    holdout_targets = {
        str(p)
        for p in adata[holdout_mask].obs["target_gene"].astype(str).unique().tolist()
        if str(p) != "non-targeting"
    }
    other_targets = {
        str(p)
        for p in adata[other_mask].obs["target_gene"].astype(str).unique().tolist()
        if str(p) != "non-targeting"
    }
    pert_targets = sorted(holdout_targets & other_targets)
    holdout_only_targets = sorted(holdout_targets - other_targets)
    if not pert_targets:
        raise ValueError(
            f"No eligible perturbations found for holdout cell line {holdout}: "
            "a test perturbation must appear in both holdout and non-holdout cell lines."
        )
    if holdout_only_targets:
        print(
            f"Excluding {len(holdout_only_targets)} holdout-only perturbations from LOCO split "
            "because they are absent from training cell lines."
        )
    
    rng = np.random.default_rng(args.seed)
    shuffled_perts = rng.permutation(pert_targets)
    n_train_perts = args.n_train_perts
    n_test_perts = args.n_test_perts
    if n_train_perts + n_test_perts > len(shuffled_perts):
        raise ValueError(
            f"n_train_perts ({n_train_perts}) + n_test_perts ({n_test_perts}) = "
            f"{n_train_perts + n_test_perts} > {len(shuffled_perts)} eligible perturbations."
        )

    # 前 n_train_perts 到 holdout-cell-line training，后 n_test_perts 到 holdout-cell-line test。
    # The test perturbation genes remain present in training through other cell lines.
    train_perts = set(shuffled_perts[:n_train_perts])
    test_perts = set(shuffled_perts[-n_test_perts:])
    missing_test_in_other = test_perts - other_targets
    if missing_test_in_other:
        raise AssertionError(f"Test perturbations missing from non-holdout training cell lines: {missing_test_in_other}")
    
    # 训练集: 其它3个细胞系全部 + holdout的前n_train_perts个扰动 + holdout的non-targeting(让模型知道基态)
    train_mask = other_mask | (holdout_mask & adata.obs['target_gene'].isin(train_perts)) | (holdout_mask & (adata.obs['target_gene'] == 'non-targeting'))
    # 零样本测试集: holdout的后n_test_perts个扰动
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
    print(f"  Holdout perturbations in Train: {len(train_perts)}")
    print(f"  Holdout perturbations in Test : {len(test_perts)}")
    print(f"  Training cells kept: {adata_train_full.n_obs}/{train_cells_before_subsample} ({args.train_cell_fraction:.2%}) before validation split.")
    print(f"  Test cells kept    : {adata_test_holdout.n_obs}/{test_cells_before_subsample} ({args.test_cell_fraction:.2%}).")
    print(f"  Using {adata.n_obs} cells for training, {adata_val.n_obs} for validation.")
    train_holdout_perts_seen = set(
        adata_train_full.obs.loc[
            (adata_train_full.obs["cell_type"] == holdout)
            & (~adata_train_full.obs[args.control_key].astype(bool)),
            "target_gene",
        ].astype(str)
    )
    test_perts_after_subsample = set(adata_test_holdout.obs["target_gene"].astype(str).unique())
    if not test_perts_after_subsample <= other_targets:
        raise AssertionError("Every test perturbation gene must be observed in non-holdout training cell lines.")
    if test_perts_after_subsample & train_holdout_perts_seen:
        raise AssertionError("Test perturbation responses leaked into the held-out cell line training subset.")
    if not (
        (adata_train_full.obs["cell_type"] == holdout)
        & adata_train_full.obs[args.control_key].astype(bool)
    ).any():
        raise AssertionError(f"Training set must include {holdout} control/basal cells.")

    cross_cell_delta_prior: dict[str, np.ndarray] = {}
    prior_weight = 0.0
    use_delta_condition = False
    if prior_weight > 0 or use_delta_condition:
        other_train = adata_train_full[adata_train_full.obs["cell_type"] != holdout].copy()
        other_ctrl = other_train[other_train.obs[args.control_key].astype(bool)].copy()
        if other_ctrl.n_obs == 0:
            raise AssertionError("Cross-cell delta feature needs non-holdout control/basal cells.")
        other_ctrl_mean = mean_expr(other_ctrl)
        delta_genes = sorted(
            set(adata_train_full.obs["target_gene"].astype(str).unique())
            | set(adata_test_holdout.obs["target_gene"].astype(str).unique())
        )
        cross_cell_delta_prior["non-targeting"] = np.zeros(adata_train_full.n_vars, dtype=np.float32)
        for gene in delta_genes:
            if gene == "non-targeting":
                continue
            same_pert = other_train[
                (~other_train.obs[args.control_key].astype(bool))
                & (other_train.obs["target_gene"].astype(str) == gene)
            ].copy()
            if same_pert.n_obs == 0:
                if gene in test_perts_after_subsample:
                    raise AssertionError(f"Cross-cell delta feature missing non-holdout training cells for {gene}.")
                cross_cell_delta_prior[gene] = np.zeros(adata_train_full.n_vars, dtype=np.float32)
                continue
            cross_cell_delta_prior[gene] = mean_expr(same_pert) - other_ctrl_mean
        if use_delta_condition:
            adata_train_full.uns["cross_cell_delta_embeddings"] = cross_cell_delta_prior
            adata_val.uns["cross_cell_delta_embeddings"] = cross_cell_delta_prior
            adata.uns["cross_cell_delta_embeddings"] = cross_cell_delta_prior
            perturbation_covariates["cross_cell_delta"] = ["target_gene"]
            perturbation_reps["cross_cell_delta"] = "cross_cell_delta_embeddings"
        print(
            f"  Cross-cell delta feature enabled: condition={use_delta_condition}, "
            f"posthoc_weight={prior_weight:.2f}, conditions={len(cross_cell_delta_prior)}."
        )
    else:
        print("  Cross-cell delta feature disabled.")

    print(f"  Heldout-cell-line test set contains {adata_test_holdout.n_obs} cells.")
    print("  Test perturbation genes are seen in other cell lines, but their heldout-cell-line responses are withheld.")
    write_json(
        out_dir / f"split_summary_{run_label}.json",
        {
            "holdout_cell_line": holdout,
            "holdout_perturbations_total": len(holdout_targets),
            "holdout_perturbations_eligible_seen_in_other_cell_lines": len(pert_targets),
            "holdout_only_perturbations_excluded": holdout_only_targets,
            "holdout_perturbations_train": len(train_perts),
            "holdout_perturbations_test": len(test_perts),
            "test_perturbations_seen_in_training_cell_lines": sorted(test_perts & other_targets),
            "train_cells_before_subsample": int(train_cells_before_subsample),
            "test_cells_before_subsample": int(test_cells_before_subsample),
            "train_cell_fraction": args.train_cell_fraction,
            "test_cell_fraction": args.test_cell_fraction,
            "control_source_split_covariates": ["cell_type"],
            "cross_cell_delta_prior_weight": prior_weight,
            "use_cross_cell_delta_condition": use_delta_condition,
            "cross_cell_delta_prior_conditions": len(cross_cell_delta_prior),
            "static_cache": {
                "cache_dir": str(Path(args.cache_dir)),
                "gene2vec_dict_file": str(gene2vec_dict_file),
            },
            "train_full_before_validation": summarize_adata_split(adata_train_full, args.control_key),
            "train_passed_to_myflow": summarize_adata_split(adata, args.control_key),
            "validation": summarize_adata_split(adata_val, args.control_key),
            "heldout_cell_line_test": summarize_adata_split(adata_test_holdout, args.control_key),
        },
    )
    # ==============================================================================

    print("Initializing MyFlow (this may import jax/flax/ott)")
    cf = MyFlow(adata, solver=args.solver)
    print("Preparing data for training")
    if not args.use_cell_type_split:
        print("Ignoring --no-use-cell-type-split: Replogle training always splits controls by cell_type.")
    split_covariates = ["cell_type"]
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
    layers_before_pool = []
    if use_delta_condition:
        layers_before_pool = {
            "gene_perturbation": [],
            "cross_cell_delta": [
                {
                    "layer_type": "mlp",
                    "dims": [256, args.cross_cell_delta_condition_dim],
                    "dropout_rate": 0.05,
                    "act_last_layer": True,
                }
            ],
        }
        if args.use_cell_type_condition:
            layers_before_pool["cell_type"] = []
    cf.prepare_model(
        seed=args.seed,
        condition_embedding_dim=args.condition_embedding_dim,
        hidden_dims=args.hidden_dims,
        decoder_dims=args.decoder_dims,
        time_encoder_dims=args.time_encoder_dims,
        cond_output_dropout=args.cond_output_dropout,
        layers_before_pool=layers_before_pool,
        cross_attn_layers=args.cross_attn_layers,
        gene_attn_dim=args.gene_attn_dim,
        gene_self_attn_layers=args.gene_self_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        probability_path={"constant_noise": args.flow_noise},
        optimizer=optax.MultiSteps(optax.chain(optax.clip_by_global_norm(args.gradient_clip_norm), optax.adamw(args.learning_rate, weight_decay=1e-5)), args.gradient_accumulation_steps),
        conditioning=args.conditioning,
        perturbation_gnn_kwargs=perturbation_gnn_kwargs,
        delta_head_enabled=args.delta_head_enabled,
        delta_head_hidden=args.delta_head_hidden,
        gene_mask_enabled=not args.no_gene_mask,
        solver_kwargs={
            "condition_combined_loss_weight": args.condition_combined_loss_weight,
            "condition_combined_sinkhorn_weight": 0.0,
            "condition_combined_energy_weight": 1.0,
            "condition_combined_epsilon": args.condition_combined_epsilon,
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
            "match_every_n": args.match_every_n,
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
    print("===== Hyperparameter Summary =====")
    print(f"  solver: {args.solver}")
    print(f"  seed: {args.seed}")
    print(f"  num_iterations: {args.num_iterations}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  match_every_n: {args.match_every_n}")
    print(f"")
    print(f"  conditioning: {args.conditioning}")
    print(f"  hidden_dims: {args.hidden_dims}")
    print(f"  decoder_dims: {args.decoder_dims}")
    print(f"  time_encoder_dims: {args.time_encoder_dims}")
    print(f"  condition_embedding_dim: {args.condition_embedding_dim}")
    print(f"  cross_attn_layers: {args.cross_attn_layers}")
    print(f"  gene_attn_dim: {args.gene_attn_dim}")
    print(f"  gene_self_attn_layers: {args.gene_self_attn_layers}")
    print(f"  cross_attn_heads: {args.cross_attn_heads}")
    print(f"  cond_output_dropout: {args.cond_output_dropout}")
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
    print("Starting prediction on heldout-cell-line perturbation responses...", flush=True)
    # 提取 holdout 细胞系的 control 作为测试集的 baseline 输入
    # (即上面划分时放进 adata_train_full 的 non-targeting 细胞)
    test_adata = adata_train_full[(adata_train_full.obs['cell_type']==holdout) & (adata_train_full.obs[args.control_key]==True)].copy()
    
    # 提取测试集中 heldout 细胞系下 withheld response 的扰动基因；
    # 这些扰动基因本身必须已在其他细胞系训练数据中出现过。
    groups = adata_test_holdout.obs.groupby("target_gene").groups

    all_X = []
    all_obs = []

    # Use a fixed prediction sample size by default to avoid repeated JAX
    # recompilations caused by hundreds of condition-specific input shapes.
    fixed_predict_n = int(args.predict_n_cells)
    size_to_genes: dict[int, list[str]] = {}
    for gene, idx in groups.items():
        sample_size = fixed_predict_n if fixed_predict_n > 0 else int(len(idx))
        size_to_genes.setdefault(sample_size, []).append(str(gene))
    bucket_sizes = sorted(size_to_genes.keys())
    print(
        f"Prediction buckets by sample_size: {len(bucket_sizes)} unique sizes "
        f"across {len(groups)} genes. Largest sample_size={max(bucket_sizes) if bucket_sizes else 0}.",
        flush=True,
    )
    if fixed_predict_n > 0:
        print(f"Using fixed predict_n_cells={fixed_predict_n} per perturbation to reduce JAX compilation.", flush=True)

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
            if prior_weight > 0:
                source_expr = dense_X(sub_adata).astype(np.float32, copy=False)
                delta = cross_cell_delta_prior.get(gene)
                if delta is None:
                    raise AssertionError(f"Missing cross-cell delta prior for {gene}.")
                anchor = np.clip(source_expr + delta[None, :], 0.0, None)
                arr = (1.0 - prior_weight) * arr + prior_weight * anchor
            all_X.append(arr)
            obs = pd.DataFrame({"perturbation": [gene] * arr.shape[0]})
            all_obs.append(obs)
    print("Prediction finished")
    X = np.vstack(all_X)
    X = np.clip(X, 0, None)  # clamp negative predicted expression to 0
    obs = pd.concat(all_obs, ignore_index=True)
    pred_dir = Path(args.output_dir) / f"predictions_{run_label}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    finite_mask = np.isfinite(X)
    diagnostics = {
        "shape": list(X.shape),
        "min": float(np.nanmin(X)),
        "max": float(np.nanmax(X)),
        "mean": float(np.nanmean(X)),
        "std": float(np.nanstd(X)),
        "q50": float(np.nanpercentile(X, 50)),
        "q75": float(np.nanpercentile(X, 75)),
        "q95": float(np.nanpercentile(X, 95)),
        "q99": float(np.nanpercentile(X, 99)),
        "q999": float(np.nanpercentile(X, 99.9)),
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "frac_gt_prediction_max_allowed": float(np.mean(X > args.prediction_max_allowed)),
        "prediction_max_allowed": float(args.prediction_max_allowed),
        "prediction_frac_gt_max_allowed": float(args.prediction_frac_gt_max_allowed),
        "all_finite": bool(finite_mask.all()),
    }
    diag_file = pred_dir / f"prediction_diagnostics_{run_label}.json"
    with open(diag_file, "w") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
    print(f"Prediction diagnostics: {diagnostics}")
    print(f"Saved prediction diagnostics: {diag_file}")

    prediction_failed = (
        not finite_mask.all()
        or diagnostics["max"] > args.prediction_max_allowed
        or diagnostics["frac_gt_prediction_max_allowed"] > args.prediction_frac_gt_max_allowed
    )
    if prediction_failed:
        raw_file = pred_dir / f"raw_unclipped_predictions_{run_label}.h5ad"
        ad.AnnData(X=X, obs=obs, var=test_adata.var.copy()).write_h5ad(raw_file)
        raise RuntimeError(
            "Raw MyFlow prediction left the expected expression range; "
            f"max={diagnostics['max']:.6g}, "
            f"frac>{args.prediction_max_allowed:g}={diagnostics['frac_gt_prediction_max_allowed']:.6g}. "
            f"Saved diagnostics to {diag_file} and raw predictions to {raw_file}."
        )

    adata_pred = ad.AnnData(X=X, obs=obs, var=test_adata.var.copy())
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

        print(f"Basic => MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
        metrics_summary.update(
            {
                "success": True,
                "mse": float(mse),
                "mae": float(mae),
                "l2": float(l2),
            }
        )
        
        print("\nCalculating per-condition DEG metrics (R², EV, PCC)...")
        deg_details = compute_deg_metrics_per_condition(
            test_adata, adata_test_holdout, adata_pred,
            real_condition_key="target_gene",
            pred_condition_key="perturbation",
        )
        if deg_details:
            deg_df = pd.DataFrame(deg_details)
            avg_r2 = float(deg_df["r2_deg"].mean())
            avg_ev = float(deg_df["ev_deg"].mean())
            avg_pcc = float(deg_df["pcc_deg"].mean())
            avg_ndegs = float(deg_df["n_degs"].mean())
            spearman_valid = deg_df["de_spearman"].dropna()
            avg_de_spearman = float(spearman_valid.mean()) if len(spearman_valid) > 0 else float("nan")
            avg_precision = float(deg_df["deg_precision"].mean())
            avg_recall = float(deg_df["deg_recall"].mean())
            avg_f1 = float(deg_df["deg_f1"].mean())
            avg_jaccard = float(deg_df["deg_jaccard"].mean())
            avg_n_pred_degs = float(deg_df["n_pred_degs"].mean())
            avg_n_overlap = float(deg_df["n_overlap_degs"].mean())
            avg_condition_ds = float(deg_df["condition_ds"].mean())
            avg_pearson_delta = float(deg_df["condition_pearson_delta"].mean())
            avg_pearson_delta_hat = float(deg_df["condition_pearson_delta_hat"].mean()) if "condition_pearson_delta_hat" in deg_df else float("nan")
            avg_pearson_delta_top20 = float(deg_df["condition_pearson_delta_top20"].mean())
            avg_pearson_delta_hat_top20 = float(deg_df["condition_pearson_delta_hat_top20"].mean()) if "condition_pearson_delta_hat_top20" in deg_df else float("nan")
            avg_pearson_delta_top50 = float(deg_df["condition_pearson_delta_top50"].mean())
            avg_pearson_delta_hat_top50 = float(deg_df["condition_pearson_delta_hat_top50"].mean()) if "condition_pearson_delta_hat_top50" in deg_df else float("nan")
            avg_pearson_delta_top100 = float(deg_df["condition_pearson_delta_top100"].mean())
            avg_pearson_delta_hat_top100 = float(deg_df["condition_pearson_delta_hat_top100"].mean()) if "condition_pearson_delta_hat_top100" in deg_df else float("nan")
            avg_pearson_delta_top1000 = float(deg_df["condition_pearson_delta_top1000"].mean())
            avg_pearson_delta_hat_top1000 = float(deg_df["condition_pearson_delta_hat_top1000"].mean()) if "condition_pearson_delta_hat_top1000" in deg_df else float("nan")
            print(f"Per-condition DEG avg => R²: {avg_r2:.4f}, EV: {avg_ev:.4f}, PCC: {avg_pcc:.4f}, condition_DS: {avg_condition_ds:.4f}, DE-Spearman: {avg_de_spearman:.4f}, avg #DEGs: {avg_ndegs:.0f}")
            print(f"Per-condition Pearson => Δ: {avg_pearson_delta:.4f}, Δ̂: {avg_pearson_delta_hat:.4f}, "
                  f"Δ20: {avg_pearson_delta_top20:.4f}, Δ̂20: {avg_pearson_delta_hat_top20:.4f}, "
                  f"Δ50: {avg_pearson_delta_top50:.4f}, Δ̂50: {avg_pearson_delta_hat_top50:.4f}, "
                  f"Δ100: {avg_pearson_delta_top100:.4f}, Δ̂100: {avg_pearson_delta_hat_top100:.4f}, "
                  f"Δ1000: {avg_pearson_delta_top1000:.4f}, Δ̂1000: {avg_pearson_delta_hat_top1000:.4f}")
            print(f"DEG Overlap => Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, Jaccard: {avg_jaccard:.4f}, avg #pred-DEGs: {avg_n_pred_degs:.0f}, avg #overlap: {avg_n_overlap:.0f}")
            metrics_summary.update(
                {
                    "r2_deg": avg_r2,
                    "ev_deg": avg_ev,
                    "pcc_deg": avg_pcc,
                    "pearson_delta": avg_pearson_delta,
                    "pearson_delta_hat": avg_pearson_delta_hat,
                    "pearson_delta_top20": avg_pearson_delta_top20,
                    "pearson_delta_hat_top20": avg_pearson_delta_hat_top20,
                    "pearson_delta_top50": avg_pearson_delta_top50,
                    "pearson_delta_hat_top50": avg_pearson_delta_hat_top50,
                    "pearson_delta_top100": avg_pearson_delta_top100,
                    "pearson_delta_hat_top100": avg_pearson_delta_hat_top100,
                    "pearson_delta_top1000": avg_pearson_delta_top1000,
                    "pearson_delta_hat_top1000": avg_pearson_delta_hat_top1000,
                    "de_spearman": avg_de_spearman,
                    "deg_precision": avg_precision,
                    "deg_recall": avg_recall,
                    "deg_f1": avg_f1,
                    "deg_jaccard": avg_jaccard,
                    "avg_n_degs": avg_ndegs,
                    "avg_n_pred_degs": avg_n_pred_degs,
                    "avg_n_overlap_degs": avg_n_overlap,
                    "condition_ds": avg_condition_ds,
                    "deg_conditions_count": len(deg_details),
                }
            )
            deg_file = out_dir / f"deg_per_condition_{run_label}.json"
            with open(deg_file, "w") as f:
                json.dump(deg_details, f, indent=2)
            print(f"Saved per-condition DEG metrics: {deg_file}")
        else:
            print("No valid per-condition DEG metrics computed.")
    except Exception as e:
        print(f"Evaluation failed (usually due to sparse matrix formatting or dimension mismatch): {e}")
        metrics_summary["error"] = str(e)
    write_json(out_dir / f"metrics_summary_{run_label}.json", metrics_summary)
    print("=" * 50)

if __name__ == "__main__":
    main()
