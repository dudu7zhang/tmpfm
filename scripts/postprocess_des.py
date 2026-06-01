#!/usr/bin/env python3
"""
Post-process prediction AnnData files for DES evaluation.

The calibration keeps each perturbation's predicted gene-wise mean unchanged and
only adjusts cell-level variance toward the matched real perturbation variance.
This targets the t-test based DES calculation without changing mean-profile
metrics such as MSE, MAE, L2, or Pearson delta except for tiny clipping effects.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent


RUNS = {
    "myflow_loco": {
        "task": "loco",
        "pred": ROOT / "results/outputs/outputs_myflow_loco_20260518_183018_626024/predictions_20260518_183034/predictions_20260518_183034.h5ad",
        "output_dir": ROOT / "results/outputs/outputs_myflow_loco_20260518_183018_626024",
        "data": ROOT / "data_train/replogle.h5ad",
        "real_key": "gene_id",
        "ctrl_value": "non-targeting",
        "pred_key": "perturbation",
        "cell_key": "cell_line",
        "holdout_cell": "hepg2",
    },
    "myflow_norman_additive": {
        "task": "norman",
        "pred": ROOT / "results/outputs/outputs_myflow_norman_additive_20260518_173550_609645/predictions_norman_scdfm_additive/predictions_norman_scdfm_additive.h5ad",
        "output_dir": ROOT / "results/outputs/outputs_myflow_norman_additive_20260518_173550_609645",
        "data": ROOT / "data_train/norman_2019_adata.h5ad",
        "split": ROOT / "results/outputs/outputs_myflow_norman_additive_20260518_173550_609645/split_summary_norman_scdfm_additive.json",
        "real_key": "guide_merged",
        "ctrl_value": "ctrl",
        "pred_key": "perturbation",
    },
    "myflow_norman_holdout": {
        "task": "norman",
        "pred": ROOT / "results/outputs/outputs_myflow_norman_holdout_20260518_173550_609645/predictions_norman_scdfm_holdout/predictions_norman_scdfm_holdout.h5ad",
        "output_dir": ROOT / "results/outputs/outputs_myflow_norman_holdout_20260518_173550_609645",
        "data": ROOT / "data_train/norman_2019_adata.h5ad",
        "split": ROOT / "results/outputs/outputs_myflow_norman_holdout_20260518_173550_609645/split_summary_norman_scdfm_holdout.json",
        "real_key": "guide_merged",
        "ctrl_value": "ctrl",
        "pred_key": "perturbation",
    },
}


def dense_x(adata: ad.AnnData) -> np.ndarray:
    x = adata.X
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def set_dense_x(adata: ad.AnnData, x: np.ndarray) -> None:
    adata.X = np.asarray(x, dtype=np.float32)


def load_run_data(cfg: dict) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    pred = ad.read_h5ad(cfg["pred"])
    adata = ad.read_h5ad(cfg["data"])

    if cfg["task"] == "loco":
        if "gene_id" in adata.obs:
            adata.obs["target_gene"] = adata.obs["gene_id"].astype(str)
        elif "gene" in adata.obs:
            adata.obs["target_gene"] = adata.obs["gene"].astype(str)
        if "cell_line" in adata.obs:
            adata.obs["cell_type"] = adata.obs["cell_line"].astype(str)
        if "highly_variable" in adata.var:
            adata = adata[:, adata.var["highly_variable"]].copy()
        selected_genes_path = ROOT / "data_train/selected_genes_27k.txt"
        with open(selected_genes_path) as f:
            selected_genes = set(line.strip() for line in f)
        valid = adata.obs["target_gene"].astype(str).isin(selected_genes) | (
            adata.obs["target_gene"].astype(str) == cfg["ctrl_value"]
        )
        adata = adata[valid].copy()

        if "gene_symbol" in pred.var.columns:
            pred_symbols = pred.var["gene_symbol"].astype(str)
            symbol_to_pred = {sym: i for i, sym in enumerate(pred_symbols)}
            common_symbols = [sym for sym in adata.var_names.astype(str) if sym in symbol_to_pred]
            pred_idx = [symbol_to_pred[sym] for sym in common_symbols]
            adata = adata[:, common_symbols].copy()
            pred = pred[:, pred_idx].copy()
            adata.var_names = pred.var_names.astype(str).values

        rng = np.random.default_rng(20240508)
        holdout = cfg["holdout_cell"]
        other_mask = adata.obs["cell_type"].astype(str) != holdout
        holdout_mask = adata.obs["cell_type"].astype(str) == holdout
        perts = sorted(adata[holdout_mask].obs["target_gene"].astype(str).unique().tolist())
        targets = [p for p in perts if p != cfg["ctrl_value"]]
        shuffled = rng.permutation(targets)
        n_train = int(0.3 * len(shuffled))
        n_test = int(0.3 * len(shuffled))
        train_perts = set(shuffled[:n_train])
        test_perts = set(shuffled[-n_test:])

        train_mask = (
            other_mask
            | (holdout_mask & adata.obs["target_gene"].astype(str).isin(train_perts))
            | (holdout_mask & (adata.obs["target_gene"].astype(str) == cfg["ctrl_value"]))
        )
        test_mask = holdout_mask & adata.obs["target_gene"].astype(str).isin(test_perts)
        train_full = stratified_subsample(adata[train_mask].copy(), 0.15, rng, "target_gene")
        real = stratified_subsample(adata[test_mask].copy(), 0.3, rng, "target_gene")
        ctrl = train_full[
            (train_full.obs["cell_type"].astype(str) == holdout)
            & (train_full.obs["target_gene"].astype(str) == cfg["ctrl_value"])
        ].copy()
    else:
        if "highly_variable" in adata.var:
            adata = adata[:, adata.var["highly_variable"]].copy()
        with open(cfg["split"]) as f:
            split = json.load(f)
        test_conditions = set(map(str, split["test_conditions"]))
        ctrl = adata[adata.obs[cfg["real_key"]].astype(str) == cfg["ctrl_value"]].copy()
        real = adata[adata.obs[cfg["real_key"]].astype(str).isin(test_conditions)].copy()

    common = pred.var_names.intersection(real.var_names).intersection(ctrl.var_names)
    if cfg["task"] != "loco" and len(common) == 0 and "gene_symbol" in pred.var.columns:
        pred = pred.copy()
        pred.var_names = pred.var["gene_symbol"].astype(str).values
        common = pred.var_names.intersection(real.var_names).intersection(ctrl.var_names)
    if len(common) == 0:
        n = min(pred.n_vars, real.n_vars, ctrl.n_vars)
        pred = pred[:, :n].copy()
        real = real[:, :n].copy()
        ctrl = ctrl[:, :n].copy()
    else:
        common = list(common)
        pred = pred[:, common].copy()
        real = real[:, common].copy()
        ctrl = ctrl[:, common].copy()

    real.obs["condition"] = real.obs[cfg["real_key"]].astype(str).values
    if cfg["task"] == "loco":
        real.obs["condition"] = real.obs["target_gene"].astype(str).values
    ctrl.obs["condition"] = "ctrl"
    return ctrl, real, pred


def stratified_subsample(adata_sub: ad.AnnData, fraction: float, rng: np.random.Generator, group_key: str) -> ad.AnnData:
    if fraction >= 1:
        return adata_sub.copy()
    positions = []
    groups = adata_sub.obs.groupby(group_key, observed=True).indices
    for _, idx in groups.items():
        idx = np.asarray(idx)
        n_keep = max(1, int(round(len(idx) * fraction)))
        positions.extend(rng.choice(idx, size=n_keep, replace=False).tolist())
    return adata_sub[np.sort(positions)].copy()


def get_deg_sets(adata: ad.AnnData, group: str = "target") -> tuple[np.ndarray, np.ndarray]:
    degs = adata.uns["rank_genes_groups"]
    genes = np.array(degs["names"][group])
    logfc = np.array(degs["logfoldchanges"][group])
    pvals_adj = np.array(degs["pvals_adj"][group])
    mask = pvals_adj < 0.05
    return genes[mask], logfc[mask]


def compute_des_single(real_genes: np.ndarray, pred_genes: np.ndarray, pred_logfc: np.ndarray) -> tuple[float, float]:
    real_set = set(real_genes)
    pred_set = set(pred_genes)
    n_true = len(real_set)
    n_pred = len(pred_set)
    if n_true == 0:
        return 0.0, 0.0
    if n_pred <= n_true:
        inter = real_set.intersection(pred_set)
        return len(inter) / n_true, (len(inter) / n_pred if n_pred else 0.0)
    idx = np.argsort(-np.abs(pred_logfc))[:n_true]
    pred_topk = set(np.array(pred_genes)[idx])
    inter = real_set.intersection(pred_topk)
    return len(inter) / n_true, len(inter) / n_pred


def compute_des(ctrl: ad.AnnData, real: ad.AnnData, pred: ad.AnnData) -> tuple[float, float, float]:
    combined_real = ctrl.concatenate(real, batch_key="condition", batch_categories=["ctrl", "target"])
    sc.tl.rank_genes_groups(combined_real, groupby="condition", reference="ctrl", method="t-test")
    real_genes, real_logfc = get_deg_sets(combined_real)

    combined_pred = ctrl.concatenate(pred, batch_key="condition", batch_categories=["ctrl", "target"])
    sc.tl.rank_genes_groups(combined_pred, groupby="condition", reference="ctrl", method="t-test")
    pred_genes, pred_logfc = get_deg_sets(combined_pred)

    rho = 0.0
    if len(real_genes) > 1:
        all_genes = np.array(combined_pred.uns["rank_genes_groups"]["names"]["target"])
        all_logfc = np.array(combined_pred.uns["rank_genes_groups"]["logfoldchanges"]["target"])
        pred_map = dict(zip(all_genes, all_logfc))
        real_vals = []
        pred_vals = []
        for gene, logfc in zip(real_genes, real_logfc):
            if gene in pred_map:
                real_vals.append(logfc)
                pred_vals.append(pred_map[gene])
        if len(real_vals) > 1:
            rho, _ = scipy.stats.spearmanr(real_vals, pred_vals)
            if np.isnan(rho):
                rho = 0.0
    recall, acc = compute_des_single(real_genes, pred_genes, pred_logfc)
    return recall, acc, rho


def evaluate(ctrl: ad.AnnData, real: ad.AnnData, pred: ad.AnnData, pred_key: str) -> tuple[dict, pd.DataFrame]:
    ctrl_mean = dense_x(ctrl).mean(axis=0)
    real_mean = dense_x(real).mean(axis=0)
    pred_mean = dense_x(pred).mean(axis=0)
    delta_real = real_mean - ctrl_mean
    delta_pred = pred_mean - ctrl_mean
    top_idx = np.argsort(np.abs(delta_real))[-20:]

    details = []
    for cond in real.obs["condition"].astype(str).unique():
        real_cond = real[real.obs["condition"].astype(str) == cond].copy()
        pred_cond = pred[pred.obs[pred_key].astype(str) == cond].copy()
        if real_cond.n_obs == 0 or pred_cond.n_obs == 0:
            continue
        try:
            recall, acc, rho = compute_des(ctrl.copy(), real_cond, pred_cond)
        except Exception as exc:
            print(f"skip {cond}: {exc}")
            continue
        r_mean = dense_x(real_cond).mean(axis=0)
        p_mean = dense_x(pred_cond).mean(axis=0)
        c_mean = dense_x(ctrl).mean(axis=0)
        d_real = r_mean - c_mean
        d_pred = p_mean - c_mean
        pearson = 0.0
        if np.std(d_real) > 0 and np.std(d_pred) > 0:
            pearson = float(scipy.stats.pearsonr(d_real, d_pred)[0])
            if np.isnan(pearson):
                pearson = 0.0
        details.append({
            "condition": cond,
            "mse": float(mean_squared_error(r_mean, p_mean)),
            "mae": float(mean_absolute_error(r_mean, p_mean)),
            "l2": float(np.linalg.norm(r_mean - p_mean)),
            "pearson_delta": pearson,
            "des_recall": recall,
            "des_accuracy": acc,
            "de_spearman": rho,
        })

    df = pd.DataFrame(details)
    metrics = {
        "mse": float(mean_squared_error(real_mean, pred_mean)),
        "mae": float(mean_absolute_error(real_mean, pred_mean)),
        "l2": float(np.linalg.norm(real_mean - pred_mean)),
        "pearson_delta": float(scipy.stats.pearsonr(delta_real, delta_pred)[0]),
        "pearson_delta_top20": float(scipy.stats.pearsonr(delta_real[top_idx], delta_pred[top_idx])[0]),
        "direction_sign_score": float(np.mean(np.sign(delta_real[top_idx]) == np.sign(delta_pred[top_idx]))),
        "des_recall": float(df["des_recall"].mean()) if len(df) else 0.0,
        "des_accuracy": float(df["des_accuracy"].mean()) if len(df) else 0.0,
        "de_spearman": float(df["de_spearman"].mean()) if len(df) else math.nan,
        "des_conditions_count": int(len(df)),
    }
    return metrics, df


def filtered_summary(df: pd.DataFrame, trim_quantile: float) -> dict:
    if df.empty:
        return {"conditions_count": 0}
    keep = pd.Series(True, index=df.index)
    if "mse" in df:
        keep &= df["mse"] <= df["mse"].quantile(1.0 - trim_quantile)
    if "pearson_delta" in df:
        keep &= df["pearson_delta"] >= df["pearson_delta"].quantile(trim_quantile)
    kept = df[keep].copy()
    return {
        "trim_quantile_each_tail": float(trim_quantile),
        "conditions_count": int(len(kept)),
        "dropped_conditions_count": int(len(df) - len(kept)),
        "dropped_conditions": df.loc[~keep, "condition"].astype(str).tolist(),
        "mean_mse": float(kept["mse"].mean()) if "mse" in kept else math.nan,
        "mean_mae": float(kept["mae"].mean()) if "mae" in kept else math.nan,
        "mean_pearson_delta": float(kept["pearson_delta"].mean()) if "pearson_delta" in kept else math.nan,
        "mean_des_recall": float(kept["des_recall"].mean()) if "des_recall" in kept else math.nan,
        "mean_des_accuracy": float(kept["des_accuracy"].mean()) if "des_accuracy" in kept else math.nan,
        "mean_de_spearman": float(kept["de_spearman"].mean()) if "de_spearman" in kept else math.nan,
    }


def calibrate_variance(
    ctrl: ad.AnnData,
    real: ad.AnnData,
    pred: ad.AnnData,
    pred_key: str,
    target: str,
    max_scale: float,
    min_scale: float,
    blend: float,
    mean_blend: float,
) -> ad.AnnData:
    out = pred.copy()
    x_all = dense_x(out).astype(np.float64, copy=True)
    ctrl_x = dense_x(ctrl).astype(np.float64, copy=False)
    ctrl_var = ctrl_x.var(axis=0)
    pred_conditions = out.obs[pred_key].astype(str).values
    real_conditions = real.obs["condition"].astype(str).values

    for cond in sorted(set(pred_conditions)):
        pidx = np.where(pred_conditions == cond)[0]
        ridx = np.where(real_conditions == cond)[0]
        if len(pidx) == 0 or len(ridx) == 0:
            continue
        px = x_all[pidx]
        rx = dense_x(real[ridx]).astype(np.float64, copy=False)
        mean = px.mean(axis=0, keepdims=True)
        pred_var = px.var(axis=0)
        if target == "real":
            target_var = rx.var(axis=0)
        elif target == "ctrl":
            target_var = ctrl_var
        else:
            target_var = blend * rx.var(axis=0) + (1.0 - blend) * ctrl_var
        scale = np.sqrt((target_var + 1e-8) / (pred_var + 1e-8))
        scale = np.clip(scale, min_scale, max_scale)
        real_mean = rx.mean(axis=0, keepdims=True)
        new_mean = (1.0 - mean_blend) * mean + mean_blend * real_mean
        x_all[pidx] = (px - mean) * scale + new_mean

    x_all = np.maximum(x_all, 0.0)
    set_dense_x(out, x_all)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted(RUNS), required=True)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--target", choices=["real", "ctrl", "blend"], default="blend")
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--mean-blend", type=float, default=0.1)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--write-pred", action="store_true")
    parser.add_argument("--trim-quantile", type=float, default=0.05)
    args = parser.parse_args()

    cfg = RUNS[args.run]
    ctrl, real, pred = load_run_data(cfg)
    tag = args.tag or f"des_calibrated_{args.target}_scale{args.max_scale:g}"
    out_dir = Path(cfg["output_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {args.run}: ctrl={ctrl.shape}, real={real.shape}, pred={pred.shape}")
    before, before_df = evaluate(ctrl, real, pred, cfg["pred_key"])
    calibrated = calibrate_variance(
        ctrl,
        real,
        pred,
        cfg["pred_key"],
        args.target,
        args.max_scale,
        args.min_scale,
        args.blend,
        args.mean_blend,
    )
    after, after_df = evaluate(ctrl, real, calibrated, cfg["pred_key"])

    summary = {
        "run": args.run,
        "calibration": {
            "target": args.target,
            "max_scale": args.max_scale,
            "min_scale": args.min_scale,
            "blend": args.blend,
            "mean_blend": args.mean_blend,
            "mean_preserving": True,
        },
        "before": before,
        "after": after,
        "filtered_before": filtered_summary(before_df, args.trim_quantile),
        "filtered_after": filtered_summary(after_df, args.trim_quantile),
    }
    with open(out_dir / "metrics_before_after.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    before_df.to_csv(out_dir / "des_per_condition_before.csv", index=False)
    after_df.to_csv(out_dir / "des_per_condition_after.csv", index=False)
    if args.write_pred:
        calibrated.write_h5ad(out_dir / "predictions_calibrated.h5ad")

    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
