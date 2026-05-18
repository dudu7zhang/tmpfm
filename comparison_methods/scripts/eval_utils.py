"""Shared evaluation metrics matching CellFlow's cal_score.py / eval.degs."""

import numpy as np
import scipy.stats
import scanpy as sc
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
        pearson_delta_top_k, _ = scipy.stats.pearsonr(
            delta_real[top_n_idx], delta_pred[top_n_idx]
        )
    else:
        pearson_delta_top_k = 0.0

    sign_real = np.sign(delta_real[top_n_idx])
    sign_pred = np.sign(delta_pred[top_n_idx])
    ds_score = np.mean([1 if r == p else 0 for r, p in zip(sign_real, sign_pred)])
    return pearson_delta, pearson_delta_top_k, ds_score


def get_deg_sets(adata, group="target"):
    degs = adata.uns["rank_genes_groups"]
    genes = np.array(degs["names"][group])
    logfc = np.array(degs["logfoldchanges"][group])
    pvals_adj = np.array(degs["pvals_adj"][group])
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
    combined_real = ctrl.concatenate(
        target, batch_key="condition", batch_categories=["ctrl", "target"]
    )
    sc.tl.rank_genes_groups(
        combined_real, groupby="condition", reference="ctrl", method="t-test"
    )
    real_genes, real_logfc = get_deg_sets(combined_real, group="target")

    if "gene_symbol" in pred.var.columns:
        pred.var.index = pred.var["gene_symbol"]
        pred.var_names = pred.var["gene_symbol"].values

    combined_pred = ctrl.concatenate(
        pred, batch_key="condition", batch_categories=["ctrl", "target"]
    )
    sc.tl.rank_genes_groups(
        combined_pred, groupby="condition", reference="ctrl", method="t-test"
    )
    pred_genes, pred_logfc = get_deg_sets(combined_pred, group="target")

    de_spearman = 0.0
    if len(real_genes) > 1:
        degs_pred_all = combined_pred.uns["rank_genes_groups"]
        all_pred_genes = np.array(degs_pred_all["names"]["target"])
        all_pred_logfc = np.array(degs_pred_all["logfoldchanges"]["target"])
        pred_logfc_map = dict(zip(all_pred_genes, all_pred_logfc))
        real_matched_logfc = []
        pred_matched_logfc = []
        for rg, r_fc in zip(real_genes, real_logfc):
            if rg in pred_logfc_map:
                real_matched_logfc.append(r_fc)
                pred_matched_logfc.append(pred_logfc_map[rg])
        if len(real_matched_logfc) > 1:
            try:
                de_spearman, _ = scipy.stats.spearmanr(
                    real_matched_logfc, pred_matched_logfc
                )
                if np.isnan(de_spearman):
                    de_spearman = 0.0
            except Exception:
                de_spearman = 0.0
    des_recall, des_acc = compute_des_single(real_genes, pred_genes, pred_logfc)
    return des_recall, des_acc, de_spearman


def compute_des_per_condition(ctrl_adata, real_adata, pred_adata,
                              real_condition_key="condition",
                              pred_condition_key="perturbation"):
    """Compute DES and DE Spearman per condition, return averaged results."""
    real_conditions = real_adata.obs[real_condition_key].unique()
    des_results = []
    for cond in real_conditions:
        real_mask = real_adata.obs[real_condition_key] == cond
        pred_mask = pred_adata.obs[pred_condition_key] == cond
        if real_mask.sum() == 0 or pred_mask.sum() == 0:
            continue
        real_cond = real_adata[real_mask].copy()
        pred_cond = pred_adata[pred_mask].copy()
        try:
            d_recall, d_acc, d_spearman = compute_des(ctrl_adata.copy(), real_cond, pred_cond)
            des_results.append({
                "condition": str(cond),
                "des_recall": float(d_recall),
                "des_accuracy": float(d_acc),
                "de_spearman": float(d_spearman) if not np.isnan(d_spearman) else None,
            })
        except Exception:
            continue

    if not des_results:
        return 0.0, 0.0, float("nan"), []

    import pandas as pd
    des_df = pd.DataFrame(des_results)
    recall_avg = des_df["des_recall"].mean()
    acc_avg = des_df["des_accuracy"].mean()
    spearman_valid = des_df["de_spearman"].dropna()
    spearman_avg = float(spearman_valid.mean()) if len(spearman_valid) > 0 else float("nan")
    return recall_avg, acc_avg, spearman_avg, des_results


def evaluate_predictions(ctrl_adata, real_adata, pred_adata, output_prefix="",
                         real_condition_key="condition",
                         pred_condition_key="perturbation"):
    """Run full evaluation and print/save results."""
    import json

    metrics = {}
    try:
        ctrl_mean = np.array(ctrl_adata.X.mean(axis=0)).flatten()
        real_mean = np.array(real_adata.X.mean(axis=0)).flatten()
        pred_mean = np.array(pred_adata.X.mean(axis=0)).flatten()

        mse, mae, l2 = cal_metric(pred_mean, real_mean)
        pearson_del, pearson_del_top20, ds = cal_delta_metric(
            ctrl_mean, real_mean, pred_mean
        )
        print(f"Basic => MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
        print(
            f"Delta => Pearson D: {pearson_del:.4f}, Pearson D20: {pearson_del_top20:.4f}, DS: {ds:.4f}"
        )
        metrics.update(
            {
                "mse": float(mse),
                "mae": float(mae),
                "l2": float(l2),
                "pearson_delta": float(pearson_del),
                "pearson_delta_top20": float(pearson_del_top20),
                "direction_sign_score": float(ds),
            }
        )

        print("Calculating per-condition DES & DE-Spearman...")
        des_recall, des_acc, de_spearman, des_details = compute_des_per_condition(
            ctrl_adata, real_adata, pred_adata,
            real_condition_key=real_condition_key,
            pred_condition_key=pred_condition_key,
        )
        print(
            f"DES (per-condition avg) => Recall: {des_recall:.4f}, Accuracy: {des_acc:.4f}, DE-Spearman rho: {de_spearman:.4f}"
        )
        metrics.update(
            {
                "des_recall": float(des_recall),
                "des_accuracy": float(des_acc),
                "de_spearman": float(de_spearman),
                "des_conditions_count": len(des_details),
            }
        )

        # Save per-condition DES details
        if output_prefix and des_details:
            des_file = f"{output_prefix}_des_per_condition.json"
            with open(des_file, "w") as f:
                json.dump(des_details, f, indent=2)
            print(f"Saved per-condition DES to {des_file}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        metrics["error"] = str(e)

    if output_prefix:
        with open(f"{output_prefix}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {output_prefix}_metrics.json")

    return metrics
