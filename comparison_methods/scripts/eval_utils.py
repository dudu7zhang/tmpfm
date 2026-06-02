"""Shared evaluation metrics matching CellFlow's cal_score.py / eval.degs."""

import numpy as np
import scipy.stats
import scanpy as sc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def cal_deg_metrics(ctrl_mean, real_mean, pred_mean, deg_indices):
    """Compute R², EV, PCC on DEG genes only (delta space)."""
    if len(deg_indices) == 0:
        return {"r2_deg": float("nan"), "ev_deg": float("nan"), "pcc_deg": float("nan")}
    delta_real = real_mean[deg_indices] - ctrl_mean[deg_indices]
    delta_pred = pred_mean[deg_indices] - ctrl_mean[deg_indices]
    # R²
    r2 = r2_score(delta_real, delta_pred)
    # Explained Variance: 1 - Var(residual) / Var(real)
    residual = delta_real - delta_pred
    ev = 1.0 - np.var(residual) / (np.var(delta_real) + 1e-10)
    # PCC
    if len(delta_real) < 2:
        pcc = 0.0
    else:
        pcc, _ = scipy.stats.pearsonr(delta_real, delta_pred)
        if np.isnan(pcc):
            pcc = 0.0
    return {"r2_deg": float(r2), "ev_deg": float(ev), "pcc_deg": float(pcc)}


def identify_degs(ctrl_mean, target_mean, alpha=0.05, n_cells_ctrl=50, n_cells_target=50):
    """Identify DEGs by comparing ctrl vs target distributions (synthetic t-test)."""
    # Use delta magnitude as proxy; genes with |delta| > threshold are DEGs
    delta = target_mean - ctrl_mean
    # Robust z-score approach
    median_delta = np.median(delta)
    mad = np.median(np.abs(delta - median_delta)) * 1.4826  # MAD to std
    if mad < 1e-10:
        mad = np.std(delta)
    if mad < 1e-10:
        return np.array([], dtype=int)
    z_scores = np.abs(delta - median_delta) / (mad + 1e-10)
    deg_mask = z_scores > 2.0  # ~p<0.05 threshold
    return np.where(deg_mask)[0]


def cal_metric(pred_mean, real_mean):
    mse = mean_squared_error(real_mean, pred_mean)
    mae = mean_absolute_error(real_mean, pred_mean)
    l2 = np.linalg.norm(real_mean - pred_mean)
    return mse, mae, l2


def cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=20, ds_top_k=None, sign_eps=1e-8):
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

    # DS is all-gene sign agreement by default.  A positive ds_top_k is kept
    # only for backward-compatible ad-hoc analyses.
    if ds_top_k is None or ds_top_k <= 0:
        ds_idx = np.arange(delta_real.shape[0])
    else:
        ds_idx = np.argsort(np.abs(delta_real))[-ds_top_k:]
    sign_real = np.where(np.abs(delta_real[ds_idx]) > sign_eps, np.sign(delta_real[ds_idx]), 0)
    sign_pred = np.where(np.abs(delta_pred[ds_idx]) > sign_eps, np.sign(delta_pred[ds_idx]), 0)
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


def _align_pred_var_names(ctrl, target, pred):
    """Keep prediction gene IDs on the same namespace as ctrl/target."""
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


def compute_des(ctrl, target, pred):
    ctrl, target, pred = _align_pred_var_names(ctrl, target, pred)
    combined_real = ctrl.concatenate(
        target, batch_key="condition", batch_categories=["ctrl", "target"]
    )
    sc.tl.rank_genes_groups(
        combined_real, groupby="condition", reference="ctrl", method="t-test"
    )
    real_genes, real_logfc = get_deg_sets(combined_real, group="target")

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
            ctrl_aligned, real_aligned, pred_aligned = _align_pred_var_names(
                ctrl_adata.copy(), real_cond, pred_cond
            )
            d_recall, d_acc, d_spearman = compute_des(ctrl_aligned.copy(), real_aligned, pred_aligned)
            ctrl_mean = np.array(ctrl_aligned.X.mean(axis=0)).flatten()
            real_mean = np.array(real_aligned.X.mean(axis=0)).flatten()
            pred_mean = np.array(pred_aligned.X.mean(axis=0)).flatten()
            c_pearson, c_pearson_top20, c_ds = cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=20)
            c_l2 = np.linalg.norm(real_mean - pred_mean)
            des_results.append({
                "condition": str(cond),
                "des_recall": float(d_recall),
                "des_accuracy": float(d_acc),
                "de_spearman": float(d_spearman) if not np.isnan(d_spearman) else None,
                "condition_pearson_delta": float(c_pearson) if not np.isnan(c_pearson) else None,
                "condition_pearson_delta_top20": float(c_pearson_top20) if not np.isnan(c_pearson_top20) else None,
                "condition_direction_sign_score": float(c_ds),
                "condition_l2": float(c_l2),
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
    condition_delta_avg = float(des_df["condition_pearson_delta"].dropna().mean())
    condition_delta_top20_avg = float(des_df["condition_pearson_delta_top20"].dropna().mean())
    condition_l2_avg = float(des_df["condition_l2"].mean())
    for row in des_results:
        row.setdefault("_summary_condition_pearson_delta_avg", condition_delta_avg)
        row.setdefault("_summary_condition_pearson_delta_top20_avg", condition_delta_top20_avg)
        row.setdefault("_summary_condition_l2_avg", condition_l2_avg)
    return recall_avg, acc_avg, spearman_avg, des_results


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
                                      real_condition_key="condition",
                                      pred_condition_key="perturbation"):
    """Compute R², EV, PCC on DEG genes per condition, return averaged results."""
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
            _, _, cond_ds = cal_delta_metric(ctrl_mean, real_mean, pred_mean, top_k=20)
            n_degs = len(deg_idx)

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
                "n_degs": int(n_degs),
                **deg_metrics,
                "deg_precision": deg_overlap["deg_precision"],
                "deg_recall": deg_overlap["deg_recall"],
                "deg_f1": deg_overlap["deg_f1"],
                "deg_jaccard": deg_overlap["deg_jaccard"],
                "n_pred_degs": deg_overlap["n_pred_degs"],
                "n_overlap_degs": deg_overlap["n_overlap_degs"],
                "de_spearman": de_spearman,
                "condition_ds": float(cond_ds),
            })
        except Exception:
            continue
    return results


def evaluate_predictions(ctrl_adata, real_adata, pred_adata, output_prefix="",
                         real_condition_key="condition",
                         pred_condition_key="perturbation"):
    """Run full evaluation and print/save results."""
    import json

    metrics = {}
    try:
        ctrl_adata, real_adata, pred_adata = _align_pred_var_names(
            ctrl_adata.copy(), real_adata.copy(), pred_adata.copy()
        )
        ctrl_mean = np.array(ctrl_adata.X.mean(axis=0)).flatten()
        real_mean = np.array(real_adata.X.mean(axis=0)).flatten()
        pred_mean = np.array(pred_adata.X.mean(axis=0)).flatten()

        mse, mae, l2 = cal_metric(pred_mean, real_mean)
        pearson_del, pearson_del_top20, _ = cal_delta_metric(
            ctrl_mean, real_mean, pred_mean, top_k=20
        )
        _, pearson_del_top50, _ = cal_delta_metric(
            ctrl_mean, real_mean, pred_mean, top_k=50
        )
        _, pearson_del_top1000, _ = cal_delta_metric(
            ctrl_mean, real_mean, pred_mean, top_k=1000
        )
        print(f"Basic => MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
        print(
            f"Delta => Pearson Δ: {pearson_del:.4f}, Δ20: {pearson_del_top20:.4f}, Δ50: {pearson_del_top50:.4f}, Δ1000: {pearson_del_top1000:.4f}"
        )
        metrics.update(
            {
                "mse": float(mse),
                "mae": float(mae),
                "l2": float(l2),
                "pearson_delta": float(pearson_del),
                "pearson_delta_top20": float(pearson_del_top20),
                "pearson_delta_top50": float(pearson_del_top50),
                "pearson_delta_top1000": float(pearson_del_top1000),
            }
        )

        # Per-condition DEG metrics (R², EV, PCC on DEG genes)
        print("Calculating per-condition DEG metrics (R², EV, PCC)...")
        deg_details = compute_deg_metrics_per_condition(
            ctrl_adata, real_adata, pred_adata,
            real_condition_key=real_condition_key,
            pred_condition_key=pred_condition_key,
        )
        if deg_details:
            import pandas as pd
            deg_df = pd.DataFrame(deg_details)
            avg_r2 = float(deg_df["r2_deg"].mean())
            avg_ev = float(deg_df["ev_deg"].mean())
            avg_pcc = float(deg_df["pcc_deg"].mean())
            avg_ndegs = float(deg_df["n_degs"].mean())
            spearman_valid = deg_df["de_spearman"].dropna()
            avg_de_spearman = float(spearman_valid.mean()) if len(spearman_valid) > 0 else float("nan")
            avg_condition_ds = float(deg_df["condition_ds"].mean())
            print(
                f"Per-condition DEG avg => R²: {avg_r2:.4f}, EV: {avg_ev:.4f}, "
                f"PCC: {avg_pcc:.4f}, condition_DS: {avg_condition_ds:.4f}, "
                f"DE-Spearman: {avg_de_spearman:.4f}, avg #DEGs: {avg_ndegs:.0f}"
            )
            metrics.update(
                {
                    "r2_deg": avg_r2,
                    "ev_deg": avg_ev,
                    "pcc_deg": avg_pcc,
                    "de_spearman": avg_de_spearman,
                    "avg_n_degs": avg_ndegs,
                    "condition_ds": avg_condition_ds,
                    "deg_conditions_count": len(deg_details),
                }
            )

            # DEG overlap averaged from per-condition results
            avg_prec = float(deg_df["deg_precision"].mean())
            avg_rec = float(deg_df["deg_recall"].mean())
            avg_f1 = float(deg_df["deg_f1"].mean())
            avg_jac = float(deg_df["deg_jaccard"].mean())
            avg_pred_degs = float(deg_df["n_pred_degs"].mean())
            avg_overlap = float(deg_df["n_overlap_degs"].mean())
            print(
                f"DEG Overlap => Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, "
                f"F1: {avg_f1:.4f}, Jaccard: {avg_jac:.4f}, "
                f"avg #pred-DEGs: {avg_pred_degs:.0f}, avg #overlap: {avg_overlap:.0f}"
            )
            metrics.update({
                "deg_precision": avg_prec,
                "deg_recall": avg_rec,
                "deg_f1": avg_f1,
                "deg_jaccard": avg_jac,
                "avg_n_pred_degs": avg_pred_degs,
                "avg_n_overlap_degs": avg_overlap,
            })

        # Save per-condition details
        if output_prefix and deg_details:
            deg_file = f"{output_prefix}_deg_per_condition.json"
            with open(deg_file, "w") as f:
                json.dump(deg_details, f, indent=2)
            print(f"Saved per-condition DEG metrics to {deg_file}")

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
