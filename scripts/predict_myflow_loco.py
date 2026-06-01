#!/usr/bin/env python3
"""
Standalone prediction script for MyFlow LOCO experiment.
Loads a trained model and generates predictions using the model's internal adata.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def mean_expr(adata):
    if hasattr(adata.X, "toarray"):
        return np.array(adata.X.mean(axis=0)).flatten()
    return np.array(adata.X.mean(axis=0)).flatten()


def dense_X(adata):
    if hasattr(adata.X, "toarray"):
        return adata.X.toarray()
    return np.array(adata.X)


def cal_metric(pred, real):
    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    l2 = np.linalg.norm(pred - real)
    return mse, mae, l2


def cal_delta_metric(ctrl, real, pred):
    delta_real = real - ctrl
    delta_pred = pred - ctrl
    pearson_del = float(np.corrcoef(delta_real, delta_pred)[0, 1])
    top20_idx = np.argsort(np.abs(delta_real))[-20:]
    pearson_top20 = float(np.corrcoef(delta_real[top20_idx], delta_pred[top20_idx])[0, 1])
    sign_real = np.sign(delta_real)
    sign_pred = np.sign(delta_pred)
    ds = float(np.mean(sign_real == sign_pred))
    return pearson_del, pearson_top20, ds


def compute_des(ctrl_adata, real_adata, pred_adata):
    ctrl_mean = mean_expr(ctrl_adata)
    real_mean = mean_expr(real_adata)
    pred_mean = mean_expr(pred_adata)
    delta_real = real_mean - ctrl_mean
    delta_pred = pred_mean - ctrl_mean
    n_genes = len(delta_real)
    k = max(1, n_genes // 20)
    real_top_idx = set(np.argsort(np.abs(delta_real))[-k:])
    pred_top_idx = set(np.argsort(np.abs(delta_pred))[-k:])
    recall = len(real_top_idx & pred_top_idx) / len(real_top_idx) if real_top_idx else 0
    accuracy = len(real_top_idx & pred_top_idx) / len(pred_top_idx) if pred_top_idx else 0
    spearman = float(scipy.stats.spearmanr(delta_real, delta_pred)[0])
    return recall, accuracy, spearman


def compute_deg_metrics(ctrl_adata, real_adata, pred_adata):
    """Compute R2, EV, PCC on DEGs only."""
    ctrl_mean = mean_expr(ctrl_adata)
    real_mean = mean_expr(real_adata)
    pred_mean = mean_expr(pred_adata)
    delta_real = real_mean - ctrl_mean
    delta_pred = pred_mean - ctrl_mean

    n_genes = len(delta_real)
    k = max(1, n_genes // 20)
    deg_idx = np.argsort(np.abs(delta_real))[-k:]

    real_deg = delta_real[deg_idx]
    pred_deg = delta_pred[deg_idx]

    ss_res = np.sum((real_deg - pred_deg) ** 2)
    ss_tot = np.sum((real_deg - np.mean(real_deg)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    var_real = np.var(real_deg)
    var_res = np.var(real_deg - pred_deg)
    ev = 1 - var_res / var_real if var_real > 0 else float('nan')

    pcc = float(np.corrcoef(real_deg, pred_deg)[0, 1])

    return r2, ev, pcc


def main():
    parser = argparse.ArgumentParser(description="MyFlow LOCO Prediction (no training)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--holdout-cell-line", type=str, default="hepg2")
    parser.add_argument("--control-key", type=str, default="control")
    parser.add_argument("--sample-rep", type=str, default="X")
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--predict-n-cells", type=int, default=64)
    parser.add_argument("--gpu-id", type=str, default="4")
    parser.add_argument("--cross-cell-delta-prior-weight", type=float, default=0.35)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import myflow
    from myflow.model import MyFlow

    run_label = args.run_name or f"myflow_pred_{Path(args.model_path).stem}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================= Load Model =================
    print(f"Loading model from {args.model_path}")
    cf = MyFlow.load(args.model_path)
    if cf.solver is not None:
        cf.solver._cached_predict_fn = None
        cf.solver._cached_predict_kwargs = None
    print("Model loaded successfully")

    # Use the model's internal adata (already has correct gene space)
    adata = cf.adata
    control_key = args.control_key
    holdout = args.holdout_cell_line

    print(f"Model adata: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Holdout cell line: {holdout}")

    # The model's adata is the training set (after validation split)
    # We need to identify:
    # 1. Control cells from the holdout cell line (for prediction baseline)
    # 2. Test perturbation cells from the holdout cell line (for evaluation)
    # 3. Cross-cell delta prior from non-holdout cell lines

    # Check if control_key exists
    if control_key not in adata.obs.columns:
        adata.obs[control_key] = adata.obs["target_gene"].astype(str) == "non-targeting"

    holdout_mask = adata.obs['cell_type'] == holdout
    other_mask = ~holdout_mask

    # Get all perturbation genes in holdout cell line (excluding non-targeting)
    holdout_perts = set(
        adata[holdout_mask].obs.loc[
            adata[holdout_mask].obs["target_gene"].astype(str) != "non-targeting",
            "target_gene"
        ].astype(str).unique()
    )
    other_perts = set(
        adata[other_mask].obs.loc[
            adata[other_mask].obs["target_gene"].astype(str) != "non-targeting",
            "target_gene"
        ].astype(str).unique()
    )

    # Test perturbations = holdout perturbations that appear in other cell lines
    # (same logic as training script)
    test_perts = sorted(holdout_perts & other_perts)
    print(f"Holdout perturbations: {len(holdout_perts)}")
    print(f"Perturbations also in other cell lines: {len(test_perts)}")

    # Check if there's a separate test adata
    # The training script saves adata_test_holdout but the model only has training adata
    # We need to reconstruct the test set from the full data
    # Actually, the model's adata is the training set, so test perturbations in holdout
    # cell line are NOT in the model's adata (they were held out during training)

    # Let's check what perturbations are in the holdout cell line within the model's adata
    holdout_train_perts = set(
        adata[holdout_mask].obs.loc[
            adata[holdout_mask].obs["target_gene"].astype(str) != "non-targeting",
            "target_gene"
        ].astype(str).unique()
    )
    print(f"Holdout perturbations in model's training adata: {len(holdout_train_perts)}")

    # The test perturbations are those NOT in the training adata for holdout cell line
    # but present in other cell lines
    test_only_perts = test_perts  # All test perturbations
    # Those that ARE in holdout training are the "train_perts" from the script
    train_holdout_perts = holdout_train_perts & set(test_perts)
    actual_test_perts = set(test_perts) - train_holdout_perts
    print(f"Train perturbations (in holdout training): {len(train_holdout_perts)}")
    print(f"Test perturbations (held out from holdout): {len(actual_test_perts)}")

    if len(actual_test_perts) == 0:
        print("WARNING: No test perturbations found. Using all test_perts.")
        actual_test_perts = set(test_perts)

    # Get control cells from holdout cell line
    ctrl_mask = holdout_mask & adata.obs[control_key].astype(bool)
    test_adata = adata[ctrl_mask].copy()
    print(f"Control cells in holdout cell line: {test_adata.n_obs}")

    # Build cross-cell delta prior from non-holdout cell lines
    cross_cell_delta_prior = {}
    prior_weight = float(np.clip(args.cross_cell_delta_prior_weight, 0.0, 1.0))
    if prior_weight > 0:
        other_train = adata[other_mask].copy()
        other_ctrl = other_train[other_train.obs[control_key].astype(bool)].copy()
        other_ctrl_mean = mean_expr(other_ctrl)
        delta_genes = sorted(set(test_perts))
        cross_cell_delta_prior["non-targeting"] = np.zeros(adata.n_vars, dtype=np.float32)
        for gene in delta_genes:
            same_pert = other_train[
                (~other_train.obs[control_key].astype(bool))
                & (other_train.obs["target_gene"].astype(str) == gene)
            ].copy()
            if same_pert.n_obs == 0:
                cross_cell_delta_prior[gene] = np.zeros(adata.n_vars, dtype=np.float32)
                continue
            cross_cell_delta_prior[gene] = mean_expr(same_pert) - other_ctrl_mean
        print(f"Cross-cell delta prior: {len(cross_cell_delta_prior)} conditions, weight={prior_weight}")

    # ================= Prediction =================
    genes_to_predict = sorted(actual_test_perts)
    print(f"\nPredicting {len(genes_to_predict)} perturbations...")

    rng = np.random.default_rng(20240508)
    fixed_predict_n = int(args.predict_n_cells)
    all_X = []
    all_obs = []

    for i, gene in enumerate(genes_to_predict):
        print(f"  [{i+1}/{len(genes_to_predict)}] {gene}", flush=True)
        sample_size = min(fixed_predict_n, test_adata.n_obs)
        if sample_size > test_adata.n_obs:
            sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=True)
        else:
            sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=False)
        sub_adata = test_adata[sampled_idx]

        covariate_data = pd.DataFrame({
            "target_gene": [gene],
            "cell_type": [holdout],
            control_key: [False],
        })
        predict_kwargs = {
            "adata": sub_adata,
            "covariate_data": covariate_data,
            "sample_rep": args.sample_rep,
            "predict_batch_size": args.predict_batch_size,
        }
        preds = cf.predict(**predict_kwargs)
        arr = np.asarray(list(preds.values())[0])
        if prior_weight > 0:
            source_expr = dense_X(sub_adata).astype(np.float32, copy=False)
            delta = cross_cell_delta_prior.get(gene)
            if delta is None:
                delta = cross_cell_delta_prior.get("non-targeting", np.zeros(adata.n_vars, dtype=np.float32))
            anchor = np.clip(source_expr + delta[None, :], 0.0, None)
            arr = (1.0 - prior_weight) * arr + prior_weight * anchor
        all_X.append(arr)
        obs = pd.DataFrame({"perturbation": [gene] * arr.shape[0]})
        all_obs.append(obs)

    print("Prediction finished")
    X = np.vstack(all_X)
    X = np.clip(X, 0, None)

    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=test_adata.var.copy())
    pred_dir = Path(args.output_dir) / f"predictions_{run_label}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_file = pred_dir / f"predictions_{run_label}.h5ad"
    adata_pred.write_h5ad(out_file)
    print(f"Saved prediction file: {out_file}")

    # ================= Evaluation =================
    print("\n" + "=" * 50)
    print("Evaluating Predictions...")

    # We need ground truth for the test perturbations
    # Since the model's adata doesn't contain test perturbation responses for holdout cell line,
    # we need to load the original data for evaluation
    # For now, save the predictions and print a summary

    metrics_summary = {
        "run_label": run_label,
        "prediction_file": str(out_file),
        "n_predictions": len(genes_to_predict),
        "genes_predicted": genes_to_predict,
        "success": True,
    }

    write_json(out_dir / f"metrics_summary_{run_label}.json", metrics_summary)
    print(f"Predicted {len(genes_to_predict)} perturbations successfully")
    print(f"Prediction shape: {X.shape}")
    print("=" * 50)
    print("NOTE: To compute evaluation metrics, run the evaluation script separately")
    print("      with the prediction file and the ground truth test adata.")


if __name__ == "__main__":
    main()
