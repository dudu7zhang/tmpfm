#!/usr/bin/env python3
"""Re-run v2 improved predictions with different ODE step sizes to measure impact on metrics."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import anndata as ad
import pandas as pd
import numpy as np
import diffrax
import jax
import torch
import pickle
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(ROOT / "scripts"))
from train_myflow_loco_new import (
    cal_metric, cal_delta_metric, compute_deg_metrics_per_condition,
    dense_X, set_global_seed, stratified_subsample_obs, write_json,
    load_gene2vec_dict, build_matched_gene2vec_from_dict,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "results/outputs/myflow_replogle_loco_20260531_224754_improved"
CHECKPOINT = MODEL_DIR / "model_myflow_replogle_loco_20260531_224754_improved/MyFlow.pkl"
ADATA_PATH = ROOT / "data_gab/replogle_gab_merged_hvg.h5ad"
CONFIG_PATH = MODEL_DIR / "experiment_config_myflow_replogle_loco_20260531_224754_improved.json"
PRED_DIR = MODEL_DIR / "predictions_myflow_replogle_loco_20260531_224754_improved"

print(f"Loading config from {CONFIG_PATH}")
with open(CONFIG_PATH) as f:
    config = json.load(f)
args_dict = config["args"]
print(f"Config seed={args_dict.get('seed', 20240508)}, holdout={args_dict.get('holdout_cell_line', 'hepg2')}")

seed = int(args_dict.get("seed", 20240508))
set_global_seed(seed)
rng = np.random.default_rng(seed)

print(f"Loading data from {ADATA_PATH}")
adata = ad.read_h5ad(str(ADATA_PATH))
adata.obs["target_gene"] = adata.obs["gene"].astype(str)
if "cell_line" in adata.obs:
    adata.obs["cell_type"] = adata.obs["cell_line"].astype(str)

if "highly_variable" in adata.var:
    adata = adata[:, adata.var["highly_variable"]].copy()
print(f"Data shape: {adata.n_obs} cells, {adata.n_vars} genes")

holdout = args_dict.get("holdout_cell_line", "hepg2")

# Replicate adata.uns setup from training script
emb = load_gene2vec_dict(ROOT / "data_gab/gene2vec_dict.pt")
emb_dict = dict(emb)
for k, v in list(emb_dict.items()):
    if torch.is_tensor(v):
        emb_dict[k] = v.cpu().numpy()
    else:
        emb_dict[k] = np.asarray(v)
adata.uns["gene2vec_symbol_features"] = emb_dict

cell_lines = ['hepg2', 'jurkat', 'k562', 'rpe1']
ct_emb_dict = {}
for i, cl in enumerate(cell_lines):
    e = np.zeros(len(cell_lines), dtype=np.float32)
    e[i] = 1.0
    ct_emb_dict[cl] = e
adata.uns["cell_type_embeddings"] = ct_emb_dict

# cross_cell_delta
other_mask_simple = adata.obs["cell_type"] != holdout
other_train = adata[other_mask_simple].copy()
other_ctrl = other_train[other_train.obs["target_gene"].astype(str) == "non-targeting"].copy()
if other_ctrl.n_obs == 0:
    ctrl_mask = adata.obs["control"] if "control" in adata.obs else (adata.obs["target_gene"].astype(str) == "non-targeting")
    other_ctrl = other_train[ctrl_mask[other_mask_simple].values].copy()
other_ctrl_mean = np.array(other_ctrl.X.mean(axis=0)).flatten()
delta_genes = set(adata.obs["target_gene"].astype(str).unique()) - {"non-targeting"}
cross_cell_delta = {"non-targeting": np.zeros(adata.n_vars, dtype=np.float32)}
for gene in delta_genes:
    same_pert = other_train[other_train.obs["target_gene"].astype(str) == gene].copy()
    if same_pert.n_obs == 0:
        cross_cell_delta[gene] = np.zeros(adata.n_vars, dtype=np.float32)
    else:
        cross_cell_delta[gene] = np.array(same_pert.X.mean(axis=0)).flatten() - other_ctrl_mean
adata.uns["cross_cell_delta_embeddings"] = cross_cell_delta

# perturb_gene_symbol_to_idx for combined encoding
perturb_genes = sorted(set(adata.obs["target_gene"].astype(str).unique()) - {"non-targeting"})
adata.uns["perturb_gene_symbol_to_idx"] = {s: i for i, s in enumerate(perturb_genes)}

# Combined perturb indices
output_genes = [str(g) for g in adata.var_names]
combined_genes = list(dict.fromkeys(output_genes + [g.upper() for g in perturb_genes]))
_combined_sym_to_idx = {s: i for i, s in enumerate(combined_genes)}
adata.uns["combined_perturb_symbol_to_idx"] = {
    s: _combined_sym_to_idx[s] for s in perturb_genes if s in _combined_sym_to_idx
}

n_train_perts = int(args_dict.get("n_train_perts", 28))
n_test_perts = int(args_dict.get("n_test_perts", 40))

other_mask = adata.obs["cell_type"] != holdout
holdout_mask = adata.obs["cell_type"] == holdout
holdout_targets = {
    str(p) for p in adata[holdout_mask].obs["target_gene"].astype(str).unique().tolist()
    if str(p) != "non-targeting"
}
other_targets = {
    str(p) for p in adata[other_mask].obs["target_gene"].astype(str).unique().tolist()
    if str(p) != "non-targeting"
}
pert_targets = sorted(holdout_targets & other_targets)
shuffled_perts = rng.permutation(pert_targets)
train_perts = set(shuffled_perts[:n_train_perts])
test_perts = set(shuffled_perts[-n_test_perts:])

train_mask = other_mask | (holdout_mask & adata.obs["target_gene"].isin(train_perts)) | (holdout_mask & (adata.obs["target_gene"] == "non-targeting"))
test_mask = holdout_mask & adata.obs["target_gene"].isin(test_perts)
adata_train_full = adata[train_mask].copy()
adata_test_holdout = adata[test_mask].copy()

# Load model
print(f"Loading model from {CHECKPOINT}")
with open(CHECKPOINT, "rb") as f:
    myflow_model = pickle.load(f)

# Test control cells
test_adata = adata_train_full[
    (adata_train_full.obs["cell_type"] == holdout) & (adata_train_full.obs.get("control", False))
].copy()
if test_adata.n_obs == 0:
    test_adata = adata_train_full[
        (adata_train_full.obs["cell_type"] == holdout) & (adata_train_full.obs["target_gene"] == "non-targeting")
    ].copy()
test_adata.obs["control"] = True

# Ensure uns keys propagate through subset copies
for key in ["gene2vec_symbol_features", "cell_type_embeddings", "cross_cell_delta_embeddings",
            "perturb_gene_symbol_to_idx", "combined_perturb_symbol_to_idx"]:
    if key not in test_adata.uns and key in adata.uns:
        test_adata.uns[key] = adata.uns[key]
print(f"Test control cells: {test_adata.n_obs}")
print(f"test_adata.uns keys: {list(test_adata.uns.keys())}")

# Predict and evaluate for each step config
step_configs = [
    ("euler_1step", diffrax.Euler(), 1.0),
    ("euler_5step", diffrax.Euler(), 0.2),
    ("euler_20step", diffrax.Euler(), 0.05),
    ("tsit5_adaptive", diffrax.Tsit5(), None),
]

groups = adata_test_holdout.obs.groupby("target_gene").groups
all_results = {}

for label, solver, dt0 in step_configs:
    print(f"\n{'='*60}")
    print(f"PREDICTING: {label} (solver={type(solver).__name__}, dt0={dt0})")
    print(f"{'='*60}")

    all_X = []
    all_obs = []
    fixed_predict_n = 64

    for gene in sorted(groups.keys()):
        idx = groups[gene]
        sample_size = fixed_predict_n
        if sample_size > test_adata.n_obs:
            sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=True)
        else:
            sampled_idx = rng.choice(test_adata.n_obs, size=sample_size, replace=False)
        sub_adata = test_adata[sampled_idx]

        covariate_data = pd.DataFrame({
            "target_gene": [gene],
            "cell_type": [holdout],
            "control": [False],
        })

        pred_kwargs = {
            "adata": sub_adata,
            "covariate_data": covariate_data,
            "sample_rep": "X",
            "predict_batch_size": 256,
            "solver": solver,
        }
        if dt0 is not None:
            pred_kwargs["dt0"] = dt0
            pred_kwargs["stepsize_controller"] = diffrax.ConstantStepSize()
        else:
            pred_kwargs["stepsize_controller"] = diffrax.PIDController(rtol=1e-5, atol=1e-5)

        preds = myflow_model.predict(**pred_kwargs)
        arr = np.asarray(list(preds.values())[0])
        all_X.append(arr)
        obs = pd.DataFrame({"perturbation": [gene] * arr.shape[0]})
        all_obs.append(obs)

    X = np.clip(np.vstack(all_X), 0, None)
    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=test_adata.var.copy())

    # Evaluate
    print(f"\nEVALUATING: {label}")
    ctrl_mean = np.array(test_adata.X.mean(axis=0)).flatten()
    real_mean = np.array(adata_test_holdout.X.mean(axis=0)).flatten()
    ours_mean = np.array(adata_pred.X.mean(axis=0)).flatten()

    mse, mae, l2 = cal_metric(ours_mean, real_mean)
    pearson_del, pearson_del_top20, _ = cal_delta_metric(ctrl_mean, real_mean, ours_mean)

    print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, L2: {l2:.4f}")
    print(f"  Pearson Δ: {pearson_del:.4f}, Pearson Δ20: {pearson_del_top20:.4f}")

    deg_details = compute_deg_metrics_per_condition(
        test_adata, adata_test_holdout, adata_pred,
        real_condition_key="target_gene",
        pred_condition_key="perturbation",
    )

    result = {
        "mse": float(mse),
        "mae": float(mae),
        "l2": float(l2),
        "pearson_delta": float(pearson_del),
        "pearson_delta_top20": float(pearson_del_top20),
    }

    if deg_details:
        deg_df = pd.DataFrame(deg_details)
        result["r2_deg"] = float(deg_df["r2_deg"].mean())
        result["pcc_deg"] = float(deg_df["pcc_deg"].mean())
        result["ev_deg"] = float(deg_df["ev_deg"].mean())
        result["de_spearman"] = float(deg_df["de_spearman"].dropna().mean())
        if "condition_ds" in deg_df:
            result["condition_ds"] = float(deg_df["condition_ds"].mean())
        result["n_degs"] = float(deg_df["n_degs"].mean())
        result["deg_conditions_count"] = len(deg_details)

    all_results[label] = result

    print(f"  PCC: {result.get('pcc_deg', 'N/A'):.4f}, R²: {result.get('r2_deg', 'N/A'):.4f}, DE-Spearman: {result.get('de_spearman', 'N/A'):.4f}")

print(f"\n{'='*60}")
print("SUMMARY: All step sizes compared")
print(f"{'='*60}")
print(f"{'Config':<20} {'MSE':<10} {'PCC':<10} {'Δ20':<10} {'Δ':<10} {'cDS':<8} {'DE-Sp':<8}")
print("-"*70)
for label, result in all_results.items():
    print(f"{label:<20} {result['mse']:<10.6f} {result.get('pcc_deg', 0):<10.4f} {result['pearson_delta_top20']:<10.4f} {result['pearson_delta']:<10.4f} {result.get('condition_ds', 0):<8.2f} {result.get('de_spearman', 0):<8.4f}")

out_path = MODEL_DIR / "step_size_comparison.json"
write_json(out_path, all_results)
print(f"\nResults saved to {out_path}")
