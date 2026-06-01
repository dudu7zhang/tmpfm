#!/usr/bin/env python3
"""
GEARS on Replogle LOCO split.
Matches data setup from train_cellflow_loco_new.py.
"""
import sys
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
from pathlib import Path
from datetime import datetime

# Add GEARS to path
GEARS_ROOT = Path(__file__).resolve().parent.parent / "GEARS-main"
sys.path.insert(0, str(GEARS_ROOT))

from gears import PertData, GEARS

# Import shared eval
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions

SEED = 20240508
HOLDOUT = "hepg2"
TRAIN_FRACTION = 1.0
TEST_FRACTION = 1.0
N_TRAIN_PERTS = 28
N_TEST_PERTS = 40
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data_gab/replogle_gab_merged_hvg.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_gears_loco"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[GEARS-LOCO] Start at {timestamp}")
print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

# ===================== Data Loading & Split (matching CellFlow) =====================
print("Loading data:", ADATA_PATH)
adata = ad.read_h5ad(ADATA_PATH)

# Column mapping - use gene symbols (not Ensembl IDs) for GEARS GO graph compatibility
if "gene" in adata.obs:
    adata.obs["target_gene"] = adata.obs["gene"].astype(str)
elif "gene_id" in adata.obs:
    adata.obs["target_gene"] = adata.obs["gene_id"].astype(str)
if "cell_line" in adata.obs:
    adata.obs["cell_type"] = adata.obs["cell_line"].astype(str)

# HVG filtering
if "highly_variable" in adata.var:
    print(f"Filtering HVG: {adata.n_vars} -> ", end="")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"{adata.n_vars}")

# Filter target genes with gene2vec features (gene2vec file uses Ensembl IDs)
g2v_genes_path = "/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt"
with open(g2v_genes_path, "r") as f:
    g2v_genes = set(line.strip() for line in f.readlines())
# Use gene_id column for filtering since gene2vec has Ensembl IDs
filter_col = adata.obs["gene_id"].astype(str) if "gene_id" in adata.obs else adata.obs["target_gene"]
valid_mask = filter_col.isin(g2v_genes) | (adata.obs["target_gene"] == "non-targeting")
adata = adata[valid_mask].copy()
print(f"After gene2vec filter: {adata.n_obs} cells")

# Control key
adata.obs["control"] = adata.obs["target_gene"] == "non-targeting"

# ===================== LOCO Split =====================
rng = np.random.default_rng(SEED)
holdout_mask = adata.obs["cell_type"] == HOLDOUT
other_mask = ~holdout_mask

# Test perturbations must also exist in training cell lines (matching MyFlow logic)
holdout_targets = {
    str(p) for p in adata[holdout_mask].obs["target_gene"].astype(str).unique()
    if str(p) != "non-targeting"
}
other_targets = {
    str(p) for p in adata[other_mask].obs["target_gene"].astype(str).unique()
    if str(p) != "non-targeting"
}
pert_targets = sorted(holdout_targets & other_targets)
holdout_only = sorted(holdout_targets - other_targets)
if holdout_only:
    print(f"Excluding {len(holdout_only)} holdout-only perturbations: {holdout_only[:5]}...")
if not pert_targets:
    raise ValueError(f"No eligible perturbations for holdout {HOLDOUT}")

shuffled_perts = rng.permutation(pert_targets)
n_train_perts = N_TRAIN_PERTS
n_test_perts = N_TEST_PERTS

if n_train_perts + n_test_perts > len(shuffled_perts):
    raise ValueError(
        f"n_train_perts ({n_train_perts}) + n_test_perts ({n_test_perts}) = "
        f"{n_train_perts + n_test_perts} > {len(shuffled_perts)} eligible perturbations."
    )
train_perts = set(shuffled_perts[:n_train_perts])
test_perts = set(shuffled_perts[-n_test_perts:])
missing_test_in_other = test_perts - other_targets
if missing_test_in_other:
    raise AssertionError(f"Test perturbations missing from non-holdout cell lines: {missing_test_in_other}")

train_mask = (
    other_mask
    | (holdout_mask & adata.obs["target_gene"].isin(train_perts))
    | (holdout_mask & (adata.obs["target_gene"] == "non-targeting"))
)
test_mask = holdout_mask & adata.obs["target_gene"].isin(test_perts)

print(f"LOCO Split ({HOLDOUT}):")
print(f"  Eligible perturbations (in both holdout & other): {len(pert_targets)}")
print(f"  Train perturbations (30%): {len(train_perts)}")
print(f"  Test perturbations (30%): {len(test_perts)}")

adata_train = adata[train_mask].copy()
adata_test = adata[test_mask].copy()

# Subsample
def stratified_subsample(adata_sub, fraction, rng, group_key="target_gene"):
    if fraction >= 1:
        return adata_sub.copy()
    positions = []
    for _, idx in adata_sub.obs.groupby(group_key, observed=True).indices.items():
        idx = np.asarray(idx)
        n_keep = max(1, int(round(len(idx) * fraction)))
        positions.extend(rng.choice(idx, size=n_keep, replace=False).tolist())
    return adata_sub[np.sort(positions)].copy()

adata_train = stratified_subsample(adata_train, TRAIN_FRACTION, rng)
adata_test = stratified_subsample(adata_test, TEST_FRACTION, rng)

print(f"Train: {adata_train.n_obs} cells, Test: {adata_test.n_obs} cells")
train_holdout_perts_seen = set(
    adata_train.obs.loc[
        (adata_train.obs["cell_type"] == HOLDOUT) & (~adata_train.obs["control"]),
        "target_gene",
    ].astype(str)
)
test_perts_after_subsample = set(adata_test.obs["target_gene"].astype(str).unique())
if not test_perts_after_subsample <= other_targets:
    raise AssertionError("Every test perturbation gene must be observed in non-holdout training cell lines.")
if test_perts_after_subsample & train_holdout_perts_seen:
    raise AssertionError("Test perturbation responses leaked into the held-out cell line training subset.")
if not ((adata_train.obs["cell_type"] == HOLDOUT) & adata_train.obs["control"]).any():
    raise AssertionError(f"Training set must include {HOLDOUT} control/basal cells.")

train_pert_count = adata_train[~adata_train.obs["control"]].obs["target_gene"].nunique()
test_pert_count = adata_test.obs["target_gene"].nunique()
print(f"Train perturbation genes: {train_pert_count}")
print(f"Test perturbation genes: {test_pert_count} seen in other cell lines, held out in {HOLDOUT}")

# ===================== Prepare GEARS-format AnnData =====================
# GEARS needs: obs['condition'] with 'ctrl'/'GENE+ctrl' format, obs['cell_type'], var['gene_name']
def prepare_gears_adata(adata_in):
    adata_g = adata_in.copy()
    # GEARS condition format
    def make_condition(row):
        if row["control"]:
            return "ctrl"
        return f"{row['target_gene']}+ctrl"
    adata_g.obs["condition"] = adata_g.obs.apply(make_condition, axis=1)
    adata_g.var["gene_name"] = adata_g.var_names
    # Note: Data is already normalized (log1p) in the input h5ad file.
    return adata_g

adata_train_gears = prepare_gears_adata(adata_train)
adata_test_gears = prepare_gears_adata(adata_test)

# Combine for GEARS processing
adata_combined = ad.concat(
    [adata_train_gears, adata_test_gears],
    join="inner",
    label="split",
    keys=["train", "test"],
)
adata_combined.obs["cell_type"] = pd.concat(
    [adata_train_gears.obs["cell_type"], adata_test_gears.obs["cell_type"]]
).values
adata_combined.obs["condition"] = pd.concat(
    [adata_train_gears.obs["condition"], adata_test_gears.obs["condition"]]
).values
adata_combined.var["gene_name"] = adata_train_gears.var_names.tolist()

# Ensure X is sparse for GEARS
from scipy import sparse
if not sparse.issparse(adata_combined.X):
    adata_combined.X = sparse.csr_matrix(adata_combined.X)

# ===================== GEARS Pipeline =====================
data_path = str(OUTPUT_DIR / "gears_data")
pert_data = PertData(data_path)
pert_data.default_pert_graph = False  # Use genes from data instead of default list to ensure all perturbations are covered
pert_data.new_data_process(dataset_name="replogle_loco", adata=adata_combined, skip_calc_de=False)
pert_data.load(data_path=os.path.join(data_path, "replogle_loco"))

# Remove perturbations from dataset_processed that were filtered out by GO graph
valid_conds = set(pert_data.adata.obs['condition'].unique())
pert_data.dataset_processed = {k: v for k, v in pert_data.dataset_processed.items()
                               if k in valid_conds}

# Custom split: train conditions vs test conditions
train_conditions = sorted(c for c in adata_train_gears.obs["condition"].unique().tolist() if c in valid_conds)
test_conditions = sorted(
    c for c in adata_test_gears.obs["condition"].unique().tolist() if c in valid_conds and c != "ctrl"
)
val_conditions = train_conditions[-max(1, len(train_conditions) // 20) :]

split_dict = {
    "train": [c for c in train_conditions if c not in val_conditions],
    "val": val_conditions,
    "test": test_conditions,
}
import pickle

split_path = OUTPUT_DIR / "custom_split.pkl"
with open(split_path, "wb") as f:
    pickle.dump(split_dict, f)

pert_data.prepare_split(split="custom", seed=SEED, split_dict_path=str(split_path))
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
gears_model = GEARS(pert_data, device=device)
gears_model.model_initialize(hidden_size=64)
gears_model.train(epochs=40, lr=1e-3)

# Save model
model_save_path = str(OUTPUT_DIR / "gears_model")
gears_model.save_model(model_save_path)

# ===================== Prediction =====================
test_pert_list = [[g] for g in sorted(adata_test.obs["target_gene"].unique()) if g != "non-targeting" and g in gears_model.pert_list]
skipped = [g for g in sorted(adata_test.obs["target_gene"].unique()) if g != "non-targeting" and g not in gears_model.pert_list]
if skipped:
    print(f"Skipping {len(skipped)} perturbations not in GEARS graph")
print(f"Predicting {len(test_pert_list)} perturbations...")
preds = gears_model.predict(test_pert_list)

# Build predicted AnnData
ctrl_adata = adata_train[
    (adata_train.obs["control"]) & (adata_train.obs["cell_type"] == HOLDOUT)
].copy()
if ctrl_adata.n_obs == 0:
    raise RuntimeError(f"No {HOLDOUT} control/basal cells found for prediction/evaluation.")

all_X = []
all_obs = []
for pert_name, pred_expr in preds.items():
    n_cells = adata_test[adata_test.obs["target_gene"] == pert_name].n_obs
    if n_cells == 0:
        # Try matching by first gene in combo
        for g in pert_name.split("_"):
            n_cells = adata_test[adata_test.obs["target_gene"] == g].n_obs
            if n_cells > 0:
                break
    n_cells = max(n_cells, 10)
    pred_repeated = np.tile(pred_expr, (n_cells, 1))
    all_X.append(pred_repeated)
    all_obs.extend([pert_name] * n_cells)

pred_X = np.vstack(all_X)
pred_X = np.clip(pred_X, 0, None)

pred_obs = pd.DataFrame({"perturbation": all_obs})
adata_pred = ad.AnnData(X=pred_X, obs=pred_obs, var=adata_train_gears.var.copy())

# Real test data (normalized)
adata_real = adata_test_gears.copy()

# Control (normalized)
ctrl_mean_adata = ctrl_adata.copy()

pred_path = OUTPUT_DIR / f"predictions_{timestamp}.h5ad"
adata_pred.write_h5ad(pred_path)
print(f"Saved predictions: {pred_path}")

# ===================== Evaluation =====================
print("\n" + "=" * 50)
print("Evaluating GEARS-LOCO predictions...")
evaluate_predictions(ctrl_mean_adata, adata_real, adata_pred, str(OUTPUT_DIR / f"gears_loco_{timestamp}"), real_condition_key="target_gene")
print("=" * 50)
print("Done.")
