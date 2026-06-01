#!/usr/bin/env python3
"""
CellFlow Baseline on Norman 2019 dataset with holdout split.
Uses PCA 50-dim space (following SCDFM paper A.5 description).
No graph fusion, no combined distribution loss - pure CellFlow.

Holdout setting:
- Hold out 12 single perturbation genes.
- Test on held-out singles and every double perturbation involving held-out genes.
- Keep all control cells in training.
"""
import sys, os, json, random, pickle
import numpy as np, pandas as pd, torch, scanpy as sc, anndata as ad
from pathlib import Path
from datetime import datetime
from scipy import sparse

CELLFLOW_ROOT = Path(__file__).resolve().parent.parent / "CellFlow-main"
sys.path.insert(0, str(CELLFLOW_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions
from split_utils import build_scdfm_norman_split

SEED = 20240508
SPLIT_SEED_BASE = 42
FOLD = 0
SPLIT_METHOD = "holdout"
HOLDOUT_GENES_COUNT = 12
N_PCA_COMPS = 50
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data_train/norman_2019_adata.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_cellflow_baseline_norman_holdout"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")
TRAIN_CELL_FRAC = 0.3
TEST_CELL_FRAC = 0.3

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[CellFlow-Baseline-Norman-Holdout] Start at {timestamp}")
print(f"Split: {SPLIT_METHOD}, fold={FOLD}, seed_base={SPLIT_SEED_BASE}, holdout_genes={HOLDOUT_GENES_COUNT}")
print(f"PCA components: {N_PCA_COMPS}")

# ===================== Data Loading =====================
adata = ad.read_h5ad(ADATA_PATH)
print(f"Raw: {adata.n_obs} x {adata.n_vars}")
adata.obs["target_gene"] = adata.obs["guide_identity"].astype(str)
adata.obs["condition"] = adata.obs["guide_merged"].astype(str)
is_ctrl = adata.obs["condition"] == "ctrl"
adata.obs["is_control"] = is_ctrl
print(f"Control: {is_ctrl.sum()}, Pert: {(~is_ctrl).sum()}")

# HVG
if "highly_variable" in adata.var:
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"After HVG: {adata.n_vars}")

# ===================== scDFM-style Holdout Split =====================
rng = np.random.default_rng(SEED)
control_mask = adata.obs["is_control"].astype(bool)

pert_conditions = (
    adata.obs.loc[~control_mask, "condition"]
    .drop_duplicates().sort_values().tolist()
)
train_conditions, test_conditions, split_info = build_scdfm_norman_split(
    conditions=pert_conditions,
    split_method=SPLIT_METHOD,
    fold=FOLD,
    test_fraction=0.3,
    holdout_genes_count=HOLDOUT_GENES_COUNT,
    seed_base=SPLIT_SEED_BASE,
)

is_train_cond = adata.obs["condition"].isin(train_conditions).to_numpy()
is_test_cond = adata.obs["condition"].isin(test_conditions).to_numpy()

train_mask = control_mask.to_numpy() | ((~control_mask.to_numpy()) & is_train_cond)
test_mask = (~control_mask.to_numpy()) & is_test_cond

adata_train_full = adata[train_mask].copy()
adata_test_holdout = adata[test_mask].copy()

# Subsample
def stratified_sub(adata_sub, frac, rng, key="condition"):
    if frac >= 1: return adata_sub.copy()
    positions = []
    for _, idx in adata_sub.obs.groupby(key, observed=True).indices.items():
        idx = np.asarray(idx)
        n = max(1, int(round(len(idx) * frac)))
        positions.extend(rng.choice(idx, size=n, replace=False).tolist())
    return adata_sub[np.sort(positions)].copy()

adata_train_full = stratified_sub(adata_train_full, TRAIN_CELL_FRAC, rng)
adata_test_holdout = stratified_sub(adata_test_holdout, TEST_CELL_FRAC, rng)

print(f"Train: {adata_train_full.n_obs}, Test: {adata_test_holdout.n_obs}")
print(f"Train conditions: {len(train_conditions)}, Test conditions: {len(test_conditions)}")
print(f"  Train singles/doubles: {split_info['train_single_conditions_count']}/{split_info['train_double_conditions_count']}")
print(f"  Test singles/doubles : {split_info['test_single_conditions_count']}/{split_info['test_double_conditions_count']}")
if "holdout_genes" in split_info:
    print(f"  Holdout genes: {split_info['holdout_genes']}")

# ===================== Normalize & Log Transform =====================
# sc.pp.normalize_total(adata_train_full, target_sum=1e4)
# sc.pp.log1p(adata_train_full)

# sc.pp.normalize_total(adata_test_holdout, target_sum=1e4)
# sc.pp.log1p(adata_test_holdout)

# ===================== PCA Reduction (50-dim) =====================
print(f"Performing PCA reduction to {N_PCA_COMPS} dimensions...")

sc.pp.pca(adata_train_full, n_comps=N_PCA_COMPS, svd_solver='arpack')
pca_components = adata_train_full.varm['PCs']
if sparse.issparse(adata_train_full.X):
    pca_mean = np.array(adata_train_full.X.mean(axis=0)).flatten()
else:
    pca_mean = np.array(adata_train_full.X.mean(axis=0)).flatten()

adata_test_holdout.obsm['X_pca'] = np.array(
    (adata_test_holdout.X - pca_mean) @ pca_components
)

print(f"PCA train: {adata_train_full.obsm['X_pca'].shape}")
print(f"PCA test: {adata_test_holdout.obsm['X_pca'].shape}")

# ===================== CellFlow in PCA Space =====================
from cellflow.model._cellflow import CellFlow

adata_train_pca = ad.AnnData(
    X=adata_train_full.obsm['X_pca'],
    obs=adata_train_full.obs.copy(),
    var=pd.DataFrame(index=[f'PC{i+1}' for i in range(N_PCA_COMPS)]),
)
adata_train_pca.uns = adata_train_full.uns.copy()
adata_train_pca.obs['is_control'] = control_mask.loc[adata_train_full.obs_names].values

unique_perts = sorted(train_conditions)
pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
adata_train_pca.obs['pert_idx'] = adata_train_pca.obs['condition'].map(pert_to_idx).fillna(-1).astype(int)

n_perts = len(unique_perts)
pert_emb = {p: np.eye(n_perts, dtype=np.float32)[i] for p, i in pert_to_idx.items()}
pert_emb['ctrl'] = np.zeros(n_perts, dtype=np.float32)
adata_train_pca.uns['perturbation_embeddings'] = pert_emb

print(f"CellFlow PCA training data: {adata_train_pca.n_obs} cells x {adata_train_pca.n_vars} genes")

cf = CellFlow(adata_train_pca, solver='otfm')

perturbation_covariates = {"perturbation": ["condition"]}
perturbation_reps = {"perturbation": "perturbation_embeddings"}

cf.prepare_data(
    sample_rep="X",
    control_key="is_control",
    perturbation_covariates=perturbation_covariates,
    perturbation_covariate_reps=perturbation_reps,
)

cf.prepare_model(
    seed=SEED,
    condition_encoder_kwargs={
        "x_graph_fusion_kwargs": {"enabled": False},
    },
    solver_kwargs={
        "condition_combined_loss_weight": 0.0,
    },
)

NUM_ITERATIONS = 30000
BATCH_SIZE = 256
print(f"Start training: iterations={NUM_ITERATIONS}, batch_size={BATCH_SIZE}")
cf.train(
    num_iterations=NUM_ITERATIONS,
    batch_size=BATCH_SIZE,
    seed=SEED,
    valid_freq=0,
)
print("Training completed.")

# ===================== Prediction =====================
print("Starting prediction...")
ctrl_adata = adata_train_full[adata_train_full.obs["is_control"]].copy()

all_X, all_obs = [], []
groups = adata_test_holdout.obs.groupby("condition").groups

for condition, idx in groups.items():
    n_cells = len(idx)
    ctrl_idx = rng.choice(ctrl_adata.n_obs, size=n_cells, replace=True)
    sub_ctrl = ctrl_adata[ctrl_idx].copy()

    covariate_data = pd.DataFrame({
        "condition": [condition],
    })

    preds = cf.predict(
        adata=sub_ctrl,
        covariate_data=covariate_data,
        sample_rep="X",
        predict_batch_size=256,
    )
    pred_pca = list(preds.values())[0]

    pred_gene = np.array(pred_pca @ pca_components.T + pca_mean)
    pred_gene = np.expm1(pred_gene)
    pred_gene = np.clip(pred_gene, 0, None)

    all_X.append(pred_gene)
    all_obs.extend([condition] * pred_gene.shape[0])

pred_X = np.vstack(all_X)
adata_pred = ad.AnnData(X=pred_X, obs=pd.DataFrame({"perturbation": all_obs}), var=adata.var.copy())
adata_pred.write_h5ad(OUTPUT_DIR / f"predictions_{timestamp}.h5ad")

# ===================== Evaluation =====================
ctrl_eval = ctrl_adata.copy()
# sc.pp.normalize_total(ctrl_eval, target_sum=1e4)
# sc.pp.log1p(ctrl_eval)

adata_real = adata_test_holdout.copy()

print("\n" + "=" * 50)
evaluate_predictions(ctrl_eval, adata_real, adata_pred, str(OUTPUT_DIR / f"cellflow_baseline_norman_holdout_{timestamp}"))
print("=" * 50)
print("Done.")
