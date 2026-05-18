#!/usr/bin/env python3
"""
GEARS on Norman 2019 dataset.
Uses scDFM-style additive split (30% double perturbations as test, all singles in train).
Matches data setup from train_cellflow_norman_scdfm.py.
"""
import sys, os, json, random, pickle
import numpy as np, pandas as pd, torch, scanpy as sc, anndata as ad
from pathlib import Path
from datetime import datetime

GEARS_ROOT = Path(__file__).resolve().parent.parent / "GEARS-backup"
sys.path.insert(0, str(GEARS_ROOT))
from gears import PertData, GEARS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions
from split_utils import build_scdfm_norman_split

SEED = 20240508
SPLIT_SEED_BASE = 42
FOLD = 0
TEST_COND_FRAC = 0.3
SPLIT_METHOD = "additive"
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data_train/norman_2019_adata.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_gears_norman_additive"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")
TRAIN_CELL_FRAC = 0.3
TEST_CELL_FRAC = 0.3

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[GEARS-Norman-Additive] Start at {timestamp}")
print(f"Split: {SPLIT_METHOD}, fold={FOLD}, seed_base={SPLIT_SEED_BASE}")

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

# ===================== scDFM-style Split =====================
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
    test_fraction=TEST_COND_FRAC,
    seed_base=SPLIT_SEED_BASE,
)

is_train_cond = adata.obs["condition"].isin(train_conditions).to_numpy()
is_test_cond = adata.obs["condition"].isin(test_conditions).to_numpy()

# Train uses controls plus train perturbations; zero-shot evaluation uses test perturbations only.
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

# ===================== Prepare GEARS Format =====================
def make_gears_condition(row):
    if row["is_control"]: return "ctrl"
    genes = [g.strip() for g in str(row["condition"]).split("+") if g.strip() and g.strip().lower() != "ctrl"]
    if len(genes) == 0: return "ctrl"
    if len(genes) == 1: return f"{genes[0]}+ctrl"
    return "+".join(sorted(genes))

adata_train_full.obs["condition_gears"] = adata_train_full.obs.apply(make_gears_condition, axis=1)
adata_test_holdout.obs["condition_gears"] = adata_test_holdout.obs.apply(make_gears_condition, axis=1)

adata_train_g = adata_train_full.copy()
sc.pp.normalize_total(adata_train_g); sc.pp.log1p(adata_train_g)
adata_test_g = adata_test_holdout.copy()
sc.pp.normalize_total(adata_test_g); sc.pp.log1p(adata_test_g)

adata_train_g.obs["condition"] = adata_train_g.obs["condition_gears"]
adata_train_g.obs["cell_type"] = "K562"
adata_train_g.var["gene_name"] = adata_train_g.var_names

adata_test_g.obs["condition"] = adata_test_g.obs["condition_gears"]
adata_test_g.obs["cell_type"] = "K562"
adata_test_g.var["gene_name"] = adata_test_g.var_names

adata_combined = ad.concat([adata_train_g, adata_test_g], join="inner", label="split", keys=["train", "test"])
adata_combined.obs["cell_type"] = pd.concat([adata_train_g.obs["cell_type"], adata_test_g.obs["cell_type"]]).values
adata_combined.obs["condition"] = pd.concat([adata_train_g.obs["condition"], adata_test_g.obs["condition"]]).values
adata_combined.var["gene_name"] = adata_train_g.var_names.tolist()

# Ensure X is sparse for GEARS
from scipy import sparse
if not sparse.issparse(adata_combined.X):
    adata_combined.X = sparse.csr_matrix(adata_combined.X)

# ===================== GEARS Pipeline =====================
data_path = str(OUTPUT_DIR / "gears_data")
pert_data = PertData(data_path)
pert_data.new_data_process(dataset_name="norman_gears", adata=adata_combined, skip_calc_de=False)
pert_data.load(data_path=os.path.join(data_path, "norman_gears"))

# Remove perturbations from dataset_processed that were filtered out by GO graph
valid_conds = set(pert_data.adata.obs['condition'].unique())
pert_data.dataset_processed = {k: v for k, v in pert_data.dataset_processed.items()
                               if k in valid_conds}

train_conds_gears = sorted(c for c in adata_train_g.obs["condition"].unique().tolist() if c in valid_conds)
test_conds_gears = sorted(c for c in adata_test_g.obs["condition"].unique().tolist() if c in valid_conds and c != "ctrl")
val_conds_gears = train_conds_gears[-max(1, len(train_conds_gears) // 20):]

split_dict = {
    "train": [c for c in train_conds_gears if c not in val_conds_gears],
    "val": val_conds_gears,
    "test": test_conds_gears,
}
split_path = OUTPUT_DIR / "custom_split.pkl"
with open(split_path, "wb") as f:
    pickle.dump(split_dict, f)

pert_data.prepare_split(split="custom", seed=SEED, split_dict_path=str(split_path))
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

device = "cuda" if torch.cuda.is_available() else "cpu"
gears_model = GEARS(pert_data, device=device)
gears_model.model_initialize(hidden_size=64)
gears_model.train(epochs=20, lr=1e-3)
gears_model.save_model(str(OUTPUT_DIR / "gears_model"))

# ===================== Prediction =====================
test_pert_list = []
for cond in sorted(test_conditions):
    genes = [g.strip() for g in cond.split("+") if g.strip() and g.strip().lower() != "ctrl"]
    if genes:
        test_pert_list.append(genes)

print(f"Predicting {len(test_pert_list)} perturbations...")
preds = gears_model.predict(test_pert_list)

ctrl_adata = adata_train_full[adata_train_full.obs["is_control"]].copy()
sc.pp.normalize_total(ctrl_adata); sc.pp.log1p(ctrl_adata)

all_X, all_obs = [], []
for pert_name, pred_expr in preds.items():
    # Map GEARS pert name back to original condition
    orig_conds = [c for c in test_conditions if pert_name.replace("_", "+") == c or pert_name == c]
    if not orig_conds:
        # Try partial match
        for c in test_conditions:
            c_genes = sorted([g.strip() for g in c.split("+") if g.strip().lower() != "ctrl"])
            if "+".join(c_genes) == pert_name.replace("_", "+"):
                orig_conds = [c]
                break
    if not orig_conds:
        orig_conds = [pert_name]
    for oc in orig_conds:
        n_cells = (adata_test_holdout.obs["condition"] == oc).sum()
        if n_cells > 0:
            all_X.append(np.tile(pred_expr, (n_cells, 1)))
            all_obs.extend([oc] * n_cells)
    if not any((adata_test_holdout.obs["condition"] == oc).sum() > 0 for oc in orig_conds):
        all_X.append(np.tile(pred_expr, (10, 1)))
        all_obs.extend([pert_name] * 10)

pred_X = np.vstack(all_X)
adata_pred = ad.AnnData(X=pred_X, obs=pd.DataFrame({"perturbation": all_obs}), var=adata_train_g.var.copy())
adata_pred.write_h5ad(OUTPUT_DIR / f"predictions_{timestamp}.h5ad")

# ===================== Evaluation =====================
print("\n" + "=" * 50)
evaluate_predictions(ctrl_adata, adata_test_g, adata_pred, str(OUTPUT_DIR / f"gears_norman_additive_{timestamp}"))
print("=" * 50)
print("Done.")
