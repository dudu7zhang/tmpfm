#!/usr/bin/env python3
"""
Squidiff on Norman 2019 dataset.
Uses scDFM-style additive split (30% double perturbations as test, all singles in train).
Supports perturbation conditioning via class labels.
"""
import sys, os, random
import numpy as np, pandas as pd, torch, scanpy as sc, anndata as ad
from pathlib import Path
from datetime import datetime

SQUIDIFF_ROOT = Path(__file__).resolve().parent.parent / "Squidiff-offical"
sys.path.insert(0, str(SQUIDIFF_ROOT))

from Squidiff.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
from Squidiff.scrna_datasets import prepared_data
from Squidiff.train_util import TrainLoop
from Squidiff.resample import create_named_schedule_sampler
import Squidiff.dist_util as dist_util
import sample_squidiff

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions
from split_utils import build_scdfm_norman_split

SEED = 20240508
SPLIT_SEED_BASE = 42
FOLD = 0
TEST_COND_FRAC = 0.3
SPLIT_METHOD = "additive"
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data/norman_2019_adata.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_squidiff_norman"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")
TRAIN_CELL_FRAC = 0.3
TEST_CELL_FRAC = 0.3

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[Squidiff-Norman] Start at {timestamp}")
print(f"Split: {SPLIT_METHOD}, fold={FOLD}, seed_base={SPLIT_SEED_BASE}")

# ===================== Data Loading =====================
adata = ad.read_h5ad(ADATA_PATH)
adata.obs["target_gene"] = adata.obs["guide_identity"].astype(str)
adata.obs["condition"] = adata.obs["guide_merged"].astype(str)
is_ctrl = adata.obs["condition"] == "ctrl"
adata.obs["is_control"] = is_ctrl
adata.obs["control"] = is_ctrl.astype(int)

if "highly_variable" in adata.var:
    adata = adata[:, adata.var["highly_variable"]].copy()

# ===================== scDFM-style Split =====================
rng = np.random.default_rng(SEED)
control_mask = adata.obs["is_control"].astype(bool)
pert_conditions = adata.obs.loc[~control_mask, "condition"].drop_duplicates().sort_values().tolist()
train_conditions, test_conditions, split_info = build_scdfm_norman_split(
    conditions=pert_conditions, split_method=SPLIT_METHOD, fold=FOLD,
    test_fraction=TEST_COND_FRAC, seed_base=SPLIT_SEED_BASE,
)

is_train = adata.obs["condition"].isin(train_conditions).to_numpy()
is_test = adata.obs["condition"].isin(test_conditions).to_numpy()
# scDFM: both splits include control cells
train_mask = control_mask.to_numpy() | ((~control_mask.to_numpy()) & is_train)
test_mask = (~control_mask.to_numpy()) & is_test

adata_train = adata[train_mask].copy()
adata_test = adata[test_mask].copy()

def stratified_sub(adata_sub, frac, rng, key="condition"):
    if frac >= 1: return adata_sub.copy()
    positions = []
    for _, idx in adata_sub.obs.groupby(key, observed=True).indices.items():
        idx = np.asarray(idx)
        n = max(1, int(round(len(idx) * frac)))
        positions.extend(rng.choice(idx, size=n, replace=False).tolist())
    return adata_sub[np.sort(positions)].copy()

adata_train = stratified_sub(adata_train, TRAIN_CELL_FRAC, rng)
adata_test = stratified_sub(adata_test, TEST_CELL_FRAC, rng)
print(f"Train: {adata_train.n_obs}, Test: {adata_test.n_obs}, Genes: {adata_train.n_vars}")

# ===================== Perturbation Label Mapping =====================
# Create mapping from perturbation condition to integer index
all_conditions = sorted(set(adata_train.obs["condition"].unique().tolist() +
                            adata_test.obs["condition"].unique().tolist()))
cond_to_idx = {cond: idx for idx, cond in enumerate(all_conditions)}
NUM_CLASSES = len(all_conditions)
print(f"Number of perturbation classes: {NUM_CLASSES}")
print(f"Conditions: {all_conditions[:10]}...")  # Show first 10

# Save mapping for later use
import json
mapping_path = OUTPUT_DIR / "perturbation_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(cond_to_idx, f, indent=2)

# ===================== Prepare Squidiff Format =====================
sc.pp.normalize_total(adata_train); sc.pp.log1p(adata_train)
sc.pp.normalize_total(adata_test); sc.pp.log1p(adata_test)

# Convert condition strings to integer indices for class conditioning
adata_train.obs["Group"] = adata_train.obs["condition"].map(cond_to_idx).astype(int)
adata_test.obs["Group"] = adata_test.obs["condition"].map(cond_to_idx).astype(int)

train_h5ad_path = OUTPUT_DIR / "squidiff_train.h5ad"
adata_train.write_h5ad(train_h5ad_path)
gene_size = adata_train.n_vars

# ===================== Train =====================
logger_path = str(OUTPUT_DIR / "logger")
checkpoint_path = str(OUTPUT_DIR / "checkpoint")
os.makedirs(logger_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

args = {}
args.update(model_and_diffusion_defaults())
args.update({
    "data_path": str(train_h5ad_path), "control_data_path": "",
    "schedule_sampler": "uniform", "lr": 1e-4, "weight_decay": 0.0,
    "lr_anneal_steps": 30000, "batch_size": 64, "microbatch": -1,
    "ema_rate": "0.9999", "log_interval": 5000, "save_interval": 5000,
    "resume_checkpoint": checkpoint_path, "use_fp16": False,
    "fp16_scale_growth": 1e-3, "gene_size": gene_size, "output_dim": gene_size,
    "num_layers": 3, "class_cond": True, "num_classes": NUM_CLASSES,
    "use_encoder": True,
    "diffusion_steps": 1000, "logger_path": logger_path,
    "use_drug_structure": False, "comb_num": 1, "use_ddim": True,
})

print("Training Squidiff with perturbation conditioning...")
model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
model.to(dist_util.dev())
schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], diffusion)
data = prepared_data(data_dir=args["data_path"], control_data_dir=None, batch_size=64, use_drug_structure=False, comb_num=1)

train_loop = TrainLoop(
    model=model, diffusion=diffusion, data=data,
    batch_size=64, microbatch=-1, lr=1e-4,
    ema_rate="0.9999", log_interval=5000, save_interval=5000,
    resume_checkpoint=checkpoint_path, use_fp16=False,
    fp16_scale_growth=1e-3, schedule_sampler=schedule_sampler,
    weight_decay=0.0, lr_anneal_steps=30000, use_drug_structure=False, comb_num=1,
)
train_loop.run_loop()

# ===================== Prediction =====================
model_path = os.path.join(checkpoint_path, "model.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(checkpoint_path, "model0.9999.pt")

sampler = sample_squidiff.sampler(model_path=model_path, gene_size=gene_size, output_dim=gene_size, use_drug_structure=False, class_cond=True, num_classes=NUM_CLASSES)

ctrl_adata = adata_train[adata_train.obs["is_control"]].copy()
test_conditions_list = sorted(test_conditions)

all_X, all_obs = [], []
for cond in test_conditions_list:
    ctrl_X = ctrl_adata.X
    if hasattr(ctrl_X, "toarray"): ctrl_X = ctrl_X.toarray()
    ctrl_tensor = torch.tensor(ctrl_X, dtype=torch.float32).to("cuda")

    # Get perturbation label index
    cond_idx = cond_to_idx.get(cond, 0)
    pert_label = torch.full((ctrl_tensor.shape[0],), cond_idx, dtype=torch.long, device="cuda")

    # Encode with perturbation conditioning
    with torch.no_grad():
        z_sem = sampler.encode_with_perturbation(ctrl_tensor, pert_label)

    n_test_cells = (adata_test.obs["condition"] == cond).sum()
    n_pred = min(n_test_cells, z_sem.shape[0])
    pred = sampler.pred(z_sem[:n_pred], gene_size=gene_size)
    pred_np = np.clip(pred.cpu().numpy(), 0, None)
    all_X.append(pred_np)
    all_obs.extend([cond] * n_pred)

pred_X = np.vstack(all_X)
adata_pred = ad.AnnData(X=pred_X, obs=pd.DataFrame({"perturbation": all_obs}), var=adata_train.var.copy())
adata_pred.write_h5ad(OUTPUT_DIR / f"predictions_{timestamp}.h5ad")

# ===================== Evaluation =====================
print("\n" + "=" * 50)
evaluate_predictions(ctrl_adata, adata_test, adata_pred, str(OUTPUT_DIR / f"squidiff_norman_{timestamp}"))
print("=" * 50)
print("Done.")
