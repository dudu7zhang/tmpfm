#!/usr/bin/env python3
"""
Squidiff on Replogle LOCO split.
Matches data setup from train_cellflow_loco_new.py.
Supports perturbation conditioning via class labels.
"""
import sys, os, random, json
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

SEED = 20240508
HOLDOUT = "hepg2"
TRAIN_FRACTION = 0.3
TEST_FRACTION = 0.3
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_squidiff_loco"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[Squidiff-LOCO] Start at {timestamp}")

# ===================== Data Loading & Split =====================
adata = ad.read_h5ad(ADATA_PATH)
if "gene_id" in adata.obs: adata.obs["target_gene"] = adata.obs["gene_id"].astype(str)
elif "gene" in adata.obs: adata.obs["target_gene"] = adata.obs["gene"].astype(str)
if "cell_line" in adata.obs: adata.obs["cell_type"] = adata.obs["cell_line"].astype(str)

if "highly_variable" in adata.var:
    adata = adata[:, adata.var["highly_variable"]].copy()

g2v_genes_path = "/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt"
with open(g2v_genes_path) as f:
    g2v_genes = set(line.strip() for line in f)
valid_mask = adata.obs["target_gene"].isin(g2v_genes) | (adata.obs["target_gene"] == "non-targeting")
adata = adata[valid_mask].copy()

adata.obs["control"] = adata.obs["target_gene"] == "non-targeting"

# LOCO split
rng = np.random.default_rng(SEED)
holdout_mask = adata.obs["cell_type"] == HOLDOUT
other_mask = ~holdout_mask

perts = sorted(adata[holdout_mask].obs["target_gene"].unique().tolist())
pert_targets = [p for p in perts if p != "non-targeting"]
shuffled = rng.permutation(pert_targets)
n_train_perts = int(TRAIN_FRACTION * len(shuffled))
n_test_perts = int(TEST_FRACTION * len(shuffled))
train_perts = set(shuffled[:n_train_perts])
test_perts = set(shuffled[-n_test_perts:])

train_mask = other_mask | (holdout_mask & adata.obs["target_gene"].isin(train_perts)) | (holdout_mask & (adata.obs["target_gene"] == "non-targeting"))
test_mask = holdout_mask & adata.obs["target_gene"].isin(test_perts)

adata_train = adata[train_mask].copy()
adata_test = adata[test_mask].copy()

def stratified_sub(adata_sub, frac, rng, key="target_gene"):
    if frac >= 1: return adata_sub.copy()
    positions = []
    for _, idx in adata_sub.obs.groupby(key, observed=True).indices.items():
        idx = np.asarray(idx)
        n = max(1, int(round(len(idx) * frac)))
        positions.extend(rng.choice(idx, size=n, replace=False).tolist())
    return adata_sub[np.sort(positions)].copy()

adata_train = stratified_sub(adata_train, TRAIN_FRACTION, rng)
adata_test = stratified_sub(adata_test, TEST_FRACTION, rng)

print(f"Train: {adata_train.n_obs}, Test: {adata_test.n_obs}, Genes: {adata_train.n_vars}")

# ===================== Perturbation Label Mapping =====================
all_conditions = sorted(set(adata_train.obs["target_gene"].unique().tolist() +
                            adata_test.obs["target_gene"].unique().tolist()))
cond_to_idx = {cond: idx for idx, cond in enumerate(all_conditions)}
NUM_CLASSES = len(all_conditions)
print(f"Number of perturbation classes: {NUM_CLASSES}")

mapping_path = OUTPUT_DIR / "perturbation_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(cond_to_idx, f, indent=2)

# ===================== Prepare Squidiff Format =====================
sc.pp.normalize_total(adata_train); sc.pp.log1p(adata_train)
sc.pp.normalize_total(adata_test); sc.pp.log1p(adata_test)

adata_train.obs["Group"] = adata_train.obs["target_gene"].map(cond_to_idx).astype(int)
adata_test.obs["Group"] = adata_test.obs["target_gene"].map(cond_to_idx).astype(int)

train_h5ad_path = OUTPUT_DIR / "squidiff_train.h5ad"
adata_train.write_h5ad(train_h5ad_path)

gene_size = adata_train.n_vars
print(f"Gene size: {gene_size}")

# ===================== Train Squidiff =====================
logger_path = str(OUTPUT_DIR / "logger")
checkpoint_path = str(OUTPUT_DIR / "checkpoint")
os.makedirs(logger_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

lr_anneal_steps = 30000
batch_size = 64

args = {}
args.update(model_and_diffusion_defaults())
args.update({
    "data_path": str(train_h5ad_path),
    "control_data_path": "",
    "schedule_sampler": "uniform",
    "lr": 1e-4,
    "weight_decay": 0.0,
    "lr_anneal_steps": lr_anneal_steps,
    "batch_size": batch_size,
    "microbatch": -1,
    "ema_rate": "0.9999",
    "log_interval": 5000,
    "save_interval": 5000,
    "resume_checkpoint": checkpoint_path,
    "use_fp16": False,
    "fp16_scale_growth": 1e-3,
    "gene_size": gene_size,
    "output_dim": gene_size,
    "num_layers": 3,
    "class_cond": True,
    "num_classes": NUM_CLASSES,
    "use_encoder": True,
    "diffusion_steps": 1000,
    "logger_path": logger_path,
    "use_drug_structure": False,
    "comb_num": 1,
    "use_ddim": True,
})

print("Creating model and diffusion...")
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.to(dist_util.dev())
schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], diffusion)

print("Creating data loader...")
data = prepared_data(
    data_dir=args["data_path"],
    control_data_dir=args["control_data_path"] if args["control_data_path"] else None,
    batch_size=args["batch_size"],
    use_drug_structure=args["use_drug_structure"],
    comb_num=args["comb_num"],
)

print("Training Squidiff...")
train_loop = TrainLoop(
    model=model, diffusion=diffusion, data=data,
    batch_size=batch_size, microbatch=-1, lr=1e-4,
    ema_rate="0.9999", log_interval=5000, save_interval=5000,
    resume_checkpoint=checkpoint_path,
    use_fp16=False, fp16_scale_growth=1e-3,
    schedule_sampler=schedule_sampler,
    weight_decay=0.0, lr_anneal_steps=lr_anneal_steps,
    use_drug_structure=False, comb_num=1,
)
train_loop.run_loop()
print("Training done.")

# ===================== Prediction =====================
model_path = os.path.join(checkpoint_path, "model.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(checkpoint_path, "model0.9999.pt")

print(f"Loading model from {model_path}")
sampler = sample_squidiff.sampler(
    model_path=model_path,
    gene_size=gene_size,
    output_dim=gene_size,
    use_drug_structure=False,
    class_cond=True,
    num_classes=NUM_CLASSES,
)

ctrl_adata = adata_train[adata_train.obs["control"]].copy()
test_perts = sorted([p for p in adata_test.obs["target_gene"].unique() if p != "non-targeting"])

all_X, all_obs = [], []
print(f"Predicting {len(test_perts)} perturbations...")
for pert in test_perts:
    ctrl_cells = ctrl_adata.X
    if hasattr(ctrl_cells, "toarray"):
        ctrl_cells = ctrl_cells.toarray()
    ctrl_tensor = torch.tensor(ctrl_cells, dtype=torch.float32).to("cuda")

    cond_idx = cond_to_idx.get(pert, 0)
    pert_label = torch.full((ctrl_tensor.shape[0],), cond_idx, dtype=torch.long, device="cuda")

    with torch.no_grad():
        z_sem = sampler.encode_with_perturbation(ctrl_tensor, pert_label)

    n_test_cells = (adata_test.obs["target_gene"] == pert).sum()
    n_pred = min(n_test_cells, z_sem.shape[0])

    pred = sampler.pred(z_sem[:n_pred], gene_size=gene_size)
    pred_np = np.clip(pred.cpu().numpy(), 0, None)

    all_X.append(pred_np)
    all_obs.extend([pert] * n_pred)

pred_X = np.vstack(all_X)
adata_pred = ad.AnnData(
    X=pred_X,
    obs=pd.DataFrame({"perturbation": all_obs}),
    var=adata_train.var.copy(),
)
pred_path = OUTPUT_DIR / f"predictions_{timestamp}.h5ad"
adata_pred.write_h5ad(pred_path)
print(f"Saved predictions: {pred_path}")

# ===================== Evaluation =====================
print("\n" + "=" * 50)
evaluate_predictions(ctrl_adata, adata_test, adata_pred, str(OUTPUT_DIR / f"squidiff_loco_{timestamp}"), real_condition_key="target_gene")
print("=" * 50)
print("Done.")
