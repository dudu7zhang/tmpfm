#!/usr/bin/env python3
"""
PerturbDiff on Norman 2019 dataset.
Uses scDFM-style additive split (30% double perturbations as test, all singles in train).
"""
import sys, os, random, types
import numpy as np, pandas as pd, torch, scanpy as sc, anndata as ad
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

PD_ROOT = Path(__file__).resolve().parent.parent / "PerturbDiff-main"
sys.path.insert(0, str(PD_ROOT))

from src.models.cross_dit.cross_dit_main import Cross_DiT
from src.models.lightning.lightning_factories import create_diffusion, create_named_schedule_sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions
from split_utils import build_scdfm_norman_split
from tqdm import tqdm

SEED = 20240508
SPLIT_SEED_BASE = 42
FOLD = 0
TEST_COND_FRAC = 0.3
SPLIT_METHOD = "additive"
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data/norman_2019_adata.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_perturbdiff_norman"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")
TRAIN_CELL_FRAC = 0.3
TEST_CELL_FRAC = 0.3

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[PerturbDiff-Norman] Start at {timestamp}")
print(f"Split: {SPLIT_METHOD}, fold={FOLD}, seed_base={SPLIT_SEED_BASE}")

# ===================== Data Loading =====================
adata = ad.read_h5ad(ADATA_PATH)
adata.obs["target_gene"] = adata.obs["guide_identity"].astype(str)
adata.obs["condition"] = adata.obs["guide_merged"].astype(str)
is_ctrl = adata.obs["condition"] == "ctrl"
adata.obs["is_control"] = is_ctrl
adata.obs["control"] = is_ctrl.astype(int)
adata.obs["perturbation"] = adata.obs["condition"].astype(str)

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

sc.pp.normalize_total(adata_train); sc.pp.log1p(adata_train)
sc.pp.normalize_total(adata_test); sc.pp.log1p(adata_test)

ctrl_adata = adata_train[adata_train.obs["is_control"]].copy()
print(f"Train: {adata_train.n_obs} (ctrl: {ctrl_adata.n_obs}), Test: {adata_test.n_obs}, Genes: {adata_train.n_vars}")

# ===================== Model =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = min(adata_train.n_vars, 2000)
HIDDEN_DIM = 512

if adata_train.n_vars > INPUT_DIM:
    X_dense = adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else np.array(adata_train.X)
    gene_vars = np.var(X_dense, axis=0)
    top_idx = np.argsort(-gene_vars)[:INPUT_DIM]
    adata_train = adata_train[:, top_idx].copy()
    adata_test = adata_test[:, top_idx].copy()
    ctrl_adata = ctrl_adata[:, top_idx].copy()

pert_list = sorted(set(adata_train.obs["perturbation"].unique().tolist() + adata_test.obs["perturbation"].unique().tolist()))
pert_dict = {p: i for i, p in enumerate(pert_list)}

model_cfg = types.SimpleNamespace(
    model_type="Cross_DiT", dit_depth=8, dit_num_heads=8, qk_norm=True,
    input_dim=INPUT_DIM, hidden_num=[INPUT_DIM, HIDDEN_DIM], dropout=0.1, output_fn="relu",
    use_orig_gene_count_as_emb=True, class_emb_gather_strategy="mean",
    class_emb_hidden_dimension=HIDDEN_DIM, use_gene_embedding=False, gene_embedding_type="linear",
    p_drop_cond=0.5, p_drop_control=0.5, steps=1000, learn_sigma=False, sigma_small=False,
    noise_schedule="linear", noise_schedule_gamma=0.3, use_kl=False, predict_xstart=True,
    rescale_timesteps=False, rescale_learned_sigmas=False, schedule_sampler="uniform",
    separate_embedder=False, use_class_silu=False, replace_batch_encoder=False, cutoff=1e-4, no_mse_loss=False,
)

model = Cross_DiT(model_cfg=model_cfg)
diffusion = create_diffusion(model_cfg)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

# ===================== Dataset =====================
ctrl_X = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, "toarray") else np.array(ctrl_adata.X)

class PerturbDiffDataset(Dataset):
    def __init__(self, adata_in, ctrl_X_in, pert_dict, batch_size=256):
        self.adata = adata_in; self.ctrl_X = ctrl_X_in; self.pert_dict = pert_dict
        self.batch_size = batch_size
        self.perts = [p for p in adata_in.obs["perturbation"].unique() if p != "ctrl"]
    def __len__(self): return 1000
    def __getitem__(self, idx):
        pert = np.random.choice(self.perts)
        cells = self.adata[self.adata.obs["perturbation"] == pert]
        X = cells.X.toarray() if hasattr(cells.X, "toarray") else np.array(cells.X)
        n = min(self.batch_size, X.shape[0])
        p_idx = np.random.choice(X.shape[0], n, replace=True)
        c_idx = np.random.choice(self.ctrl_X.shape[0], n, replace=True)
        return (torch.tensor(X[p_idx], dtype=torch.float32),
                torch.tensor(self.ctrl_X[c_idx], dtype=torch.float32),
                torch.full((n,), self.pert_dict.get(pert, 0), dtype=torch.long))

dataloader = DataLoader(PerturbDiffDataset(adata_train, ctrl_X, pert_dict), batch_size=1, shuffle=False, num_workers=4)

# ===================== Training =====================
TRAIN_STEPS = 30000
print(f"Training PerturbDiff for {TRAIN_STEPS} steps...")
model.train(); pbar = tqdm(range(TRAIN_STEPS)); iteration = 0
while iteration < TRAIN_STEPS:
    for batch in dataloader:
        pert_emb, cont_emb, cov_pert = [b.squeeze(0).to(device) for b in batch]
        B = pert_emb.shape[0]; t, _ = schedule_sampler.sample(B, device)
        output = model(pert_emb, t, cont_emb, cov_pert)
        loss = ((output - pert_emb) ** 2).mean()
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if iteration % 5000 == 0 and iteration > 0:
            torch.save(model.state_dict(), OUTPUT_DIR / f"model_{iteration}.pt")
        pbar.update(1); pbar.set_description(f"loss: {loss.item():.4f}")
        iteration += 1
        if iteration >= TRAIN_STEPS: break

torch.save(model.state_dict(), OUTPUT_DIR / "model_final.pt")

# ===================== Prediction =====================
def ddim_sample(model, n_samples, input_dim, cont, cov, n_steps=100, device="cuda"):
    """DDIM deterministic sampling (eta=0) from noise to data."""
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    alpha_cumprod_prev = torch.cat([torch.ones(1, device=device), alpha_cumprod[:-1]])

    step_size = 1000 // n_steps
    timesteps = torch.arange(0, 1000, step_size, device=device).flip(0)

    x = torch.randn(n_samples, input_dim, device=device)
    model.eval()

    with torch.no_grad():
        for t in timesteps:
            t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
            pred_x0 = model(x, t_batch, cont, cov)
            pred_x0 = torch.clamp(pred_x0, min=0)

            ab = alpha_cumprod[t]
            ab_prev = alpha_cumprod_prev[t]
            eps = (x - torch.sqrt(ab) * pred_x0) / torch.sqrt(1 - ab)
            x = torch.sqrt(ab_prev) * pred_x0 + torch.sqrt(1 - ab_prev) * eps

    return torch.clamp(x, min=0)

model.eval()
test_perts_sorted = sorted(test_conditions)
all_X, all_obs_list = [], []
with torch.no_grad():
    for pert in test_perts_sorted:
        n_test = (adata_test.obs["perturbation"] == pert).sum()
        n_pred = min(n_test, ctrl_X.shape[0])
        c_idx = np.random.choice(ctrl_X.shape[0], n_pred, replace=False)
        cont = torch.tensor(ctrl_X[c_idx], dtype=torch.float32).to(device)
        cov = torch.full((n_pred,), pert_dict.get(pert, 0), dtype=torch.long).to(device)
        pred = ddim_sample(model, n_pred, INPUT_DIM, cont, cov, n_steps=100, device=device)
        pred_np = pred.cpu().numpy()
        all_X.append(pred_np); all_obs_list.extend([pert] * n_pred)

adata_pred = ad.AnnData(X=np.vstack(all_X), obs=pd.DataFrame({"perturbation": all_obs_list}), var=adata_test.var.copy())
adata_pred.write_h5ad(OUTPUT_DIR / f"predictions_{timestamp}.h5ad")

# ===================== Evaluation =====================
print("\n" + "=" * 50)
evaluate_predictions(ctrl_adata, adata_test, adata_pred, str(OUTPUT_DIR / f"perturbdiff_norman_{timestamp}"))
print("=" * 50)
print("Done.")
