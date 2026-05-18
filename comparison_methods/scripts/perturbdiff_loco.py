#!/usr/bin/env python3
"""
PerturbDiff on Replogle LOCO split.
Directly instantiates model/diffusion without Hydra.
Matches data setup from train_cellflow_loco_new.py.
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
from src.models.lightning.lightning_factories import create_diffusion, create_named_schedule_sampler, model_init_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions
from tqdm import tqdm

SEED = 20240508
HOLDOUT = "hepg2"
TRAIN_FRACTION = 0.3
TEST_FRACTION = 0.3
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_perturbdiff_loco"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[PerturbDiff-LOCO] Start at {timestamp}")

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

adata.obs["control"] = (adata.obs["target_gene"] == "non-targeting").astype(int)
adata.obs["perturbation"] = adata.obs["target_gene"].astype(str)

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

adata_train_full = adata[train_mask].copy()
adata_test = adata[test_mask].copy()

def stratified_sub(adata_sub, frac, rng, key="target_gene"):
    if frac >= 1: return adata_sub.copy()
    positions = []
    for _, idx in adata_sub.obs.groupby(key, observed=True).indices.items():
        idx = np.asarray(idx)
        n = max(1, int(round(len(idx) * frac)))
        positions.extend(rng.choice(idx, size=n, replace=False).tolist())
    return adata_sub[np.sort(positions)].copy()

adata_train_full = stratified_sub(adata_train_full, TRAIN_FRACTION, rng)
adata_test = stratified_sub(adata_test, TEST_FRACTION, rng)

# Normalize
sc.pp.normalize_total(adata_train_full); sc.pp.log1p(adata_train_full)
sc.pp.normalize_total(adata_test); sc.pp.log1p(adata_test)

ctrl_adata = adata_train_full[adata_train_full.obs["control"].astype(bool)].copy()
print(f"Train: {adata_train_full.n_obs} (ctrl: {ctrl_adata.n_obs}), Test: {adata_test.n_obs}, Genes: {adata_train_full.n_vars}")

# ===================== Model Config =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = min(adata_train_full.n_vars, 2000)
HIDDEN_DIM = 512

# Truncate genes by variance if needed
if adata_train_full.n_vars > INPUT_DIM:
    X_dense = adata_train_full.X.toarray() if hasattr(adata_train_full.X, 'toarray') else np.array(adata_train_full.X)
    gene_vars = np.var(X_dense, axis=0)
    top_idx = np.argsort(-gene_vars)[:INPUT_DIM]
    adata_train_full = adata_train_full[:, top_idx].copy()
    adata_test = adata_test[:, top_idx].copy()
    ctrl_adata = ctrl_adata[:, top_idx].copy()
    print(f"Truncated to {INPUT_DIM} genes")

# Create model_cfg namespace
model_cfg = types.SimpleNamespace(
    model_type="Cross_DiT",
    dit_depth=8,
    dit_num_heads=8,
    qk_norm=True,
    input_dim=INPUT_DIM,
    hidden_num=[INPUT_DIM, HIDDEN_DIM],
    dropout=0.1,
    output_fn="relu",
    use_orig_gene_count_as_emb=True,
    class_emb_gather_strategy="mean",
    class_emb_hidden_dimension=HIDDEN_DIM,
    use_gene_embedding=False,
    gene_embedding_type="linear",
    p_drop_cond=0.5,
    p_drop_control=0.5,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    noise_schedule_gamma=0.3,
    use_kl=False,
    predict_xstart=True,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    schedule_sampler="uniform",
    separate_embedder=False,
    use_class_silu=False,
    replace_batch_encoder=False,
    cutoff=1e-4,
    no_mse_loss=False,
)

# Build perturbation dict
pert_list = sorted(set(adata_train_full.obs["perturbation"].unique().tolist() + adata_test.obs["perturbation"].unique().tolist()))
pert_dict = {p: i for i, p in enumerate(pert_list)}
n_perts = len(pert_list)

# ===================== Model & Diffusion =====================
print("Creating model...")
model = Cross_DiT(model_cfg=model_cfg)
diffusion = create_diffusion(model_cfg)
model = model.to(device)

TRAIN_STEPS = 30000
BATCH_SIZE = 256
LR = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

# ===================== Dataset =====================
ctrl_X = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, "toarray") else np.array(ctrl_adata.X)

class PerturbDiffDataset(Dataset):
    def __init__(self, adata_in, ctrl_X_in, pert_dict, batch_size=256):
        self.adata = adata_in
        self.ctrl_X = ctrl_X_in
        self.pert_dict = pert_dict
        self.batch_size = batch_size
        self.perts = [p for p in adata_in.obs["perturbation"].unique() if p != "non-targeting"]

    def __len__(self): return 1000

    def __getitem__(self, idx):
        pert = np.random.choice(self.perts)
        pert_cells = self.adata[self.adata.obs["perturbation"] == pert]
        X = pert_cells.X.toarray() if hasattr(pert_cells.X, "toarray") else np.array(pert_cells.X)
        n = min(self.batch_size, X.shape[0])
        p_idx = np.random.choice(X.shape[0], n, replace=True)
        c_idx = np.random.choice(self.ctrl_X.shape[0], n, replace=True)
        return (
            torch.tensor(X[p_idx], dtype=torch.float32),
            torch.tensor(self.ctrl_X[c_idx], dtype=torch.float32),
            torch.full((n,), self.pert_dict.get(pert, 0), dtype=torch.long),
        )

dataset = PerturbDiffDataset(adata_train_full, ctrl_X, pert_dict, BATCH_SIZE)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# ===================== Training =====================
print(f"Training PerturbDiff for {TRAIN_STEPS} steps...")
model.train()
pbar = tqdm(range(TRAIN_STEPS))
iteration = 0

while iteration < TRAIN_STEPS:
    for batch in dataloader:
        pert_emb, cont_emb, cov_pert = [b.squeeze(0).to(device) for b in batch]
        B = pert_emb.shape[0]
        t, _ = schedule_sampler.sample(B, device)

        # PerturbDiff predicts x_start
        output = model(pert_emb, t, cont_emb, cov_pert)
        loss = ((output - pert_emb) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iteration % 5000 == 0 and iteration > 0:
            torch.save(model.state_dict(), OUTPUT_DIR / f"model_{iteration}.pt")

        pbar.update(1)
        pbar.set_description(f"loss: {loss.item():.4f}")
        iteration += 1
        if iteration >= TRAIN_STEPS: break

torch.save(model.state_dict(), OUTPUT_DIR / "model_final.pt")
print("Training done.")

# ===================== DDIM Sampling =====================
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

# ===================== Prediction =====================
model.eval()
test_perts_sorted = sorted([p for p in adata_test.obs["perturbation"].unique() if p != "non-targeting"])
all_X, all_obs_list = [], []

print(f"Predicting {len(test_perts_sorted)} perturbations...")
with torch.no_grad():
    for pert in test_perts_sorted:
        n_test = (adata_test.obs["perturbation"] == pert).sum()
        n_pred = min(n_test, ctrl_X.shape[0])
        c_idx = np.random.choice(ctrl_X.shape[0], n_pred, replace=False)
        cont = torch.tensor(ctrl_X[c_idx], dtype=torch.float32).to(device)
        cov = torch.full((n_pred,), pert_dict.get(pert, 0), dtype=torch.long).to(device)

        pred = ddim_sample(model, n_pred, INPUT_DIM, cont, cov, n_steps=100, device=device)
        pred_np = pred.cpu().numpy()
        all_X.append(pred_np)
        all_obs_list.extend([pert] * n_pred)

adata_pred = ad.AnnData(X=np.vstack(all_X), obs=pd.DataFrame({"perturbation": all_obs_list}), var=adata_test.var.copy())
adata_pred.write_h5ad(OUTPUT_DIR / f"predictions_{timestamp}.h5ad")

# ===================== Evaluation =====================
print("\n" + "=" * 50)
evaluate_predictions(ctrl_adata, adata_test, adata_pred, str(OUTPUT_DIR / f"perturbdiff_loco_{timestamp}"))
print("=" * 50)
print("Done.")
