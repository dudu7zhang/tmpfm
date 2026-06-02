#!/usr/bin/env python3
"""
scDFM on Replogle LOCO (Leave-One-Cell-Line-Out) split.
Matches data setup from train_myflow_loco_new.py and other comparison methods.

LOCO setting:
- Hold out HepG2 cell line for testing.
- Train on all other cell lines + a fraction of HepG2 perturbations.
- Test on remaining HepG2 perturbations.
"""
import sys, os, random
import numpy as np, pandas as pd, torch, scanpy as sc, anndata as ad
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

SCDFM_ROOT = Path(__file__).resolve().parent.parent / "scDFM-main"
sys.path.insert(0, str(SCDFM_ROOT))

from src.data_process.data import TrainSampler, TestDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.scheduler import CondOTScheduler
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.utils.utils import (
    save_checkpoint, set_requires_grad_for_p_only,
    build_gene_coexpression_graph, sorted_pad_mask,
)
from accelerate import Accelerator, DistributedDataParallelKwargs
import torchdiffeq
from tqdm import tqdm
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions

SEED = 20240508
HOLDOUT = "hepg2"
TRAIN_FRACTION = 1.0
TEST_FRACTION = 1.0
N_TRAIN_PERTS = 28
N_TEST_PERTS = 40
INFER_TOP_GENE = 1000  # Training: random 1000-gene subset per step. Inference: chunked over all genes.
K_TOPK = 30
STEPS = int(os.environ.get("SCDFM_STEPS", "5000"))
BATCH_SIZE = int(os.environ.get("SCDFM_BATCH_SIZE", "16"))
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data_gab/replogle_gab_merged_hvg.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_scdfm_loco"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[scDFM-LOCO] Start at {timestamp}")
print(f"Holdout cell line: {HOLDOUT}")
print(f"Steps: {STEPS}, Batch size: {BATCH_SIZE}")

# ===================== Data Loading =====================
print("Loading data:", ADATA_PATH)
adata = ad.read_h5ad(ADATA_PATH)
print(f"Raw: {adata.n_obs} x {adata.n_vars}")

# Column mapping - use gene symbols for scDFM compatibility
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

adata.var_names = [str(g) for g in adata.var_names]

# Control key
adata.obs["control"] = (adata.obs["target_gene"] == "non-targeting").astype(int)
adata.obs["is_control"] = adata.obs["target_gene"] == "non-targeting"
print(f"Control: {adata.obs['control'].sum()}, Pert: {(~adata.obs['is_control']).sum()}")

# ===================== LOCO Split (matching other methods) =====================
rng = np.random.default_rng(SEED)
holdout_mask = adata.obs["cell_type"] == HOLDOUT
other_mask = ~holdout_mask

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

adata_train = stratified_subsample(adata[train_mask].copy(), TRAIN_FRACTION, rng)
adata_test = stratified_subsample(adata[test_mask].copy(), TEST_FRACTION, rng)

train_holdout_perts_seen = set(
    adata_train.obs.loc[
        (adata_train.obs["cell_type"] == HOLDOUT) & (~adata_train.obs["is_control"]),
        "target_gene",
    ].astype(str)
)
test_perts_after_subsample = set(adata_test.obs["target_gene"].astype(str).unique())
if not test_perts_after_subsample <= other_targets:
    raise AssertionError("Every test perturbation must be observed in non-holdout cell lines.")
if test_perts_after_subsample & train_holdout_perts_seen:
    raise AssertionError("Test perturbation responses leaked into the HepG2 training subset.")
if not ((adata_train.obs["cell_type"] == HOLDOUT) & adata_train.obs["control"]).any():
    raise AssertionError(f"Training set must include {HOLDOUT} control cells.")

print(f"LOCO Split ({HOLDOUT}):")
print(f"  Eligible perturbations (in both holdout & other): {len(pert_targets)}")
print(f"  Train perturbations: {len(train_perts)}, Test perturbations: {len(test_perts)}")
print(f"Train: {adata_train.n_obs} cells, Test: {adata_test.n_obs} cells")

# ===================== Format for scDFM =====================
# scDFM expects: Drug1, Drug2 columns, condition = Drug1+Drug2
# For single-gene perturbations: Drug1=gene, Drug2=control
# For control cells: Drug1=control, Drug2=control -> perturbation_covariates="control+control"

def format_scdfm_columns(adata_in):
    adata_out = adata_in.copy()
    adata_out.obs["Drug1"] = adata_out.obs.apply(
        lambda x: "control" if x["is_control"] else x["target_gene"], axis=1
    )
    adata_out.obs["Drug2"] = "control"
    adata_out.obs["condition"] = adata_out.obs.apply(
        lambda x: f"{x['Drug1']}+{x['Drug2']}", axis=1
    )
    return adata_out

adata_train = format_scdfm_columns(adata_train)
adata_test = format_scdfm_columns(adata_test)

# Ensure sparse X for scDFM
if not sparse.issparse(adata_train.X):
    adata_train.X = sparse.csr_matrix(adata_train.X)
if not sparse.issparse(adata_test.X):
    adata_test.X = sparse.csr_matrix(adata_test.X)

# Build perturbation dict (all unique Drug1 values + "control")
all_perts = set()
for a in [adata_train, adata_test]:
    for p in a.obs["Drug1"].unique():
        all_perts.add(str(p))
all_perts = sorted(all_perts)
perturbation_dict = {p: i for i, p in enumerate(all_perts)}

# ===================== Build Co-expression Mask =====================
mask_path = OUTPUT_DIR / "coexpression_mask.pt"

if mask_path.exists():
    coexp_mask = torch.load(mask_path)
    print(f"Loaded co-expression mask from {mask_path}")
else:
    print("Building gene co-expression graph...")
    X_dense = adata_train.X.toarray()
    coexp_mask = build_gene_coexpression_graph(
        X_dense, method="pearson", wgcna_beta=None,
        sparsify="topk", k=K_TOPK, use_negative_edge=True,
    )
    coexp_mask = sorted_pad_mask(coexp_mask, pad_size=4, gene_names=list(adata_train.var_names))
    torch.save(coexp_mask, mask_path)
    print(f"Co-expression mask saved to {mask_path}")

# ===================== Train/Test Samplers =====================
train_sampler = TrainSampler("replogle", adata_train, ["Drug1", "Drug2"], perturbation_dict)
test_sampler = TestDataset("replogle", adata_test, ["Drug1", "Drug2"], perturbation_dict)

print(f"Train perturbations: {len(train_sampler._perturbation_covariates)}")
print(f"Test perturbations: {len(test_sampler._perturbation_covariates)}")

# ===================== Vocab =====================
vocab = GeneVocab.from_file(str(SCDFM_ROOT / "src" / "tokenizer" / "replogle_vocab.json"))
for tok in ["<pad>", "<cls>", "<mask>"]:
    if tok not in vocab:
        vocab.insert_token(tok, len(vocab))

# Add missing expression genes and perturbation tokens
all_gene_names = (
    list(adata_train.var_names) + list(adata_test.var_names)
    + list(perturbation_dict.keys()) + ["ctrl", "control"]
)
for gene in sorted(set(map(str, all_gene_names))):
    if gene not in vocab:
        vocab.insert_token(gene, len(vocab))

print(f"Vocab size: {len(vocab)}")

# ===================== Model =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 512

vf = instantiate_model(
    "origin", ntoken=len(vocab),
    d_model=D_MODEL, d_perturbation=D_MODEL,
    fusion_method="differential_perceiver", perturbation_function="crisper",
    use_perturbation_interaction=False,
    mask_path=str(mask_path),
)

gene_ids = torch.tensor(vocab.encode(list(adata_train.var_names)), dtype=torch.long, device=device)
assert int(gene_ids.max()) < len(vocab), f"gene id {int(gene_ids.max())} exceeds vocab size {len(vocab)}"
inverse_dict = {v: str(k) for k, v in perturbation_dict.items()}

# ===================== Training =====================
ot_sampler = OTPlanSampler(method="exact")
path_obj = AffineProbPath(scheduler=CondOTScheduler())

class PerturbationDataset(Dataset):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler; self.batch_size = batch_size
    def __len__(self): return 1000
    def __getitem__(self, idx):
        return self.sampler.get_batch(self.batch_size)

def custom_collate(batch):
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(values[0], pd.Index):
            collated[key] = values[0].tolist()
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            collated[key] = torch.from_numpy(np.stack(values))
        else:
            collated[key] = values
    return collated

dataloader = DataLoader(
    PerturbationDataset(train_sampler, BATCH_SIZE),
    batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate,
)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
optimizer = torch.optim.Adam(vf.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)
vf, optimizer, scheduler, dataloader = accelerator.prepare(vf, optimizer, scheduler, dataloader)

save_path = str(OUTPUT_DIR / "checkpoints"); os.makedirs(save_path, exist_ok=True)

def train_step(source, target, perturbation_id):
    B = source.shape[0]; dev = accelerator.device
    n_genes = source.shape[-1]
    if INFER_TOP_GENE < n_genes:
        input_gene_ids = torch.randperm(n_genes, device=dev)[:INFER_TOP_GENE]
        src, tgt = source[:, input_gene_ids], target[:, input_gene_ids]
        gene_input = gene_ids.repeat(B, 1)[:, input_gene_ids].to(dev)
    else:
        src, tgt = source, target
        gene_input = gene_ids.repeat(B, 1).to(dev)
    t = torch.rand(B, device=dev)
    noise = torch.randn_like(src)
    path_sample = path_obj.sample(t=t, x_0=noise, x_1=tgt)
    pred_vel = vf(gene_input, path_sample.x_t, path_sample.t, src, perturbation_id, gene_input, mode="predict_y")
    return ((pred_vel - path_sample.dx_t) ** 2).mean()

print("Training scDFM...")
pbar = tqdm(total=STEPS); iteration = 0
while iteration < STEPS:
    for batch_data in dataloader:
        source = batch_data['src_cell_data'].squeeze(0).to(device)
        target = batch_data['tgt_cell_data'].squeeze(0).to(device)
        perturbation_id = batch_data['condition_id'].squeeze(0).to(device)
        pert_name = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
        perturbation_id = torch.tensor(vocab.encode(pert_name), dtype=torch.long, device=device).repeat(source.shape[0], 1)
        set_requires_grad_for_p_only(vf, p_only="predict_y")
        loss = train_step(source, target, perturbation_id)
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        if iteration % 5000 == 0 and iteration > 0:
            save_checkpoint(
                model=accelerator.unwrap_model(vf),
                optimizer=optimizer, scheduler=scheduler,
                iteration=iteration, eval_score=None,
                save_path=save_path, is_best=False,
            )
        pbar.update(1); pbar.set_description(f'loss: {loss.item():.4f}')
        iteration += 1
        if iteration >= STEPS: break
pbar.close()
print("Training done.", flush=True)

# Clean up training objects to free GPU memory
del dataloader
torch.cuda.empty_cache()

# ===================== Prediction =====================
@torch.no_grad()
def generate_sample(wrapped_vf, source, condition_vec=None, vf_model=None,
                    gene_ids_local=None, gene_all=None, steps=20):
    noise = torch.randn_like(source)
    traj = torchdiffeq.odeint(
        lambda t, x: wrapped_vf(x, t, source, condition_vec, vf_model, gene_ids_local, gene_all),
        noise, torch.linspace(0, 1, steps).to(source.device),
        atol=1e-4, rtol=1e-4, method="rk4",
    )
    return torch.clamp(traj[-1], min=0)

def wrapped_vf_fn(target, t, source, perturbation_id, vf_model, g_ids, g_all):
    gene = g_ids.repeat(source.shape[0], 1).to(device)
    return vf_model(gene, target, t, source, perturbation_id, g_all)

vf.eval()
gene_ids_all = torch.tensor(vocab.encode(list(adata_test.var_names)), dtype=torch.long, device=device)
n_genes_total = gene_ids_all.numel()
gene_chunks = [
    torch.arange(i, min(i + INFER_TOP_GENE, n_genes_total), device=device)
    for i in range(0, n_genes_total, INFER_TOP_GENE)
]

# Use HepG2 control cells from training set as source for prediction
ctrl_mask = adata_train.obs["is_control"] & (adata_train.obs["cell_type"] == HOLDOUT)
ctrl_train_adata = adata_train[ctrl_mask]
if ctrl_train_adata.n_obs == 0:
    raise RuntimeError(f"No {HOLDOUT} control cells in training set for prediction.")
ctrl_X = ctrl_train_adata.X.toarray() if sparse.issparse(ctrl_train_adata.X) else ctrl_train_adata.X
src_ctrl = torch.tensor(ctrl_X, dtype=torch.float32, device=device)

all_X, all_obs_list = [], []
print(f"Predicting {len(test_sampler._perturbation_covariates)} perturbations...", flush=True)
for pert_name in test_sampler._perturbation_covariates:
    print(f"  Predicting: {pert_name}", flush=True)
    pert_data = test_sampler.get_perturbation_data(pert_name)
    pert_id = pert_data['condition_id'].to(device)
    pert_name_crisper = [inverse_dict[int(p_id)] for p_id in pert_id[0].cpu().numpy()]
    pert_id_enc = torch.tensor(vocab.encode(pert_name_crisper), dtype=torch.long, device=device)
    idx = torch.randperm(src_ctrl.shape[0]); src = src_ctrl[idx][:128]
    preds = []
    for i in range(0, src.shape[0], BATCH_SIZE):
        batch_src = src[i:i+BATCH_SIZE]
        batch_pert = pert_id_enc.repeat(batch_src.shape[0], 1)
        chunk_preds = []
        for chunk_idx in gene_chunks:
            chunk_src = batch_src[:, chunk_idx]
            chunk_gene_ids = gene_ids_all[chunk_idx]
            chunk_pred = generate_sample(
                wrapped_vf_fn, chunk_src, batch_pert, vf,
                gene_ids_local=chunk_gene_ids, gene_all=chunk_gene_ids,
            )
            chunk_preds.append(chunk_pred)
        pred = torch.cat(chunk_preds, dim=1)
        preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0).numpy()
    all_X.append(preds)
    all_obs_list.extend([pert_name] * preds.shape[0])

pred_X = np.vstack(all_X)
pred_X = np.clip(pred_X, 0, None)

# Map perturbation names from "GENE+control" -> "GENE" for evaluation
pert_names_clean = [p.replace("+control", "") for p in all_obs_list]
adata_pred = ad.AnnData(
    X=pred_X, obs=pd.DataFrame({"perturbation": pert_names_clean}),
    var=adata_test.var.copy(),
)
pred_path = OUTPUT_DIR / f"predictions_{timestamp}.h5ad"
adata_pred.write_h5ad(pred_path)
print(f"Saved predictions: {pred_path}")

# ===================== Evaluation =====================
ctrl_eval = ctrl_train_adata.copy()
adata_real = adata_test.copy()

print("\n" + "=" * 50)
print("Evaluating scDFM-LOCO predictions...")
evaluate_predictions(
    ctrl_eval, adata_real, adata_pred,
    str(OUTPUT_DIR / f"scdfm_loco_{timestamp}"),
    real_condition_key="target_gene",
)
print("=" * 50)
print("Done.")
