#!/usr/bin/env python3
"""
scDFM on Replogle LOCO split.
Matches data setup from train_cellflow_loco_new.py.
scDFM needs custom data handling since its Data class only supports 'norman'/'combosciplex'.
"""
import sys, os, random
import numpy as np, pandas as pd, torch, scanpy as sc, anndata as ad
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

SCDFM_ROOT = Path(__file__).resolve().parent.parent / "scDFM-main"
sys.path.insert(0, str(SCDFM_ROOT))
os.chdir(str(SCDFM_ROOT))  # scDFM uses relative paths

from src.data_process.data import TrainSampler, TestDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.scheduler import CondOTScheduler
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.utils.utils import (
    load_checkpoint, save_checkpoint, make_lognorm_poisson_noise,
    process_vocab, set_requires_grad_for_p_only, build_gene_coexpression_graph, sorted_pad_mask,
)
from accelerate import Accelerator, DistributedDataParallelKwargs
import torchdiffeq
from tqdm import trange, tqdm
import accelerate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions

SEED = 20240508
HOLDOUT = "hepg2"
TRAIN_FRACTION = 0.3
TEST_FRACTION = 0.3
ADATA_PATH = "/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad"
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_scdfm_loco"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[scDFM-LOCO] Start at {timestamp}")

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
adata.obs["condition"] = adata.obs["target_gene"].astype(str).str.replace("non-targeting", "ctrl")

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

# Normalize
sc.pp.normalize_total(adata_train); sc.pp.log1p(adata_train)
sc.pp.normalize_total(adata_test); sc.pp.log1p(adata_test)

# Ensure X is sparse for scDFM
from scipy import sparse
if not sparse.issparse(adata_train.X):
    adata_train.X = sparse.csr_matrix(adata_train.X)
if not sparse.issparse(adata_test.X):
    adata_test.X = sparse.csr_matrix(adata_test.X)

# Prepare Drug1/Drug2 columns for scDFM
# scDFM expects control cells to have Drug1="control", Drug2="control" -> perturbation_covariates="control+control"
adata_train.obs["Drug1"] = adata_train.obs["condition"].apply(lambda x: "control" if x == "ctrl" else (x.split("+")[0] if "+" in str(x) else str(x)))
adata_train.obs["Drug2"] = adata_train.obs["condition"].apply(lambda x: "control" if x == "ctrl" else (x.split("+")[-1] if "+" in str(x) else "control"))
adata_train.obs["is_control"] = adata_train.obs["control"].astype(bool)

adata_test.obs["Drug1"] = adata_test.obs["condition"].apply(lambda x: "control" if x == "ctrl" else (x.split("+")[0] if "+" in str(x) else str(x)))
adata_test.obs["Drug2"] = adata_test.obs["condition"].apply(lambda x: "control" if x == "ctrl" else (x.split("+")[-1] if "+" in str(x) else "control"))
adata_test.obs["is_control"] = adata_test.obs["control"].astype(bool)

print(f"Train: {adata_train.n_obs}, Test: {adata_test.n_obs}, Genes: {adata_train.n_vars}")

# ===================== Build perturbation dict =====================
all_conditions = list(adata_train.obs["condition"].unique()) + list(adata_test.obs["condition"].unique())
unique_perturbation = []
for c in all_conditions:
    unique_perturbation.extend(str(c).split("+"))
unique_perturbation = sorted(set(unique_perturbation) | {"control"})  # ensure "control" is included
perturbation_dict = {p: i for i, p in enumerate(unique_perturbation)}

# Build gene co-expression graph mask
mask_path = OUTPUT_DIR / "coexpression_mask.pt"
if not mask_path.exists():
    X_train = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else np.array(adata_train.X)
    mask = build_gene_coexpression_graph(X_train, method="pearson", wgcna_beta=None, sparsify="topk", k=30, use_negative_edge=True)
    mask = sorted_pad_mask(mask, pad_size=4, gene_names=list(adata_train.var_names))
    torch.save(mask, mask_path)
    print("Saved co-expression mask")

# Build vocab - always recreate to ensure all genes are included
vocab_path = SCDFM_ROOT / "src" / "tokenizer" / "replogle_vocab.json"
gene_names = (
    list(adata_train.var_names)
    + list(adata_test.var_names)
    + list(perturbation_dict.keys())
    + ["ctrl", "control", "<pad>", "<cls>", "<mask>"]
)
unique_genes = list(dict.fromkeys(gene_names))  # Preserve order, remove duplicates
vocab_dict = {name: i for i, name in enumerate(unique_genes)}
import json
with open(vocab_path, "w") as f:
    json.dump(vocab_dict, f)

vocab = GeneVocab.from_file(str(vocab_path))
for tok in ["<pad>", "<cls>", "<mask>"]:
    if tok not in vocab:
        vocab.insert_token(tok, len(vocab))

# ===================== Model Setup =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
N_TOP_GENES = min(adata_train.n_vars, 5000)
INFER_TOP_GENE = min(N_TOP_GENES, int(os.environ.get("SCDFM_INFER_TOP_GENE", "500")))
D_MODEL = 128
STEPS = int(os.environ.get("SCDFM_STEPS", "30000"))
LR = 5e-5
BATCH_SIZE = int(os.environ.get("SCDFM_BATCH_SIZE", "2"))

vf = instantiate_model(
    "origin", ntoken=len(vocab), d_model=D_MODEL, d_perturbation=D_MODEL,
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

train_sampler = TrainSampler("norman", adata_train, ["Drug1", "Drug2"], perturbation_dict)
test_sampler = TestDataset("norman", adata_test, ["Drug1", "Drug2"], perturbation_dict)

class PerturbationDataset(Dataset):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
    def __len__(self): return 1000
    def __getitem__(self, idx):
        batch = self.sampler.get_batch(self.batch_size)
        # Remove pandas.Index fields that DataLoader can't collate
        return {k: v for k, v in batch.items() if k not in ('src_cell_id', 'tgt_cell_id')}

train_dataset = PerturbationDataset(train_sampler, BATCH_SIZE)
dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(vf.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)

vf, optimizer, scheduler, dataloader = accelerator.prepare(vf, optimizer, scheduler, dataloader)
save_path = str(OUTPUT_DIR / "checkpoints")
os.makedirs(save_path, exist_ok=True)

pbar = tqdm(total=STEPS, initial=0)
iteration = 0

def train_step(source, target, perturbation_id, vf, criterion, accelerator):
    B = source.shape[0]
    dev = accelerator.device
    input_gene_ids = torch.randperm(source.shape[-1], device=dev)[:INFER_TOP_GENE]
    src = source[:, input_gene_ids]
    tgt = target[:, input_gene_ids]
    gene_input = gene_ids.repeat(B, 1)[:, input_gene_ids].to(dev)

    t = torch.rand(B, device=dev)
    target_noise = torch.randn_like(src)
    path_sample = path_obj.sample(t=t, x_0=target_noise, x_1=tgt)

    pred_vel = vf(gene_input, path_sample.x_t, path_sample.t, src, perturbation_id, gene_input, mode="predict_y")
    loss = ((pred_vel - path_sample.dx_t) ** 2).mean()
    return loss

print("Training scDFM...")
while iteration < STEPS:
    for batch_data in dataloader:
        source = batch_data['src_cell_data'].squeeze(0).to(device)
        target = batch_data['tgt_cell_data'].squeeze(0).to(device)
        perturbation_id = batch_data['condition_id'].squeeze(0).to(device)

        if True:  # crisper mode
            pert_name = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
            perturbation_id = torch.tensor(vocab.encode(pert_name), dtype=torch.long, device=device)
            perturbation_id = perturbation_id.repeat(source.shape[0], 1)

        set_requires_grad_for_p_only(vf, p_only="predict_y")
        loss = train_step(source, target, perturbation_id, vf, criterion, accelerator)
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        if iteration % 5000 == 0 and iteration > 0:
            save_checkpoint(model=accelerator.unwrap_model(vf), optimizer=optimizer, scheduler=scheduler, iteration=iteration, eval_score=None, save_path=save_path, is_best=False)
        pbar.update(1)
        pbar.set_description(f'loss: {loss.item():.4f}')
        iteration += 1
        if iteration >= STEPS:
            break

print("Training done.")

# ===================== Prediction =====================
@torch.no_grad()
def generate_sample(wrapped_vf, source, condition_vec=None, vf=None, gene_ids_local=None, gene_all=None, steps=20, method="rk4"):
    target_noise = torch.randn_like(source)
    traj = torchdiffeq.odeint(
        lambda t, x: wrapped_vf(x, t, source, condition_vec, vf, gene_ids_local, gene_all),
        target_noise,
        torch.linspace(0, 1, steps).to(source.device),
        atol=1e-4, rtol=1e-4, method=method,
    )
    return torch.clamp(traj[-1], min=0)

def wrapped_vf_fn(target, t, source, perturbation_id, vf_model, g_ids, g_all):
    gene = g_ids.repeat(source.shape[0], 1).to(device)
    return vf_model(gene, target, t, source, perturbation_id, g_all)

vf.eval()
gene_ids_test = torch.tensor(vocab.encode(list(adata_test.var_names)), dtype=torch.long, device=device)
predict_gene_idx = torch.arange(min(INFER_TOP_GENE, gene_ids_test.numel()), device=device)

# LOCO test set has no control cells - use training set's holdout cell line controls as source
ctrl_train_adata = adata_train[(adata_train.obs["is_control"]) & (adata_train.obs["cell_type"] == HOLDOUT)]
if ctrl_train_adata.n_obs == 0:
    print(f"Warning: No control cells from {HOLDOUT} in training set, using all training controls")
    ctrl_train_adata = adata_train[adata_train.obs["is_control"]]
if ctrl_train_adata.n_obs == 0:
    raise RuntimeError("No control cells found in training set at all")
ctrl_X = ctrl_train_adata.X.toarray() if sparse.issparse(ctrl_train_adata.X) else ctrl_train_adata.X
src_ctrl = torch.tensor(ctrl_X, dtype=torch.float32, device=device)

perturbation_name_list = test_sampler._perturbation_covariates
all_X, all_obs_list = [], []

print(f"Predicting {len(perturbation_name_list)} perturbations...")
for pert_name in perturbation_name_list:
    pert_data = test_sampler.get_perturbation_data(pert_name)
    target = pert_data['tgt_cell_data']
    pert_id = pert_data['condition_id'].to(device)

    pert_name_crisper = [inverse_dict[int(p_id)] for p_id in pert_id[0].cpu().numpy()]
    pert_id_encoded = torch.tensor(vocab.encode(pert_name_crisper), dtype=torch.long, device=device)

    idx = torch.randperm(src_ctrl.shape[0])
    src = src_ctrl[idx][:128]

    pred_expressions = []
    for i in range(0, src.shape[0], BATCH_SIZE):
        batch_src = src[i:i+BATCH_SIZE]
        batch_src_subset = batch_src[:, predict_gene_idx]
        gene_ids_subset = gene_ids_test[predict_gene_idx]
        batch_pert = pert_id_encoded.repeat(batch_src.shape[0], 1)
        pred = generate_sample(wrapped_vf_fn, batch_src_subset, batch_pert, vf, gene_ids_local=gene_ids_subset, gene_all=gene_ids_subset)
        pred_full = batch_src.clone()
        pred_full[:, predict_gene_idx] = pred
        pred_expressions.append(pred_full.cpu())

    pred_expressions = torch.cat(pred_expressions, dim=0).numpy()
    all_X.append(pred_expressions)
    all_obs_list.extend([pert_name] * pred_expressions.shape[0])

pred_X = np.vstack(all_X)
adata_pred = ad.AnnData(X=pred_X, obs=pd.DataFrame({"perturbation": all_obs_list}), var=adata_test.var.copy())
adata_pred.write_h5ad(OUTPUT_DIR / f"predictions_{timestamp}.h5ad")

# ===================== Evaluation =====================
ctrl_eval = adata_train[adata_train.obs["is_control"]].copy()
print("\n" + "=" * 50)
evaluate_predictions(ctrl_eval, adata_test, adata_pred, str(OUTPUT_DIR / f"scdfm_loco_{timestamp}"))
print("=" * 50)
print("Done.")
