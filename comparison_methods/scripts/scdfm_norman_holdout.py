#!/usr/bin/env python3
"""
scDFM on Norman 2019 dataset with holdout split.
Matches data setup from train_cellflow_norman_scdfm_holdout.py.
"""
import sys, os, random
import numpy as np, pandas as pd, torch, scanpy as sc, anndata as ad
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

SCDFM_ROOT = Path(__file__).resolve().parent.parent / "scDFM-main"
sys.path.insert(0, str(SCDFM_ROOT))
os.chdir(str(SCDFM_ROOT))

from src.data_process.data import Data, TrainSampler, TestDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.scheduler import CondOTScheduler
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.utils.utils import (
    save_checkpoint, make_lognorm_poisson_noise,
    set_requires_grad_for_p_only, build_gene_coexpression_graph, sorted_pad_mask,
)
from accelerate import Accelerator, DistributedDataParallelKwargs
import torchdiffeq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate_predictions

SEED = 20240508
FOLD = 0
SPLIT_METHOD = "unseen"
N_TOP_GENES = 5000
INFER_TOP_GENE = int(os.environ.get("SCDFM_INFER_TOP_GENE", "1000"))
K_TOPK = 30
ADATA_PATH = os.environ.get("NORMAN_ADATA_PATH", "/home/zhangshibo24s/cell_flow/data/norman_2019_adata.h5ad")
if not os.path.exists(ADATA_PATH):
    for _path in (
        "/home/zhangshibo24s/cell_flow/data_train/norman_2019_adata.h5ad",
        "/home/zhangshibo24s/cell_flow/data_gab/norman_2019_adata.h5ad",
    ):
        if os.path.exists(_path):
            ADATA_PATH = _path
            break
_BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "outputs_scdfm_norman_holdout"
_RUN_ID = os.environ.get("CELLFLOW_RUN_ID")
OUTPUT_DIR = _BASE_OUTPUT_DIR if not _RUN_ID else _BASE_OUTPUT_DIR.with_name(f"{_BASE_OUTPUT_DIR.name}_{_RUN_ID}")

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[scDFM-Norman-Holdout] Start at {timestamp}")
print(f"Split: {SPLIT_METHOD}, fold={FOLD}")


def load_scdfm_norman_adata(src_path: str) -> ad.AnnData:
    """Load Norman data and add the obs columns expected by scDFM."""
    from scipy import sparse
    adata = ad.read_h5ad(src_path)
    if "guide_merged" not in adata.obs:
        raise KeyError("Norman input must contain obs['guide_merged'] for scDFM condition labels.")

    # Convert var_names from numeric indices to actual gene names
    if 'gene_name' in adata.var.columns:
        adata.var_names = adata.var['gene_name'].values

    # Ensure X is sparse for scDFM
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    adata.obs["condition"] = adata.obs["guide_merged"].astype(str)
    adata.obs["control"] = (adata.obs["condition"] == "ctrl").astype(int)
    if "guide_identity" in adata.obs:
        adata.obs["target_gene"] = adata.obs["guide_identity"].astype(str)
    return adata


# ===================== Use scDFM's own Data class =====================
scdfm_data_dir = OUTPUT_DIR / "scdfm_data"
scdfm_data_dir.mkdir(parents=True, exist_ok=True)

data_manager = Data(str(scdfm_data_dir))
data_manager.data_name = "norman"
data_manager.adata = load_scdfm_norman_adata(ADATA_PATH)
print(f"Loaded scDFM-compatible Norman data from {ADATA_PATH}")
data_manager.process_data(
    n_top_genes=N_TOP_GENES,
    split_method=SPLIT_METHOD,
    fold=FOLD,
    use_negative_edge=True,
    k=K_TOPK,
)

# Ensure adata_train and adata_test have sparse X after process_data
from scipy import sparse
if not sparse.issparse(data_manager.adata_train.X):
    data_manager.adata_train.X = sparse.csr_matrix(data_manager.adata_train.X)
if not sparse.issparse(data_manager.adata_test.X):
    data_manager.adata_test.X = sparse.csr_matrix(data_manager.adata_test.X)

train_sampler, test_sampler, _ = data_manager.load_flow_data(batch_size=2)

print(f"Train: {data_manager.adata_train.n_obs}, Test: {data_manager.adata_test.n_obs}")
print(f"Genes: {data_manager.adata_train.n_vars}")
print(f"Test conditions: {len(test_sampler._perturbation_covariates)}")

# ===================== Model Setup =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 128
STEPS = int(os.environ.get("SCDFM_STEPS", "30000"))
BATCH_SIZE = int(os.environ.get("SCDFM_BATCH_SIZE", "2"))

vocab = GeneVocab.from_file(str(SCDFM_ROOT / "src" / "tokenizer" / "norman_5000_highly_vocab.json"))
for tok in ["<pad>", "<cls>", "<mask>"]:
    if tok not in vocab: vocab.insert_token(tok, len(vocab))

# Add missing expression genes and perturbation tokens to vocabulary.
all_gene_names = (
    list(data_manager.adata_train.var_names)
    + list(data_manager.adata_test.var_names)
    + list(data_manager.perturbation_dict.keys())
    + ["ctrl", "control"]
)
for gene in sorted(set(map(str, all_gene_names))):
    if gene not in vocab:
        vocab.insert_token(gene, len(vocab))

mask_path = os.path.join(str(scdfm_data_dir), "norman",
    f"mask_fold_{FOLD}topk_{K_TOPK}{SPLIT_METHOD}_negative_edge.pt")

vf = instantiate_model(
    "origin", ntoken=len(vocab),
    d_model=D_MODEL, d_perturbation=D_MODEL,
    fusion_method="differential_perceiver", perturbation_function="crisper",
    use_perturbation_interaction=False,
    mask_path=mask_path,
)

gene_ids = torch.tensor(vocab.encode(list(data_manager.adata_train.var_names)), dtype=torch.long, device=device)
assert int(gene_ids.max()) < len(vocab), f"gene id {int(gene_ids.max())} exceeds vocab size {len(vocab)}"
inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}

# ===================== Training =====================
ot_sampler = OTPlanSampler(method="exact")
path_obj = AffineProbPath(scheduler=CondOTScheduler())

class PerturbationDataset(Dataset):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler; self.batch_size = batch_size
    def __len__(self): return 1000
    def __getitem__(self, idx): return self.sampler.get_batch(self.batch_size)

def custom_collate(batch):
    """Custom collate function to handle pandas.Index objects."""
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

dataloader = DataLoader(PerturbationDataset(train_sampler, BATCH_SIZE), batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(vf.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)
vf, optimizer, scheduler, dataloader = accelerator.prepare(vf, optimizer, scheduler, dataloader)

save_path = str(OUTPUT_DIR / "checkpoints"); os.makedirs(save_path, exist_ok=True)

def train_step(source, target, perturbation_id):
    B = source.shape[0]; dev = accelerator.device
    input_gene_ids = torch.randperm(source.shape[-1], device=dev)[:INFER_TOP_GENE]
    src, tgt = source[:, input_gene_ids], target[:, input_gene_ids]
    gene_input = gene_ids.repeat(B, 1)[:, input_gene_ids].to(dev)
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
        optimizer.zero_grad(set_to_none=True); accelerator.backward(loss); optimizer.step(); scheduler.step()
        if iteration % 5000 == 0 and iteration > 0:
            save_checkpoint(model=accelerator.unwrap_model(vf), optimizer=optimizer, scheduler=scheduler, iteration=iteration, eval_score=None, save_path=save_path, is_best=False)
        pbar.update(1); pbar.set_description(f'loss: {loss.item():.4f}')
        iteration += 1
        if iteration >= STEPS: break

# ===================== Prediction =====================
@torch.no_grad()
def generate_sample(wrapped_vf, source, condition_vec=None, vf_model=None, gene_ids_local=None, gene_all=None, steps=20):
    noise = torch.randn_like(source)
    traj = torchdiffeq.odeint(lambda t, x: wrapped_vf(x, t, source, condition_vec, vf_model, gene_ids_local, gene_all),
                              noise, torch.linspace(0, 1, steps).to(source.device), atol=1e-4, rtol=1e-4, method="rk4")
    return torch.clamp(traj[-1], min=0)

def wrapped_vf_fn(target, t, source, perturbation_id, vf_model, g_ids, g_all):
    gene = g_ids.repeat(source.shape[0], 1).to(device)
    return vf_model(gene, target, t, source, perturbation_id, g_all)

vf.eval()
gene_ids_test = torch.tensor(vocab.encode(list(data_manager.adata_test.var_names)), dtype=torch.long, device=device)
predict_gene_idx = torch.arange(min(INFER_TOP_GENE, gene_ids_test.numel()), device=device)
control_data = test_sampler.get_control_data()
src_ctrl = control_data['src_cell_data'].to(device)

all_X, all_obs_list = [control_data['src_cell_data'].numpy()], ['control'] * control_data['src_cell_data'].shape[0]
for pert_name in test_sampler._perturbation_covariates:
    pert_data = test_sampler.get_perturbation_data(pert_name)
    target = pert_data['tgt_cell_data']
    pert_id = pert_data['condition_id'].to(device)
    pert_name_crisper = [inverse_dict[int(p_id)] for p_id in pert_id[0].cpu().numpy()]
    pert_id_enc = torch.tensor(vocab.encode(pert_name_crisper), dtype=torch.long, device=device)
    idx = torch.randperm(src_ctrl.shape[0]); src = src_ctrl[idx][:128]
    preds = []
    for i in range(0, src.shape[0], BATCH_SIZE):
        batch_src = src[i:i+BATCH_SIZE]
        batch_src_subset = batch_src[:, predict_gene_idx]
        gene_ids_subset = gene_ids_test[predict_gene_idx]
        batch_pert = pert_id_enc.repeat(batch_src.shape[0], 1)
        pred = generate_sample(wrapped_vf_fn, batch_src_subset, batch_pert, vf, gene_ids_local=gene_ids_subset, gene_all=gene_ids_subset)
        pred_full = batch_src.clone()
        pred_full[:, predict_gene_idx] = pred
        preds.append(pred_full.cpu())
    preds = torch.cat(preds, dim=0).numpy()
    all_X.append(preds); all_obs_list.extend([pert_name] * preds.shape[0])

adata_pred = ad.AnnData(X=np.vstack(all_X), obs=pd.DataFrame({"perturbation": all_obs_list}), var=data_manager.adata_test.var.copy())
adata_pred.write_h5ad(OUTPUT_DIR / f"predictions_{timestamp}.h5ad")

# ===================== Evaluation =====================
# adata_test was further filtered to INFER_TOP_GENE HVGs inside process_data,
# so ctrl_eval (from adata_train, 2000 genes) and adata_real (from adata_test, 1000 genes)
# have different gene counts.  Align everything to the common gene set.
adata_real = data_manager.adata_test.copy()
train_var = set(str(g) for g in data_manager.adata_train.var_names)
test_var = set(str(g) for g in adata_real.var_names)
common_genes = sorted(train_var & test_var)
if not common_genes:
    raise RuntimeError("No common genes between adata_train and adata_test")
ctrl_eval = data_manager.adata_train[data_manager.adata_train.obs["is_control"], common_genes].copy()
sc.pp.normalize_total(ctrl_eval); sc.pp.log1p(ctrl_eval)
adata_pred = adata_pred[:, common_genes].copy()
adata_real = adata_real[:, common_genes].copy()

print("\n" + "=" * 50)
evaluate_predictions(ctrl_eval, adata_real, adata_pred, str(OUTPUT_DIR / f"scdfm_norman_holdout_{timestamp}"))
print("=" * 50)
print("Done.")
