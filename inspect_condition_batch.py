#!/usr/bin/env python3
"""Quick check of CellFlow training batch condition format.

Loads k562_hvg.h5ad, registers ESM perturbation embeddings,
runs prepare_data, samples one batch, and prints condition fields.
"""

from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from cellflow.model import CellFlow
from cellflow.data._dataloader import TrainSampler


def main() -> None:
    root = Path(__file__).resolve().parent
    adata_path = root / "data_train" / "k562_hvg.h5ad"
    esm_path = root / "data" / "ESM2_pert_features.pt"

    print(f"Loading adata: {adata_path}")
    adata = sc.read_h5ad(str(adata_path))

    emb = torch.load(str(esm_path), map_location="cpu")
    emb_dict = dict(emb)
    for k, v in list(emb_dict.items()):
        if torch.is_tensor(v):
            emb_dict[k] = v.cpu().numpy()
        else:
            emb_dict[k] = np.asarray(v)

    rep_key = "ESM2_pert_features"
    adata.uns[rep_key] = emb_dict

    control_key = "control"
    if control_key not in adata.obs:
        adata.obs[control_key] = adata.obs["target_gene"].astype(str) == "non-targeting"

    cf = CellFlow(adata, solver="otfm")
    cf.prepare_data(
        sample_rep="X",
        control_key=control_key,
        perturbation_covariates={"gene_perturbation": ["target_gene"]},
        perturbation_covariate_reps={"gene_perturbation": rep_key},
    )

    sampler = TrainSampler(cf.train_data, batch_size=25600)
    batch = sampler.sample(np.random.default_rng(0))

    print("\nBatch keys:", list(batch.keys()))
    print("src shape:", batch["src_cell_data"].shape)
    print("tgt shape:", batch["tgt_cell_data"].shape)

    condition = batch.get("condition")
    condition_idx = batch.get("condition_idx")

    print("\ncondition type:", type(condition))
    if isinstance(condition, dict):
        for k, v in condition.items():
            print(f"condition[{k}] shape: {v.shape}, dtype: {v.dtype}")
            print(f"condition[{k}] sample[0, :5]:", v[0, :5])

    print("\ncondition_idx:", condition_idx, "dtype:", getattr(condition_idx, "dtype", None))
    if condition_idx is not None:
        cond_tuple = cf.train_data.perturbation_idx_to_covariates[int(condition_idx)]
        print("condition_idx -> perturbation covariates:", cond_tuple)


if __name__ == "__main__":
    main()
