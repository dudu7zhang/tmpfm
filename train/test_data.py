#!/usr/bin/env python3
"""Minimal training script for `MyFlow`.

Notes:
- This script expects to be run from the repository root (`/home/zhangshibo24s/cell_project`).
- It adds the local `myflow/src` package to `sys.path` so you don't need to install the package.
- Provide `--perturbation-covariates` as a JSON string mapping covariate-name -> list-of-obs-columns.
- Optionally provide `--perturbation-reps` as a JSON dict mapping covariate-name -> key-in-adata.uns holding embeddings.
"""

import argparse
from curses.ascii import ctrl
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
import scanpy as sc
from datetime import datetime

from myflow.utils import build_condition_gene_masks
# from eval import compute_des
# Make local package importable (src)
ROOT = Path(__file__).resolve().parent.parent
# Ensure JAX/CUDA sees only one GPU by default (can override with env var)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    try:
        from myflow.model._myflow import MyFlow
        import anndata as ad
        import numpy as np
        import pandas as pd
        from myflow.data._dataloader import ValidationSampler
        from myflow.metrics import compute_metrics, compute_mean_metrics
    except Exception as e:
        raise ImportError(
            "Failed to import local `myflow` package. Make sure you run this from project root and that dependencies are installed.\n"
            f"Original error: {e}"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adata",
        default="/home/zhangshibo24s/cell_flow/data_train",
        required=False,
        help="Path to anndata (.h5ad) or directory containing *hvg.h5ad files"
    )
    p.add_argument(
        "--sample-rep",
        default="X",
        help="Key in adata.obsm to use as sample representation (default: X)",
    )
    p.add_argument("--control-key", default="control", help="obs column marking control samples (default: control)")
    p.add_argument(
        "--perturbation-reps",
        default="{}",
        help="JSON mapping perturbation_name -> adata.uns key containing embeddings (optional)",
    )
    p.add_argument("--num-iterations", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    # p.add_argument("--valid-freq", type=int, default=500)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    # p.add_argument("--preset", choices=["jurkat"], default=None, help="Optional preset for known datasets (loads embeddings automatically)")
    return p.parse_args()


def main():
    args = parse_args()
    adata_path = Path(args.adata)
    if not adata_path.exists():
        raise FileNotFoundError(f"adata path not found: {adata_path}")

    if adata_path.is_dir():
        h5ad_files = sorted([p for p in adata_path.iterdir() if p.name.endswith("hvg.h5ad")])
        if len(h5ad_files) < 4:
            raise ValueError(
                f"Need at least 4 *hvg.h5ad files in directory for 3-train/1-test split, found {len(h5ad_files)}"
            )
        train_files = h5ad_files[:3]
        # 剩rpe1文件作为测试集
        # Load and concatenate training files
        adatas = [ad.read_h5ad(str(p)) for p in train_files]
        print("Concatenating training AnnData objects...")
        try:
            adata = ad.concat(adatas, join="outer", label="batch", keys=[p.stem for p in train_files])
        except Exception:
            adata = ad.concat(adatas)
 
        print("Training adata.obs columns:", list(adata.obs.columns))
    else:
        print("Loading data:", adata_path)
        adata = ad.read_h5ad(str(adata_path))
        print("adata.obs columns:", list(adata.obs.columns))
    

    esm_path = ROOT / "data" / "ESM2_pert_features.pt"
    emb = torch.load(str(esm_path), map_location="cpu")
    emb_dict = dict(emb)
    for k, v in list(emb_dict.items()):
        if torch.is_tensor(v):
            emb_dict[k] = v.cpu().numpy()
        else:
            emb_dict[k] = np.asarray(v)
    rep_key = "ESM2_pert_features"
    adata.uns[rep_key] = emb_dict
    perturbation_covariates = {"gene_perturbation": ["target_gene"]}
    perturbation_reps = {"gene_perturbation": rep_key}
    if args.control_key not in adata.obs:
        adata.obs[args.control_key] = adata.obs["target_gene"].astype(str) == "non-targeting"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Initializing MyFlow (this may import jax/flax/ott)")
    cf = MyFlow(adata, solver=args.solver)
    print("Preparing data for training")
    cf.prepare_data(
        sample_rep=args.sample_rep,
        control_key=args.control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_reps,
    )
    grn_path = ROOT / "data" / "trrust_rawdata.human.tsv"
    cond_gene_masks = build_condition_gene_masks(
        adata=adata,
        perturbation_idx_to_covariates=cf.train_data.perturbation_idx_to_covariates,
        trrust_path=str(grn_path),
        collectri_organism="human",
    )
    print("cond_gene_masks shape:", cond_gene_masks.shape)
    print(cond_gene_masks)
    
    # # ---------------- 探索数据区 ----------------
    # # cell_data 是通过 cf._dm._get_cell_data(cf.adata) 计算出来的细胞表示矩阵 (如 X 或 obsm 里的向量)
    # cell_data = cf._dm._get_cell_data(cf.adata)
    # print("\n" + "="*40)
    # print(">>> 1. Cell Data Shape:", cell_data.shape)
    # print(">>> (First 2 cells, first 5 features):")
    # print(cell_data[:2, :5])
    
    # # cond_data 是刚才我们看得很细的 _get_condition_data 的返回值 ReturnData
    # cond_data = cf._dm._get_condition_data(adata=cf.adata)
    # print("\n" + "="*40)
    # print(">>> 2. Condition Data Info")
    
    # # 看看都有哪些 key 以及它们各自张量的形状
    # print(f"- condition_data dict keys: {cond_data.condition_data.keys()}")
    # for k, v in cond_data.condition_data.items():
    #     print(f"  * {k}: shape {np.shape(v)}")
        
    # print(f"\n- split_covariates_mask shape: {cond_data.split_covariates_mask.shape}")
    # print(f"- perturbation_covariates_mask shape: {cond_data.perturbation_covariates_mask.shape}")
    
    # print("\n- perturbation_idx_to_covariates (first 5):")
    # for idx, cov in list(cond_data.perturbation_idx_to_covariates.items())[:5]:
    #     print(f"  {idx}: {cov}")
        
    # print("\n- control_to_perturbation mapping (first 2 control groups):")
    # for k, v in list(cond_data.control_to_perturbation.items())[:2]:
    #     print(f"  Control_idx {k} points to {len(v)} target permutations. First 5 targets: {v[:5]}")
    
    # # 如果你想详细看看最终的 TrainingData 也可以这么看
    # # td = cf.train_data
    # # print(td)
    # print("="*40 + "\n")
    # ------------------------------------------
    


if __name__ == "__main__":
    main()
