#!/usr/bin/env python3
"""Minimal training script for `CellFlow`.

Notes:
- This script expects to be run from the repository root (`/home/zhangshibo24s/cell_project`).
- It adds the local `cellflow/src` package to `sys.path` so you don't need to install the package.
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
# from eval import compute_des
# Make local package importable (src)
ROOT = Path(__file__).resolve().parent
# Ensure JAX/CUDA sees only one GPU by default (can override with env var)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    try:
        from cellflow.model._cellflow import CellFlow
        import anndata as ad
        import numpy as np
        import pandas as pd
        from cellflow.data._dataloader import ValidationSampler
        from cellflow.metrics import compute_metrics, compute_mean_metrics
    except Exception as e:
        raise ImportError(
            "Failed to import local `cellflow` package. Make sure you run this from project root and that dependencies are installed.\n"
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
    p.add_argument("--num-iterations", type=int, default=6000)
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
    print("Initializing CellFlow (this may import jax/flax/ott)")
    cf = CellFlow(adata, solver=args.solver)
    print("Preparing data for training")
    cf.prepare_data(
        sample_rep=args.sample_rep,
        control_key=args.control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_reps,
    )
    print("Preparing model (default architecture). This may take a few seconds")
    # cf.prepare_model(seed=args.seed)
    cf.prepare_model(
        seed=args.seed,
        grn_path=str(ROOT / "data" / "trrust_rawdata.human.tsv"),
        use_collectri=True,
        collectri_organism="human",       
        use_nonlinear_path=False,
        solver_kwargs={
            "condition_deg_top_frac": 0.2,
            "condition_change_eps": 1e-8,
            "condition_mask_smooth": 0.1,
            "condition_mask_kl_mix_change": 0.3,
            "condition_change_weight": 0.2,
            "condition_mask_aux_weight": 0.2,
            "condition_fused_mode": "adaptive",
        },
    )
    print(f"Start training: iterations={args.num_iterations}, batch_size={args.batch_size}")
    cf.train(num_iterations=args.num_iterations, batch_size=args.batch_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out_path = out_dir / f"model_{timestamp}"
    model_out_path.mkdir(parents=True, exist_ok=True)
    cf.save(str(model_out_path), file_prefix=None, overwrite=args.overwrite)
    print("Saving model to output directory")
    print("Training finished. Model saved.")
    
    
    print("Starting prediction...")
    test_adata_path = ROOT / "data" / "jurkat_ctrl.h5ad"
    test_adata = sc.read_h5ad(str(test_adata_path))
    test_adata.obs[args.control_key] = True
    test_adata.uns[rep_key] = emb_dict
    groups = test_adata.obs.groupby("target_gene").groups

    all_X = []
    all_obs = []

    for gene, idx in groups.items():
        sub_adata = test_adata[idx].copy()
        covariate_data = pd.DataFrame({
            "target_gene": [gene],
            args.control_key: [False]
        })
        predict_kwargs = {
            "adata": sub_adata,
            "covariate_data": covariate_data,
            "sample_rep": args.sample_rep,
        }
        if args.solver == "otfm":
            predict_kwargs["predict_batch_size"] = 256
        preds = cf.predict(**predict_kwargs)
        arr = list(preds.values())[0]
        arr = np.asarray(arr)
        all_X.append(arr)
        obs = pd.DataFrame({
            "perturbation": [gene] * arr.shape[0]
        })
        all_obs.append(obs)
    print("Prediction finished")
    X = np.vstack(all_X)
    obs = pd.concat(all_obs, ignore_index=True)
    adata_pred = ad.AnnData(X=X, obs=obs, var=test_adata.var.copy())
    pred_dir = Path(args.output_dir) / f"predictions_{timestamp}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_file = pred_dir / f"predictions_{timestamp}.h5ad"
    adata_pred.write_h5ad(out_file)
    print(f"Saved prediction file: {out_file}")
    
    
    # ctrl = sc.read_h5ad('/home/zhangshibo24s/cell_flow/data/jurkat_ctrl.h5ad')
    # target = sc.read_h5ad('/home/zhangshibo24s/cell_flow/data/jurkat_validation.h5ad')
    # #state_20000, ours_w
    # pred = adata_pred
    # des_recall,  des_acc= compute_des(ctrl, target, pred)
    # print("DES score =", des_recall, des_acc)


if __name__ == "__main__":
    main()
