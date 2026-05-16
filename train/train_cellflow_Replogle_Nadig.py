#!/usr/bin/env python3
"""Minimal training script for `CellFlow`.

Notes:
- This script expects to be run from the repository root (`/home/zhangshibo24s/cell_project`).
- It adds the local `cellflow/src` package to `sys.path` so you don't need to install the package.
- Provide `--perturbation-covariates` as a JSON string mapping covariate-name -> list-of-obs-columns.
- Optionally provide `--perturbation-reps` as a JSON dict mapping covariate-name -> key-in-adata.uns holding embeddings.
"""

import argparse
import os
from pathlib import Path
import re

# Set GPU/JAX environment before importing torch/jax/cellflow.
ROOT = Path(__file__).resolve().parent
# Ensure only one GPU is visible by default (can override from shell env).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
# Avoid large up-front JAX memory preallocation that can look like multi-GPU usage.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import anndata as ad
import mygene
import pandas as pd
import torch
import numpy as np
import scanpy as sc
from datetime import datetime
from cellflow.model._cellflow import CellFlow
from cellflow.training import Metrics



ENSG_PATTERN = re.compile(r"^ENSG\d+$", re.IGNORECASE)


def _extract_ensembl_id(entry) -> str | None:
    if entry is None:
        return None
    if isinstance(entry, list):
        for item in entry:
            if isinstance(item, dict) and "gene" in item:
                val = str(item["gene"]).strip().upper()
                if val:
                    return val
            elif isinstance(item, str):
                val = item.strip().upper()
                if val:
                    return val
        return None
    if isinstance(entry, dict):
        gene = entry.get("gene")
        if gene is not None:
            return str(gene).strip().upper()
        return None
    return str(entry).strip().upper()


def build_symbol_to_ensembl(symbols: list[str]) -> dict[str, str]:
    symbols = [str(s).strip() for s in symbols]
    unique_symbols = list(dict.fromkeys(symbols))
    symbol_to_ensembl: dict[str, str] = {}

    already_ensg = [s for s in unique_symbols if ENSG_PATTERN.match(s)]
    for s in already_ensg:
        symbol_to_ensembl[s] = s.upper()

    unresolved = [s for s in unique_symbols if s not in symbol_to_ensembl]
    if unresolved:
        mg = mygene.MyGeneInfo()
        query = mg.querymany(
            unresolved,
            scopes="symbol,alias",
            fields="ensembl.gene",
            species="human",
            as_dataframe=False,
            returnall=False,
            verbose=False,
        )
        for row in query:
            q = str(row.get("query", "")).strip()
            ensembl_id = _extract_ensembl_id(row.get("ensembl"))
            if q and ensembl_id:
                symbol_to_ensembl[q] = ensembl_id
    return symbol_to_ensembl


def align_adata_to_selected_ensembl(
    adata: ad.AnnData,
    symbol_to_ensembl: dict[str, str],
) -> ad.AnnData:
    original_symbols = [str(g).strip() for g in adata.var_names]
    mapped_ids = [symbol_to_ensembl.get(s, s).upper() for s in original_symbols]

    keep_idx = []
    seen: set[str] = set()
    for i, gid in enumerate(mapped_ids):
        # We don't drop non-ENSG anymore, to guarantee all provided input genes are kept (except duplicates)
        if gid in seen:
            continue
        seen.add(gid)
        keep_idx.append(i)

    if not keep_idx:
        raise ValueError("No valid genes left.")

    adata = adata[:, keep_idx].copy()
    kept_ids = [mapped_ids[i] for i in keep_idx]
    kept_symbols = [original_symbols[i] for i in keep_idx]
    adata.var["gene_symbol"] = kept_symbols
    adata.var_names = kept_ids

    return adata


def build_matched_gene2vec(
    selected_gene_ids_file: Path,
    selected_gene2vec_file: Path,
    ordered_ids: list[str],
    save_dir: Path,
) -> tuple[Path, Path]:
    with open(selected_gene_ids_file, "r", encoding="utf-8") as f:
        all_ids = [line.strip().upper() for line in f if line.strip()]
    id_to_idx = {g: i for i, g in enumerate(all_ids)}
    full_vec = np.load(selected_gene2vec_file)
    dim = full_vec.shape[1]

    matched_vecs = []
    for g in ordered_ids:
        if g in id_to_idx:
            matched_vecs.append(full_vec[id_to_idx[g]].astype(np.float32))
        else:
            # If the gene is not found in gene2vec dictionary, initialize with zeros
            matched_vecs.append(np.zeros(dim, dtype=np.float32))
            
    matched_vec = np.stack(matched_vecs)

    save_dir.mkdir(parents=True, exist_ok=True)
    gene_ids_out = save_dir / "selected_gene_ids_matched.txt"
    gene2vec_out = save_dir / "selected_gene2vec_matched.npy"

    with open(gene_ids_out, "w", encoding="utf-8") as f:
        for g in ordered_ids:
            f.write(f"{g}\n")
    np.save(gene2vec_out, matched_vec)

    return gene_ids_out, gene2vec_out


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
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--skip-prediction", action="store_true")
    # p.add_argument("--valid-freq", type=int, default=500)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--solver", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    # p.add_argument("--condition-combined-loss-weight", type=float, default=0.1)
    # p.add_argument("--preset", choices=["jurkat"], default=None, help="Optional preset for known datasets (loads embeddings automatically)")
    return p.parse_args()


def main():
    args = parse_args()
    adata_path = Path(args.adata)
    if not adata_path.exists():
        raise FileNotFoundError(f"adata path not found: {adata_path}")

    if adata_path.is_dir():
        # h5ad_files = sorted([p for p in adata_path.iterdir() if p.name.endswith("hvg.h5ad")])
        # if len(h5ad_files) < 0:
        #     raise ValueError(
        #         f"Need at least 4 *hvg.h5ad files in directory for 3-train/1-test split, found {len(h5ad_files)}"
        #     )
        # train_files = h5ad_files[:3]
        # print("Using training files:", [p.name for p in train_files])
        # 剩rpe1文件作为测试集
        # Load and concatenate training files
        train_files = [
            "/home/zhangshibo24s/cell_flow/data_train/hepg2_hvg.h5ad",
            "/home/zhangshibo24s/cell_flow/data_train/jurkat_hvg.h5ad",
            # "/home/zhangshibo24s/cell_flow/data_train/k562_hvg.h5ad",
            "/home/zhangshibo24s/cell_flow/data_train/rpe1_hvg.h5ad"
        ]
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
    

    selected_gene_ids_file = ROOT / "data_train" / "selected_genes_27k.txt"
    selected_gene2vec_file = ROOT / "data_train" / "selected_gene2vec_27k.npy"
    gene2go_graph_file = ROOT / "data_train" / "human_ens_gene2go_graph.csv"

    print("Mapping var_names to Ensembl IDs via mygene and aligning to ontology gene list...")
    symbol_to_ensembl = build_symbol_to_ensembl([str(g) for g in adata.var_names])
    adata = align_adata_to_selected_ensembl(
        adata=adata,
        symbol_to_ensembl=symbol_to_ensembl,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_ids_file, matched_gene2vec_file = build_matched_gene2vec(
        selected_gene_ids_file=selected_gene_ids_file,
        selected_gene2vec_file=selected_gene2vec_file,
        ordered_ids=[str(g).upper() for g in adata.var_names],
        save_dir=out_dir,
    )
    print(f"Aligned genes: {adata.n_vars}")

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

    print(f"Total cells before split: {adata.n_obs}")
    n_total = adata.n_obs
    rng_val = np.random.default_rng(args.seed)
    val_indices = rng_val.choice(n_total, int(n_total * 0.05), replace=False)
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[val_indices] = True

    adata_val = adata[val_mask].copy()
    adata = adata[~val_mask].copy()
    print(f"Using {adata.n_obs} cells for training and {adata_val.n_obs} cells for validation.")

    print("Initializing CellFlow (this may import jax/flax/ott)")
    cf = CellFlow(adata, solver=args.solver)
    print("Preparing data for training")
    cf.prepare_data(
        sample_rep=args.sample_rep,
        control_key=args.control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_reps,
    )
    print("Preparing validation data")
    cf.prepare_validation_data(
        adata_val,
        name="val",
        n_conditions_on_log_iteration=5,
    )
    print("Preparing model (default architecture). This may take a few seconds")
    # cf.prepare_model(seed=args.seed)
    cf.prepare_model(
        seed=args.seed,
        condition_encoder_kwargs={
            "x_graph_fusion_kwargs": {
                "enabled": True,
                "dim": int(np.load(matched_gene2vec_file).shape[1]),
                "max_seq_len": int(adata.n_vars),
                "max_edges": 80000,
                "gene2vec_file": str(matched_gene2vec_file),
                "gene_ids_file": str(matched_ids_file),
                "gene2go_graph_file": str(gene2go_graph_file),
            }
            
        },
        solver_kwargs={
            "condition_combined_loss_weight": 0.01,
        },
        # solver_kwargs={
        #     "condition_change_eps": 1e-8,
        #     "condition_mask_smooth": 0.1,
        #     "condition_mask_kl_mix_change": 0.3,
        #     "condition_change_weight": 0.2,
        #     "condition_mask_aux_weight": 0.2,
        #     "condition_fused_mode": "adaptive",
        # },
    )
    print(f"Start training: iterations={args.num_iterations}, batch_size={args.batch_size}")
    metrics_cb = Metrics(metrics=["r_squared", "mmd"])
    cf.train(
        num_iterations=args.num_iterations, 
        batch_size=args.batch_size,
        valid_freq=500,
        callbacks=[metrics_cb],
        monitor_metrics=["val_r_squared_mean", "val_mmd_mean"]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out_path = out_dir / f"model_{timestamp}"
    model_out_path.mkdir(parents=True, exist_ok=True)
    cf.save(str(model_out_path), file_prefix=None, overwrite=args.overwrite)
    print("Saving model to output directory")
    print("Training finished. Model saved {}.".format(model_out_path))

    
    # if args.skip_prediction:
    #     print("Skipping prediction stage due to --skip-prediction flag.")
    #     return

    print("Starting prediction...")
    test_adata_path = ROOT / "data" / "k562_ctrl.h5ad"
    test_adata = sc.read_h5ad(str(test_adata_path))
    test_adata = align_adata_to_selected_ensembl(
        adata=test_adata,
        symbol_to_ensembl=symbol_to_ensembl,
    )
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
            predict_kwargs["predict_batch_size"] = args.predict_batch_size
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
    
    
    # ctrl = sc.read_h5ad('/home/zhangshibo24s/cell_flow/data/k562_ctrl.h5ad')
    # target = sc.read_h5ad('/home/zhangshibo24s/cell_flow/data/k562_validation.h5ad')
    # #state_20000, ours_w
    # pred = adata_pred
    # des_recall,  des_acc= compute_des(ctrl, target, pred)
    # print("DES score =", des_recall, des_acc)


if __name__ == "__main__":
    main()
