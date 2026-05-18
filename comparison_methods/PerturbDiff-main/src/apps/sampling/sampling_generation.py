"""Sampling generation core split from rawdata_diffusion_sample.py (logic-preserving)."""

import gc
import time

import anndata
import numpy as np
import torch
from geomloss import SamplesLoss
from pytorch_lightning.utilities import move_data_to_device
from sklearn.metrics import r2_score

from src.apps.sampling.sampling_io import save_adata
from src.apps.sampling.sampling_generation_helpers import (
    build_obs_data,
    build_self_condition,
    build_x_ctrl,
    build_gene_embedding_cache,
    collect_batch_covariates,
    load_ctrl_adata,
    load_selected_genes,
    resolve_sampling_runner,
)
def generate_samples(
    model,
    diffusion,
    cfg, 
    device, 
    logger, 
    datamodule, 
    pca_for_decode=None,
    store_noised_truth=False,
):
    """
    Dispatcher that preserves original behavior:
      - If `pca_for_decode` is None -> sample directly in **original gene space**.
      - If `pca_for_decode` is provided -> delegate to `generate_samples_pca` and return its outputs.

    Gene-space mode:
      - pert_emb and samples are [B, S, G] (G ~ 2000)
      - R^2 and MMD are computed in gene space (this is the working space, not a 'decoded' space)
      - Returns (truths, samples, trajectories, decoded=None)
    """
    # Use test split dataloader.
    dataloader = datamodule.val_dataloader()[1] # for test split

    dataloader_iter = iter(dataloader)
    model = model.to(device)
    model.eval()

    # Model type helpers
    model_type = getattr(getattr(model, "model_cfg", None), "model_type", None)
    if model_type is None:
        model_type = getattr(cfg.model, "model_type", None)

    # Set up sampling parameters
    batch_size = cfg.sampling.batch_size
    if cfg.data.use_cell_set is not None:
        batch_size = int(batch_size // cfg.data.use_cell_set)
    num_sampled_batches = cfg.sampling.num_sampled_batches
    if num_sampled_batches is None:
        num_sampled_batches = len(dataloader)
    num_samples = len(dataloader.sampler)
    
    use_ddim = cfg.sampling.use_ddim
    clip_denoised = cfg.sampling.clip_denoised
    progress = cfg.sampling.progress

    # Resolve sampling dimensionality from config if available.
    input_dim = (
        getattr(cfg.model, "input_dim", None)
        or getattr(cfg.model, "output_size", None)
        or getattr(cfg.model, "gene_dim", None)
    )

    logger.info(f"Generating {num_samples} samples with batch size {batch_size}")
    logger.info(f"Using {'DDIM' if use_ddim else 'DDPM'} sampling")
    
    all_truths = []
    all_samples = []
    all_trajectories = []
    all_covariates = []
    
    # Generate samples in batches

    cell_set_number = cfg.data.use_cell_set if hasattr(cfg.data, 'use_cell_set') and cfg.data.use_cell_set is not None else 1

    data_args = getattr(dataloader.dataset, "data_args", None)
    normalize_counts = getattr(data_args, "normalize_counts", None) if data_args is not None else None

    assert store_noised_truth == False
    
    _genes = load_selected_genes(cfg)
    assert len(_genes) == 2000

    with torch.no_grad():
        for batch_idx in range(num_sampled_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            logger.info(f"Generating batch {batch_idx + 1}/{num_sampled_batches} (size: {current_batch_size})")
            
            sample_fn, sampling_kwargs = resolve_sampling_runner(cfg, diffusion, use_ddim)

            batch_data = next(dataloader_iter)
            
            batch_data = move_data_to_device(batch_data, device)

            batch_data["batch_emb"] = model._encode_covariates(batch_data)
            pert_emb = batch_data["pert_emb"]
            # Fallback to runtime batch shape when config does not expose a sampling dim.
            resolved_input_dim = int(input_dim) if input_dim is not None else int(pert_emb.shape[-1])

            if len(pert_emb) != current_batch_size:
                logger.warning("batch size mismatch detected; this may be the last batch")
            current_batch_size = len(pert_emb)
            device = pert_emb.device
            gene_emb = build_gene_embedding_cache(model, batch_data, device)
            self_condition = build_self_condition(cfg, model, batch_data, gene_emb)
            # Generate samples
            start_time = time.time()
            mmd_loss = SamplesLoss(loss="energy", blur=0.05, scaling=0.5).to(device)

            mask = ~batch_data["is_padded_list"].bool()

            sample, traj = sample_fn(
                model.model,
                (current_batch_size, cell_set_number, resolved_input_dim),
                self_condition=self_condition,
                clip_denoised=clip_denoised,
                device=device,
                progress=progress,
                **sampling_kwargs
            )

            # remove duplicated sample that are used for padding
            pert_emb = pert_emb[mask]
            if sample is not None:
                sample = sample[mask]

            logger.debug("sample: %s", sample)
            logger.debug("pert_emb: %s", pert_emb)

            pert_emb_cpu = pert_emb.detach().cpu()
            if sample is not None:
                sample_cpu = sample.detach().cpu()
            
            # Store samples
            np_mask = np.isin(batch_data["col_genes"][0], _genes)
            truth_np = pert_emb_cpu.numpy()[:, np_mask]
            if sample is not None:
                sample_np = sample_cpu.numpy()[:, np_mask]
            
            if sample is not None:

                r2_metric = r2_score(truth_np.mean(0), sample_np.mean(0))
                torch_mask = torch.as_tensor(np_mask, device=device, dtype=torch.bool)
                sample_eval = sample[:, torch_mask]
                truth_eval = pert_emb[:, torch_mask]
                mmd_metric = mmd_loss(sample_eval, truth_eval).item()

                batch_time = time.time() - start_time
                logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s, r2_metric for this batch: {r2_metric}, mmd_metric for this batch: {mmd_metric}")
            
            all_truths.append(truth_np)
            if sample is not None:
                all_samples.append(sample_np)

            # covariates
            all_covariates.extend(collect_batch_covariates(batch_data, dataloader, datamodule, mask))

    # Concatenate all samples
    all_truths = np.concatenate(all_truths, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)

    logger.info(f"Generated {all_samples.shape[0]} samples with shape {all_samples.shape}")

    r2_metric = r2_score(all_truths.mean(0), all_samples.mean(0))
    logger.info(f"Overall r2_metric: {r2_metric}")


    all_pert = np.concatenate([x[0] for x in all_covariates], axis=0)
    all_celltype = np.concatenate([x[1] for x in all_covariates], axis=0)
    all_batch = np.concatenate([x[2] for x in all_covariates], axis=0)
        
    ctrl_adata, var_index = load_ctrl_adata(cfg)

    # Getting X_ctrl.
    var_index = _genes

    X_ctrl = build_x_ctrl(ctrl_adata, _genes, cfg)

    if normalize_counts is not None:
        all_samples *= normalize_counts
        all_truths *= normalize_counts

    obs = build_obs_data(cfg, all_pert, all_celltype, all_batch, ctrl_adata)
    
    if len(np_mask) > 2000: # i.e., when cfg.data.data_name = "Tahoe100mPBMCPretrain"
        logger.info("Restricting to downstream data selected genes only...")
        assert (ctrl_adata.var.index[ctrl_adata.var.highly_variable] == _genes).all()

        # unmerged data setting, no need of using sel_mask
        sel_mask = np.isin(batch_data["col_genes"][0], _genes)
        if all_samples.shape[-1] != len(_genes):
        
            assert sel_mask.sum() == 2000
            all_samples = all_samples[:, sel_mask]
            all_truths = all_truths[:, sel_mask]

        # reorder
        cur_genes = np.array(batch_data["col_genes"][0])[sel_mask].tolist()
        sort_idx = [cur_genes.index(g) for g in _genes]
        assert (np.array(cur_genes)[sort_idx] == _genes).all()
        all_samples = all_samples[:, sort_idx]
        all_truths = all_truths[:, sort_idx]

        var_index = _genes

    pred_adata = anndata.AnnData(X=np.concatenate([all_samples, X_ctrl]), obs=obs)
    true_adata = anndata.AnnData(X=np.concatenate([all_truths, X_ctrl]), obs=obs)
    if var_index is not None:
        pred_adata.var.index = var_index
        true_adata.var.index = var_index

    # Use a shared timestamp so that auxiliary files align with the main outputs.
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    save_adata(pred_adata, true_adata, cfg, logger, timestamp=run_timestamp)

    gc.collect()

    return all_truths, all_samples, all_trajectories, None
