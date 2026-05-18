"""Lightning training module split from lightning_module (logic-preserving)."""

import gc
import pickle
import sys
import time
import tracemalloc

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from geomloss import SamplesLoss
from pytorch_lightning.utilities import grad_norm
from transformers.trainer_pt_utils import get_parameter_names

from src.common.utils import get_short_dsname
from src.models.covariate_encoding import CovEncoder

from src.models.lightning.lightning_factories import (
    create_diffusion,
    create_named_schedule_sampler,
    get_optimizer,
    model_init_fn,
)

class PlModel(pl.LightningModule):
    """Plmodel implementation used by the PerturbDiff pipeline."""
    def __init__(self, cov_encoding_cfg, model_cfg, py_logger, optimizer_cfg, trainer_cfg, all_split_names):
        """
        Initialize the class instance.

        :param cov_encoding_cfg: Configuration object for this component.
        :param model_cfg: Model configuration.
        :param py_logger: Input `py_logger` value.
        :param optimizer_cfg: Optimizer configuration.
        :param trainer_cfg: Trainer configuration.
        :param all_split_names: Input `all_split_names` value.
        :return: None.
        """
        super().__init__()
        self.cov_encoding_cfg = cov_encoding_cfg
        self.model_cfg = model_cfg
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        self.trainer_cfg = trainer_cfg
        self.all_split_names = all_split_names

        for split in self.all_split_names:
            setattr(self, f"validation_{split}_step_outputs", [])

        self.cov_encoder = CovEncoder(self.cov_encoding_cfg)
        self.gene_embedding = {}
        if self.model_cfg.use_gene_embedding:
            for gene_emb_file in self.cov_encoding_cfg.gene_embedding_path:
                # read pickle file
                with open(gene_emb_file, "rb") as f:
                    gene_emb = pickle.load(f)
                    # transform to tensor
                    gene_emb = {k: torch.tensor(v, dtype=torch.float32) for k, v in gene_emb.items()}
                    self.gene_embedding.update(gene_emb)
        else:
            self.gene_embedding = None
        self.gene_name_embedding_cache = {}

        self.model = model_init_fn(self.model_cfg, self.cov_encoding_cfg)
        self.diffusion = create_diffusion(self.model_cfg)

        sampler_name = getattr(self.model_cfg, "schedule_sampler", "uniform")
        self.schedule_sampler = create_named_schedule_sampler(sampler_name, self.diffusion)

        self.model_cfg.p_drop_cond = getattr(self.model_cfg, "p_drop_cond", 0.0)

        self._last_logged_batch_start_time = time.monotonic()
        self.validation_step_outputs = [] 

        blur = self.optimizer_cfg.get("blur", 0.05)
        self.loss_fn = SamplesLoss(loss="energy", blur=blur)

        self.save_hyperparameters()  # Save hparams for checkpointing
    
    def log_data(self,
                log_dict,
                train=True):
        """
        Log data.

        :param log_dict: Dictionary containing mapped values.
        :param train: Whether to run in training mode.
        :return: Computed output(s) for this function.
        """
        if train:
            self.log_dict(
                    log_dict,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    batch_size=self.optimizer_cfg.micro_batch_size,
                    logger=True,
                    sync_dist=True,
                )
        else:
            self.log_dict(
                    log_dict,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=self.optimizer_cfg.micro_batch_size,
                    logger=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: Current batch tensor(s).
        :param batch_idx: Index value used for lookup or slicing.
        :return: Computed output(s) for this function.
        """
        loss_dict = self._compute_loss(batch)
        loss = (loss_dict["loss"] * loss_dict["weights"]).mean()

        log_dict = {"training_loss_step": loss}

        for k,v in loss_dict.items():
            if k.startswith("dataset_loss_"):
                log_dict[k] = v

        assert self.model.model_name == "Cross_DiT"
        log_dict["training_Pert_loss_step"] = loss_dict["loss1"]
        log_dict["training_MMD_Pert_loss_step"] = loss_dict["mmd1"]

        self.log_data(log_dict, train=True)

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step.

        :param batch: Current batch tensor(s).
        :param batch_idx: Index value used for lookup or slicing.
        :param dataloader_idx: Index value used for lookup or slicing.
        :return: Computed output(s) for this function.
        """
        split = self.all_split_names[dataloader_idx]

        loss_dict = self._compute_loss(batch)
        loss = (loss_dict["loss"] * loss_dict["weights"]).mean()

        log_dict = {f"validation_{split}_loss": loss}

        for k,v in loss_dict.items():
            if k.startswith("dataset_loss_"):
                log_dict[k] = v

        assert self.model.model_name == "Cross_DiT"
        log_dict[f"validation_{split}_Pert_loss"] = loss_dict["loss1"]

        log_dict[f"validation_{split}_MMD_Pert_loss"] = loss_dict["mmd1"]
        self.log_data(log_dict, train=False)

        self.validation_step_outputs.append({f"validation_{split}_loss": loss})
        getattr(self, f"validation_{split}_step_outputs").append({f"validation_{split}_loss": loss})
        return {f"validation_{split}_loss": loss}

    def on_validation_epoch_end(self):
        """Execute `on_validation_epoch_end` and return values used by downstream logic."""
        for split in self.all_split_names:
            arr = getattr(self, f"validation_{split}_step_outputs")
            if len(arr) > 0:
                vals = torch.stack([o[f"validation_{split}_loss"] for o in arr])
                mean_val = vals.mean()
                self.log(
                    f"validation_{split}_loss_epoch",
                    mean_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            getattr(self, f"validation_{split}_step_outputs").clear()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Execute `on_train_batch_end` and return values used by downstream logic."""
        if batch_idx > 0 and batch_idx % self.trainer_cfg.log_every_n_steps == 0:
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            self._last_logged_batch_start_time = time.monotonic()
            time_per_step = elapsed_time / self.trainer_cfg.log_every_n_steps
            self.log("sec/step", time_per_step, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)

    def _encode_covariates(self, batch):
        """Execute `_encode_covariates` and return values used by downstream logic."""
        cov_reprs = self.cov_encoder(batch["cov_pert"], batch["cov_celltype"], batch["cov_batch"])

        return cov_reprs

    def _compute_loss(self, batch):
        """
        Compute a training loss value.

        :param batch: Current batch.
        :return: Loss tensor(s) and/or scalar metrics.
        """
        batch["batch_emb"] = self._encode_covariates(batch)
        pert_emb = batch["pert_emb"]
        device = pert_emb.device

        if self.gene_embedding is None:
            gene_emb = None
        else:
            gene_emb = []
            for i, dataset_name in enumerate(batch["ds_name"]):
                dataset_name = dataset_name[0]
                if dataset_name not in self.gene_name_embedding_cache:
                    self.gene_name_embedding_cache[dataset_name] = torch.stack([self.gene_embedding.get(id, torch.zeros(5120)) for id in batch["col_genes"][i]])
                gene_emb.append(self.gene_name_embedding_cache[dataset_name])

            gene_emb = torch.stack(gene_emb).to(device)
        #batch["gene_emb"] = torch.stack([self.gene_embedding.get(id, torch.zeros(5120)) for id in batch["col_genes"]])
        #batch["gene_emb"] = batch["gene_emb"].expand(pert_emb.shape[0], -1, -1).to(device)

        cond = {"batch_emb": batch["batch_emb"],
                "cont_emb": batch["cont_emb"],
                "gene_emb": gene_emb,
                "cov_celltype": batch["cov_celltype"],
                "cov_pert": batch["cov_pert"],
                "ds_name": batch["ds_name"],
                }

        t, weights = self.schedule_sampler.sample(pert_emb.shape[0], device)

        losses = self.diffusion.training_losses(
            self.model, 
            pert_emb, 
            t, 
            self_condition=cond, 
            model_kwargs=None, 
            noise=None,
            p_drop_cond=self.model_cfg.p_drop_cond,
            MMD_loss_fn=self.loss_fn,
        )
        return_dict = {}
        
        keys = list(set([(get_short_dsname(x)) for x in self.model.model_cfg.dataset_dict]))
        name_arr = np.array([get_short_dsname(x) for x in cond["ds_name"]])
        for ds_name in keys:
            if (name_arr == ds_name).any():
                return_dict[f"dataset_loss_mse1_{ds_name}"] = losses["loss1"][name_arr == ds_name].nanmean().item()
                return_dict[f"dataset_loss_mmd1_{ds_name}"] = losses["mmd1_list"][name_arr == ds_name].nanmean().item()
            else:
                return_dict[f"dataset_loss_mse1_{ds_name}"] =  0
                return_dict[f"dataset_loss_mmd1_{ds_name}"] = 0
        
        assert self.model.model_name == "Cross_DiT"
        losses["loss"] = losses["loss1"]
        
        if not self.optimizer_cfg.use_mse_loss:
            losses["loss"] = 0

        losses["loss"] += losses["mmd1"] * self.optimizer_cfg.MMD_loss_factor
        return_dict["mmd1"] = losses["mmd1"]

        return_dict["loss1"] = (losses["loss1"] * weights).mean()

        if hasattr(self.schedule_sampler, "update_with_local_losses"):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        return_dict["loss"] = losses["loss"]
        return_dict["weights"] = weights

        return return_dict

    def configure_optimizers(self):
        """
        return optimizers and schedulers
        """
        decay_names = set(get_parameter_names(self.model, [torch.nn.LayerNorm]))
        decay_names_cov = set(get_parameter_names(self.cov_encoder, [torch.nn.LayerNorm]))
        decay_names.update(decay_names_cov)
        decay_names = {n for n in decay_names if "bias" not in n and "layer_norm" not in n and "layernorm" not in n}

        params_decay, params_nodecay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (params_decay if n in decay_names else params_nodecay).append(p)

        for n, p in self.cov_encoder.named_parameters():
            if not p.requires_grad:
                continue
            (params_decay if n in decay_names else params_nodecay).append(p)

        main_groups = [
            {"params": params_decay, "weight_decay": self.optimizer_cfg.optimizer.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        opt_main = get_optimizer(main_groups, self.optimizer_cfg.optimizer)
        
        sched_main = hydra.utils.call(self.optimizer_cfg.scheduler, optimizer=opt_main)
        return [opt_main], [{"scheduler": sched_main, "interval": "step", "frequency": 1, "monitor": "validation_loss", }]

    def on_before_optimizer_step(self, optimizer):
        """Execute `on_before_optimizer_step` and return values used by downstream logic."""
        norms = grad_norm(self.model, norm_type=2)
        device = next(self.parameters()).device
        norms = {k:v.to(device) for k,v in norms.items()}

        
        self.log_dict(
            norms,
            prog_bar=True,
            sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size,
            add_dataloader_idx=False,
        )
