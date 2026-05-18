"""Factory and builder utilities split from lightning_module (logic-preserving)."""

import os
import torch
from torch.optim import AdamW, Adam

from src.models.cross_dit.cross_dit_main import Cross_DiT
from src.models.diffusion.diffusion_core import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from src.models.diffusion.diffusion_schedules import get_named_beta_schedule
from src.models.resampling import UniformSampler
from src.models.weight_averaging_callback import WeightAveraging


class EMAWeightAveraging(WeightAveraging):
    """Emaweightaveraging implementation used by the PerturbDiff pipeline."""
    def __init__(self, decay=0.999, update_steps=100):
        """Special method `__init__`."""
        self._update_steps = update_steps
        super().__init__(avg_fn=torch.optim.swa_utils.get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        """Execute `should_update` and return values used by downstream logic."""
        if epoch_idx is not None:
            return True
        elif step_idx is not None:
            return step_idx % self._update_steps == 0
        else:
            raise ValueError("step_idx and epoch_idx cannot be both None.")


def model_init_fn(model_cfg, cov_cfg=None):
    """Execute `model_init_fn` and return values used by downstream logic."""
    model_type = model_cfg.model_type.lower()
    if model_type == "unet":
        raise NotImplementedError("unet model_type is removed from src.")
    elif model_type == "dit":
        raise NotImplementedError("dit model_type is removed from src.")
    elif model_type == "cross_dit":
        model = Cross_DiT(model_cfg=model_cfg)
    elif model_type == "linear":
        raise NotImplementedError("linear model_type is removed from src.")
    else:
        raise NotImplementedError(f"Model type {model_cfg.model_type} not implemented.")
    return model


def create_diffusion(model_cfg):
    """
    Create diffusion.

    :param model_cfg: Model configuration.
    :return: Requested object(s) for downstream use.
    """
    assert not model_cfg.learn_sigma, "model.learn_sigma=True is no longer supported in src."
    betas = get_named_beta_schedule(model_cfg.noise_schedule, model_cfg.steps, model_cfg.noise_schedule_gamma)
    assert not model_cfg.use_kl, "LossType.KL/RESCALED_KL is no longer supported in src."
    if model_cfg.rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=(
            ModelVarType.FIXED_LARGE
            if not model_cfg.sigma_small
            else ModelVarType.FIXED_SMALL
        ),
        loss_type=loss_type,
        rescale_timesteps=model_cfg.rescale_timesteps,
    )


def get_optimizer(optim_groups, optimizer_cfg):
    """Execute `get_optimizer` and return values used by downstream logic."""
    optim_cls = AdamW if optimizer_cfg.adam_w_mode else Adam
    if hasattr(optimizer_cfg, "lr"):
        return optim_cls(
            optim_groups,
            lr=optimizer_cfg.lr,
            eps=optimizer_cfg.eps,
            betas=(optimizer_cfg.betas[0], optimizer_cfg.betas[1]),
        )
    else:
        return optim_cls(
            optim_groups,
            eps=optimizer_cfg.eps,
            betas=(optimizer_cfg.betas[0], optimizer_cfg.betas[1]),
        )


def create_named_schedule_sampler(name, diffusion):
    """Execute `create_named_schedule_sampler` and return values used by downstream logic."""
    if name == "uniform":
        return UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
