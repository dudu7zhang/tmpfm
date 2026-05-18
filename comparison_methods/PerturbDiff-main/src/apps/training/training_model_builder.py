"""Model builders and model-init flows split from rawdata_diffusion_training.py (logic-preserving)."""

import hydra
from pytorch_lightning.utilities import model_summary

from src.apps.training.training_model_checkpoint import (
    load_plmodel_checkpoint,
    maybe_load_and_patch_checkpoint_model as _maybe_load_and_patch_checkpoint_model,
)
from src.apps.training.training_model_compare import maybe_compare_model_representation as _maybe_compare_model_representation

def build_model(cfg, logger, datamodule):
    """
    Build model.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param datamodule: Data module providing datasets and loaders.
    :return: Requested object(s) for downstream use.
    """
    model = hydra.utils.instantiate(
        cfg.lightning.model_module,
        _recursive_=False,
        cov_encoding_cfg=cfg.cov_encoding,
        model_cfg=cfg.model,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        trainer_cfg=cfg.trainer,
        all_split_names=datamodule.all_split_names,
    )

    summary = model_summary.ModelSummary(model, max_depth=2)
    logger.info(summary)

    model.model.group_mean = None
    model.model.group_mean_ctrl = None

    return model

def maybe_compare_model_representation(cfg, logger, datamodule):
    """Execute `maybe_compare_model_representation` and return values used by downstream logic."""
    return _maybe_compare_model_representation(
        cfg,
        logger,
        datamodule,
        load_plmodel_checkpoint=load_plmodel_checkpoint,
    )

def maybe_load_and_patch_checkpoint_model(cfg, model, logger):
    """Execute `maybe_load_and_patch_checkpoint_model` and return values used by downstream logic."""
    return _maybe_load_and_patch_checkpoint_model(cfg, model, logger)

def maybe_reinitialize_from_scratch(cfg, model, logger):
    """Execute `maybe_reinitialize_from_scratch` and return values used by downstream logic."""
    if cfg.model.reinitial_all_from_scratch:
        model.model.initialize_weights()
        logger.info("Re-initialize all layers")
