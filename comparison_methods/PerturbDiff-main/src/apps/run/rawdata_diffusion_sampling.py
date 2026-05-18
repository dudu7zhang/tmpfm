"""Sampling entrypoint split from rawdata_diffusion_sample.py (logic-preserving)."""

import os
import sys
import logging
import hydra
import omegaconf
import pytorch_lightning as pl

# local imports
exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(exc_dir)
module_logger = logging.getLogger(__name__)
module_logger.info("exc_dir: %s", exc_dir)


from src.common.utils import setup_loggings
from src.apps.sampling.sampling_generation import generate_samples
from src.apps.sampling.sampling_utils import setup_device
from src.apps.sampling.sampling_setup import build_sampling_datamodule, load_sampling_model, populate_covariate_cfg
from src.apps.training.training_runtime import setup_trainer

@hydra.main(version_base=None, config_path="../../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg):
    """
    Load a pretrained diffusion model and generate samples.
    """
    omegaconf.OmegaConf.resolve(cfg)

    
    # Set up logging
    logger = setup_loggings(cfg)
    logger.info("Starting diffusion model sampling")
    logger.info(f"Config: {omegaconf.OmegaConf.to_yaml(cfg)}")
    
    # set seed before initializing models
    pl.seed_everything(cfg.optimization.seed)

    datamodule = build_sampling_datamodule(cfg, logger)

    populate_covariate_cfg(cfg, datamodule)

    '''
    Below is functionality to automatically detect and load the correct model parameters from the checkpoint 
    without having to manually specify the model parameters etc. in the hydra config. 
    This will be enabled once we finalize what config variables we want to use.
    '''
    model = load_sampling_model(cfg, logger, datamodule)

    from pytorch_lightning.utilities import model_summary
    summary = model_summary.ModelSummary(model, max_depth=2)
    logger.info(summary)

    device = setup_device(cfg, logger)

    # Generate samples
    logger.info("Starting sample generation...")
    datamodule.setup_dataset()

    truths, samples, trajectories, decoded = generate_samples(
        model, model.diffusion, cfg, device, logger, datamodule, 
        pca_for_decode=None,
    )
    
    # Print sample statistics
    logger.info(f"Sample statistics:")
    logger.info(f"  Shape: {samples.shape}")
    logger.info(f"  Mean: {samples.mean():.4f}")
    logger.info(f"  Std: {samples.std():.4f}")
    logger.info(f"  Min: {samples.min():.4f}")
    logger.info(f"  Max: {samples.max():.4f}")

if __name__ == "__main__":
    main()
