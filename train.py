import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

logger = logging.getLogger(__name__)


def save_stats(dataset, run_dir):
    """
    Compute and save motion and text embedding statistics for normalization.
    
    Args:
        dataset: The training dataset to compute statistics from
        run_dir: Directory to save the statistics
    """
    is_training = dataset.is_training
    # don't drop anything
    dataset.is_training = False

    motion_stats_dir = os.path.join(run_dir, "motion_stats")
    os.makedirs(motion_stats_dir, exist_ok=True)

    text_stats_dir = os.path.join(run_dir, "text_stats")
    os.makedirs(text_stats_dir, exist_ok=True)

    from tqdm import tqdm
    import torch
    from src.normalizer import Normalizer

    logger.info("Compute motion embedding stats")
    motionfeats = torch.cat([x["x"] for x in tqdm(dataset)])
    mean_motionfeats = motionfeats.mean(0)
    std_motionfeats = motionfeats.std(0)

    motion_normalizer = Normalizer(base_dir=motion_stats_dir, disable=True)
    motion_normalizer.save(mean_motionfeats, std_motionfeats)

    logger.info("Compute text embedding stats")
    textfeats = torch.cat([x["tx"]["x"] for x in tqdm(dataset)])
    mean_textfeats = textfeats.mean(0)
    std_textfeats = textfeats.std(0)

    text_normalizer = Normalizer(base_dir=text_stats_dir, disable=True)
    text_normalizer.save(mean_textfeats, std_textfeats)

    # re enable droping
    dataset.is_training = is_training


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    """
    Main training function for STMC with optional keyframe conditioning.
    
    Args:
        cfg: Hydra configuration object containing all training parameters
    """
    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        resume_dir = cfg.resume_dir
        max_epochs = cfg.trainer.max_epochs
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        cfg.trainer.max_epochs = max_epochs
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{resume_dir}")
    else:
        resume_dir = None
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    import src.prepare  # noqa
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")
    print(cfg.data)
    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")

    # Log keyframe conditioning settings if enabled
    if hasattr(cfg, 'diffusion') and hasattr(cfg.diffusion, 'keyframe_conditioned'):
        if cfg.diffusion.keyframe_conditioned:
            logger.info("Keyframe conditioning enabled with following settings:")
            logger.info(f"  - Selection scheme: {cfg.diffusion.keyframe_selection_scheme}")
            logger.info(f"  - Keyframe mask probability: {cfg.diffusion.keyframe_mask_prob}")
            logger.info(f"  - Number of keyframes: {cfg.diffusion.n_keyframes}")
        else:
            logger.info("Keyframe conditioning disabled")
    
    if resume_dir is not None:
        logger.info("Computing statistics")
        save_stats(train_dataset, cfg.run_dir)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    logger.info("Loading the model")
    diffusion = instantiate(cfg.diffusion)
    
    # Log model architecture details for keyframe conditioning
    if hasattr(diffusion, 'keyframe_conditioned') and diffusion.keyframe_conditioned:
        logger.info("Diffusion model initialized with keyframe conditioning")
        logger.info(f"  - Denoiser type: {type(diffusion.denoiser).__name__}")
        if hasattr(diffusion.denoiser, 'keyframe_conditioned'):
            logger.info(f"  - Denoiser keyframe support: {diffusion.denoiser.keyframe_conditioned}")

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
