import logging
from collections import defaultdict
from tqdm import tqdm

import torch

from .gaussian import GaussianDiffusion, masked
from .time_sampler import MeanBeta, MeanBetaCfg, Independent, IndependentCfg
from ..data.collate import length_to_mask, collate_tensor_with_padding
from src.utils.keyframe_masking import (
    get_keyframes_mask, 
    apply_keyframe_conditioning,
    create_keyframe_loss_mask,
    get_random_keyframe_dropout_mask
)


logger = logging.getLogger(__name__)


class SRMGaussianDiffusion(GaussianDiffusion):
    """
    SRM-enhanced Gaussian Diffusion for motion generation.
    
    This extends the base GaussianDiffusion with:
    - Frame-level time sampling (instead of sequence-level)
    - Loss weighting based on empirical p(t) estimation
    - Support for different sequentialization strategies
    """
    
    name = "srm_gaussian"

    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        prediction: str = "x",
        lr: float = 2e-4,
        # Keyframe conditioning parameters
        keyframe_conditioned: bool = False,
        keyframe_selection_scheme: str = "random_frames",
        keyframe_mask_prob: float = 0.1,
        zero_keyframe_loss: bool = False,
        n_keyframes: int = 10,
        keyframe_loss_weight: float = 2.0,
        # SRM-specific parameters
        time_sampler_type: str = "mean_beta",  # "mean_beta" or "independent"
        time_sampler_sharpness: float = 1.0,
        max_frames: int = 81,  # Maximum number of frames to expect
        motion_features: int = 205,  # Number of motion features per frame
        enable_frame_level_conditioning: bool = True,  # Whether to condition on frame-level times
    ):
        super().__init__(
            denoiser=denoiser,
            schedule=schedule,
            timesteps=timesteps,
            motion_normalizer=motion_normalizer,
            text_normalizer=text_normalizer,
            prediction=prediction,
            lr=lr,
            keyframe_conditioned=keyframe_conditioned,
            keyframe_selection_scheme=keyframe_selection_scheme,
            keyframe_mask_prob=keyframe_mask_prob,
            zero_keyframe_loss=zero_keyframe_loss,
            n_keyframes=n_keyframes,
            keyframe_loss_weight=keyframe_loss_weight,
        )
        
        # SRM-specific parameters
        self.max_frames = max_frames
        self.motion_features = motion_features
        self.enable_frame_level_conditioning = enable_frame_level_conditioning
        
        # Initialize time sampler
        if time_sampler_type == "mean_beta":
            time_sampler_cfg = MeanBetaCfg(
                name="mean_beta",
                sharpness=time_sampler_sharpness,
            )
            self.time_sampler = MeanBeta(
                cfg=time_sampler_cfg,
                resolution=(max_frames, motion_features)
            )
        elif time_sampler_type == "independent":
            time_sampler_cfg = IndependentCfg(name="independent")
            self.time_sampler = Independent(
                cfg=time_sampler_cfg,
                resolution=(max_frames, motion_features)
            )
        else:
            raise ValueError(f"Unknown time sampler type: {time_sampler_type}")
        
        logger.info(f"Initialized SRM with {time_sampler_type} time sampler")

    def q_sample_frame_level(self, xstart, t_frames, noise=None, mask=None):
        """
        Sample from q(xt | xstart) with frame-level time values.
        
        Args:
            xstart: Clean motion data [batch_size, max_frames, n_features]
            t_frames: Time values per frame [batch_size, max_frames] (0.0 to 1.0)
            noise: Noise tensor (optional)
            mask: Valid frame mask [batch_size, max_frames]
            
        Returns:
            Noisy motion data with frame-specific noise levels
            
        Note: Frames with t=0 will remain exactly as xstart (perfect conditioning)
        """
        if noise is None:
            noise = torch.randn_like(xstart)
        
        # Ensure noise respects mask
        if mask is not None:
            noise = masked(noise, mask)
        
        batch_size, max_frames, n_features = xstart.shape
        
        # Expand time values to feature dimension for broadcasting
        t_expanded = t_frames.unsqueeze(-1).expand(-1, -1, n_features)  # [B, T, F]
        
        # SRM: Handle t=0 case explicitly for perfect conditioning
        zero_time_mask = (t_expanded <= 1e-7)  # Use small epsilon for numerical stability
        
        # For non-zero times, compute noise schedule coefficients
        # Clamp to avoid indexing issues
        t_indices = (t_expanded * (self.timesteps - 1)).long().clamp(0, self.timesteps - 1)
        
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t_indices]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t_indices]
        
        # Apply forward diffusion per frame
        xt = sqrt_alphas_cumprod * xstart + sqrt_one_minus_alphas_cumprod * noise
        
        # SRM: Ensure frames with t≈0 remain exactly as clean input (perfect conditioning)
        xt = torch.where(zero_time_mask, xstart, xt)
        
        # Ensure padded regions remain zeros
        if mask is not None:
            xt = masked(xt, mask)
        
        return xt

    def diffusion_step(self, batch, batch_idx, training=False):
        """Enhanced diffusion step with SRM frame-level time sampling and noise-level conditioning."""
        mask = batch["mask"]  # Valid motion mask [batch_size, max_frames]
        batch_size, max_frames = mask.shape

        # Normalization
        x = masked(self.motion_normalizer(batch["x"]), mask)
        
        # Prepare base conditioning
        y = {
            "length": batch["length"],
            "mask": mask,
            "tx": self.prepare_tx_emb(batch["tx"]),
        }

        # SRM: Sample frame-level time values and loss weights
        if hasattr(self.time_sampler, 'get_frame_level_times'):
            # Use the frame-level method for better efficiency
            t_frames, loss_weights = self.time_sampler.get_frame_level_times(
                batch_size, max_frames, device=x.device
            )
        else:
            # Fallback to full sampling and extract frame values
            t_full, weights_full = self.time_sampler(batch_size, device=x.device)
            t_frames = t_full.squeeze(1).mean(-1)  # Average over features
            loss_weights = weights_full.squeeze(1).mean(-1)
        
        # Ensure time values are properly masked and clamped
        t_frames = t_frames * mask.float()  # Zero out padded frames
        t_frames = torch.clamp(t_frames, 0.0, 1.0)
        loss_weights = loss_weights * mask.float()

        # SRM Keyframe Conditioning: Set noise level to 0 for keyframes instead of using masks
        keyframe_mask = None
        if self.keyframe_conditioned and training:
            keyframe_mask = get_keyframes_mask(
                data=batch["x"],  
                lengths=batch["length"],
                edit_mode=self.keyframe_selection_scheme,
                n_keyframes=self.n_keyframes,
                device=x.device
            )
            
            if self.keyframe_mask_prob > 0.0:
                keyframe_mask = get_random_keyframe_dropout_mask(
                    keyframe_mask, 
                    dropout_prob=self.keyframe_mask_prob
                )
            
            # SRM: Set time to 0 for keyframes (no noise = perfect conditioning)
            keyframe_mask = keyframe_mask & mask  # Ensure keyframes are within valid sequence
            t_frames[keyframe_mask] = 0.0  # Zero noise level for conditioning
            
            # Store keyframe information for logging and optional denoiser conditioning
            y["keyframe_mask"] = keyframe_mask
            y["keyframe_x0"] = x

        # Create noisy version with frame-level times (keyframes will have t=0, so no noise)
        noise = masked(torch.randn_like(x), mask)
        xt = self.q_sample_frame_level(xstart=x, t_frames=t_frames, noise=noise, mask=mask)
        
        # Note: No additional keyframe conditioning needed here since t=0 frames are already clean
        # The q_sample_frame_level will preserve original values where t_frames=0

        # Convert frame times to format expected by denoiser
        # For compatibility, we'll pass the mean time per sequence (excluding keyframes if present)
        if keyframe_mask is not None:
            # Compute mean excluding keyframes for a more representative sequence time
            non_keyframe_mask = mask & (~keyframe_mask)
            if non_keyframe_mask.sum() > 0:
                t_sequence = (t_frames * non_keyframe_mask.float()).sum(dim=1) / non_keyframe_mask.sum(dim=1).clamp(min=1)
            else:
                t_sequence = (t_frames * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            t_sequence = (t_frames * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        t_sequence_discrete = (t_sequence * (self.timesteps - 1)).long().clamp(0, self.timesteps - 1)
        
        # Add frame-level time information to conditioning if enabled
        if self.enable_frame_level_conditioning:
            y["frame_times"] = t_frames
            y["frame_time_mask"] = mask
        
        # Denoise
        output = masked(self.denoiser(xt, y, t_sequence_discrete), mask)

        # Predictions
        xstart = masked(self.output_to("x", output, xt, t_sequence_discrete), mask)
        
        # Compute frame-level reconstruction loss
        reconstruction_loss = self.reconstruction_loss(xstart, x)  # [B, T, F]
        
        # Apply SRM loss weighting combined with keyframe weighting
        frame_loss_weights = loss_weights.unsqueeze(-1)  # [B, T, 1]
        
        if self.keyframe_conditioned and keyframe_mask is not None:
            # Apply keyframe-specific loss weighting
            keyframe_weights = mask.float().clone()
            if self.zero_keyframe_loss:
                # Zero loss for keyframes (they should be perfectly predicted anyway)
                keyframe_weights[keyframe_mask] = 0.0
            else:
                # Higher weight for keyframes to ensure they're well-preserved
                keyframe_weights[keyframe_mask] = self.keyframe_loss_weight
            
            # Combine SRM weights with keyframe weights
            keyframe_weights_expanded = keyframe_weights.unsqueeze(-1)
            final_weights = frame_loss_weights * keyframe_weights_expanded
        else:
            # Use only SRM weights
            final_weights = frame_loss_weights
        
        # Compute weighted loss
        weighted_loss = reconstruction_loss * final_weights
        total_loss = weighted_loss.sum()
        valid_elements = final_weights.sum()
        xloss = total_loss / (valid_elements + 1e-8)
        
        loss = {"loss": xloss}
        
        # Additional logging for SRM
        if training:
            # Log time sampling statistics
            mean_frame_time = (t_frames * mask.float()).sum() / mask.sum().clamp(min=1)
            loss["mean_frame_time"] = mean_frame_time
            loss["time_variance"] = ((t_frames - mean_frame_time) ** 2 * mask.float()).mean()
            
            # Log loss weighting statistics
            mean_loss_weight = (loss_weights * mask.float()).sum() / mask.sum().clamp(min=1)
            loss["mean_loss_weight"] = mean_loss_weight
            
            if self.keyframe_conditioned and keyframe_mask is not None:
                keyframe_ratio = keyframe_mask.float().mean()
                loss["keyframe_ratio"] = keyframe_ratio
                
                # Log statistics about zero-noise conditioning
                zero_noise_frames = (t_frames == 0.0) & mask
                loss["zero_noise_ratio"] = zero_noise_frames.float().mean()
        
        return loss 