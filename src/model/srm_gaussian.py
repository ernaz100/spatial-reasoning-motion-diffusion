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
        prediction: str = "eps",
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
        variance_loss_weight: float = 0.1,  # Weight for variance loss (small as suggested in paper)
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
        self.variance_loss_weight = variance_loss_weight
        
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

    def output_to_frame_level(self, target_space, output, xt, t_frames, mask=None):
        """
        Frame-level version of output_to that handles different timesteps per frame.
        
        Args:
            target_space: Target space ('x', 'xstart', 'eps', 'score')
            output: Model output [batch_size, max_frames, n_features]
            xt: Noisy input [batch_size, max_frames, n_features]  
            t_frames: Frame-level time values [batch_size, max_frames] (0.0 to 1.0)
            mask: Valid frame mask [batch_size, max_frames] (optional)
            
        Returns:
            Converted output in target space with frame-specific transformations
        """
        batch_size, max_frames, n_features = output.shape
        
        # Convert continuous time values to discrete indices
        t_discrete = (t_frames * (self.timesteps - 1)).long().clamp(0, self.timesteps - 1)
        
        # Expand to match feature dimensions for coefficient extraction
        t_expanded = t_discrete.unsqueeze(-1).expand(-1, -1, n_features)  # [B, T, F]
        
        # Apply frame-level conversion
        if self.prediction == target_space or target_space == "output":
            result = output
        elif self.prediction == "eps" and target_space in ["x", "xstart"]:
            # Convert eps to xstart frame by frame
            inv_sqrt_alphas_cumprod = self.inv_sqrt_alphas_cumprod[t_expanded]
            sqrt_inv_alphas_cumprod_minus_one = self.sqrt_inv_alphas_cumprod_minus_one[t_expanded]
            result = inv_sqrt_alphas_cumprod * xt - sqrt_inv_alphas_cumprod_minus_one * output
        elif self.prediction in ["x", "xstart"] and target_space == "eps":
            # Convert xstart to eps frame by frame  
            sqrt_inv_one_minus_alphas_cumprod = self.sqrt_inv_one_minus_alphas_cumprod[t_expanded]
            sqrt_alphas_cumprod_over_sqrt_one_minus_alphas_cumprod = self.sqrt_alphas_cumprod_over_sqrt_one_minus_alphas_cumprod[t_expanded]
            result = sqrt_inv_one_minus_alphas_cumprod * xt - sqrt_alphas_cumprod_over_sqrt_one_minus_alphas_cumprod * output
        else:
            # For other conversions, fall back to frame-by-frame processing
            # This is less efficient but handles all cases
            result = torch.zeros_like(output)
            for b in range(batch_size):
                for t in range(max_frames):
                    if mask is None or mask[b, t]:
                        t_val = t_discrete[b, t:t+1]  # [1] 
                        frame_output = output[b:b+1, t:t+1]  # [1, 1, F]
                        frame_xt = xt[b:b+1, t:t+1]  # [1, 1, F]
                        
                        # Use parent class method for single frame
                        frame_result = self.output_to(target_space, frame_output, frame_xt, t_val)
                        result[b, t] = frame_result[0, 0]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # [batch_size, max_frames]
                mask_expanded = mask.unsqueeze(-1)  # [batch_size, max_frames, 1]
            else:  # Already [batch_size, max_frames, 1] or [batch_size, max_frames, n_features]
                mask_expanded = mask
            result = result * mask_expanded.float()
            
        return result

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
        
        # Expand time values to feature dimension for broadcasting -> values from t_frames[B, T] are copied to each position int_expanded[B, T, :]
        t_expanded = t_frames.unsqueeze(-1).expand(-1, -1, n_features)  # [B, T, F]
        
        # SRM: Handle t=0 case explicitly for perfect conditioning
        zero_time_mask = (t_expanded <= 1e-7)  # Use small epsilon for numerical stability
        
        # For non-zero times, compute noise schedule coefficients
        # Clamp to avoid indexing issues
        t_expanded = t_expanded.clamp(0.0, 1.0)
        # Convert continuous time values (0-1) to discrete indices for noise schedule lookup
        # Scale to range [0, timesteps-1], convert to integers, and clamp to valid bounds
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

    def _calculate_primary_loss(
        self, 
        denoiser_main_output: torch.Tensor, 
        true_noise: torch.Tensor, 
        true_x0: torch.Tensor, 
        xt: torch.Tensor, 
        t_frames: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the primary supervised loss based on the model's prediction mode.

        Args:
            denoiser_main_output: The main output of the denoiser model (predicted noise or x0).
            true_noise: The ground truth noise that was added to x0.
            true_x0: The ground truth clean data.
            xt: The noisy input to the denoiser.
            t_frames: Frame-level time values.
            mask: Valid frame mask.

        Returns:
            A tensor representing the frame-level primary loss [B, T, F].
        """
        if self.prediction == "eps":
            # If the model predicts noise (eps), the loss is between the predicted noise and the actual noise.
            primary_loss_frames = self.reconstruction_loss(denoiser_main_output, true_noise)
        elif self.prediction in ["x", "xstart"]:
            # If the model predicts the clean data (x or xstart), the loss is between
            # the predicted clean data and the actual clean data.
            primary_loss_frames = self.reconstruction_loss(denoiser_main_output, true_x0)
        else:
            raise ValueError(
                f"Unexpected prediction type '{self.prediction}'. "
                f"Expected one of: 'eps', 'x', 'xstart'"
            )
        return primary_loss_frames

    def _calculate_variance_loss(
        self, 
        denoiser_main_output: torch.Tensor, 
        denoiser_log_variance: torch.Tensor,
        true_noise: torch.Tensor, 
        xt: torch.Tensor, 
        t_frames: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Calculates the variance loss (NLL) if log_variance is provided by the denoiser.

        Args:
            denoiser_main_output: The main output of the denoiser model.
            denoiser_log_variance: The predicted log variance from the denoiser.
            true_noise: The ground truth noise that was added to x0.
            xt: The noisy input to the denoiser.
            t_frames: Frame-level time values.
            mask: Valid frame mask.

        Returns:
            A tensor representing the frame-level variance loss [B, T, F], or None.
        """
        if denoiser_log_variance is None:
            raise ValueError(
                f"No log_variance provided in denoiser output. "
            )

        # Determine the predicted noise based on the model's prediction mode
        if self.prediction == "eps":
            predicted_noise_for_variance = denoiser_main_output
        elif self.prediction == "x": # Assumes "x" means x0 prediction
            # Convert from x0 prediction to noise prediction for variance loss
            predicted_noise_for_variance = self.output_to_frame_level("eps", denoiser_main_output, xt, t_frames, mask)
        else:
            raise ValueError(
                f"Unexpected prediction type '{self.prediction}'. "
                f"Expected one of: 'eps', 'x', 'xstart'"
            )

        # Compute NLL components for variance loss
        # NLL loss: -log N(ε | ε_θ(x^t), σ_θ(x^t)^2 I)
        # This equals: 0.5 * (log(2π) + log_variance + (noise - predicted_noise)^2 / variance)
        # Constant term log(2π) is omitted as it doesn't affect gradients.
        noise_error_sq = (true_noise - predicted_noise_for_variance) ** 2  # [B, T, F]
        
        # Ensure variance is not zero to avoid division by zero. Add small epsilon.
        variance = torch.exp(denoiser_log_variance) + 1e-8 
        variance_term = denoiser_log_variance + noise_error_sq / variance  # [B, T, F]
        
        variance_loss_frames = 0.5 * variance_term  # [B, T, F]
        return variance_loss_frames

    def _combine_and_weight_losses(
        self,
        primary_loss_frames: torch.Tensor,
        srm_loss_weights: torch.Tensor,
        mask: torch.Tensor,
        variance_loss_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies SRM loss weights and combines primary and variance losses.

        Args:
            primary_loss_frames: Frame-level primary loss [B, T, F].
            srm_loss_weights: SRM time sampler loss weights [B, T].
            mask: Valid frame mask [B, T].
            variance_loss_frames: Frame-level variance loss [B, T, F] (optional).

        Returns:
            A tuple containing:
            - total_loss (scalar): The final combined and weighted loss.
            - primary_loss_avg (scalar): The averaged weighted primary loss.
            - variance_loss_avg (scalar, optional): The averaged weighted variance loss, or None.
        """
        # Expand SRM loss weights to match feature dimensions
        frame_srm_weights = srm_loss_weights.unsqueeze(-1)  # [B, T, 1]
        
        final_weights = frame_srm_weights * mask.unsqueeze(-1).float() # Ensure masked regions have 0 weight

        # Compute weighted primary (reconstruction) loss
        weighted_primary_loss = primary_loss_frames * final_weights
        total_primary_loss_sum = weighted_primary_loss.sum()
        
        valid_elements_count = final_weights.sum() # Sums over B, T, and the singleton dimension

        primary_loss_avg = total_primary_loss_sum / (valid_elements_count + 1e-8)
        
        total_loss = primary_loss_avg

        weighted_variance_loss = variance_loss_frames * final_weights
        total_variance_loss_sum = weighted_variance_loss.sum()
        variance_loss_avg = total_variance_loss_sum / (valid_elements_count + 1e-8)
        
        total_loss = primary_loss_avg + self.variance_loss_weight * variance_loss_avg
        
        return total_loss, primary_loss_avg, variance_loss_avg

    def diffusion_step(self, batch, batch_idx, training=False):
        """Enhanced diffusion step with SRM frame-level time sampling and noise-level conditioning."""
        mask = batch["mask"]  # Valid motion mask [batch_size, max_frames]
        batch_size, max_frames = mask.shape

        # Normalization
        x = masked(self.motion_normalizer(batch["x"]), mask)

        if(max_frames == 0):
            return {"loss": None, "reconstruction_loss": None, "variance_loss": None}
        
        t_bar = torch.rand(batch_size, device=x.device)  # Random values between 0 and 1 for each batch element

        # SRM: Sample frame-level time values and loss weights
        t_frames, _ = self.time_sampler.get_times_for_t_bar(   
            t_bar = t_bar,
            num_frames=max_frames,
            device=x.device,
            calculate_weights=False
        )
        
        # Ensure time values are properly masked and clamped
        t_frames = t_frames * mask.float()  # Zero out padded frames
        t_frames = torch.clamp(t_frames, 0.0, 1.0)

        # Create noisy version with frame-level times
        noise = masked(torch.randn_like(x), mask)
        xt = self.q_sample_frame_level(xstart=x, t_frames=t_frames, noise=noise, mask=mask)
        
        # Prepare conditioning
        y = {
            "length": batch["length"],
            "mask": mask,
            "tx": self.prepare_tx_emb(batch["tx"]),
            "frame_times": t_frames,
            "t_bar": t_bar,
        }

        # Denoise
        denoiser_output_dict = self.denoiser(xt, y) 
        
        # Extract outputs from the denoiser
        denoiser_main_output = masked(denoiser_output_dict["noise_prediction"], mask)

        # 1. Calculate Primary Loss (reconstruction or noise prediction loss)
        primary_loss_frames = self._calculate_primary_loss(
            denoiser_main_output=denoiser_main_output,
            true_noise=noise,
            true_x0=x,
            xt=xt,
            t_frames=t_frames,
            mask=mask
        )

        # Construct the loss dictionary for output and logging
        loss_dict = {"loss": primary_loss_frames.mean()}
                
        return loss_dict 