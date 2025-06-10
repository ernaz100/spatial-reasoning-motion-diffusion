import torch
from dataclasses import dataclass
from typing import Union
from jaxtyping import Float
from torch import device, Tensor
from torch.distributions import Beta

from .time_sampler import TimeSampler, TimeSamplerCfg


@dataclass
class MeanBetaCfg(TimeSamplerCfg):
    """Configuration for the MeanBeta time sampler that implements uniform t̄ sampling."""
    sharpness: float = 1.0  # Controls concentration around mean
    name: str = "mean_beta"


class MeanBeta(TimeSampler[MeanBetaCfg]):
    """
    MeanBeta time sampler implementing the uniform t̄ strategy from SRM paper.
    
    This ensures the mean noise level t̄ follows a uniform distribution during training,
    which matches what happens during inference for both parallel and autoregressive
    generation strategies.
    """

    def get_time(
        self,
        batch_size: int,
        num_samples: int = 1,
        device: Union[device, str] = "cpu",
    ) -> Tensor:
        """
        Sample time values using uniform t̄ strategy.
        
        First samples mean time t̄ ~ U(0, 1), then uses recursive allocation
        to generate individual time values t_i around that mean.
        """
        # Sample uniform mean time
        t_bar = torch.rand(batch_size, num_samples, device=device)
        
        # For motion data, we treat each frame as a spatial variable
        # Resolution is (num_frames, num_features)
        num_frames, num_features = self.resolution #TODO: num_features is not 205 but 1 here?
        
        # Generate time values for each frame using recursive allocation
        t_frames = self._recursive_allocation_sampling(
            t_bar, num_frames, device
        )
        
        # Expand to include feature dimension (all features in a frame share same t)
        t = t_frames.unsqueeze(-1).expand(-1, -1, -1, num_features)
        
        return t

    def _recursive_allocation_sampling(
        self,
        t_bar: Float[Tensor, "batch sample"],
        num_frames: int,
        device: Union[device, str] = "cpu",
    ) -> Tensor:
        """
        Recursive allocation sampling to generate frame-wise time values.
        
        This algorithm ensures that the mean of generated times equals t_bar
        while allowing individual frame times to vary according to sharpness.
        """
        batch_size, num_samples = t_bar.shape
        
        # Total sum constraint: sum(t_i) = num_frames * t_bar
        total_sum = num_frames * t_bar
        
        # Recursive allocation for each batch element
        result = torch.zeros(batch_size, num_samples, num_frames, device=device)
        
        for b in range(batch_size):
            for s in range(num_samples):
                result[b, s] = self._get_sum_constrained_vector(
                    total_sum[b, s].item(), num_frames, device
                )
        
        return result

    def _get_sum_constrained_vector(
        self,
        target_sum: float,
        dimension: int,
        device: Union[device, str] = "cpu",
    ) -> Tensor:
        """
        Generate a vector with specified sum using recursive binary splitting.
        
        Args:
            target_sum: The desired sum of all elements
            dimension: Number of elements in the vector
            device: Device to create tensor on
            
        Returns:
            Vector of length 'dimension' with sum equal to 'target_sum'
        """
        if dimension == 1:
            return torch.tensor([target_sum], device=device)
        
        # Split into two halves
        d1 = dimension // 2
        d2 = dimension - d1
        
        # Compute valid range for first half's contribution
        s_max1 = min(target_sum, d1)  # Maximum if all elements = 1
        s_max2 = min(target_sum, d2)
        s_min1 = max(0, target_sum - s_max2)  # Minimum to leave room for second half
        
        # Sample split point using Beta distribution for natural behavior
        # Higher sharpness -> more concentrated around midpoint
        alpha = beta = (dimension - 1 - (dimension % 2)) ** 1.05 * self.cfg.sharpness
        split_ratio = Beta(alpha, beta).sample().item()
        
        # Allocate sum to first half
        s1 = s_min1 + (s_max1 - s_min1) * split_ratio
        s2 = target_sum - s1
        
        # Recursively generate sub-vectors
        vec1 = self._get_sum_constrained_vector(s1, d1, device)
        vec2 = self._get_sum_constrained_vector(s2, d2, device)
        
        return torch.cat([vec1, vec2])

    def get_frame_level_times(
        self,
        batch_size: int,
        num_frames: int,
        device: Union[device, str] = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """
        Convenience method to get frame-level time values and weights.
        
        Returns:
            t: Time values for each frame [batch_size, num_frames]
            weights: Loss weights for each frame [batch_size, num_frames]
        """
        # Temporarily set resolution for frame-level sampling
        original_resolution = self.resolution
        self.resolution = (num_frames, 1)
        
        t_full, weights_full = self(batch_size, num_samples=1, device=device)
        
        # Extract frame-level values (squeeze feature dimension)
        t = t_full.squeeze(1).squeeze(-1)  # [batch_size, num_frames]
        weights = weights_full.squeeze(1).squeeze(-1)  # [batch_size, num_frames]
        
        # Restore original resolution
        self.resolution = original_resolution
        
        return t, weights

    def get_times_for_t_bar(
        self,
        t_bar: Float[Tensor, "batch"],
        num_frames: int,
        device: Union[device, str] = "cpu",
        calculate_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Generates frame-level time values for a given t_bar.

        Args:
            t_bar: The mean time values, one for each batch element.
            num_frames: The number of frames for which to generate times.
            device: The device to create tensors on.

        Returns:
            t: Time values for each frame [batch_size, num_frames]
            weights: Loss weights for each frame [batch_size, num_frames]
        """
        if t_bar.ndim == 1:
            # Add sample dimension
            t_bar = t_bar.unsqueeze(1)

        # Generate time values for each frame using recursive allocation
        t_frames = self._recursive_allocation_sampling(
            t_bar, num_frames, device
        )

        # remove sample dimension
        t = t_frames.squeeze(1)
        weights = None
        if calculate_weights:
            # Calculate weights
            weights = self.get_normalization_weights(t)

        return t, weights