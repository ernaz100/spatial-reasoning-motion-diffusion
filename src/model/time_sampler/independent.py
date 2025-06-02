import torch
from dataclasses import dataclass
from typing import Union
from jaxtyping import Float
from torch import device, Tensor

from .time_sampler import TimeSampler, TimeSamplerCfg


@dataclass
class IndependentCfg(TimeSamplerCfg):
    """Configuration for the Independent time sampler (baseline)."""
    name: str = "independent"


class Independent(TimeSampler[IndependentCfg]):
    """
    Independent time sampler - the baseline approach used in Diffusion Forcing.
    
    This samples each frame's time value independently from U(0, 1), which leads
    to the mean time t̄ following a Bates distribution concentrated around 0.5.
    This is what SRM identified as problematic for training.
    """

    def get_time(
        self,
        batch_size: int,
        num_samples: int = 1,
        device: Union[device, str] = "cpu",
    ) -> Float[Tensor, "batch sample height width"]:
        """
        Sample time values independently for each frame.
        
        Each frame gets an independent time value t_i ~ U(0, 1).
        This is the standard approach but leads to biased t̄ distribution.
        """
        # For motion data, resolution is (num_frames, num_features)
        num_frames, num_features = self.resolution
        
        # Sample independent time values for each frame
        t = torch.rand(batch_size, num_samples, num_frames, num_features, device=device)
        
        return t 