import torch
import numpy as np
from typing import List, Tuple, Optional


def get_keyframes_mask(
    data: torch.Tensor, 
    lengths: torch.Tensor, 
    edit_mode: str = 'random_frames', 
    trans_length: int = 10, 
    n_keyframes: int = 5,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Computes the keyframe observation mask for motion data.
    
    Args:
        data (torch.Tensor): Input motion of shape [batch_size, max_frames, n_features].
        lengths (torch.Tensor): Actual lengths of the input motions of shape [batch_size].
        edit_mode (str): Defines which frames should be observed as keyframes.
                        Options: 'random_frames', 'benchmark_sparse', 'benchmark_clip', 
                                'pelvis_only', 'uncond', 'gmd_keyframes'
        trans_length (int): Transition length for benchmark tasks.
        n_keyframes (int): Number of keyframes to be observed for certain modes.
        device (torch.device): Device to create tensors on.
        
    Returns:
        torch.Tensor: Keyframe mask of shape [batch_size, max_frames], where True indicates
                     observed keyframes and False indicates frames to be inpainted.
    """
    
    if device is None:
        device = data.device
        
    batch_size, max_frames, n_features = data.shape
    
    # Initialize mask with all False (no frames observed initially)
    keyframe_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool, device=device)
    
    if edit_mode == 'random_frames':
        # Pick random frames throughout the sequence for training
        # This is the most general approach for keyframe conditioning
        for i, length in enumerate(lengths.cpu().numpy()):
            length = int(length)
            if length <= 1:
                continue
                
            # Randomly choose number of keyframes (between 1 and min(20, length))
            max_keyframes = min(n_keyframes, length)
            num_keyframes = np.random.randint(1, max_keyframes + 1)
            
            # Randomly select keyframe indices
            keyframe_indices = np.random.choice(range(length), num_keyframes, replace=False)
            keyframe_mask[i, keyframe_indices] = True

    elif edit_mode == 'random_joints':
        JOINTS_DIM = 22
        for i, length in enumerate(lengths.cpu().numpy()):
            length = int(length)
            num_keyframes = np.random.randint(1, length)
            gt_indices = np.random.choice(range(length), num_keyframes, replace=False)
            # random joint selection
            num_joints = np.random.randint(0, (JOINTS_DIM)*num_keyframes)
            rand_bin_mask = get_random_binary_mask(JOINTS_DIM, num_keyframes, num_joints).to(data.device) # 22, num_keyframes
            keyframe_mask[i, :, :, gt_indices] = rand_bin_mask.unsqueeze(1)  # set joints in keyframes
            keyframe_mask[i, 0, :, gt_indices] = True # set root joint
        
    else:
        raise ValueError(f"Unknown edit_mode: {edit_mode}")
    
    return keyframe_mask

def get_random_binary_mask(dim1, dim2, n):
    """
    Get a random binary mask of size (dim1, dim2) with n ones.
    """
    mask = torch.zeros((dim1, dim2), dtype=torch.bool)
    ones_indices = torch.randperm(dim1 * dim2)[:n]
    mask.view(-1)[ones_indices] = True

def apply_keyframe_conditioning(
    motion_batch: torch.Tensor,
    keyframe_mask: torch.Tensor,
    noise_scale: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply keyframe conditioning to a motion batch.
    
    Args:
        motion_batch (torch.Tensor): Motion data of shape [batch_size, max_frames, n_features]
        keyframe_mask (torch.Tensor): Keyframe mask of shape [batch_size, max_frames]
        noise_scale (float): Amount of noise to add to non-keyframe regions
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - conditioned_motion: Motion with non-keyframe regions optionally noised
            - keyframe_mask: The keyframe mask (passed through for convenience)
    """
    
    conditioned_motion = motion_batch.clone()
    
    if noise_scale > 0.0:
        # Add noise to non-keyframe regions
        noise = torch.randn_like(motion_batch) * noise_scale
        # Apply noise only where keyframe_mask is False
        non_keyframe_mask = ~keyframe_mask.unsqueeze(-1)  # [batch_size, max_frames, 1]
        conditioned_motion = motion_batch + noise * non_keyframe_mask.float()
        
    return conditioned_motion, keyframe_mask


def create_keyframe_loss_mask(
    keyframe_mask: torch.Tensor,
    motion_mask: torch.Tensor,
    zero_keyframe_loss: bool = False
) -> torch.Tensor:
    """
    Create a loss mask for keyframe conditioning.
    
    Args:
        keyframe_mask (torch.Tensor): Keyframe mask of shape [batch_size, max_frames]
        motion_mask (torch.Tensor): Valid motion mask of shape [batch_size, max_frames]
        zero_keyframe_loss (bool): If True, zero out loss on keyframe regions
        
    Returns:
        torch.Tensor: Loss mask of shape [batch_size, max_frames]
    """
    
    if zero_keyframe_loss:
        # Zero loss on keyframe regions - only train on non-keyframe regions
        loss_mask = motion_mask & (~keyframe_mask)
    else:
        # Normal loss computation on all valid regions
        loss_mask = motion_mask
        
    return loss_mask


def get_random_keyframe_dropout_mask(
    keyframe_mask: torch.Tensor,
    dropout_prob: float = 0.1
) -> torch.Tensor:
    """
    Randomly drop some keyframes during training for robustness.
    
    Args:
        keyframe_mask (torch.Tensor): Original keyframe mask [batch_size, max_frames]
        dropout_prob (float): Probability of dropping each keyframe
        
    Returns:
        torch.Tensor: Modified keyframe mask with some keyframes dropped
    """
    
    if dropout_prob <= 0.0:
        return keyframe_mask
        
    # Generate random dropout mask
    dropout_mask = torch.bernoulli(
        torch.ones_like(keyframe_mask, dtype=torch.float) * (1.0 - dropout_prob)
    ).bool()
    
    # Apply dropout: keep keyframes only where both original mask and dropout mask are True
    return keyframe_mask & dropout_mask


# Keyframe selection schemes for different training strategies
KEYFRAME_MODES = {
    'random_frames': 'Random sparse keyframes for general training',
    'random_joints': 'Random selection of N>=1 keyframes and random selection of J>=1 joints in each keyframe',
} 