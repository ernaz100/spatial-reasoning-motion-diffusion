import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding, TimestepEmbedder
from einops import repeat


class SRMTransformerDenoiser(nn.Module):
    """
    SRM-adapted Transformer Denoiser for motion generation.
    
    Key differences from standard MDM:
    - Supports frame-level time conditioning (different noise levels per frame)
    - Spatially-aware time embedding per frame instead of sequence-level
    - Compatible with SRM training strategies
    """
    name = "srm_transformer"

    def __init__(
        self,
        nfeats: int,
        tx_dim: int,
        latent_dim: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        nb_registers: int = 2,
        activation: str = "gelu",
        keyframe_conditioned: bool = False,

        max_timesteps: int = 1000,
    ):
        super().__init__()

        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.nb_registers = nb_registers
        self.tx_dim = tx_dim
        self.keyframe_conditioned = keyframe_conditioned

        self.max_timesteps = max_timesteps

        # Linear layer for the condition
        self.tx_embedding = nn.Sequential(
            nn.Linear(tx_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Linear layer for the skeletons
        self.skel_embedding = nn.Linear(nfeats, latent_dim)
        
        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, max_len=max_timesteps, batch_first=True
        )

        # Global timestep embedder
        self.time_embed = TimestepEmbedder(latent_dim, self.sequence_pos_encoding)

        # SRM: Frame-level timestep encoder - now using same TimestepEmbedder
        self.frame_time_embed = TimestepEmbedder(latent_dim, self.sequence_pos_encoding)
        
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            norm_first=True,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        # Final layer to go back to skeletons
        self.to_skel_layer = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, nfeats)
        )
        
    def forward(self, x, y):
        """
        Forward pass with SRM frame-level conditioning.
        
        Args:
            x: Noisy motion data [batch_size, max_frames, n_features]
            y: Conditioning dictionary containing:
                - mask: Valid frame mask [batch_size, max_frames]
                - tx: Text conditioning
                - frame_times: Frame-level time values [batch_size, max_frames] (required for SRM)
                - t_bar: Overall time value the individual frame times are based on
            
        Returns:
            Predicted noise/motion [batch_size, max_frames, n_features]
        """
        # Ensure consistent float32 dtype
        x = x.float()
        device = x.device
        x_mask = y["mask"]
        bs, nframes, nfeats = x.shape

        # SRM: Frame-level time conditioning
        frame_times = y["frame_times"]  # [bs, nframes] - continuous values [0,1]
        
        # Discretize frame times to indices in [0, max_timesteps-1]
        frame_time_indices = (frame_times * (self.max_timesteps - 1)).long()
        
        # Get frame-level time embeddings using the same mechanism as overall timestep
        frame_time_emb = self.frame_time_embed(frame_time_indices)  # [bs, nframes, latent_dim]
        
        # Mask invalid frame times
        frame_time_emb = frame_time_emb * x_mask.unsqueeze(-1).float()
        
        # Condition part (can be text/action etc)
        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"]
        t_bar = y["t_bar"]

        # Scale t_bar to match frame timesteps range
        t_bar = (t_bar * (self.max_timesteps - 1)).long()

        tx_emb = self.tx_embedding(tx_x)
        time_emb = self.time_embed(t_bar)

        # Process motion embeddings
        x_emb = self.skel_embedding(x)  # [bs, nframes, latent_dim]
        x_emb = self.sequence_pos_encoding(x_emb)
        
        # SRM: Add frame-level time conditioning to motion embeddings
        # Each frame gets its own time embedding based on its individual noise level
        x_emb = x_emb + frame_time_emb
        
        # Concatenate text and motion embeddings
        full_emb = torch.cat([tx_emb, x_emb, time_emb.unsqueeze(1)], dim=1)
        
        # Create a combined mask for the concatenated sequence
        full_mask = torch.cat([tx_mask, x_mask, torch.ones(bs, time_emb.unsqueeze(1).shape[1], device=device, dtype=bool)], dim=1)

        final = self.seqTransEncoder(full_emb, src_key_padding_mask=~full_mask)
        
        # Extract the motion part of the output sequence
        n_text_tokens = tx_emb.shape[1]
        motion_output = final[:, n_text_tokens : n_text_tokens + nframes, :]

        # SRM: Predict main output
        noise_prediction = self.to_skel_layer(motion_output)  # [bs, nframes, nfeats]
        
        # Return as dictionary to include main output
        return {
            "noise_prediction": noise_prediction,
        } 