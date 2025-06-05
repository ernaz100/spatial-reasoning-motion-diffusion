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
        
        # Keyframe conditioning components
        if self.keyframe_conditioned:
            # Learnable embeddings for keyframe status
            self.keyframe_token_embedding = nn.Embedding(2, latent_dim)  # 0: non-keyframe, 1: keyframe
            
            # Optional: separate embedding for observed keyframe data
            self.keyframe_data_embedding = nn.Linear(nfeats, latent_dim)

        # register for aggregating info
        if nb_registers > 0:
            self.registers = nn.Parameter(torch.randn(nb_registers, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        # SRM: Frame-level timestep encoder - maps continuous time values [0,1] to embeddings
        self.frame_timestep_encoder = nn.Sequential(
            nn.Linear(1, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, latent_dim),
        )
        


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
        self.to_skel_layer = nn.Linear(latent_dim, nfeats)
        
        # SRM: Additional head for variance/uncertainty prediction
        self.to_variance_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, x, y):
        """
        Forward pass with SRM frame-level conditioning.
        
        Args:
            x: Noisy motion data [batch_size, max_frames, n_features]
            y: Conditioning dictionary containing:
                - mask: Valid frame mask [batch_size, max_frames]
                - tx: Text conditioning
                - frame_times: Frame-level time values [batch_size, max_frames] (required for SRM)
                - frame_time_mask: Mask for valid frame times (optional, defaults to motion mask)
                - keyframe_mask: Keyframe indicators (if keyframe conditioning enabled)
                - keyframe_x0: Clean keyframe data (if keyframe conditioning enabled)
            
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
        frame_time_mask = y.get("frame_time_mask", x_mask)
        
        # Embed frame-level times
        # Reshape for linear layer: [bs, nframes] -> [bs, nframes, 1]
        frame_times_input = frame_times.unsqueeze(-1)  # [bs, nframes, 1]
        frame_time_emb = self.frame_timestep_encoder(frame_times_input)  # [bs, nframes, latent_dim]
        
        # Mask invalid frame times
        if frame_time_mask.dim() == 3:  # [bs, nframes, 1]
            frame_time_emb = frame_time_emb * frame_time_mask.float()
        else:  # [bs, nframes]
            frame_time_emb = frame_time_emb * frame_time_mask.unsqueeze(-1).float()
        
        # Initialize sequence-level info embeddings (no global timestep needed)
        info_emb = torch.empty(bs, 0, self.latent_dim, device=device)  # [bs, 0, latent_dim]
        info_mask = torch.empty(bs, 0, dtype=bool, device=device)  # [bs, 0]

        assert "tx" in y

        # Condition part (can be text/action etc)
        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"]

        tx_emb = self.tx_embedding(tx_x)

        info_emb = torch.cat((info_emb, tx_emb), 1)
        info_mask = torch.cat((info_mask, tx_mask), 1)

        # add registers
        if self.nb_registers > 0:
            registers = repeat(self.registers, "nbtoken dim -> bs nbtoken dim", bs=bs)
            registers_mask = torch.ones(
                (bs, self.nb_registers), dtype=bool, device=device
            )
            # add the register
            info_emb = torch.cat((info_emb, registers), 1)
            info_mask = torch.cat((info_mask, registers_mask), 1)

        # Process motion embeddings
        x_emb = self.skel_embedding(x)  # [bs, nframes, latent_dim]
        
        # SRM: Add frame-level time conditioning to motion embeddings
        # Each frame gets its own time embedding based on its individual noise level
        x_emb = x_emb + frame_time_emb
                    
        number_of_info = info_emb.shape[1]

        # adding the embedding token for all sequences
        xseq = torch.cat((info_emb, x_emb), 1)

        # add positional encoding to all the tokens
        xseq = self.sequence_pos_encoding(xseq)
 
        # create a bigger mask, to allow attend to time and condition as well
        # Handle different mask dimensions
        if x_mask.dim() == 3:  # [bs, nframes, 1]
            motion_mask = x_mask.squeeze(-1)  # [bs, nframes]
        else:  # [bs, nframes]
            motion_mask = x_mask
            
        aug_mask = torch.cat((info_mask, motion_mask), 1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # extract the important part (motion predictions)
        motion_features = final[:, number_of_info:]  # [bs, nframes, latent_dim]
        
        # SRM: Predict both main output and variance
        main_output = self.to_skel_layer(motion_features)  # [bs, nframes, nfeats]
        log_variance = self.to_variance_layer(motion_features)  # [bs, nframes, nfeats]
        
        # Return as dictionary to include both predictions
        return {
            "output": main_output,
            "log_variance": log_variance,
            "variance": torch.exp(log_variance),  # For convenience
        } 