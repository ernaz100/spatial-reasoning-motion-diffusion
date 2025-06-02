import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding, TimestepEmbedder
from einops import repeat


class TransformerDenoiser(nn.Module):
    name = "transformer"

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
    ):
        super().__init__()

        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.nb_registers = nb_registers
        self.tx_dim = tx_dim
        self.keyframe_conditioned = keyframe_conditioned

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

        # MLP for the timesteps
        self.timestep_encoder = TimestepEmbedder(latent_dim, self.sequence_pos_encoding)

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

    def forward(self, x, y, t):
        device = x.device
        x_mask = y["mask"]
        bs, nframes, nfeats = x.shape

        # Time embedding
        time_emb = self.timestep_encoder(t)
        time_mask = torch.ones(bs, dtype=bool, device=device)

        # put all the additionnal here
        info_emb = time_emb[:, None]
        info_mask = time_mask[:, None]

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
        x_emb = self.skel_embedding(x)
        
        # Add keyframe conditioning if enabled
        if self.keyframe_conditioned and "keyframe_mask" in y:
            keyframe_mask = y["keyframe_mask"]  # [bs, nframes] - True for keyframes
            
            # Ensure keyframe mask is consistent with motion mask
            keyframe_mask = keyframe_mask & x_mask.squeeze(-1)  # Remove last dimension if present
            
            # Create keyframe token embeddings (binary indicator)
            keyframe_tokens = keyframe_mask.long()  # Convert to long for embedding lookup
            keyframe_emb = self.keyframe_token_embedding(keyframe_tokens)  # [bs, nframes, latent_dim]
            
            # Add keyframe information to motion embeddings
            x_emb = x_emb + keyframe_emb
            
            #  For keyframe regions, also incorporate the clean data
            if "keyframe_x0" in y:
                keyframe_x0 = y["keyframe_x0"]  # Clean motion data [bs, nframes, nfeats]
                
                # Create separate embedding for clean keyframe data
                keyframe_data_emb = self.keyframe_data_embedding(keyframe_x0)
                
                # Only add keyframe data embedding where we have keyframes
                keyframe_mask_expanded = keyframe_mask.unsqueeze(-1)  # [bs, nframes, 1]
                x_emb = x_emb + keyframe_data_emb * keyframe_mask_expanded.float()
                
                # Alternative approach: Replace embeddings entirely at keyframe positions
                # This ensures the model gets strong signal about observed keyframes
                # x_emb = x_emb * (~keyframe_mask_expanded).float() + keyframe_data_emb * keyframe_mask_expanded.float()

        number_of_info = info_emb.shape[1]

        # adding the embedding token for all sequences
        xseq = torch.cat((info_emb, x_emb), 1)

        # add positional encoding to all the tokens
        xseq = self.sequence_pos_encoding(xseq)

        # create a bigger mask, to allow attend to time and condition as well
        aug_mask = torch.cat((info_mask, x_mask), 1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # extract the important part
        output = self.to_skel_layer(final[:, number_of_info:])
        return output
