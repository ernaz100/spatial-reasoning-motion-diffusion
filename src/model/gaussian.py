import logging
from collections import defaultdict
from tqdm import tqdm

import torch

from .diffusion_base import DiffuserBase
from ..data.collate import length_to_mask, collate_tensor_with_padding
from src.stmc import combine_features_intervals, interpolate_intervals
from src.utils.keyframe_masking import (
    get_keyframes_mask, 
    apply_keyframe_conditioning,
    create_keyframe_loss_mask,
    get_random_keyframe_dropout_mask
)


# Inplace operator: return the original tensor
# work with a list of tensor as well
def masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = 0.0
    return tensor


logger = logging.getLogger(__name__)


def remove_padding_to_numpy(x, length):
    x = x.detach().cpu().numpy()
    return [d[:l] for d, l in zip(x, length)]


class GaussianDiffusion(DiffuserBase):
    name = "gaussian"

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
    ):
        super().__init__(schedule, timesteps)

        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.lr = lr
        self.prediction = prediction

        self.reconstruction_loss = torch.nn.MSELoss(reduction="none")  # Changed to 'none' for masking

        # normalization
        self.motion_normalizer = motion_normalizer
        self.text_normalizer = text_normalizer
        
        # Keyframe conditioning parameters
        self.keyframe_conditioned = keyframe_conditioned
        self.keyframe_selection_scheme = keyframe_selection_scheme
        self.keyframe_mask_prob = keyframe_mask_prob
        self.zero_keyframe_loss = zero_keyframe_loss
        self.n_keyframes = n_keyframes
        self.keyframe_loss_weight = keyframe_loss_weight

    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def prepare_tx_emb(self, tx_emb):
        # Text embedding normalization
        if "mask" not in tx_emb:
            tx_emb["mask"] = length_to_mask(tx_emb["length"], device=self.device)
        tx = {
            "x": masked(self.text_normalizer(tx_emb["x"]), tx_emb["mask"]),
            "length": tx_emb["length"],
            "mask": tx_emb["mask"],
        }
        return tx

    def diffusion_step(self, batch, batch_idx, training=False):
        mask = batch["mask"]  # Valid motion mask [batch_size, max_frames]

        # normalization
        x = masked(self.motion_normalizer(batch["x"]), mask)
        
        # Prepare base conditioning
        y = {
            "length": batch["length"],
            "mask": mask,
            "tx": self.prepare_tx_emb(batch["tx"]),
            # the condition is already dropped sometimes in the dataloader
        }

        # Add keyframe conditioning if enabled
        keyframe_mask = None
        if self.keyframe_conditioned and training:
            # Generate keyframe mask during training
            keyframe_mask = get_keyframes_mask(
                data=batch["x"],  # Use original (non-normalized) data for masking
                lengths=batch["length"],
                edit_mode=self.keyframe_selection_scheme,
                n_keyframes=self.n_keyframes,
                device=x.device
            )
            
            # Apply keyframe dropout for robustness
            if self.keyframe_mask_prob > 0.0:
                keyframe_mask = get_random_keyframe_dropout_mask(
                    keyframe_mask, 
                    dropout_prob=self.keyframe_mask_prob
                )
            
            # Add keyframe information to conditioning
            y["keyframe_mask"] = keyframe_mask
            y["keyframe_x0"] = x  # Provide clean motion data for keyframes
            
            # Ensure keyframe mask is consistent with motion mask
            keyframe_mask = keyframe_mask & mask

        bs = len(x)
        # Sample a diffusion step between 0 and T-1
        # 0 corresponds to noising from x0 to x1
        # T-1 corresponds to noising from xT-1 to xT
        t = torch.randint(0, self.timesteps, (bs,), device=x.device)

        # Create a noisy version of x
        # no noise for padded region
        noise = masked(torch.randn_like(x), mask)
        xt = self.q_sample(xstart=x, t=t, noise=noise)
        xt = masked(xt, mask)
        
        # Apply keyframe conditioning to noisy input if enabled
        if self.keyframe_conditioned and keyframe_mask is not None:
            keyframe_mask_expanded = keyframe_mask.unsqueeze(-1)  # [batch_size, max_frames, 1]            
            # This line replaces the noisy motion data (xt) with clean motion data (x) at keyframe positions. Specifically:
            # xt * (~keyframe_mask_expanded): Keeps the noisy motion data for non-keyframe regions
            # x * keyframe_mask_expanded: Uses clean motion data for keyframe regions
            xt = xt * (~keyframe_mask_expanded) + x * keyframe_mask_expanded

        # denoise it
        # no drop cond -> this is done in the training dataloader already
        # give "" instead of the text
        # denoise it
        output = masked(self.denoiser(xt, y, t), mask)

        # Predictions
        xstart = masked(self.output_to("x", output, xt, t), mask)
        
        # Compute loss with optional keyframe masking
        reconstruction_loss = self.reconstruction_loss(xstart, x)  # [batch_size, max_frames, n_features]
        
        # Create loss mask based on keyframe conditioning
        if self.keyframe_conditioned and keyframe_mask is not None:
            # Create a float mask: keyframes get keyframe_loss_weight, others get 1.0
            loss_mask = mask.float().clone()
            if self.zero_keyframe_loss:
                # Zero out keyframe loss
                loss_mask[keyframe_mask] = 0.0
            else:
                # Increase keyframe loss at keyframe positions
                loss_mask[keyframe_mask] = self.keyframe_loss_weight
        else:
            loss_mask = mask.float()
            
        # Expand mask for features
        loss_mask_expanded = loss_mask.unsqueeze(-1)  # [batch_size, max_frames, 1]
        masked_loss = reconstruction_loss * loss_mask_expanded
        
        # Compute mean loss over valid (non-masked) elements
        total_loss = masked_loss.sum()
        valid_elements = loss_mask_expanded.sum()
        xloss = total_loss / (valid_elements + 1e-8)  # Add epsilon to avoid division by zero
        
        loss = {"loss": xloss}
        
        # Add additional logging for keyframe conditioning
        if self.keyframe_conditioned and keyframe_mask is not None and training:
            # Log keyframe statistics
            keyframe_ratio = keyframe_mask.float().mean()
            loss["keyframe_ratio"] = keyframe_ratio
            
            if self.zero_keyframe_loss:
                # Log loss only on non-keyframe regions
                non_keyframe_mask = mask & (~keyframe_mask)
                if non_keyframe_mask.sum() > 0:
                    non_keyframe_loss_mask = non_keyframe_mask.unsqueeze(-1)
                    non_keyframe_loss = (reconstruction_loss * non_keyframe_loss_mask.float()).sum()
                    non_keyframe_elements = non_keyframe_loss_mask.sum()
                    loss["non_keyframe_loss"] = non_keyframe_loss / (non_keyframe_elements + 1e-8)
            
        return loss

    def training_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx, training=True)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )

        return loss["loss"]

    def on_train_epoch_end(self):
        dico = {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.global_step),
        }
        # reset losses
        self._saved_losses = defaultdict(list)
        self.losses = []
        self.log_dict(dico)

    # dispatch
    def forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        if "timeline" in infos:
            ff = self.stmc_forward
            if "baseline" in infos:
                if "sinc" in infos["baseline"]:
                    ff = self.sinc_baseline
            # for the other baselines stmc handle it
            # STMC generalize one text forward and DiffCollage
        else:
            ff = self.text_forward
        return ff(tx_emb, tx_emb_uncond, infos, progress_bar=progress_bar)

    def text_forward(
        self,
        tx_emb,
        tx_emb_uncond,
        infos,
        progress_bar=tqdm,
    ):
        # normalize text embeddings first
        device = self.device

        lengths = infos["all_lengths"]
        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        duration = max(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, duration, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def p_sample(self, xt, y, t):
        # guided forward
        output_cond = self.denoiser(xt, y, t)

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()  # not a deep copy
            y_uncond["tx"] = y_uncond["tx_uncond"]

            output_uncond = self.denoiser(xt, y_uncond, t)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart

    def stmc_forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        device = self.device

        # the lengths of all the crops + uncondionnal
        lengths = infos["all_lengths"]
        n_frames = infos["n_frames"]
        n_seq = infos["n_seq"]

        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        nfeats = self.denoiser.nfeats

        shape = n_seq, n_frames, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t_seq = torch.full((n_seq,), diffusion_step)
            t_bs = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample_stmc(xt, y, t_seq, t_bs)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def p_sample_stmc(self, xt, y, t_seq, t_bs):
        all_intervals = y["infos"]["all_intervals"]

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        x_lst = []
        for idx, intervals in enumerate(all_intervals):
            x_lst.extend([xt[idx, x.start : x.end] for x in intervals])

        lengths = [len(x) for x in x_lst]
        assert lengths == y["length"]

        xx = collate_tensor_with_padding(x_lst)
        output = self.denoiser(xx, y, t_bs)

        if guidance_weight != 1.0:
            output_cond = output

            y_uncond = y.copy()  # not a deep copy
            y_uncond["tx"] = y_uncond["tx_uncond"]

            output_uncond = self.denoiser(xx, y_uncond, t_bs)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        output_xt = 0 * xt
        combine_features_intervals(output, y["infos"], output_xt)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(
            output_xt, xt, t_seq
        )

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise

        xstart = output_xt
        return x_out, xstart

    def sinc_baseline(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        device = self.device

        # the lengths of all the crops + uncondionnal
        lengths = infos["all_lengths"]
        n_frames = infos["n_frames"]
        n_seq = infos["n_seq"]

        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, max(lengths), nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t_bs = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t_bs)

        # at the end recombine
        shape = n_seq, n_frames, nfeats
        output = torch.zeros(shape, device=device)

        xstart = combine_features_intervals(xstart, infos, output)

        if "lerp" in infos["baseline"] or "interp" in infos["baseline"]:
            # interpolate to smooth the results
            xstart = interpolate_intervals(xstart, infos)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart
