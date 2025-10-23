#!/usr/bin/env python
# coding: utf-8

# Source #1: https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/e33606ed54102bd872c820346f3d90dfc954b71a/mmd_vae.py#L99)

import logging
from typing import TypeVar

import torch
import torch.nn as nn

from quest_qso.mlconfig import MLConfig
from quest_qso.utils import utilities

from .vae import BaseSpecVAE

Tensor = TypeVar("torch.tensor")


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


logger = logging.getLogger(__name__)

## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


# MMD Loss after https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
#  with minimal changes
class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        exp_ = torch.arange(n_kernels) - n_kernels // 2
        self.register_buffer("bandwidth_multipliers", mul_factor**exp_)
        self.bandwidth = bandwidth

    # ======================================================================= #

    def get_bandwidth(self, L2_distances):
        # I think I should remove the diagonal from the sum
        #  but they given I am computing the cdist between X and X
        #  all the diagonal elements are zero (afaik)
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            # The detach is very much required here, otherwise (I think)
            #  I end up optimizing the bandwidth, which is not what I want at all
            return L2_distances.detach().sum() / (n_samples * (n_samples - 1))

        return self.bandwidth

    # ======================================================================= #

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)  # sum seems to be more stable


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


class MMDLoss(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    # ======================================================================= #

    def forward(self, X, Y):
        assert X.shape[0] == Y.shape[0], "X and Y have different dimensions along dim 0"

        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]

        # used biased estimate to avoid negative values
        # size of each one of these is (batch_size, batch_size)
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

        # Below unbiased, which seems to do more harm than good
        #  so essentially kept for historical reasons
        #  or just in case we get the itch of experimenting again

        # # Calculate means of kernel submatrices
        # XX = K[:X_size, :X_size]
        # XY = K[:X_size, X_size:]
        # YY = K[X_size:, X_size:]

        # # Exclude diagonal terms for unbiased estimate
        # XX_mean = (XX.sum() - XX.diag().sum()) / (X_size * (X_size - 1))
        # YY_mean = (YY.sum() - YY.diag().sum()) / (X_size * (X_size - 1))
        # XY_mean = XY.mean()

        # # Note that this is the unbiased version, and returns the MMD^2
        # # Also note that the unbiased MMD can be negative, for (afaik)
        # #  computational reasons.
        # # We avoid that by setting a small smoothign factor
        # return XX_mean - 2 * XY_mean + YY_mean


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


class InfoSpecVAE(BaseSpecVAE):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation_function: str,
        hidden_dims=None,
        device="mps",
        dispersion=None,
        reg_weight_rec=1e-3,
        reg_weight_mmd=1e-3,
        reg_weight_kld=1e-3,
        rel_weight_mmd_kld=1.0,
        random_mask=None,
        coverage_mask=None,
        scaling_factor=None,
        dataset_dir=None,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            activation_function,
            hidden_dims=hidden_dims,
            device=device,
            random_mask=random_mask,
            coverage_mask=coverage_mask,
            dispersion=dispersion,
            scaling_factor=scaling_factor,
            dataset_dir=dataset_dir,
        )

        self.mmd = MMDLoss(kernel=RBF()).to(self.device)

        self.rel_weight_mmd_kld = rel_weight_mmd_kld
        self.reg_weight_rec = reg_weight_rec
        self.reg_weight_mmd = reg_weight_mmd
        self.reg_weight_kld = reg_weight_kld

    ## ===================================================================== ##

    def loss_function(
        self,
        input_: Tensor,
        output: Tensor,
        ivar: Tensor,
        coverage_mask: Tensor,
        mu: Tensor,
        log_var: Tensor,
        z: Tensor,
        params: MLConfig,
        epoch: int,
        validation=False,
    ) -> dict:
        # determine whether we are good with the epoch condition or not
        if validation:
            epoch_condition = False
        elif not validation and params.line_loss_epoch is None:
            epoch_condition = False
        elif not validation and params.line_loss_epoch is not None:
            epoch_condition = epoch > params.line_loss_epoch

        #  Reconstruction loss, see discussion here: https://stats.stackexchange.com/q/485488
        #  which would imply that the mse loss should be sum of the single mse losses.
        # However:
        #  1 - that would depend on the batch size, which we do not like
        #  2 - in the original infoVAE implementation, the loss is averaged (see: sources#1 above)
        # Taking the mean should be consistent across batch sizes.

        # TODO: Add an assert in the _validate stage of the parameters
        #  to make sure that some reconstruction loss is computed

        # 0 - Normalisation factor for the reconstruction loss
        # Note that we only count the valid pixels!
        rec_loss_norm = coverage_mask.sum()

        # 1 - Compute reconstruction loss (chi square/MSE) stat.
        #      Note that the chi square is multipied by whatever the last argument is, in this case
        #      the coverage_mask, inside the function itself.
        if params.loss_type == "chisq":
            rec_loss = utilities.chisq_stat(
                input_,
                output,
                ivar,
                coverage_mask,
            )
        elif params.loss_type == "mse":
            rec_loss = nn.functional.mse_loss(
                input_ * coverage_mask,
                output * coverage_mask,
                reduction="none",
            )

        # validation loss, using the same function to avoid code duplications and be consistent
        #  Note that in this case I don't care at all about the line loss or any other shenanigans!
        #  We only want to compute the reconstruction loss
        if validation:
            validation_loss = rec_loss.sum() / rec_loss_norm
            validation_loss /= params.latent_dim
            validation_loss *= self.reg_weight_rec
            return {"valid_loss": validation_loss}

        # After this, several possible cases (made as explicit as possible)
        # - I don't care about the line loss at all, I just want the reconstruction loss
        #   -> sum_line_loss == None (default)
        if params.sum_line_loss is None:
            reconstruction_loss = rec_loss.sum() / rec_loss_norm
            reconstruction_loss /= params.latent_dim
            reconstruction_loss *= self.reg_weight_rec

        # - I want to sum the line loss to the overall reconstruction loss
        #   -> sum_line_loss == True
        elif params.sum_line_loss is not None and params.sum_line_loss:
            reconstruction_loss = rec_loss.sum() / rec_loss_norm
            reconstruction_loss /= params.latent_dim
            reconstruction_loss *= self.reg_weight_rec

        # - I want to only use the line loss and not sum it to the overall
        #   reconstruction loss but I have yet to reach the epoch I set
        #   -> sum_line_loss == False, epoch not reached
        elif (
            params.sum_line_loss is not None
            and not params.sum_line_loss
            and not epoch_condition
        ):
            reconstruction_loss = rec_loss.sum() / rec_loss_norm
            reconstruction_loss /= params.latent_dim
            reconstruction_loss *= self.reg_weight_rec

        # - I want to only use the line loss and not sum it to the overall
        #    reconstruction loss but I have reached the epoch I set
        #   -> sum_line_loss == False, epoch reached
        elif (
            not params.sum_line_loss
            and params.sum_line_loss is not None
            and epoch_condition
        ):
            reconstruction_loss = 0.0

        # 2 - Line loss, to force the model to focus on the emission lines
        if (
            params.sum_line_loss is not None
            and params.line_loss_weight is not None
            and params.line_loss_weight > 0.0
            and epoch_condition
        ):
            line_mask = utilities.gen_line_mask(self.dispersion)
            line_loss = (rec_loss * line_mask).sum() / (coverage_mask * line_mask).sum()
            line_loss /= params.latent_dim
            line_loss *= self.reg_weight_rec
            line_loss *= params.line_loss_weight
            line_loss *= utilities.sigmoid(epoch, params.line_loss_epoch + 10)
        else:
            line_loss = 0.0

        # Latent space losses, computed only if needed
        # 3 - If we want to include the warm up, we do it here
        if params.do_warm_up:
            warm_up_scaling = utilities.sigmoid(epoch, params.warm_up_epoch)
        else:
            warm_up_scaling = 1.0

        rel_weight_kld_mmd = 1 - self.rel_weight_mmd_kld

        # 4 - KLD Latent losses
        # See examples from this page: https://stackoverflow.com/a/74869158,
        if rel_weight_kld_mmd > 0.0:
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            kld_loss /= params.latent_dim
            kld_loss *= self.reg_weight_kld
            kld_loss *= rel_weight_kld_mmd
            kld_loss *= warm_up_scaling
        else:
            kld_loss = 0.0

        # 5 - MMD loss, (actually) sampling Gaussian distributed numbers...
        # Note that this will be likely bad for small batch sizes
        #  as we are sampling, say, 8 numbers and those will not be a gaussian at all...
        if self.rel_weight_mmd_kld > 0.0:
            true_samples = torch.randn_like(z, requires_grad=False, device=self.device)
            mmd_loss = self.mmd(true_samples, z) * input_.shape[0]
            # still not fully clear to me why this ^ is needed
            mmd_loss /= params.latent_dim
            mmd_loss *= self.reg_weight_mmd
            mmd_loss *= self.rel_weight_mmd_kld
            mmd_loss *= warm_up_scaling
        else:
            mmd_loss = 0.0

        # Total loss, things should already be normalised wrt batch_size
        # Note that I am multiplying the relative MMD/KLD weight above
        #  to save ourselves a multiplication
        loss = reconstruction_loss + line_loss + kld_loss + mmd_loss

        loss_dict = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kld_loss": kld_loss,
            "mmd_loss": mmd_loss,
        }

        return loss_dict
