#!/usr/bin/env python
# coding: utf-8

import logging
import tarfile
import tempfile
from typing import Any, List, TypeVar

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from tqdm import tqdm

from quest_qso.utils import resources, utilities

Tensor = TypeVar("torch.tensor")

logger = logging.getLogger(__name__)


# see this very helpful comment: https://stackoverflow.com/a/57013056
# and this other helpful discussion
# https://discuss.pytorch.org/t/how-to-define-a-parametrized-tanh-activation-function-with-2-trainable-parameters/161720
# and this repo: https://github.com/pmelchior/spender/blob/main/spender/model.py
class SpeculatorActivation(nn.Module):
    def __init__(
        self,
        n_parameter=1,
        plus_one=False,
        device=torch.device("cpu"),
        init_alpha=0.5,
        init_beta=0.5,
        use_norm=False,
        clip_grad=None,
    ):
        super().__init__()
        self.plus_one = plus_one
        self.use_norm = use_norm
        self.clip_grad = clip_grad

        # Better initialization with controlled values instead of random
        self.alpha = nn.Parameter(
            torch.ones(n_parameter, device=device) * init_alpha,
            requires_grad=True,
        )
        self.beta = nn.Parameter(
            torch.ones(n_parameter, device=device) * init_beta,
            requires_grad=True,
        )

    def forward(self, x):
        # Apply optional gradient clipping during training
        if self.training and self.clip_grad is not None:
            with torch.no_grad():
                self.alpha.data.clamp_(-self.clip_grad, self.clip_grad)
                self.beta.data.clamp_(0.0, 1.0)
                # Beta should typically be between 0 and 1

        activation = (self.beta + torch.sigmoid(self.alpha * x) * (1.0 - self.beta)) * x

        # Optional normalization to control output range
        if self.use_norm:
            activation = nn.functional.layer_norm(
                activation,
                normalized_shape=activation.shape[1:],
            )

        if self.plus_one:
            return activation + 1.0

        return activation


## =========================================================================== ##
## =========================================================================== ##
## =========================================================================== ##


activation_functions_dict = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leakyrelu": nn.LeakyReLU,
    "speculator": SpeculatorActivation,
}

activation_functions_param_dict = {
    "relu": {},
    "tanh": {},
    "sigmoid": {},
    "leakyrelu": {"negative_slope": 0.1},
    "speculator": {},  # "clip_grad": 1.5},
}

## =========================================================================== ##
## =========================================================================== ##
## =========================================================================== ##


class BaseSpecVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation_function: str,
        hidden_dims=None,
        device="mps",
        random_mask=None,
        coverage_mask=None,
        dispersion=None,
        # TODO: Possibly remove this scaling factor
        scaling_factor=None,
        dataset_dir=None,
    ):
        """Initialize the BaseSpecVAE class

        :param input_dim:
        :param latent_dim:
        :param hidden_dims:
        :param device:
        :param relu_alpha:
        :param use_mask:
        """
        super().__init__()

        logger.info(f"Using activation function: {activation_function}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.device = device

        self.activation_function = activation_function

        # Only to load the correct normalisation
        self.dataset_dir = dataset_dir

        # TODO: I would really like to find a way around using dispersion as a model
        #  attribute, it does not make much sense, but it is also very convoluted to
        #  change this and/or pass the dispersion around otherwise.
        # Stays for the time being, I have other things to fix first and this is ok
        #  for now.
        self.dispersion = dispersion

        # Same for the scaling factor, even though it makes more sense than the
        #  dispersion in my opinion
        self.scaling_factor = scaling_factor

        # add random mask to input
        if random_mask is not None:
            input_dim += random_mask

        if coverage_mask is not None:
            input_dim += coverage_mask

        # Set the hidden dimensions
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        self.input_scaling_factor = torch.from_numpy(
            resources.load_median_spec(parent_dir=self.dataset_dir)
        ).to(self.device, dtype=torch.float)

        # Define the encoder
        modules = []
        for n, h_dim in enumerate(hidden_dims):
            # TODO: There are for sure cleaner ways to do this, but for the time being
            #  this is ok
            if self.activation_function == "speculator":
                activation_functions_param_dict[activation_function]["n_parameter"] = (
                    h_dim
                )
                activation_functions_param_dict[activation_function]["device"] = (
                    self.device
                )

            modules.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=input_dim,
                        out_features=h_dim,
                        device=self.device,
                    ),
                    nn.BatchNorm1d(h_dim, device=self.device),
                    activation_functions_dict[activation_function](
                        **activation_functions_param_dict[activation_function],
                    ),
                )
            )
            input_dim = h_dim

        # Set the encoder
        self.encoder = nn.Sequential(*modules)

        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim, device=self.device)
        self.fc_var = nn.Linear(hidden_dims[-1], self.latent_dim, device=self.device)

        # Define the decoder input layer
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1], device=device)

        # Define the decoder
        modules = []

        # Reverse the order of the hidden dimensions
        hidden_dims.reverse()

        # Define the decoder
        for idx in range(len(hidden_dims) - 1):
            if self.activation_function == "speculator":
                activation_functions_param_dict[activation_function]["n_parameter"] = (
                    hidden_dims[idx + 1]
                )

            modules.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=hidden_dims[idx],
                        out_features=hidden_dims[idx + 1],
                        device=self.device,
                    ),
                    nn.BatchNorm1d(hidden_dims[idx + 1], device=self.device),
                    activation_functions_dict[activation_function](
                        **activation_functions_param_dict[activation_function],
                    ),
                )
            )

        # Set the decoder
        self.decoder = nn.Sequential(*modules)

        if self.activation_function == "speculator":
            activation_functions_param_dict[activation_function]["n_parameter"] = (
                self.input_dim
            )

        # Define the final layer
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.input_dim, device=self.device),
            activation_functions_dict[activation_function](
                **activation_functions_param_dict[activation_function],
            ),
        )

    ## ========================================================================= ##

    def encode(self, input: Tensor) -> List[Tensor]:
        """Encode the input into the latent space representation

        :param input:
        :type input: Tensor
        :return:
        """
        # TODO: This is dangerous...
        input[:, : self.input_dim] = (
            input[:, : self.input_dim] / self.input_scaling_factor - 1.0
        )
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        #  of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    ## ========================================================================= ##

    def decode(self, z: Tensor) -> Any:
        """Decode the latent space representation into the original space

        :param z:
        :return:
        """
        result = self.decoder_input(z)

        result = self.decoder(result)
        result = self.final_layer(result)
        result = (1 + result) * self.input_scaling_factor

        # Scale back the results to the original values
        # if self.scaling_factor is not None:
        #     result *= self.scaling_factor

        return result.to(self.device)

    ## ========================================================================= ##

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterize the latent space representation

        :param mu:
        :param log_var:
        :return:
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return z

    ## ========================================================================= ##

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """Forward pass through the network

        :param input:
        :param kwargs:
        :return:
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)

        return [output, input, z, mu, log_var]

    ## ========================================================================= ##

    def loss_function(
        self,
        input: Tensor,
        output: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> dict:
        """Compute the loss function

        :param input:
        :param output:
        :param mu:
        :param log_var:
        :return:
        """
        # Reconstruction loss (mean squared error)
        reconstruction_loss = nn.functional.mse_loss(output, input, reduction="mean")

        # Latent loss (KLD loss)
        kld_loss = -0.5 * torch.sum(1 + (log_var - mu.pow(2) - log_var.exp()))

        # Total loss
        loss = reconstruction_loss + kld_loss

        loss_dict = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kld_loss": -kld_loss,
        }

        return loss_dict

    ## ========================================================================= ##

    # TODO: review whether this is actually needed now that our dimensions look
    #  quite decent (20250204)
    def sample(self, num_samples: int, gmm=None) -> list[Tensor]:
        """Sample from the latent space

        :param num_samples: _description_
        :type num_samples: int
        :param gmm: _description_, defaults to None
        :type gmm: _type_, optional
        :return: _description_
        :rtype: list[Tensor]
        """
        self.eval()
        with torch.no_grad():
            if gmm is not None:
                z = torch.tensor(
                    gmm.sample(num_samples)[0], dtype=torch.float32, device=self.device
                )
            else:
                z = torch.randn(num_samples, self.latent_dim, device=self.device)

            return [self.decode(z), z]

    ## ========================================================================= ##

    def sample_latent_space_dimensions(
        self,
        num_samples: int,
        dimension: int,
        ls_range=None,
        med_z=None,
    ) -> list[Tensor]:
        # set the model in evaluation mode
        self.eval()

        with torch.no_grad():
            # Create a latent space input
            if ls_range is None:
                ls_range = [0, 1]

            if med_z is not None:
                z = torch.tensor(
                    med_z,
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(num_samples, 1)
            else:
                z = torch.zeros(
                    num_samples,
                    self.latent_dim,
                    dtype=torch.float32,
                    device=self.device,
                )

            z[:, dimension] = torch.linspace(
                ls_range[0], ls_range[1], num_samples, device=self.device
            )
            out = self.decode(z)

            if self.scaling_factor is not None:
                out *= self.scaling_factor

            return [out.detach().cpu().numpy(), z[:, dimension]]

    ## ========================================================================= ##

    def reconstruct(
        self,
        input: Tensor,
        n_samples=1,
        compute_median=True,
    ) -> NDArray:
        """Reconstruct the input spectrum

        :param input:
        :return:
        """

        self.eval()

        if not torch.is_tensor(input):
            input = torch.from_numpy(np.atleast_2d(input)).to(
                device=self.device,
                dtype=torch.float32,
            )

        with torch.no_grad():
            mu, log_var = self.encode(input)

            if n_samples == 1:
                z = self.reparameterize(mu, log_var)
                output = self.decode(z).detach()
            else:
                output = []
                for _ in range(n_samples):
                    z = self.reparameterize(mu, log_var)
                    output.append(self.decode(z).detach())

                if compute_median:
                    output = torch.stack(output).median(dim=0)[0].detach()
                else:
                    output = torch.stack(output).detach()

            # This is a tad bit annoying, as I cannot make output an array without many complications
            #  for the time being!
            return (
                output * self.scaling_factor
                if self.scaling_factor is not None
                else output
            )

    ## ========================================================================= ##

    def reconstruct_all(
        self,
        data_loader,
        flux_autofit=False,
        random_mask_flag=False,
        coverage_mask_flag=False,
        n_samples=1,
        compute_median=True,
        percentile_values=[0.01, 0.99],
        return_percentiles=False,
        return_raw_input=False,
        return_stacked_input=False,
    ) -> NDArray:
        """Reconstruct the input spectrum

        :param input:
        :return:
        """
        self.eval()
        output = {}

        with torch.no_grad():
            output_, stacked_input_, raw_input_, percentiles_ = [], [], [], []
            for x in tqdm(data_loader):
                stacked_input, _, _, _, raw_input, _, _, _ = utilities.prepare_input(
                    x,
                    device=self.device,
                    flux_autofit=flux_autofit,
                    random_mask_flag=random_mask_flag,
                    coverage_mask_flag=coverage_mask_flag,
                )

                if return_stacked_input:
                    stacked_input_.append(stacked_input.detach())

                if return_raw_input:
                    raw_input_.append(raw_input.detach())

                # VAE loop
                mu, log_var = self.encode(stacked_input)

                # generate a bunch of samples
                sampled_output = []
                for _ in range(n_samples):
                    z = self.reparameterize(mu, log_var)
                    sampled_output.append(self.decode(z).detach())

                # collect all samples and stack the output
                sampled_output = torch.stack(sampled_output)

                # compute the median, if requested (defaults to true)
                if compute_median:
                    output_.append(sampled_output.median(dim=0)[0].detach())
                else:
                    output_.append(sampled_output.detach())

                # compute percentiles, if requested (defaults to false)
                if return_percentiles:
                    percentiles_.append(
                        sampled_output.quantile(
                            torch.tensor(percentile_values, device=self.device),
                            dim=0,
                        ).detach()
                    )

            # split based on the options that I pass
            # this is needed
            output["reconstruction"] = torch.vstack(output_)
            # only if return_raw_input is true
            if return_raw_input:
                output["raw_input"] = torch.vstack(raw_input_)
            # only if return_stacked_input is true
            if return_stacked_input:
                output["stacked_input"] = torch.vstack(stacked_input_)
            # only if return_percentiles is true
            if return_percentiles:
                # Stack all percentile tensors together
                all_percentiles = torch.cat(
                    percentiles_, dim=1
                )  # shape: [num_percentiles, batch_total, ...]

                # Create a dictionary with percentile values as keys
                # From quick tests in any case these look essentially the same as the median reconstruction
                #  using 16-84. Slighly more spread with .01 and .99
                percentiles_dict = {}
                for i, pval in enumerate(percentile_values):
                    # Each entry contains data for one percentile across all batches
                    percentiles_dict[f"{pval}".replace(".", "p")] = all_percentiles[i]

                output["percentiles"] = percentiles_dict

            # at some point I might remove this
            # flatten dictionary, this is the new outpuy
            flattened_output = {}
            for key in output.keys():
                if isinstance(output[key], dict):
                    for inner_key in output[key]:
                        flattened_output[f"{key}_{inner_key}"] = output[key][inner_key]
                else:
                    flattened_output[key] = output[key]

            # skip this for loop if it is not needed, wasted CPU cycles
            if self.scaling_factor is not None:
                for key in flattened_output.keys():
                    flattened_output[key] = flattened_output[key] * self.scaling_factor

            return flattened_output

    ## ========================================================================= ##

    def latent_space_posterior(
        self,
        data_loader,
        flux_autofit=False,
        coverage_mask_flag=False,
        random_mask_flag=True,
        return_stacked_input=False,
        return_flux_ivar=False,
    ):
        # model is in evaluation mode
        self.eval()

        # outputs
        out = {}

        # stacking vectors
        z_vec, mu_vec, logvar_vec = [], [], []

        # optinal
        if return_stacked_input:
            stacked_input_vec = []

        if return_flux_ivar:
            flux_vec, ivar_vec = [], []

        with torch.no_grad():
            for _, x in enumerate(data_loader):
                stacked_input, _, _, _, flux, ivar, _, _ = utilities.prepare_input(
                    x,
                    device=self.device,
                    flux_autofit=flux_autofit,
                    random_mask_flag=random_mask_flag,
                    coverage_mask_flag=coverage_mask_flag,
                )

                mu, log_var = self.encode(stacked_input)
                # there is randomness in the output due to the eps here
                z = self.reparameterize(mu, log_var)

                z_vec.append(z.cpu().detach().numpy())
                mu_vec.append(mu.cpu().detach().numpy())
                logvar_vec.append(log_var.cpu().detach().numpy())

                if return_stacked_input:
                    stacked_input_vec.append(stacked_input.cpu().detach().numpy())

                if return_flux_ivar:
                    flux_vec.append(flux.cpu().detach().numpy())
                    ivar_vec.append(ivar.cpu().detach().numpy())

            # Collect outputs
            out["z"] = np.vstack(z_vec)
            out["mu"] = np.vstack(mu_vec)
            out["logvar"] = np.vstack(logvar_vec)

            if return_stacked_input:
                out["stacked_input"] = np.vstack(stacked_input_vec)

            if return_flux_ivar:
                out["flux"] = np.vstack(flux_vec)
                out["ivar"] = np.vstack(ivar_vec)

            return out

    ## ========================================================================= ##

    # TODO: Test these functions!
    def save_checkpoint(self, fname, epoch, loss, optimizer, compress=False):
        """Save the model checkpoint. Note that this requires to save
        also the state dict of the optimiser!

        :param fname:
        :param epoch:
        :param loss:
        :param optimizer:
        :return:
        """
        # follows the general information given here:
        #  https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        # this is probably the least amount of information we need to save
        logger.info(f"Saving model to: {fname}")

        # use timestamps for checkpointing
        timestamp = utilities.get_timestamp()
        if fname.suffix == "tar":
            fname = fname.parent / (fname.stem + timestamp + fname.suffix)
        else:
            logger.warning("Using tar as checkpoint extension.")
            fname = fname.parent / (fname.stem + timestamp + ".tar")

        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        torch.save(state_dict, fname)
        logger.info("Model saved.")

        if compress:
            logger.info("Compressing the model checkpoint.")

            with tempfile.TemporaryDirectory() as tmpdirname:
                with tarfile.open(fname) as f:
                    f.extractall(tmpdirname)

                cfname = fname.parent / (fname.stem + ".tar.gz")
                with tarfile.open(cfname, "w:gz") as f:
                    f.add(tmpdirname, ".")

                fname.unlink()

        return 0

    ## ========================================================================= ##

    def load_checkpoint(self, fname, optimizer, train=True):
        """Load the model checkpoint. Note that this requires to load
        also the state dict of the optimiser!

        :param fname:
        :param optimizer:
        :return:
        """
        # decompress checkpoint if needed
        if fname.suffix == ".tar.gz":
            logger.info("Decompressing the model checkpoint.")
            cfname = fname
            with tempfile.TemporaryDirectory() as tmpdirname:
                with tarfile.open(cfname, "r:gz") as f:
                    f.extractall(tmpdirname)

                fname = cfname.parent / (cfname.stem + ".tar")
                with tarfile.open(fname, "w") as f:
                    f.add(tmpdirname, ".")

        logger.info(
            f"Loading model from: {fname} in {'train' if train else 'eval'} mode."
        )
        checkpoint = torch.load(fname, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if train:
            self.train()
        else:
            self.eval()

        return optimizer, checkpoint["epoch"], checkpoint["loss"]
