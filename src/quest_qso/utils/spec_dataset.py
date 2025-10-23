#!/usr/bin/env python
# coding: utf-8
import logging

import numpy as np
import torch

from quest_qso.utils import resources, utilities

logger = logging.getLogger(__name__)


class SpecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath,
        data=None,
        subsample=1,
        replace_nan=False,
        replace_val=1e-10,
        snr_min=None,
        labels=None,
        return_label=False,
        device=None,
        scale_spectra=False,
        scale_method=None,
        scale_by=None,
    ):
        """Initialize the SpecDataset

        Columns in generated training data are:
        dispersion, flux_res, ivar_res,
        cont_final_tot, gpm_cont_tot, coverage_mask_res

        :param filepath: Path to the spectrum data file
        :type filepath: string
        :param subsample: Integer to indicate how the spectrum should be
         subsampled. A value of 3 will choose every third spectral data point.
        """
        valid_scale_methods = [
            "mult_by",
            "divide_by_mean",
            "divide_by_median",
            "log",
        ]
        if scale_spectra:
            assert scale_method in valid_scale_methods, (
                f"Invalid scale method: {scale_method}. "
                f"Valid options are: {', '.join(valid_scale_methods)}"
            )

        # set the device
        if device is None:
            device = utilities.set_device()
        else:
            self.device = device

        # whether to also return the labels (useful for conditioning)
        self.labels = labels
        self.return_label = return_label

        # Read the spectra data file
        if data is None:
            logger.info(f"Loading dataset from {filepath}")
            spec_data = np.load(filepath)
        else:
            logger.warning("Loading dataset from provided data and ignoring filepath!")
            spec_data = data

        # Set the dispersion
        self.dispersion = spec_data["dispersion"][0]

        # Set the spectral good pixel masks
        gpm = np.array(spec_data["gpm"][::subsample], dtype=bool)

        # Set the flux values
        flux = spec_data["flux"][::subsample]
        flux_autofit = spec_data["flux_autofit"][::subsample]
        ivar = spec_data["ivar"][::subsample]

        # Set the flux err values
        sigma = 1.0 / np.sqrt(ivar)
        snr = flux / sigma

        # Set the coverage masks
        coverage_mask = spec_data["coverage_mask"][::subsample]

        # Set flux values outside of coverage mask to NaN
        # Will be set to 0 later on
        flux[~gpm] = np.nan
        flux_autofit[coverage_mask < 1] = np.nan

        if snr_min is not None:
            coverage_mask[snr < snr_min] = 0
            flux[snr < snr_min] = np.nan

        # Scale spectra if requested
        # TODO: This needs to be scaled back to the original values when I compute the
        #  reconstruction loss. The smart thing to do would be to force the user to
        #  provide both a forward and backward scaling method, so that I can use that
        #  to scale back and forth.
        # Given that for now I am not touching this scaling, I will keep it like this
        #  but it needs to be done!
        self.scaling_factor = None

        if scale_spectra and scale_method == "mult_by":
            # TODO: make this a parameter that the user can set?
            if scale_by is None:
                logger.info("`scale_by` not set, defaulting to 0.25.")
                scale_by = 0.25

            self.scaling_factor = 1 / scale_by
            logger.warning(f"Scaling requested: dividing by {self.scaling_factor}.")

        elif scale_spectra and scale_method == "log":
            raise NotImplementedError(
                "Log scaling produces artefacts and should not be used for now."
            )

        elif scale_spectra and scale_method == "divide_by_median":
            logger.warning("Scaling requested: normalise by median spectrum.")
            if scale_by is not None:
                logger.warning("Ignoring parameter `scale_by`.")

            self.scaling_factor = resources.load_median_spec()

        elif scale_spectra and scale_method == "divide_by_mean":
            logger.warning("Scaling requested: normalise by mean spectrum.")
            if scale_by is not None:
                logger.warning("Ignoring parameter `scale_by`.")

            self.scaling_factor = resources.load_mean_spec()

        # actually divide the spectra by the scaling factor if needed
        if self.scaling_factor is not None:
            logger.warning(f"Scaling applied! Scaling method {scale_method}")
            flux /= self.scaling_factor
            flux_autofit /= self.scaling_factor
            ivar *= self.scaling_factor**2

        # Replace NaN with a value
        # TODO: the coverage mask should be a bool!
        if replace_nan is not None:
            flux = np.nan_to_num(flux, nan=replace_val)
            flux_autofit = np.nan_to_num(flux_autofit, nan=replace_val)
            gpm = np.nan_to_num(gpm, nan=replace_val)
            coverage_mask = np.nan_to_num(coverage_mask, nan=replace_val)
            ivar = np.nan_to_num(ivar, nan=replace_val)

        # Move to torch tensor
        stacked_data = np.stack((flux, gpm, flux_autofit, coverage_mask, ivar), axis=1)
        self.input = torch.from_numpy(stacked_data).to(self.device, dtype=torch.float)
        self.scaling_factor = (
            torch.from_numpy(self.scaling_factor).to(self.device, dtype=torch.float)
            if self.scaling_factor is not None
            else None
        )

    def apply_scaling(self):
        """Reset the scaling of the dataset

        :return:
        """
        if self.scaling_factor is None:
            logger.warning("Scaling was not originally applied, nothing to do.")
            return

        self.input[:, 0] /= self.scaling_factor
        self.input[:, 2] /= self.scaling_factor
        self.input[:, 4] *= self.scaling_factor**2

    def reset_scaling(self):
        """Reset the scaling of the dataset

        :return:
        """
        if self.scaling_factor is None:
            logger.warning("Scaling was not originally applied, nothing to do.")
            return

        self.input[:, 0] *= self.scaling_factor
        self.input[:, 2] *= self.scaling_factor
        self.input[:, 4] /= self.scaling_factor**2

    def __len__(self):
        """Get the length of the SpecDataset

        :return:
        """

        return self.input.shape[0]

    def __getitem__(self, idx):
        """Return one spectrum of the SpecDataset with index idx

        :param idx: Index of spectrum to return
        :return: Spectrum from SpecDataset
        """

        # Get spectrum with index idx
        if self.return_label:
            return self.input[idx, :], self.labels[idx]
        else:
            return self.input[idx, :]


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


# Convenience function to load the dataset; tries to download it as needed
def load_dataset(path, dset_fname, **kwargs):
    """
    Load the dataset used for training and evaluation.

    :param path: Path to the directory containing the dataset.
    :type path: str

    :return: Dataset directory path, SpecDataset object.
    :rtype: tuple
    """
    filepath = path / dset_fname
    if not filepath.exists():
        logger.info(f"Dataset file {filepath} not found locally.")
        logger.info("Attempting to download dataset from remote resource.")

        resources.download_dataasets_from_drive()

    logger.info("Loading dataset.")
    out = SpecDataset(filepath, **kwargs)
    logger.info("Dataset loaded.")

    return out, out.dispersion, out.scaling_factor


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


if __name__ == "__main__":
    print("Nothing to do!")
    # test_specdataset(device="mps")
