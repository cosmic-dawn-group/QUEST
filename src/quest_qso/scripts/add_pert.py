# imports and stuff
import argparse
from pathlib import Path

import astropy.units as units
import numpy as np
import pandas as pd

from quest_qso.photometry import generate_photometry as gp
from quest_qso.utils import resources, utilities

# just package some repeated functions

# =========================================================================== #
# =========================================================================== #
# ===== The following is only needed if you want to do the perturbation ===== #
# =========================================================================== #
# =========================================================================== #

parser = argparse.ArgumentParser(description="Generate photometry configuration")

parser.add_argument(
    "-p",
    "--catalogue_path",
    type=str,
    help="Path to the input catalogue containing the magnitudes to be perturbed.",
)

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def do_all(input_catalogue):
    mags = pd.read_hdf(input_catalogue)

    # pandas does not like units, so we temporarily go back to numpy arrays
    mags_np = mags[
        [
            "sdss2010-u",
            "sdss2010-g",
            "sdss2010-r",
            "sdss2010-i",
            "sdss2010-z",
        ]
    ].to_numpy()
    flux_np = gp.AB_to_flux(mags_np)

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # load the error functions, units are known beforehand (see function itself)
    error_function_sdss = (
        utilities.pandas_to_recarray(resources.load_sdss_error_functions())
        * units.microJansky
    )

    perturbed_photometry = {}
    for n, band in enumerate(["U", "G", "R", "I", "Z"]):
        # the output are in this order:
        # band_perturbed, band_err_perturbed, band_flux_perturbed, band_snr
        perturbed_photometry[f"SDSS_{band}"] = gp.generate_perturbed_magnitudes(
            flux_np[:, n],
            error_function_sdss,
            f"SDSS_{band}",
            flag=99.0,
        )

    # _pert -> perturbed magnitudes
    # _sigma -> errors on perturbed magnitudes
    # _flux -> perturbed flux from which I compute the magnitudes

    mags_perturbed = utilities.numpy_to_pandas(
        [
            "SDSS_U_pert",
            "SDSS_U_sigma",
            "SDSS_U_flux",
            "SDSS_U_snr",
            "SDSS_G_pert",
            "SDSS_G_sigma",
            "SDSS_G_flux",
            "SDSS_G_snr",
            "SDSS_R_pert",
            "SDSS_R_sigma",
            "SDSS_R_flux",
            "SDSS_R_snr",
            "SDSS_I_pert",
            "SDSS_I_sigma",
            "SDSS_I_flux",
            "SDSS_I_snr",
            "SDSS_Z_pert",
            "SDSS_Z_sigma",
            "SDSS_Z_flux",
            "SDSS_Z_snr",
        ],
        np.array(
            (
                perturbed_photometry["SDSS_U"][0].value,
                perturbed_photometry["SDSS_U"][1].value,
                perturbed_photometry["SDSS_U"][2].value,
                perturbed_photometry["SDSS_U"][3].value,
                perturbed_photometry["SDSS_G"][0].value,
                perturbed_photometry["SDSS_G"][1].value,
                perturbed_photometry["SDSS_G"][2].value,
                perturbed_photometry["SDSS_G"][3].value,
                perturbed_photometry["SDSS_R"][0].value,
                perturbed_photometry["SDSS_R"][1].value,
                perturbed_photometry["SDSS_R"][2].value,
                perturbed_photometry["SDSS_R"][3].value,
                perturbed_photometry["SDSS_I"][0].value,
                perturbed_photometry["SDSS_I"][1].value,
                perturbed_photometry["SDSS_I"][2].value,
                perturbed_photometry["SDSS_I"][3].value,
                perturbed_photometry["SDSS_Z"][0].value,
                perturbed_photometry["SDSS_Z"][1].value,
                perturbed_photometry["SDSS_Z"][2].value,
                perturbed_photometry["SDSS_Z"][3].value,
            )
        ).T,
    )

    catalogue = pd.concat((mags, mags_perturbed), axis=1)

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # format filename
    outfilepath = input_catalogue.parent
    outfilename = input_catalogue.stem.replace("no_pert", "w_pert") + ".hdf5"

    # save catalogues
    catalogue.to_hdf(
        outfilepath / outfilename,
        key="data",
    )

    print(f"[INFO] Catalogue saved to {outfilepath / outfilename}")


if __name__ == "__main__":
    # I am sure this is not the intended way to do this...
    input_catalogue = Path(parser.parse_args().catalogue_path)

    if input_catalogue.is_file():
        do_all(input_catalogue)
    else:
        for file in input_catalogue.glob("*.hdf5"):
            do_all(file)
