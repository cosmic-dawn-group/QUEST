# imports and stuff
import os
from datetime import datetime
from pathlib import Path

import astropy.units as units
import numba_extinction.numba_extinction as ne
import numpy as np
import pandas as pd
import speclite.filters as filters
from astropy.cosmology import Planck18 as cosmology
from atelier import lumfun

from quest_qso import mlconfig as cfg
from quest_qso.photometry import generate_photometry as gp

# just package some repeated functions
from quest_qso.utils import generate_photometry_utils as gp_qa

# =========================================================================== #
# ================================= Globals ================================= #
# =========================================================================== #


def get_cfg():
    out = {}

    # Parse from command line
    USER_CFG = gp_qa.parse_config()

    # default folder for everything
    out["LOCAL_PATH"] = Path(os.getenv("QUEST_LOCALPATH"))
    out["MODEL_PATH"] = Path(USER_CFG.model_path)

    # timestamps
    out["DAY"] = datetime.today().strftime("%Y%m%d")
    out["TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # clean intermediate products, usefyl if running on laptop especially
    out["CLEAR_MEMORY"] = USER_CFG.clear_memory

    # make QA plots
    out["MAKE_QA_PLOTS"] = USER_CFG.make_qa_plots
    out["N_EXAMPLES"] = USER_CFG.n_examples

    # Gid parameters
    out["N_PER_BIN"] = USER_CFG.n_per_bin

    out["N_Z"] = USER_CFG.n_z
    out["LOW_Z_LIM"] = USER_CFG.low_z_lim
    out["HIGH_Z_LIM"] = USER_CFG.high_z_lim

    out["N_M1450"] = USER_CFG.n_m1450
    out["FAINT_M1450_LIM"] = USER_CFG.faint_m1450_lim
    out["BRIGHT_M1450_LIM"] = USER_CFG.bright_m1450_lim

    # Reddening parameters
    out["REDDENING_MODEL"] = ne.Go23
    out["B_BOUNDS"] = 15 * units.deg
    out["R_V"] = 3.1

    # Sampling mode
    out["SAMPLE_MODE"] = USER_CFG.sample_mode
    out["CHOSEN_LF"] = USER_CFG.luminosity_function
    # This has to be the same name as in lumfun as I am using getattr
    # Maybe using a dictionary would be better?
    out["LF_SKY_AREA"] = USER_CFG.sky_area

    # Seed, which I should make sure I am properly propagating everywhere
    out["SEED"] = USER_CFG.seed

    return out


# =========================================================================== #


def print_cfg_summary(dict_):
    print()
    print("[INFO] Summary of input parameters:")
    print("+-------------------------+---------------------------+")
    print("| Parameter               | Value                     |")
    print("+-------------------------+---------------------------+")
    print(f"| Clear memory            | {str(dict_['CLEAR_MEMORY']):^25} |")
    print("+-------------------------+---------------------------+")
    print(f"| Make QA plots           | {str(dict_['MAKE_QA_PLOTS']):^25} |")
    print(f"| Number of QA examples   | {str(dict_['N_EXAMPLES']):^25} |")
    print("+-------------------------+---------------------------+")
    print(f"| Number per bin          | {str(dict_['N_PER_BIN']):^25} |")
    print("+-------------------------+---------------------------+")
    print(f"| Redshift bins           | {str(dict_['N_Z']):^25} |")
    print(f"| Redshift low            | {str(dict_['LOW_Z_LIM']):^25} |")
    print(f"| Redshift high           | {str(dict_['HIGH_Z_LIM']):^25} |")
    print("+-------------------------+---------------------------+")
    print(f"| M1450 bins              | {str(dict_['N_M1450']):^25} |")
    print(f"| M1450 bright            | {str(dict_['BRIGHT_M1450_LIM']):^25} |")
    print(f"| M1450 faint             | {str(dict_['FAINT_M1450_LIM']):^25} |")
    print("+-------------------------+---------------------------+")
    print(f"| Sampling mode           | {str(dict_['SAMPLE_MODE']):^25} |")
    print(f"| Chosen LF               | {str(dict_['CHOSEN_LF']):^25} |")
    print(f"| LF sky area [deg^2]     | {str(dict_['LF_SKY_AREA']):^25} |")
    print(f"| Random seed             | {str(dict_['SEED']):^25} |")
    print("+-------------------------+---------------------------+")
    print()


# =========================================================================== #


def generate_target_filters():
    # Example of how to generate custom filters, based on the SVO database
    # cf.generate_custom_filter("CTIO", "DECam", "z", "DES_SVO")
    # cf.generate_custom_filter("Subaru", "HSC", "z", "HSC_SVO")
    # SVO_custom = filters.load_filters("DES_SVO-z", "HSC_SVO-z")

    # As a default, just return the SDSS filter set
    return filters.load_filters("sdss2010-*")


# =========================================================================== #


def generate_grid_params(dict_):
    # Generate the grid -- At the moment we are using the unform sampling so
    #  so commenting out the LF call
    if dict_["SAMPLE_MODE"] == "lf" and dict_["CHOSEN_LF"] is not None:
        lf_sampling_param_dict = {
            "lf": getattr(lumfun, dict_["CHOSEN_LF"])(),
            "cosmology": cosmology,
            "sky_area": ["LF_SKY_AREA"],
            "seed": ["SEED"],
        }
        lf_sampling_param_dict["lf"].parameters["alpha"].value = -1.5
    else:
        lf_sampling_param_dict = None

    current_grid_params = {
        "M1450_range": [dict_["BRIGHT_M1450_LIM"], dict_["FAINT_M1450_LIM"]],
        "redshift_range": [dict_["LOW_Z_LIM"], dict_["HIGH_Z_LIM"]],
        "n_M1450_bins": dict_["N_Z"],
        "n_redshift_bins": dict_["N_M1450"],
        "n_per_bin": dict_["N_PER_BIN"],
        "sample": dict_["SAMPLE_MODE"],
        "lf_sampling_params": lf_sampling_param_dict,
    }

    return current_grid_params


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def main():
    globals_ = get_cfg()
    print_cfg_summary(globals_)
    current_grid_params = generate_grid_params(globals_)

    # set up custom filters
    target_filters = generate_target_filters()

    # Instantiate the parameter grid
    m1450_z_grid = gp.generate_grid(params=current_grid_params)

    # Load the model and sample from it - hardcoded
    model_param_fname = globals_["MODEL_PATH"] / "params.json"
    model_params = cfg.MLConfig().from_json(model_param_fname)

    # Sample unperturbed spectra from the VAE model
    spectra, dispersion = gp.sample_from_VAE(m1450_z_grid, model_params)

    gp_qa.plot_example_spectra(
        globals_["N_EXAMPLES"],
        spectra,
        dispersion,
        "GenPhotExample.png",
        ylabel="Flux Density [A.U.]",
        dir=globals_["LOCAL_PATH"] / "QA" / str(globals_["DAY"]) / "GeneratePhotometry",
        make_qa=globals_["MAKE_QA_PLOTS"],
    )

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # computes the scaling factor for each spectra, based on its assigned redshift and M1450
    # NB: it modifies grid.grid_data in place, adding the corresponding columns
    # The scaling is done to produce a spectrum in erg / (cm^2 s AA)
    # Cosmology is currently Planck 18
    gp.compute_scale_factor(dispersion, spectra, m1450_z_grid.grid_data, cosmology)

    # Scale each spectrum based on the scale value compute in the previous cell
    scaled_spectra = gp.scale_VAE_spectra(spectra, m1450_z_grid.grid_data["scale"])

    gp_qa.plot_example_spectra(
        globals_["N_EXAMPLES"],
        scaled_spectra,
        dispersion,
        "GenPhotExample_Scaled.png",
        ylabel=r"Flux density [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",
        dir=globals_["LOCAL_PATH"] / "QA" / str(globals_["DAY"]) / "GeneratePhotometry",
        make_qa=globals_["MAKE_QA_PLOTS"],
    )

    if globals_["CLEAR_MEMORY"]:
        print("[INFO] Clearing memory and removing intermediate products.")
        del spectra

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # Resample and shift to observed frame the spectra on a new wavelength grid.

    # The user can specify the desired wavelength grid, or the grid can be computed based
    #  on the requested filters.
    # In the first case, the easiest thing is to use utilities.gen_wave_grid(), as follows:
    #  new_rest_frame_dispersion = utilities.gen_wave_grid(7000 * units.AA, 10000 * units.AA, 140 * utilities.kms)
    # Note that the new grid will be equally spaced in velocity space, and units are required.
    #  the returned array is a quantity array!
    # Otherwise, if no grid is provided, then the grid is computed based on the filters set used
    # Likewise, the wavelength grid has constant bin width in velocity space

    # returns the resampled spectra, and the new (common) wavelength grid
    resampled_spectra, new_dispersion = gp.resample_on_wavelength_grid(
        target_filters,  # Set of filter response functions
        dispersion,  # Current dispersion (rest frame, it is multiplied by redshif)
        m1450_z_grid.grid_data["redshift"].to_numpy(),  # redshfit per each object
        scaled_spectra,  # Spectra in physical units
        new_rest_frame_dispersion=None,  # Computed based on filters, otherwise see above
    )

    gp_qa.plot_example_spectra(
        globals_["N_EXAMPLES"],
        resampled_spectra,
        new_dispersion,
        "GenPhotExample_Scaled_Resampled.png",
        ylabel=r"Flux density [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",
        dir=globals_["LOCAL_PATH"] / "QA" / str(globals_["DAY"]) / "GeneratePhotometry",
        make_qa=globals_["MAKE_QA_PLOTS"],
    )

    # clear out some memory, otherwise things get very, very slow
    if globals_["CLEAR_MEMORY"]:
        print("[INFO] Clearing memory and removing intermediate products.")
        del scaled_spectra

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # Compute the IGM transmission from SimQSO
    # This return a matrix where each line is the transmission of the IGM (i.e.,
    #  a [0, 1] function) that will then be multiplied to the scaled and
    #  resampled spectra

    # The following should be much faster for a large number of sighlines at least,
    #  as it does not need to precompute but applies things in chunks
    resampled_spectra_applied_IGM = gp.compute_apply_IGM_simqso(
        new_dispersion, m1450_z_grid, resampled_spectra
    )

    gp_qa.plot_example_spectra(
        globals_["N_EXAMPLES"],
        resampled_spectra_applied_IGM,
        new_dispersion,
        "GenPhotExample_Scaled_Resampled_IGMApplied.png",
        ylabel=r"Flux density [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",
        dir=globals_["LOCAL_PATH"] / "QA" / str(globals_["DAY"]) / "GeneratePhotometry",
        make_qa=globals_["MAKE_QA_PLOTS"],
    )

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # Re-apply reddening to the spectra as we use dereddened spectra while training
    gp.redden_sampled_spectra(
        new_dispersion,
        resampled_spectra_applied_IGM,
        globals_["REDDENING_MODEL"],
        globals_["R_V"],
        b_bounds=globals_["B_BOUNDS"],
    )

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # save array of generated spectra and dataframe grid
    generated_spectra_output = globals_["LOCAL_PATH"] / "generated_spectra"
    if not generated_spectra_output.exists():
        generated_spectra_output.mkdir(parents=True)

    np.save(
        generated_spectra_output
        / f"VAE_Spectra_with_IGM_dispersion_{globals_['TIMESTAMP']}.npy",
        new_dispersion.value,
    )

    np.save(
        generated_spectra_output / f"VAE_Spectra_with_IGM_{globals_['TIMESTAMP']}.npy",
        resampled_spectra_applied_IGM,
    )

    m1450_z_grid.grid_data.to_hdf(
        generated_spectra_output / f"Param_grid_{globals_['TIMESTAMP']}.hdf5",
        key="data",
    )

    # =========================================================================== #
    # =========================================================================== #
    # =========================================================================== #

    # Finally, generate new magnitudes based on the output of the previous cells
    mags = pd.concat(
        (
            m1450_z_grid.grid_data,
            target_filters.get_ab_magnitudes(
                resampled_spectra_applied_IGM, new_dispersion
            ).to_pandas(),
        ),
        axis=1,
    )

    # Needed anyway
    # TODO: Fix this based on the number of bands used
    bands = gp_qa.merge_bands(mags.columns[5:])

    # format filename
    outfilepath = globals_["LOCAL_PATH"] / "generated_photometry" / str(globals_["DAY"])
    if not outfilepath.exists():
        outfilepath.mkdir(parents=True)

    # TODO: Fix the filename
    outfilename = (
        f"GeneratedPhot-no_pert-{bands}-"
        + f"{globals_['SAMPLE_MODE']}-"
        + (f"{globals_['CHOSEN_LF']}" if globals_["SAMPLE_MODE"] == "lf" else "")
        + (f"{globals_['LF_SKY_AREA']}-" if globals_["SAMPLE_MODE"] == "lf" else "")
        + f"SEED_{globals_['SEED']}-"
        + f"{str(globals_['LOW_Z_LIM']).replace('.', 'p')}to{str(globals_['HIGH_Z_LIM']).replace('.', 'p')}_"
        + f"M1450-m{np.abs(globals_['BRIGHT_M1450_LIM'])}tom{np.abs(globals_['FAINT_M1450_LIM'])}_{globals_['TIMESTAMP']}.hdf5"
    )

    # save catalogues
    mags.to_hdf(
        outfilepath / outfilename,
        key="data",
    )

    print(f"[INFO] Catalogue saved to {outfilepath / outfilename}")
    print("[INFO] Photometry generation completed!")
    print(
        "[INFO] Edit and run `add_pert.py` with the appropriate filters to add noise."
    )


if __name__ == "__main__":
    main()
