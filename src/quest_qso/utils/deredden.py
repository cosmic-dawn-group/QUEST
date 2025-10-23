import logging
from pathlib import Path

import dustmaps.config as cfg
import dustmaps.csfd
import numpy as np
import pandas as pd
from astropy import units
from astropy.coordinates import SkyCoord
from dustmaps.csfd import CSFDQuery
from numba_extinction import numba_extinction as ne

logger = logging.getLogger(__name__)

## ============================================================================= ##
## ============================================================================= ##
## ============================================================================= ##


def setup_dust_maps(path):
    print(path)
    if not (Path.home() / ".dustmapsrc").exists():
        print("\n =========================== \n")
        logger.info("Running automatic configuration!")
        logger.info("Generate `dustmaps` config file.")
        cfg.config.reset()
        cfg.config["data_dir"] = str(path)

        logger.info("Downloading dust map.")
        dustmaps.csfd.fetch()


## ============================================================================= ##
## ============================================================================= ##
## ============================================================================= ##


# 3.1 is a common choice for MW
def deredden(
    spectra_path,
    metadata_path,
    dust_map_path,
    save_path,
    r_v=3.1,
    ra_column="RA",
    dec_column="DEC",
    z_column="Z_PIPE",
):
    logger.info("Loading data.")
    spec_data = dict(np.load(spectra_path))

    logger.info("Loading metadata.")
    metadata = pd.read_hdf(metadata_path)

    # make coordindates for dust de-reddening
    sky_coords = SkyCoord(
        metadata[ra_column].to_numpy() * units.deg,
        metadata[dec_column].to_numpy() * units.deg,
        frame="icrs",
    )

    # take wavelength and compute back in observed frame for each object
    dispersion_obs_frame = (
        spec_data["dispersion"]
        * (1 + metadata[z_column].to_numpy()[:, None])
        * units.AA
    )

    # deredden these objects - this is the dust map we use
    setup_dust_maps(dust_map_path)
    csfd = CSFDQuery()
    # 0.86 comes from Schlafly+2010 (2011? depends on who you ask)
    #  and is not included in the CSFD map
    # See https://arxiv.org/pdf/2306.03926, page 17
    a_v = csfd(sky_coords) * r_v * 0.86

    logger.info("Computing exctinction.")
    a_lambda = ne.Go23(dispersion_obs_frame, r_v=r_v, a_v=a_v)
    for key in ["flux_orig", "flux_autofit", "flux"]:
        spec_data[key + "_deredden"] = ne.deredden(a_lambda, spec_data[key])

    # ivar goes the other way around
    spec_data["ivar_deredden"] = spec_data["ivar"] * 10 ** (-0.8 * a_lambda)

    logger.info(f"Saving dataset to: {save_path}.")
    np.savez(
        save_path,
        **spec_data,
    )

    return a_v, a_lambda, dispersion_obs_frame, spec_data


## ============================================================================= ##
## ============================================================================= ##
## ============================================================================= ##
