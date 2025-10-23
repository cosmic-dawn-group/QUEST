# improve parsing and default arguments
import importlib.resources
import json
import logging

import gdown
import numpy as np
import pandas as pd

from quest_qso import LOCAL_PATH as local_path

# =========================================================================== #


RESOURCE_PATH = importlib.resources.files("quest_qso")
PRE_TRAINED_MODELS_PATH = RESOURCE_PATH / "data" / "models"

# =========================================================================== #
# =============== Download pre-trained models if not available ============== #
# =========================================================================== #


logger = logging.getLogger(__name__)

# =========================================================================== #


def load_json_resource(json_fname, verbose=False):
    # This might become slow as the database grows
    #  Do I want to trim the cache after a certain size?
    #  For example remove all entries older than a month?
    # Probably problematic if done automatically, but maybe add a function to do that
    # See below
    try:
        with open(RESOURCE_PATH / "data" / json_fname) as f:
            data = json.load(f)
            return data

    except FileNotFoundError:
        logger.error(f"File {json_fname} not found in the resources.")

        if verbose:
            print("Files available in prefix: ")
            for file in (RESOURCE_PATH / "data").iterdir():
                if file.suffix == ".json":
                    print("\t" + file.name)

    except json.decoder.JSONDecodeError as e:
        logger.error(f"Error while decoding {json_fname}: {e}")

    return None


# =========================================================================== #


def load_median_spec(parent_dir=None):
    if parent_dir is None:
        # TODO: Fix the underscore, I made a mistake intionally
        #  to catch functions that default to the old file
        parent_dir = RESOURCE_PATH / "data_"
    else:
        parent_dir = local_path / parent_dir

    fname = "MedianSpec.npy"
    logger.info(f"Loading input scaling factor from {parent_dir / fname}")

    return np.load(parent_dir / fname)


# =========================================================================== #


def load_mean_spec(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data_"
    else:
        parent_dir = local_path / parent_dir

    fname = "MedianSpec.npy"
    logger.info(f"Loading input scaling factor from {parent_dir / fname}")
    return np.load(parent_dir / fname)


# =========================================================================== #


def load_sdss_error_functions(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    fname = "ErrorFunction_SDSS.hdf5"
    logger.info(f"Loading SDSS error function from {parent_dir / fname}")
    return pd.read_hdf(parent_dir / fname)


# =========================================================================== #


def load_2mass_error_functions(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    fname = "ErrorFunction_2Mass.hdf5"
    logger.info(f"Loading TwoMass error function from {parent_dir / fname}")
    return pd.read_hdf(parent_dir / fname)


# =========================================================================== #


def load_vikings_error_functions(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    fname = "ErrorFunction_VIKINGS.hdf5"
    logger.info(f"Loading VIKINGS error function from {parent_dir / fname}")
    return pd.read_hdf(parent_dir / fname)


# =========================================================================== #


def load_ukidss_error_functions(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    fname = "ErrorFunction_UKIDSS.hdf5"
    logger.info(f"Loading UKIDSS error function from {parent_dir / fname}")
    return pd.read_hdf(parent_dir / fname)


# =========================================================================== #


def load_panstarrs_error_functions(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    fname = "ErrorFunction_PanSTARRS_DR2.hdf5"
    logger.info(f"Loading PanSTARRS error function from {parent_dir / fname}")
    return pd.read_hdf(parent_dir / fname)


# =========================================================================== #


def load_unwise_error_functions(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    fname = "ErrorFunction_unWISE.hdf5"
    logger.info(f"Loading unWISE error function from {parent_dir / fname}")
    return pd.read_hdf(parent_dir / fname)


# =========================================================================== #


def load_vhs_error_functions(parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    fname = "ErrorFunction_VHS.hdf5"
    logger.info(
        f"Loading Vista Hemisphere Survey error function from {parent_dir / fname}"
    )
    return pd.read_hdf(parent_dir / fname)


# =========================================================================== #


def load_vdb_template(fname="vandenberk_template.csv", parent_dir=None):
    if parent_dir is None:
        parent_dir = RESOURCE_PATH / "data"
    else:
        parent_dir = local_path / parent_dir

    return np.loadtxt(
        parent_dir / fname,
        delimiter=",",
    )


# =========================================================================== #


def download_datasets_from_drive():
    md5 = "md5:496e922df03a0d4072424b166987a07c"
    output_fname = local_path / "datasets.zip"

    gdown.cached_download(
        "https://drive.google.com/uc?id=1Pho53tK8Ve3IQQWwEaNMdavzkKqBiAKH",
        str(output_fname),
        hash=md5,
        postprocess=gdown.extractall,
    )


# =========================================================================== #


def download_models_from_drive():
    # GP
    gdown.cached_download(
        "https://drive.google.com/uc?id=1u-s7oASkP5naYJpLFjyfpndQ_BBEqdvP",
        str(PRE_TRAINED_MODELS_PATH / "GP" / "InfoSpecVAE" / "GP_model_params.zip"),
        hash="md5:f9f6e6a8084ba9f0fb377354f45187e5",
        postprocess=gdown.extractall,
    )

    # FOR
    gdown.cached_download(
        "https://drive.google.com/uc?id=1FJDjJbxN3TRZTUGrmgJKasBrkaReuV5L",
        str(PRE_TRAINED_MODELS_PATH / "FOR" / "InfoSpecVAE" / "FOR_model_params.zip"),
        hash="md5:38ae7d0cd13ab92df3547cf2b29a3805",
        postprocess=gdown.extractall,
    )

    # FOB
    gdown.cached_download(
        "https://drive.google.com/uc?id=1ct4egEKCGbP1EB-2skbTlMoSCMWThibB",
        str(PRE_TRAINED_MODELS_PATH / "FOB" / "InfoSpecVAE" / "FOB_model_params.zip"),
        hash="md5:eb4ea9a5f7cf4a1c568645f2f001bfcb",
        postprocess=gdown.extractall,
    )

    logger.info("Pre-trained models downloaded successfully.")


# =========================================================================== #


def download_error_functions_from_drive():
    gdown.cached_download(
        "https://drive.google.com/uc?id=1Fv7mEUerVK1u9EiRmOdfESGACtrRz2s3",
        str(RESOURCE_PATH / "data" / "error_functions.zip"),
        hash="md5:991dd3745665427a9d51af0f058d4a26",
        postprocess=gdown.extractall,
    )
