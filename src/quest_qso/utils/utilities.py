# logger for everything
import datetime
import logging

import astropy.units as units
import numpy as np

# misc utilities
import pandas as pd
import requests
import torch
from astropy.coordinates import SkyCoord
from sklearn.mixture import GaussianMixture

from quest_qso import DEVICE
from quest_qso import LOCAL_PATH as local_path

## ========================================================================= ##

# convenience
kms = units.km / units.s
AA = units.AA
c_ms = 299792458.0 * (units.m / units.s)
# ^ PDG 2023, hopefully this will not change anytime soon
c_kms = c_ms.to(kms)


## ========================================================================= ##

# Logging

logger = logging.getLogger(__name__)

## ========================================================================= ##


def generate_masks(
    data, n_masks=3, mask_start_min=60, mask_width_min=40, mask_width_max=100, **kwargs
):
    masks = np.ones(shape=data.shape)

    for i in range(len(data)):
        for _ in range(np.random.randint(1, n_masks)):
            width = np.random.randint(mask_width_min, mask_width_max)
            loc = np.random.randint(mask_start_min, len(data[0]) - width)
            masks[i, loc : loc + width] = 0.0

    return torch.tensor(masks, **kwargs)


## ========================================================================= ##


def set_device(force=None):
    if force is not None:
        return force
    return DEVICE


## ========================================================================= ##


def prepare_input(
    x,
    device,
    flux_autofit=False,
    coverage_mask_flag=False,
    random_mask_flag=True,
):
    """

    :param x:
    :return:
    """
    # Get the flux values, the inverse variance and the coverage mask
    if flux_autofit:
        flux = x[:, 2, :].detach().clone()
        coverage_mask = x[:, 3, :].detach().clone()
        ivar = x[:, 4, :].detach().clone()
    else:
        flux = x[:, 0, :].detach().clone()
        coverage_mask = x[:, 1, :].detach().clone()
        ivar = x[:, 4, :].detach().clone()

    # From here on, prepare input for the models following the user input
    #  Quote of the day: "this part is hard" - Github Copilot, 20250129

    # default output, nothing at all
    output_flux = flux.detach().clone()
    output_ivar = ivar.detach().clone()
    output_mask = coverage_mask.detach().clone()

    # besides making ivar and flux consistent with each other
    output_ivar[flux == 0] = 0

    # Apply the random mask, if requested
    # Note that the random mask is multiplied to both the output flux and the output mask!
    if random_mask_flag:
        random_mask = generate_masks(flux, device=device, dtype=torch.float)
        output_flux *= random_mask
        output_ivar *= random_mask
        output_mask *= random_mask
    else:
        # this is just for return purposes if I don't request a mask
        random_mask = torch.ones_like(flux, device=device, dtype=torch.float)

    # Apply the coverage mask, if requested
    if coverage_mask_flag:
        output_flux *= coverage_mask
        output_ivar *= coverage_mask

    # TODO: stack nothing at all?
    stacked_input = torch.hstack((output_flux, output_mask))

    return (
        stacked_input,  # Input to the VAE. Possibly masked by both coverage and random masks depending on the arguments
        output_flux,  # Flux part of stacked input
        output_ivar,  # Inverse variance part of stacked input
        output_mask,  # Mask part of stacked input
        flux,  # Original flux, untouched and taken from dataloader
        ivar,  # Original ivar, untouched and taken from dataloader
        coverage_mask,  # Coverage mask, untouched
        random_mask,  # Random mask, generated on the fly
    )


## ========================================================================= ##


def create_latent_space_gmm(model, data_loader, n_components=3):
    z = model.latent_space_posterior(data_loader)["z"]

    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(z)

    return gmm, z


## ========================================================================= ##


def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


## ========================================================================= ##


def numpy_to_pandas(colnames, *args):
    df = pd.DataFrame(np.hstack((args)), columns=colnames)
    return df


## ========================================================================= ##


@units.quantity_input
def gen_wave_grid(
    wmin: units.AA,
    wmax: units.AA,
    dv: units.Quantity = 50 * kms,
    z=0,
    extend_right=True,
) -> units.AA:  # 50 kms, wave in AA
    w_em = 1215.67 * (1 + z) * units.AA
    v_min, v_max = (
        np.log(wmin / w_em) * c_kms,
        np.log(wmax / w_em) * c_kms,
    )
    if extend_right:
        return (
            np.exp(
                np.arange(v_min.value, v_max.value + dv.value, dv.value) * kms / c_kms
            )
            * w_em
        )
    else:
        return (
            np.exp(np.arange(v_min.value, v_max.value, dv.value) * kms / c_kms) * w_em
        )


## ========================================================================= ##


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype("string")
    return df


## ========================================================================= ##


def table_to_pandas(
    tbl,
    unroll_multicols=False,
    drop_multicols=False,
    convert_string_dtype=False,
    **kwargs,
):
    # this trows a warning about fragmentation. Solution:
    # just ignore the issue but add a warning if too many rows or cols
    safe_names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
    unsafe_names = set(tbl.colnames) - set(safe_names)
    assert not (unroll_multicols and drop_multicols), "Incompatible options."
    df = tbl[safe_names].to_pandas(**kwargs)

    if drop_multicols:
        return df
    elif unroll_multicols:
        newcols, newcolnames = None, []
        for colname in unsafe_names:
            content = tbl[colname]
            newcols = (  # data
                np.array(content)
                if newcols is None
                else np.hstack((newcols, tbl[colname]))
            )
            for n in range(content.shape[1]):  # column names
                newcolnames += [f"{colname}_{n}"]

        df = pd.concat((df, pd.DataFrame(newcols, columns=newcolnames)), axis=1)
        if convert_string_dtype:
            df = convert_column_types(df)

        return df
    else:
        for colname in unsafe_names:
            df[colname] = list(tbl[colname].data)

    if df.shape[1] > 100:
        logger.info("The resulting df has more than 100 columns.")

    return df.copy()


## ========================================================================= ##


def download_lusso(outpath=local_path, force_download=False):
    outpath = outpath / "Lusso15_table1.dat"

    def download_file(path=outpath):
        url = "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/txt?J/MNRAS/449/4204/table1.dat"
        req = requests.get(url)
        if req.status_code == 200:
            with open(path, "w") as f:
                f.write(req.content.decode(req.encoding))
        else:
            raise ValueError(
                "Could not download the Lusso+15 table. "
                f"Manually download it to {path} from {url}, or try again."
            )

    # try except seems to be recommended in these cases to avoid situations where the file is
    #  deleted between checking if it exists and actually opening it, see
    #  https://stackoverflow.com/a/82852
    try:
        if force_download:
            logger.info("Forced re-downloading.")
            download_file(path=outpath)
            logger.info("File downloaded.")

        out = np.genfromtxt(
            outpath,
            delimiter="|",
            skip_header=5,
            skip_footer=1,
        )
        logger.info("Using cached version of the file.")
        return out
    except FileNotFoundError:
        logger.info("File not found in cache, downloading it.")
        download_file(path=outpath)
        return np.genfromtxt(
            outpath,
            delimiter="|",
            skip_header=5,
            skip_footer=1,
        )


## ============================================================================= ##


def load_lusso_15(
    cut_at=1020, outpath=local_path, force_download=False
):  # aa, in units of the provided data
    qso_model = download_lusso(outpath=outpath, force_download=force_download)

    # uses the All version
    wave, flux, err = (qso_model[:, 0], qso_model[:, 1], qso_model[:, 2])

    return (
        wave[wave < cut_at] * units.AA,
        flux[wave < cut_at] * units.dimensionless_unscaled,
        err[wave < cut_at] * units.dimensionless_unscaled,
    )


## ============================================================================= ##


def pandas_to_recarray(df):
    """Convert Pandas DataFrame to numpy recarray

    :param df: _description_
    :type df: _type_
    :return: _description_
    :rtype: _type_
    """
    records = df.to_records(index=False)
    return np.array(records, dtype=records.dtype.descr)


## ============================================================================= ##


def limited_uniform(min, max, bounds, rng, size=None):
    if is_number(bounds):
        bounds = [-bounds, bounds]

    x = rng.uniform(min, max, size)
    return np.array(
        [
            _x
            if (_x <= bounds[0] or _x >= bounds[1])
            else limited_uniform(min, max, bounds, rng, 1)[0]
            for _x in x
        ]
    )


## ========================================================================= ##


# https://stackoverflow.com/a/77243426
def almost_square(number):
    """
    Make a square_ish assembly of plots.
    """
    factor1, factor2 = 0, number
    while factor1 + 1 <= factor2:
        factor1 += 1
        if number % factor1 == 0:
            factor2 = number // factor1

    return factor1, factor2


## ========================================================================= ##


# https://stackoverflow.com/a/77243426
def alt_almost_square(number):
    """
    Make a square_ish assembly of plots.
    """
    while True:
        factor1, factor2 = almost_square(number)
        if (
            1 / 2 * factor1 <= factor2
        ):  # the fraction in this line can be adjusted to change the threshold aspect ratio
            break
        number += 1
    return factor2, factor1


## ========================================================================= ##


def truncate(f, n):
    # thank you random stranger: https://stackoverflow.com/a/783927
    """Truncates/pads a float f to n decimal places without rounding"""
    s = "{}".format(f)
    if "e" in s or "E" in s:
        return "{0:.{1}f}".format(f, n)
    i, p, d = s.partition(".")
    return ".".join([i, (d + "0" * n)[:n]])


## ========================================================================= ##


def make_sdss_name(df, ra_str, dec_str):
    # ugly, bad and I am ashamed of it
    # TODO: Make a function that converts in house, so that I
    #  do not depend on astropy coordinates and units
    coords = SkyCoord(
        ra=df[ra_str].to_numpy() * units.deg,
        dec=df[dec_str].to_numpy() * units.deg,
        frame="icrs",
    )
    out = []
    r, d = coords.ra.hms, coords.dec.dms
    for _rh, _rm, _rs, _dd, _dm, _ds in zip(r.h, r.m, r.s, d.d, d.m, d.s):
        ra_formatted = f"{int(_rh):02d}" + f"{int(_rm):02d}" + truncate(_rs, 2)
        str_ds = truncate(abs(_ds), 1)
        if len(str_ds) == 3:
            str_ds = "0" + str_ds
        de_formatted = f"{int(abs(_dd)):02d}" + f"{int(abs(_dm)):02d}" + str_ds
        sign = "+" if np.sign(_dd) > 0 else "-"
        out.append(ra_formatted + sign + de_formatted)

    return out


## ========================================================================= ##


def gen_line_mask(dispersion):
    regions = [[1175, 1275], [1500, 1575], [1850, 1950], [2750, 2850], [4850, 5025]]
    weights = np.zeros_like(dispersion, dtype=np.float32)
    for region in regions:
        weights[(dispersion > region[0]) & (dispersion < region[1])] = 1.0
    return torch.tensor(weights, device=set_device())[None, :]


## ========================================================================= ##


def get_timestamp():
    """
    Get a timestamp for the current time.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%Sp%f")[:18]


## ========================================================================= ##


def chisq_stat(input_, output, ivar, coverage_mask, eps=-3):
    """Computes chi square statistics.

    :param input_: Original tensor
    :type input_: torch.Tensor
    :param output: Model output tensor
    :type output: torch.Tensor
    :param ivar: Inverse variance tensor
    :type ivar: torch.Tensor
    :param coverage_mask: Coverage mask relative to input_
    :type coverage_mask: torch.Tensor
    :return: Chi square statistics
    :rtype: torch.Tensor
    """
    # I think there is an issue with the size of the ivar
    #  as some values are so large compared to the others that they trow everything off
    # However, I cannot simply scale the ivar to the relative maximum value
    #  as this would lost the information about the relative importance of the different spectra
    # I could:
    #  1 - Scale the ivar to the relative maximum value (worse option I guess)
    #  2 - Scale the ivar by the median value (possibly even worse? Does not solve the above issue)
    #  3 - Take the log of the ivar and clip it so that it is always > 0
    #      3.1 - Tricky! Need to be careful about zeros
    #      3.note - excluding non-covered regions, there would be 23027 out of 36466029 total
    #                masked values
    #  4 - Take the log of the chi-square (but probably better to do this later on to avoid negative values?)
    #       it is not the same as it would be the log of the sum vs the sum of the logs...

    # This below is a bad way to deal with the issue, but improves the situation
    #  ivar_ = ivar / torch.max(ivar, axis=1)[0][..., None]

    # Before changing to the log this was:
    #  return (input_ - output) ** 2 * ivar * coverage_mask

    out = ((input_ - output).pow(2) * ivar).log1p()
    out *= coverage_mask
    # problem here ^ is that the log of 0 is -inf and I am not doing things properly
    #  and this now means that where there is no coverage, there is something
    #  however later on things are summed, so my understanding is that this is essentially
    #  equivalent to masking out the -inf that result later on
    # in addition, I cannot really take the log of the coverage mask, otherwise everything
    #  goes to -inf or 0, and that is, well, bad.

    # make sure we have no inf and replace them with a very small value
    if torch.any(~torch.isfinite(out)):
        out[~torch.isfinite(out)] = eps

    return out


## ========================================================================= ##


def sigmoid(x, x_not=0):
    return 1 / (1 + np.exp(-x + x_not))


## ========================================================================= ##


def sample_from_2d_hist(np_hist, y_bins, n_samples=100000, rng=None):
    """Given a 2D histogram, sample from it.

    The histogram should be creating using something like
    ```
    x_bins = np.linspace(1.0, 2.0, 50)
    y_bins = np.linspace(1.0, 2.0, 50)

    np_hist = np.histogram2d(
        df["x_col"],
        df["y_col"],
        bins=[x_bins, y_bins]
    )

    sampled_z, sampled_mi = sample_from_2d_hist(np_hist, y_bins, 10000000)
    ```

    :param np_hist: _description_
    :type np_hist: _type_
    :param n_samples: _description_, defaults to 100000
    :type n_samples: int, optional
    :param rng: _description_, defaults to None
    :type rng: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    hist_2d, x_edges, y_edges = np_hist

    if rng is None:
        logger.info("No random number generator provided, using default.")
        rng = np.random.default_rng()

    hist_2d_norm = hist_2d / np.sum(hist_2d)
    flat_hist = hist_2d_norm.flatten()
    cumulative_dist = np.cumsum(flat_hist)

    # Generate random numbers
    random_vals = rng.uniform(0, 1, n_samples)

    # Find corresponding indices in cumulative distribution
    indices = np.searchsorted(cumulative_dist, random_vals)

    # Convert flat indices back to 2D coordinates
    row_indices = indices // len(y_bins[:-1])
    col_indices = indices % len(y_bins[:-1])

    # Get bin centers and add random offset within bins
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    x_width = x_edges[1] - x_edges[0]
    y_width = y_edges[1] - y_edges[0]

    sampled_x = x_centers[row_indices] + rng.uniform(
        -x_width / 2, x_width / 2, n_samples
    )
    sampled_y = y_centers[col_indices] + rng.uniform(
        -y_width / 2, y_width / 2, n_samples
    )

    return sampled_x, sampled_y
