# Computed the error function and saves it to a file
# NB: this assumes a way (any way) to open the data. I am not making any assumption on how someone does that,
#  for the purpose of this I will just assume that the user can read the data needed for this, one way or
#  another

from random import choices

import numpy as np
from csaps import csaps
from tqdm.notebook import tqdm

from quest_qso.utils import utilities

## ============================================================================= ##
## ============================  Functions  ==================================== ##
## ============================================================================= ##


# generate confidence interval based on binned data
def get_binned_conf_int_idx(column, n=10**6, lp=16, up=84):
    """Computes the indices that allow to recover the confidence interval,
    given binned data. Uses the median as centre.

    :param column: _description_
    :type column: _type_
    :param n: _description_, defaults to 10**6
    :type n: _type_, optional
    :param lp: _description_, defaults to 16
    :type lp: int, optional
    :param up: _description_, defaults to 84
    :type up: int, optional
    :return: _description_
    :rtype: _type_
    """
    # lower percentile -> lp
    # upper percentile -> up
    column = column / np.nansum(column)
    column[np.isnan(column)] = 0
    # TODO: check if this is required
    # for now simple warning
    if np.sum(column) == 0:
        # print("[WARN] All zero weights, this should not happen!")
        return (np.nan, np.nan, np.nan)

    x_num = np.arange(len(column))
    data = choices(x_num, column, k=n)
    return (
        int(np.nanpercentile(data, lp)),
        # median instead of argmax, better!
        int(np.nanpercentile(data, 50)),
        int(np.nanpercentile(data, up)),
    )


## ============================================================================= ##


def get_values(arr_1, arr_2, i, idx):
    """Given arr_2 and arr_2, returns arr_1[i] and arr_2[idx], where idx are
    the indexes related to a given confidence interval (defaults to 16, 50
    and 84, or median pm roughly one sigma)

    :param arr_1: _description_
    :type arr_1: _type_
    :param arr_2: _description_
    :type arr_2: _type_
    :param i: _description_
    :type i: _type_
    :param idx: _description_
    :type idx: _type_
    :return: _description_
    :rtype: _type_
    """
    if np.isfinite(idx[1]):
        return np.array([arr_1[i], arr_2[idx[1]], arr_2[idx[0]], arr_2[idx[2]]])
    else:
        return np.array([arr_1[i], np.nan, np.nan, np.nan])


## ============================================================================= ##


def get_data_to_spline(
    data_x,
    data_y,
    cmin=None,
    bins=200,
    binned=True,
    cut_threshold=False,
    **kwargs,
):
    """Generates the data that will then be splined to obtain the error
    function.


    :param data_x: _description_
    :type data_x: _type_
    :param data_y: _description_
    :type data_y: _type_
    :param cmin: _description_, defaults to None
    :type cmin: _type_, optional
    :param bins: _description_, defaults to 200
    :type bins: int, optional
    :param binned: _description_, defaults to True
    :type binned: bool, optional
    :param sigma_clip: _description_, defaults to False
    :type sigma_clip: bool, optional
    :return: _description_
    :rtype: _type_
    """
    safe = ~(np.isnan(data_x) | np.isnan(data_y))
    data_x = data_x[safe]
    data_y = data_y[safe]

    mat = np.histogram2d(data_x, data_y, bins=bins)
    if cmin:
        mat[0][mat[0] < cmin] = np.nan

    bin_size_x = np.diff(mat[1])
    bin_size_y = np.diff(mat[2])

    # out
    out = np.zeros((mat[0].shape[0], 4))

    for i in tqdm(range(mat[0].shape[0])):
        if np.all(np.isnan(mat[0][i, :])):
            out[i] = [(mat[1][:-1] + bin_size_x / 2)[i], np.nan, np.nan, np.nan]

        # nth confidence
        # TODO: Try to use se data for this, the current implementation is possibly iffy!
        #  or at least add a switch
        if binned:
            inds = get_binned_conf_int_idx(mat[0][i, :], n=50000)
            out[i] = get_values(
                mat[1][:-1] + bin_size_x / 2, mat[2][:-1] + bin_size_y / 2, i, inds
            )

    if cut_threshold:
        # very manual
        out = out.T
        for row in out[1:]:
            row[row > kwargs.get("manual_th", np.inf)] = np.nan
        return out.T

    return out


## ============================================================================= ##


def gen_log_xy_bins(min_x, max_x, min_y, max_y, nbin, loglim=False):
    """Generate bins in logspace for both the x and y axis. By default
    limits are computed as `np.logspace(np.log10(min_x), np.log10(max_x), nbin)`.

    :param min_x: _description_
    :type min_x: _type_
    :param max_x: _description_
    :type max_x: _type_
    :param min_y: _description_
    :type min_y: _type_
    :param max_y: _description_
    :type max_y: _type_
    :param nbin: _description_
    :type nbin: _type_
    :param loglim: _description_, defaults to False
    :type loglim: bool, optional
    :return: _description_
    :rtype: _type_
    """
    if not loglim:
        log_bins_x = np.logspace(np.log10(min_x), np.log10(max_x), nbin)
        log_bins_y = np.logspace(np.log10(min_y), np.log10(max_y), nbin)
    else:
        log_bins_x = np.logspace(min_x, max_x, nbin)
        log_bins_y = np.logspace(min_y, max_y, nbin)

    return log_bins_x, log_bins_y


## ============================================================================= ##


def gen_log_x_bins(min_x, max_x, min_y, max_y, nbin, loglim=False):
    """Generate bins in logspace only for the x axis. By default
    limits are computed as `np.logspace(np.log10(min_x), np.log10(max_x), nbin)`.

    :param min_x: _description_
    :type min_x: _type_
    :param max_x: _description_
    :type max_x: _type_
    :param min_y: _description_
    :type min_y: _type_
    :param max_y: _description_
    :type max_y: _type_
    :param nbin: _description_
    :type nbin: _type_
    :param loglim: _description_, defaults to False
    :type loglim: bool, optional
    :return: _description_
    :rtype: _type_
    """
    if not loglim:
        log_bins_x = np.logspace(np.log10(min_x), np.log10(max_x), nbin)
        log_bins_y = np.linspace(min_y, max_y, nbin)
    else:
        log_bins_x = np.logspace(min_x, max_x, nbin)
        log_bins_y = np.linspace(min_y, max_y, nbin)

    return log_bins_x, log_bins_y


## ============================================================================= ##


def gen_log_y_bins(min_x, max_x, min_y, max_y, nbin, loglim=False):
    """Generate bins in logspace only for the y axis. By default
    limits are computed as `np.logspace(np.log10(min_y), np.log10(max_y), nbin)`.

    :param min_x: _description_
    :type min_x: _type_
    :param max_x: _description_
    :type max_x: _type_
    :param min_y: _description_
    :type min_y: _type_
    :param max_y: _description_
    :type max_y: _type_
    :param nbin: _description_
    :type nbin: _type_
    :param loglim: _description_, defaults to False
    :type loglim: bool, optional
    :return: _description_
    :rtype: _type_
    """
    if not loglim:
        log_bins_x = np.linspace(min_x, max_x, nbin)
        log_bins_y = np.logspace(np.log10(min_y), np.log10(max_y), nbin)
    else:
        log_bins_x = np.linspace(min_x, max_x, nbin)
        log_bins_y = np.logspace(min_y, max_y, nbin)

    return log_bins_x, log_bins_y


## ============================================================================= ##


def get_error_function_spline(
    data,
    i,
    x_grid=None,
    smooth=None,
):
    safe = np.where(np.isfinite(data[:, 0]) & np.isfinite(data[:, 1]))[0]

    if x_grid is None:
        x_grid = data[safe, 0]

    if smooth is not None:
        return (
            x_grid,
            csaps(data[:, 0][safe], data[:, i][safe], x_grid, smooth=smooth),
            smooth,
        )
    else:
        out = csaps(data[:, 0][safe], data[:, i][safe], x_grid)
        return x_grid, out.values, out.smooth


## ============================================================================= ##


def get_mu_sigma(data, x_grid=None, smooth=None, N=10000):
    """_summary_

    :param data: _description_
    :type data: _type_
    :param spline_evaluation_points: _description_
    :type spline_evaluation_points: _type_
    :param smooth: _description_, defaults to 0.95
    :type smooth: float, optional
    :return: _description_
    :rtype: _type_
    """
    if utilities.is_number(smooth) or smooth is None:
        smooth = (smooth, smooth, smooth)

    if x_grid is None:
        safe = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
        x_grid = np.logspace(np.log10(data[safe, 0][0]), np.log10(data[safe, 0][-1]), N)

    # return get_error_function_spline(data, 1, x_grid=x_grid, smooth=smooth[0])
    x_grid, mu, _ = get_error_function_spline(data, 1, x_grid=x_grid, smooth=smooth[0])
    sigma = (
        get_error_function_spline(data, 3, x_grid=x_grid, smooth=smooth[2])[1]
        - get_error_function_spline(data, 2, x_grid=x_grid, smooth=smooth[1])[1]
    ) / 2
    return x_grid, mu, sigma
