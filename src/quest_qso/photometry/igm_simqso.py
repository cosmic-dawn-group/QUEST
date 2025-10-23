# Code in this script was taken with some modifications from https://github.com/imcgreer/simqso
# and related forks. The original license is given below.

# ================================================================================= #
# ================================================================================= #

# BSD 3-Clause License

# Copyright (c) 2017, Ian McGreer
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ================================================================================= #
# ================================================================================= #

# The code is mostly the same as the original code, with some clean up and rearrangements
#  I further stripped unnecessary classes and definitions, and changed a few functions to match
#  modern recommendations
# Code is accelerated through numba and should be a factor of about 10-ish faster than
# the original implementation for sum_of_voigts(...), which is the main bottleneck anyway

import logging

import astropy.constants as const
import astropy.units as au
import numpy as np
import requests
from astropy.io import fits
from numba import jit
from scipy.stats import poisson
from tqdm import tqdm

from quest_qso import LOCAL_PATH as local_path

logger = logging.getLogger(__name__)


## ============================================================================= ##
## ==============================  Defaults  =================================== ##
## ============================================================================= ##

# constants
URL = "https://github.com/imcgreer/simqso/raw/master/simqso/data/all_lin.fits"

# shorthands
c_kms = const.c.to(au.km / au.s).value
sqrt_pi = np.sqrt(np.pi)
sigma_c = 6.33e-18  # cm^-2
fourpi = 4 * np.pi

# default is to go up to 32 -> 1
default_lyman_series_range = (2, 33)


def _get_line_list_data(path=local_path):
    # Line list obtained from Prochaska's XIDL code
    # https://svn.ucolick.org/xidl/trunk/Spec/Lines/all_lin.fits
    fname = path / "all_lin.fits"
    try:
        linelist = fits.getdata(fname)
    except FileNotFoundError:
        logger.info("Line list not found, downloading from Github.")
        # get the file from github if not there
        with requests.get(URL) as r:
            byte_content = r.content
            with open(fname, "wb") as f:
                f.write(byte_content)
            linelist = fits.getdata(fname)

    H_lines = np.array([i for i in range(linelist.size) if "HI" in linelist.ION[i]])
    # if b'HI' in linelist.ION[i]])
    transition_params = {}
    for n, idx in enumerate(H_lines[::-1], start=2):
        transition_params[n] = (
            linelist.WREST[idx],
            linelist.F[idx],
            linelist.GAMMA[idx],
        )
    return transition_params


transitionParams = _get_line_list_data()


## ============================================================================= ##
## ==============================  Classes  ==================================== ##
## ============================================================================= ##


class IGMTransmissionGrid(object):
    # TODO: Change documentation style
    # TODO: Fix documentation, does not reflect the current state of the function
    """
    Generate a library of forest transmission spectra, by mapping an array
    of emission redshifts to a set of sightlines.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid (must be at fixed resolution!).
    z_em : `~numpy.ndarray`
        Array containing emission redshifts.
    nlos : int
        Number of lines-of-sight to generate.
    losMap : sequence
        Optional mapping from z_em to LOS. Must have the same number of
        elements and be in the range 0..nlos-1.
        If not provided and nlos>0, losMap is randomly generated.

    Returns
    -------
    spectra: dict
    T : `~numpy.ndarray`
        transmission spectra with shape (N(z),N(wave))
    z : `~numpy.ndarray`
        emission redshift for each spectrum
    losMap : `~numpy.ndarray`
        map of z_em <-> line-of-sight
    wave : `~numpy.ndarray`
        input wavelength grid
    voigtcache : bool
        use a lookup table of Voigt profiles to speed computation (def: True)
    """

    def __init__(self, wave, forest_model, num_sight_lines, **kwargs):
        self.forest_model = forest_model
        self.num_sight_lines = num_sight_lines
        self.seed = kwargs.get("seed", 42)  # Default value for reproducibility
        self.voigtkwargs = {}

        # pad the lower redshift by just a bit
        self.z_min = wave.min() / 1215.7 - 1.01
        self.z_max = kwargs.get("z_max", 10)

        # Generate the lines-of-sight first, to preserve random generator order
        logger.info("Generating {} sightlines".format(self.num_sight_lines))
        logger.info("Using random seed {}".format(self.seed))

        # Changed: random seed is now fixed.
        # Also use generator instead of seed as per recommendation
        self.rng = np.random.default_rng(seed=self.seed)

        # This returns (z, logNHI, b) for each absorption system, randomly generated.
        logger.info("Generating LOS.")
        self.sight_lines = [
            generate_los(self.forest_model, self.z_min, self.z_max, self.rng)
            for _ in tqdm(range(self.num_sight_lines))
        ]

        # default is 10 km/s
        forest_R_min = kwargs.get("R_min", 3e4)

        logwave = np.log(wave)
        dloglam = np.diff(logwave)

        if not np.allclose(dloglam, dloglam[0]):
            raise ValueError("Must have constant dloglam")

        spec_R = dloglam[0] ** -1
        self.n_rebin = int(np.ceil(forest_R_min / spec_R))
        if self.n_rebin > 1:
            logger.warning(
                f"Requested resolution lower than minimum. "
                "If this is what you want, decrease the limit by passing a different `R_min`. "
                f"The IGM transmission spectrum will be computed at resolution {int(spec_R * self.n_rebin)}"
            )

        self.forest_R = spec_R * self.n_rebin

        # go a half pixel below the minimum wavelength
        wave_min = np.exp(logwave[0] - 0.5 / spec_R)
        # go well beyond LyA to get maximum wavelength
        wave_max = min(wave[-1], 1250 * (1 + self.z_max))

        self.n_spec_pix = np.searchsorted(wave, wave_max, side="right")

        # now make sure it is an integer multiple
        wave_max = wave[self.n_spec_pix - 1]
        self.n_pix = self.n_spec_pix * self.n_rebin

        # Make the wavelength grid
        # As far as I understand this ^ is all to make a wavelength grid that
        #  is constant in resolution and is wide enough.
        # Before this was computed twice!
        self.forest_wave = np.exp(
            np.log(wave_min) + self.forest_R**-1 * np.arange(self.n_pix)
        )

        self.tau = np.zeros(self.n_pix)
        self.current_sight_line_num = -1

    ## ===================================================================== ##

    def next_spec(self, sight_line, z):
        # make a new sightline
        if self.current_sight_line_num != sight_line:
            logger.debug(f"Finished sightline: {self.current_sight_line_num}")
            self.current_sight_line = self.sight_lines[sight_line]
            self.current_sight_line_num = sight_line
            self.tau[:] = 0.0
            self.zi = 0

        zi1 = self.zi
        los = self.current_sight_line
        zi2 = np.searchsorted(los["z"], min(z, self.z_max))
        logger.debug(f"Extending sightline {sight_line} to z={z:.4f}")

        if zi2 < zi1:
            raise ValueError(
                "For performance reasons, you must generate"
                "sightline in increasing redshift!"
            )

        self.zi = zi2
        tau = calc_tau_lambda(
            self.forest_wave, los[zi1:zi2], tau_in=self.tau, **self.voigtkwargs
        )

        # Go back to the input resolution
        self.T = np.exp(-tau).reshape(-1, self.n_rebin).mean(axis=1)
        return self.T


## ========================================================================= ##
## ===========================  Functions  ================================= ##
## ========================================================================= ##


# jitting these is useless, they have numpy vectorisation anyway
# kept as it is cleaner!
def invert_z(x, z1, z2, gamma_p_1):
    return (1 + z1) * ((((1 + z2) / (1 + z1)) ** gamma_p_1 - 1) * x + 1) ** (
        1 / gamma_p_1
    ) - 1


## ========================================================================= ##


def invert_NHI(x, NHI_min, NHI_max, m_beta_p_1):
    return np.log10(
        NHI_min * (1 + x * ((NHI_max / NHI_min) ** m_beta_p_1 - 1)) ** (1 / m_beta_p_1)
    )


## ========================================================================= ##


def invert_b(x, b_sig, b_min, b_max):
    b_exp_min = np.exp(-((b_min / b_sig) ** -4))
    b_exp_max = np.exp(-((b_max / b_sig) ** -4))

    return b_sig * (-np.log((b_exp_max - b_exp_min) * x + b_exp_min)) ** (-1.0 / 4)


## ========================================================================= ##


def generate_los(model, z_min, z_max, rng):
    """
    Given a model for the distribution of absorption systems, generate
    a random line-of-sight populated with absorbers.
    returns (z, logNHI, b) for each absorption system.
    """
    abs_dtype = [("z", np.float64), ("logNHI", np.float64), ("b", np.float64)]
    absorbers = []
    for _, p in model.items():
        if z_min > p["zrange"][1] or z_max < p["zrange"][0]:
            # outside the redshift range of this forest component
            continue
        # parameters for the forest component (LLS, etc.) absorber distribution
        NHI_min, NHI_max = p["logNHrange"]
        NHI_min, NHI_max = 10**NHI_min, 10**NHI_max
        z1 = max(z_min, p["zrange"][0])
        z2 = min(z_max, p["zrange"][1])
        beta = p["beta"]

        # shorthands
        m_beta_p_1 = -beta + 1
        gamma_p_1 = p["gamma"] + 1

        # The following is just a lot of inverting distributions

        # Expectation for the number of absorbers at this redshift
        #  (inverting n(z) = N0*(1+z)^gamma)
        N = (p["N0"] / gamma_p_1) * ((1 + z2) ** gamma_p_1 - (1 + z1) ** gamma_p_1)
        # sample from a Poisson distribution for <N>
        n = poisson.rvs(N, size=1)[0]

        x = rng.random(3 * n)  # Generate random numbers all at once

        # 1 - Invert the dN/dz CDF to get the sample redshifts
        z = invert_z(x[0:n], z1, z2, gamma_p_1)

        # 2 - Invert the NHI CDF to get the sample column densities
        log10_NHI = invert_NHI(x[n : 2 * n], NHI_min, NHI_max, m_beta_p_1)

        # 3 - Invert the b CDF to get the sample column densities OR
        #  decide to take b as a constant and make a simplified model
        try:
            b = np.array([p["b"]] * n, dtype=np.float64)
        except KeyError:
            # dn/db ~ b^-5 exp(-(b/bsig)^-4) (Hui & Rutledge 1999)
            b_sig = p["bsig"]
            b_min, b_max = p["brange"]
            b = invert_b(x[2 * n :], b_sig, b_min, b_max)

        absorber = np.empty(n, dtype=abs_dtype)
        absorber["z"] = z
        absorber["logNHI"] = log10_NHI
        absorber["b"] = b
        absorbers.append(absorber)
    absorbers = np.concatenate(absorbers)

    # return sorted by redshift
    return absorbers[absorbers["z"].argsort()]


## ========================================================================= ##


def calc_tau_lambda(wave, los, **kwargs):
    """
    Compute the absorption spectrum, in units of optical depth, for
    a series of absorbers along a line-of-sight (los).
    """
    lyman_series_range = kwargs.get("lyman_series_range", default_lyman_series_range)
    tau_min = kwargs.get("tau_min", 1e-5)
    tau_max = kwargs.get("tau_max", 15.0)
    tau_lam = kwargs.get("tau_in", np.zeros_like(wave))

    # Arrays of absorber properties
    NHI = 10 ** los["logNHI"]
    z1 = 1 + los["z"]
    b = los["b"]

    # First apply continuum blanketing. The dense systems will saturate
    # a lot of the spectrum, obviating the need for calculations of
    # discrete transition profiles

    # !! Modifies tau_lam in place !!
    # Also makes in impossible to parallelise...
    sum_of_continuum_absorption(wave, tau_lam, NHI, z1, tau_min, tau_max)
    # Now loop over Lyman series transitions and add up Voigt profiles
    # keep track of time taken to run things - is the slowdown here?
    for transition in range(*lyman_series_range):
        # Transition properties
        lambda0, F, Gamma = transitionParams[transition]
        # Doppler width
        nu_D = b / (lambda0 * 1e-13)
        # Voigt a parameter
        a = Gamma / (fourpi * nu_D)
        # Wavelength of transition at absorber redshift
        # this can never be equal to zero
        lambda_z = lambda0 * z1
        # Coefficient of absorption strength (central tau)
        c_voigt = 0.014971475 * NHI * F / nu_D
        # All the values used to calculate tau, now just needs line profile

        # Numba makes this quite quick, despite everything, and solves the issue of being
        # forced to go through a fixed initialisation step
        # !! Modifies tau_lam in place !!
        #  Also makes in impossible to parallelise...
        sum_of_voigts(wave, tau_lam, c_voigt, a, lambda_z, b, tau_min, tau_max)

    return tau_lam


## ========================================================================= ##


@jit
def jit_any(arr, th):
    for item in arr:
        if item < th:
            return True
    return False


## ========================================================================= ##


# This is the fastest implementation I can currently find, despite many attempts and changes
#  to the order, combination, saving of variables and so on. Can't do much!
# Original implementation, goes bananas for small x
# @jit
# def voigt(a, x):
#     return np.exp(-(x**2)) - (a / sqrt_pi) / (x * x) * (
#         np.exp(-(x**2)) ** 2 * (4 * x**2 * x**2 + 7 * x**2 + 4 + 1.5 / x**2)
#         - 1.5 / x**2
#         - 1
#     )


# safer implementation, should never go negative
@jit
def voigt_safe(a, x):
    f1 = np.exp(-(x**2))
    return_ = f1 - (a / sqrt_pi) / (x * x) * (
        np.exp(-(x**2)) ** 2 * (4 * x**2 * x**2 + 7 * x**2 + 4 + 1.5 / x**2)
        - 1.5 / x**2
        - 1
    )
    return f1 if jit_any(return_, 0) else return_


## ========================================================================= ##


@jit
def sum_of_voigts(wave, tau_lam, c_voigt, a, lambda_z, b, tau_min, tau_max):
    """
    Given arrays of parameters, compute the summed optical depth
    spectrum of absorbers using Voigt profiles.
    Uses the Tepper-Garcia 2006 approximation for the Voigt function.
    """
    # make sure that I am not running anby of this if every item in tau_lam
    #  is already greater than tau_max
    if np.all(tau_lam > tau_max):
        tau_lam[tau_lam > tau_max] = tau_max + 1e-3
        return

    u_max = np.clip(np.sqrt(c_voigt * (a / sqrt_pi) / tau_min), 5.0, np.inf)

    # ***assumes constant velocity bin spacings***
    dv = (wave[1] - wave[0]) / (0.5 * (wave[0] + wave[1])) * c_kms
    du = dv / b
    b_norm = b / c_kms
    n_pix = (u_max / du).astype(np.int32)

    w0 = np.searchsorted(wave, lambda_z)
    w0_m_npix = w0 - n_pix
    w0_p_npix = w0 + n_pix
    len_wave = len(wave)

    # at the moment, this is as optimised as I can think of.
    # The only other obvious thing to check would be to consider
    #  all the regions to be skipped, one after the other, but I
    #  feel like that adds a lot of complexity to an already messy
    #  and complex piece of code, so we skip it.
    i1_i2 = np.empty((len(a), 2), dtype=np.int32)
    for i in range(len(a)):
        i1 = w0_m_npix[i] if w0_m_npix[i] > 0 else 0
        i2 = w0_p_npix[i] if w0_p_npix[i] < len_wave else len_wave
        i1_i2[i] = i1, i2

    # sort the regions by size, so that the ones with the largest
    #  span are looked at first
    i1_i2_sorting_inds = np.argsort(i1_i2[:, 1] - i1_i2[:, 0])
    i1_checked, i2_checked = len_wave, 0

    # now re-run, starting from the largest region
    # despite the large amount of ifs, this is slighly faster over 100
    #  iteration than just updating the regions without thinking about
    #  what was already checked, or than using a mask.
    for i in i1_i2_sorting_inds[::-1]:
        i1, i2 = i1_i2[i]

        # if we have already checked this region, just skip this altogether
        # Cache these conditions, we need them later too
        A = i1 < i1_checked
        B = i2 > i2_checked
        if A or B:
            # if there is at least one value for tau_lam below tau_max, update
            if jit_any(tau_lam[i1:i2], tau_max):
                # the clip is to prevent division by zero errors
                # note: this does not seems to be sufficient to solve the issue!
                # TODO: Investigate this more
                u = np.clip(
                    np.abs((wave[i1:i2] / lambda_z[i] - 1) / b_norm[i]), 1e-5, np.inf
                )
                tau_lam[i1:i2] += c_voigt[i] * voigt_safe(a[i], u)
            # otherwise, this region has no points that need to be updated,
            #  so I try to extend the checked region
            else:
                # the `and` here makes sure the regions are not disconnected,
                #  otherwise I incur in a bug where there might be points to be updated
                #  between the two regions which I do not take into account
                i1_checked = i1 if A and i2 > i1_checked else i1_checked
                i2_checked = i2 if B and i1 < i2_checked else i2_checked

    tau_lam[tau_lam > tau_max] = tau_max + 1e-3
    # no need to return tau_lambda, it is modified in place anyway


## ========================================================================= ##


@jit
def sum_of_continuum_absorption(wave, tau_lam, NHI, zp1, tau_min, tau_max):
    """
    Compute the summed optical depth for Lyman continuum blanketing
    given a series of absorbers with column densities NHI and
    redshifts zp1 (=1+z).
    """
    tau_c_lim = sigma_c * NHI
    lambda_z_c = 912.0 * zp1
    ii = np.where((lambda_z_c > wave[0]) & (tau_c_lim > tau_min))[0]

    # sort by decreasing column density to start with highest tau systems
    ii = ii[NHI[ii].argsort()[::-1]]

    # ending pixel (wavelength at onset of continuum absorption)
    i_end = np.searchsorted(wave, lambda_z_c[ii], side="right")

    # starting pixel - wavelength where tau drops below tauMin
    wave_start = (tau_min / tau_c_lim[ii]) ** 0.333 * wave[i_end]
    i_start = np.searchsorted(wave, wave_start)

    # now do the sum
    for i, i1, i2 in zip(ii, i_start, i_end):
        # ... only if pixels aren't already saturated
        if np.any(tau_lam[i1:i2] < tau_max):
            l1l0 = wave[i1:i2] / lambda_z_c[i]
            tau_lam[i1:i2] += tau_c_lim[i] * l1l0 * l1l0 * l1l0

    # no need to return tau_lam, it is modified in place and saves some time


## ========================================================================= ##
## ===========================  Forest Models  ============================= ##
## ========================================================================= ##

Fan99_model = {
    "forest": {
        "zrange": (0.0, 6.0),
        "logNHrange": (13.0, 17.3),
        "N0": 50.3,
        "gamma": 2.3,
        "beta": 1.41,
        "b": 30.0,
    },
    "LLS": {
        "zrange": (0.0, 6.0),
        "logNHrange": (17.3, 20.5),
        "N0": 0.27,
        "gamma": 1.55,
        "beta": 1.25,
        "b": 70.0,
    },
    "DLA": {
        "zrange": (0.0, 6.0),
        "logNHrange": (20.5, 22.0),
        "N0": 0.04,
        "gamma": 1.3,
        "beta": 1.48,
        "b": 70.0,
    },
}

WP11_model = {
    "forest0": {
        "zrange": (0.0, 1.5),
        "logNHrange": (12.0, 19.0),
        "gamma": 0.2,
        "beta": 1.55,
        "B": 0.0170,
        "N0": 340.0,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "forest1": {
        "zrange": (1.5, 4.6),
        "logNHrange": (12.0, 14.5),
        "gamma": 2.04,
        "beta": 1.50,
        "B": 0.0062,
        "N0": 102.0,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "forest2": {
        "zrange": (1.5, 4.6),
        "logNHrange": (14.5, 17.5),
        "gamma": 2.04,
        "beta": 1.80,
        "B": 0.0062,
        "N0": 4.05,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "forest3": {
        "zrange": (1.5, 4.6),
        "logNHrange": (17.5, 19.0),
        "gamma": 2.04,
        "beta": 0.90,
        "B": 0.0062,
        "N0": 0.051,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "SLLS": {
        "zrange": (0.0, 4.6),
        "logNHrange": (19.0, 20.3),
        "N0": 0.0660,
        "gamma": 1.70,
        "beta": 1.40,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "DLA": {
        "zrange": (0.0, 4.6),
        "logNHrange": (20.3, 22.0),
        "N0": 0.0440,
        "gamma": 1.27,
        "beta": 2.00,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
}

McG13hiz_model = {
    "forest1": {
        "zrange": (1.5, 10.1),
        "logNHrange": (12.0, 14.5),
        "gamma": 3.5,
        "beta": 1.50,
        "N0": 8.5 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "forest2": {
        "zrange": (1.5, 10.1),
        "logNHrange": (14.5, 17.2),
        "gamma": 3.5,
        "beta": 1.70,
        "N0": 0.33 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "LLS": {
        "zrange": (1.5, 10.1),
        "logNHrange": (17.2, 20.3),
        "gamma": 2.0,
        "beta": 1.3,
        "N0": 0.13 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "subDLA": {
        "zrange": (0.0, 10.1),
        "logNHrange": (20.3, 21.0),
        "N0": 0.13 / 7.5 * 1.1,
        "gamma": 1.70,
        "beta": 1.28,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "DLA": {
        "zrange": (0.0, 10.1),
        "logNHrange": (21.0, 22.0),
        "N0": 0.13 / 33 * 1.1,
        "gamma": 2.0,
        "beta": 1.40,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
}

zpivot = 5.6
DoublePowerLaw_model = {
    "forest1": {
        "zrange": (1.5, zpivot),
        "logNHrange": (12.0, 14.5),
        "gamma": 3.5,
        "beta": 1.50,
        "N0": 8.5 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "forest1_eor": {
        "zrange": (zpivot, 10.1),
        "logNHrange": (12.0, 14.5),
        "gamma": 3.5,
        "beta": 1.50,
        "N0": 8.5 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "forest2": {
        "zrange": (1.5, zpivot),
        "logNHrange": (14.5, 17.2),
        "gamma": 3.5,
        "beta": 1.70,
        "N0": 0.33 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "forest2_eor": {
        "zrange": (zpivot, 10.1),
        "logNHrange": (14.5, 17.2),
        "gamma": 3.5,
        "beta": 1.70,
        "N0": 0.33 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "LLS": {
        "zrange": (1.5, 10.1),
        "logNHrange": (17.2, 20.3),
        "gamma": 2.0,
        "beta": 1.3,
        "N0": 0.13 * 1.1,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "subDLA": {
        "zrange": (0.0, 10.1),
        "logNHrange": (20.3, 21.0),
        "N0": 0.13 / 7.5 * 1.1,
        "gamma": 1.70,
        "beta": 1.28,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
    "DLA": {
        "zrange": (0.0, 10.1),
        "logNHrange": (21.0, 22.0),
        "N0": 0.13 / 33 * 1.1,
        "gamma": 2.0,
        "beta": 1.40,
        "brange": (10.0, 100.0),
        "bsig": 24.0,
    },
}

DYhiz_model = DoublePowerLaw_model.copy()
DYhiz_model["forest1"]["gamma"] = DYhiz_model["forest2"]["gamma"] = 3.5
DYhiz_model["forest1"]["N0"] = 8.5 * 1.1
DYhiz_model["forest2"]["N0"] = 0.33 * 1.1

DYhiz_model["forest1_eor"]["gamma"] = DYhiz_model["forest2_eor"]["gamma"] = 4.0
DYhiz_model["forest1_eor"]["N0"] = 4
DYhiz_model["forest2_eor"]["N0"] = 0.2

DYhiz_model["forest1"]["N0"] = 8.5 * 1.05
DYhiz_model["forest2"]["N0"] = 0.33 * 1.05
