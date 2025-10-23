#!/usr/bin/env python
# coding: utf-8
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy as dc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy import constants as const
from astropy import units
from astropy.coordinates import SkyCoord
from dustmaps.csfd import CSFDQuery
from numba_extinction import numba_extinction as ne
from scipy.integrate import trapezoid
from scipy.special import expit
from spectres import spectres
from torch.utils.data import DataLoader
from tqdm import tqdm

import quest_qso.photometry.igm_simqso as igm
from quest_qso import LOCAL_PATH as local_path
from quest_qso.models.info_vae import InfoSpecVAE
from quest_qso.utils import deredden, load, utilities
from quest_qso.utils.spec_dataset import SpecDataset

logger = logging.getLogger(__name__)

# TODO: This entire file is a mess, and needs to be cleaned up

default_rng = np.random.default_rng(42)

default_grid_params = {
    "M1450_range": [-27, -22],
    "redshift_range": [7, 9],
    "n_M1450_bins": 10,
    "n_redshift_bins": 10,
    "n_per_bin": 10,
    "sample": "uniform",
}

kms = utilities.kms


## ============================================================================= ##
## ==============================  Classes  ==================================== ##
## ============================================================================= ##


class SpectrumGrid(object):
    def __init__(
        self, M1450_range, redshift_range, n_M1450_bins, n_redshift_bins, n_per_bin
    ):
        self.M1450_range = M1450_range
        self.redshift_range = redshift_range
        self.n_M1450_bins = n_M1450_bins
        self.n_redshift_bins = n_redshift_bins
        self.n_per_bin = n_per_bin
        self._init_grid()

    ## ========================================================================= ##

    def _init_grid(self):
        M_edges = np.linspace(
            self.M1450_range[0], self.M1450_range[1], self.n_M1450_bins + 1
        )
        redshift_edges = np.linspace(
            self.redshift_range[0], self.redshift_range[1], self.n_redshift_bins + 1
        )

        self.M_edges = M_edges
        self.redshift_edges = redshift_edges

        self.grid_shape = [self.n_M1450_bins, self.n_redshift_bins, self.n_per_bin]

        self.grid_dim = len(self.grid_shape) - 1
        self.grid_edges = (M_edges, redshift_edges)
        self.grid_edges = np.meshgrid(*self.grid_edges, indexing="ij")

        # Names of the columns of the to-be dataframe
        self.grid_edge_names = ["M1450", "redshift"]

    ## ========================================================================= ##

    def uniform_sampling(self, rng=default_rng):
        grid_data = {}

        for i, edges in enumerate(self.grid_edges):
            s = [slice(0, n, 1) for n in self.grid_shape[: self.grid_dim]]

            # This is what shifts the redshift and the magnitudes later on
            x = rng.random(self.grid_shape)

            edge_values = edges[tuple(s)][..., None]
            bin_size = np.diff(edges, axis=i)

            # Modify the slice to ignore the i-th dimension
            s[i] = slice(None)
            grid_points = x * bin_size[tuple(s)][..., None] + edge_values

            grid_data[self.grid_edge_names[i]] = grid_points.flatten()

        self.grid_data = pd.DataFrame(grid_data)  # Recast as a DataFrame

    ## ========================================================================= ##

    def lf_sampling(self, lf=None, cosmology=None, sky_area=0, **kwargs):
        """kwargs are passed to `lf.sample`

        :param lf: _description_
        :type lf: _type_
        :param rng: _description_, defaults to default_rng
        :type rng: _type_, optional
        """
        grid_data = {}
        M1450_sample, redshift_sample = lf.sample_mcmc(
            self.M1450_range, self.redshift_range, cosmology, sky_area, **kwargs
        )

        # grid_data["i_id"] = np.arange(len(M1450_sample))
        grid_data[self.grid_edge_names[0]] = M1450_sample
        grid_data[self.grid_edge_names[1]] = redshift_sample

        self.grid_data = pd.DataFrame(grid_data)

    ## ========================================================================= ##

    def hist2d_sampling(self, df, redshift_col_name, M_col_name):
        """Sample the grid using a 2D histogram of the provided dataframe.
        Bins and the number of object to sample are automatically determined
        based on the edges provided at init time and used to create an histogram
        using the x_col and y_col columns in the dataframe.

        :param df: DataFrame containing the data to sample from
        :type df: pandas.DataFrame
        :param z_col_name: Column name for the redshift column
        :type z_col_name: str
        :param M_col_name: Column name for the absolute magnitude column
        :type M_col_name: str
        """
        grid_data = {}

        print(
            "Note that for the 2D histogram sampling, the total number of objects is given by:\n"
        )
        print("\tN = n_per_bin * n_M1450_bins * n_redshift_bins.\n")
        print(
            "The final distribution will match the provided edges, but the number of objects per bin"
        )
        print(
            " will NOT match the parameters `n_per_bin` and will instead reflect the parent distribution.\n"
        )
        N = self.n_per_bin * self.n_M1450_bins * self.n_redshift_bins

        np_hist = np.histogram2d(
            df[redshift_col_name],
            df[M_col_name],
            bins=[self.redshift_edges, self.M_edges],
        )

        # Actually does the sampling
        sampled_z, sampled_M = utilities.sample_from_2d_hist(np_hist, self.M_edges, N)

        grid_data[self.grid_edge_names[0]] = sampled_M
        grid_data[self.grid_edge_names[1]] = sampled_z
        self.grid_data = pd.DataFrame(grid_data)

    ## ========================================================================= ##

    def __str__(self):
        outstr = """Grid params:\n\tM1450: {}\n\tRedshift: {}\n\tn_1450 bins: {}\n\tn_redshift bins {}\n\tn per bin: {}""".format(
            self.M1450_range,
            self.redshift_range,
            self.n_M1450_bins,
            self.n_redshift_bins,
            self.n_per_bin,
        )
        return outstr

    ## ========================================================================= ##

    def __repr__(self):
        outstr = """Grid params:\n\tM1450: {}\n\tRedshift: {}\n\tn_1450 bins: {}\n\tn_redshift bins {}\n\tn per bin: {}""".format(
            self.M1450_range,
            self.redshift_range,
            self.n_M1450_bins,
            self.n_redshift_bins,
            self.n_per_bin,
        )
        return outstr


## ============================================================================= ##
## =========================  JT functions  ==================================== ##
## ============================================================================= ##


def calc_fwav_from_appmag(appmag, dispersion):
    """Calculate the monochromatic flux density from the apparent magnitude.

    :param appmag: apparent magnitude
    :param dispersion:
    :return:
    """
    f_nu = (appmag * units.ABmag).to(units.erg / units.s / units.cm**2 / units.Hz)

    f_wav = f_nu.to(
        units.erg / units.s / units.AA / units.cm**2,
        equivalencies=units.spectral_density(dispersion),
    )

    return f_wav


## ============================================================================= ##


def calc_appmag_from_absmag(absmag, redshift, cosmology):
    """Calculate the apparent magnitude from the absolute magnitude. Includes
    the K-correction.

    :param absmag:
    :param redshift:
    :param cosmology:
    :return:
    """
    dm = cosmology.distmod(redshift).value
    kcorr = 2.5 * np.log10(1 + redshift)

    return absmag + dm + kcorr


## ============================================================================= ##
## =========================  FG functions  ==================================== ##
## ============================================================================= ##


# u.ABmag if I want to implement magnitudes using astropy
def to_flux(mags) -> units.Jy:
    return np.exp(-(mags + 48.6) / 2.5)


## ============================================================================= ##


def get_common_dispersion(filters, dv=140 * kms):
    logger.info(f"Using a dv of {dv} for the wavelength grid.")
    wavelengths = np.concatenate([_filter.wavelength for _filter in filters])
    min_wavelength = np.min(wavelengths) * units.AA
    max_wavelength = np.max(wavelengths) * units.AA
    new_wave_grid = utilities.gen_wave_grid(min_wavelength, max_wavelength, dv=dv)
    logger.info(
        "The new dispersion grid goes from "
        f"{new_wave_grid[0]:.2f} to {new_wave_grid[-1]:.2f}, with {new_wave_grid.shape[0]} pixels."
    )
    return utilities.gen_wave_grid(min_wavelength, max_wavelength, dv=dv)


## ============================================================================= ##


def resample_single(new_disp_value, current_disp_value_rest, redshift, spectrum):
    # Resample a single spectrum
    return spectres(
        new_disp_value,
        current_disp_value_rest * (1 + redshift),
        spectrum,
        # TODO: Should I fill with zeros, or better to fill with nans?
        # In principle zeros should be fine, as it will be on the very blue
        # and the IGM likely eats everything over there
        fill=0.0,  #
        verbose=False,
    )


def resample_on_wavelength_grid(
    filters,
    current_rest_frame_dispersion,
    redshifts,
    spectra,
    new_rest_frame_dispersion=None,
):
    if new_rest_frame_dispersion is None:
        new_dispersion = get_common_dispersion(filters)
    else:
        new_dispersion = new_rest_frame_dispersion

    with ProcessPoolExecutor(max_workers=2) as executor:
        # I am not really interested in the order, but given it can be preserved without too much effort, I'll
        #  keep this
        tasks = {
            executor.submit(
                resample_single,
                new_dispersion.value,
                current_rest_frame_dispersion.value,
                redshifts[i],
                spectra[i],
            ): i
            for i in range(spectra.shape[0])
        }
        # The list is used to put things back into order
        results = [None] * spectra.shape[0]
        # Process completed tasks and store results
        for future in tqdm(as_completed(tasks), total=spectra.shape[0]):
            idx = tasks[future]  # Get the original index of the task
            try:
                resample_result = future.result()  # Get the result
            except Exception as exc:
                logger.error(f"Spectrum at index {idx} generated an exception: {exc}")
            else:
                results[idx] = resample_result  # Store result in correct position

    resampled_spectra = np.array(results)

    return resampled_spectra, new_dispersion


## ============================================================================= ##


def compute_scale_factor(
    dispersion, spectra, grid, cosmology, ref_wave=1450 * units.AA, delta=10 * units.AA
):
    grid["m_intr"] = calc_appmag_from_absmag(grid["M1450"], grid["redshift"], cosmology)
    grid["f_1450"] = calc_fwav_from_appmag(grid["m_intr"].to_numpy(), ref_wave).value

    # scaling factor
    grid["scale"] = grid["f_1450"] / [
        np.nanmedian(
            i[(dispersion > ref_wave - delta) & (dispersion < ref_wave + delta)]
        )
        for i in spectra
    ]  # again, no way out of this
    return None


## ============================================================================= ##


def scale_VAE_spectra(
    spectra, scale, units=units.erg / (units.AA * units.cm**2 * units.s)
):
    return spectra * scale.to_numpy().reshape(-1, 1) * units


## ============================================================================= ##


def compute_apply_IGM_zero(
    dispersion, grid, resampled_spectra, cutoff=1215.67 * units.AA, inplace=False
):
    mult = np.ones_like(resampled_spectra)
    redshift = (grid.grid_data["redshift"].to_numpy().reshape(-1, 1),)
    mult[dispersion < cutoff * (1 + redshift)] = 0.0

    if inplace:
        resampled_spectra *= mult
        return resampled_spectra
    else:
        return resampled_spectra * mult


## ============================================================================= ##


def compute_apply_IGM_simqso(
    dispersion,
    grid,
    spectra,
    R=40000,
    n_sightlines=None,
    n_div=100,
    forest_model=igm.DYhiz_model,
):
    # Applies the IGM chunk by chunk to speed up the computation,
    #  array become large, so memory concerns are significant in this case
    # construct the wavelength based on the dispersion itself
    lower_wave_lim = dispersion[0]
    upper_wave_lim = dispersion[-1]

    wave = utilities.gen_wave_grid(
        lower_wave_lim, upper_wave_lim, utilities.c_kms / R / 2
    )
    # ^ Don't think it matters at all, but sample with two pixels
    #    each resolution element
    # Also, SimQSO does not like quantities and will enforce a minimum resolution

    # forest model, from SimQSO
    # the number of sightlines defaults to n_spec // 100
    if n_sightlines is None:
        n_sightlines = np.max(grid.grid_data.index // n_div) + 1

    transmission_grid = igm.IGMTransmissionGrid(wave.value, forest_model, n_sightlines)

    # generate transmission spectra
    #  1: assigns a given number of M1450 - z to a sightline
    #  2: sort in z due to SimQSO constrain
    #  3: generate transmisison spectra, resort them to match the original sorting,
    #      and make a single array
    #  4: shift to restframe and sample to the same grid as the sampled spectra
    logger.info("Generating transmission spectra.")

    out = None
    for sightline_num, idx in (
        pbar := tqdm(enumerate(np.array_split(grid.grid_data.index, n_sightlines)))
    ):
        sub_df = grid.grid_data.loc[idx]
        pbar.set_description(f"Working on sightline: {sightline_num}")

        # Sort the subdf in z - the column i_id are now the indeces
        #  I need to use to sort the sightlines back
        # also reset the index to create a copy and make sure things are
        #  not borked in general. We just reassing then!

        # first get the sorting inds for the original spectra
        # and sort in redshift
        redshift_sorting_inds = np.argsort(sub_df["redshift"])
        sorted_sub_df = sub_df.iloc[redshift_sorting_inds].reset_index(drop=True)

        # transmission grid for this particular sightline
        igm_transmission = np.zeros(
            (sorted_sub_df.shape[0], transmission_grid.forest_wave.shape[0])
        )

        for row_idx, row in sorted_sub_df.iterrows():
            sightline_redshift = row["redshift"]
            logger.debug(
                f"Working on sightline {row_idx} at redshift {sightline_redshift}"
            )
            igm_transmission[row_idx, :] = transmission_grid.next_spec(
                sightline_num, sightline_redshift
            )

        # The transmission spectra are shorter than the wave array I am giving as input.
        # I just let spectres handle this and fill with ones
        resampled_igm_transmission = spectres(
            dispersion.value,
            transmission_grid.forest_wave,
            igm_transmission,
            fill=1.0,
            verbose=False,
        )

        # correct first column that is accidentally set to one due to spectres
        resampled_igm_transmission[:, :1] = 0.0

        # multiply with current set of spectra
        processed = resampled_igm_transmission * spectra[idx[redshift_sorting_inds], :]

        # sort back to the original order
        processed = processed[np.argsort(redshift_sorting_inds)]

        # accumulate the results
        if out is None:
            out = processed.copy()
        else:
            out = np.concatenate((out, processed))

    # For some spectra, the IGM goes completely bananas (for a single pixel)
    # This is essentially random and is just caused by a (randomly generated...)
    # unfortunate transition that gives a 1/~zero.
    # To avoid slowing things down further, I just mask it out here
    # Masking is easy, as these values are generally so off compared to the median
    #  that one can spot them immediatelys
    # out[out > 100 * np.median(out)] = 0.0

    return out


## ============================================================================= ##


def graft_lusso_15(spectra, dispersion):
    # this could most likely be faster, so be it
    # input
    wave_lusso, flux_lusso, _ = utilities.load_lusso_15()

    # redisperse Lusso template on the same pixel scale as the spectra, including the left part
    # If I do not do this, it becomes a mess with indeces later on, so we do it once and forget about
    #  it.
    # First part of the grid, from the beginning of the template to the beginning of the dispersion
    #  array (excluding the start itself)

    current_dv = (
        -1 * np.diff(np.log(dispersion / (1215.67 * units.AA)) * const.c.to(kms))[0]
    )
    new_wave_lusso = np.concatenate(
        (
            utilities.gen_wave_grid(
                dispersion[0], wave_lusso[0], dv=current_dv, extend_right=False
            )[:0:-1],
            dispersion,
        )
    )
    # resample on dispersion
    # this is what we use for scaling, as it is cut at the dispersion
    new_flux_lusso = spectres(
        new_wave_lusso.value,
        wave_lusso.value,
        flux_lusso,
        fill=np.nanmedian(flux_lusso),
        verbose=False,
    )

    # hardcoded parameters
    left_cut = 980 * units.AA
    right_cut = 1020 * units.AA
    sigmoid_wave_cut = 1000.0 * units.AA
    sigmoid_wave_width = 2.5 * units.AA

    # defive overlap range and normalisation, indices are now the same for all the spectra
    inds = np.where((dispersion > left_cut) & (dispersion < right_cut))[0]
    inds_lusso = np.where((new_wave_lusso > left_cut) & (new_wave_lusso < right_cut))[0]

    # scaling factor, #1
    average_flux_lusso = trapezoid(
        new_flux_lusso[inds_lusso], new_wave_lusso[inds_lusso]
    )

    # new wavelength axis
    wave_left = new_wave_lusso[new_wave_lusso < left_cut]
    wave_mid = dispersion[inds]
    wave_right = dispersion[dispersion > right_cut]

    old_dispersion = np.concatenate((wave_left, wave_mid, wave_right))

    # make output
    new_dispersion = utilities.gen_wave_grid(
        old_dispersion[0],
        old_dispersion[-1],
        dv=current_dv * -1,
    )

    # grafted spectra
    out = np.empty((spectra.shape[0], new_dispersion.shape[0]))
    for n, sp in tqdm(enumerate(spectra)):
        # scalign factor, #2
        average_flux_sp = trapezoid(sp[inds], dispersion[inds])

        # scale Lusso flux
        scaled_flux_lusso = average_flux_sp / average_flux_lusso * new_flux_lusso

        # make some patchwork now
        # left side, only lusso
        flux_left = scaled_flux_lusso[new_wave_lusso < left_cut]

        # centre, sigmoid
        wave_sigmoid = expit(
            ((dispersion - sigmoid_wave_cut) / sigmoid_wave_width).value
        )

        flux_mid = (
            scaled_flux_lusso[inds_lusso] * (1.0 - wave_sigmoid[inds])
            + sp[inds] * wave_sigmoid[inds]
        )

        # right, the spectrum
        flux_right = sp[dispersion > right_cut]

        # concatenate everything
        # TODO: Fix this properly
        out[n] = np.concatenate((flux_left, flux_mid, flux_right, [0]))

    return spectres(
        new_dispersion.value,
        old_dispersion.value,
        out,
        fill=0,
        verbose=False,
    ), new_dispersion


## ============================================================================= ##


def sample_from_VAE(
    grid,
    model_params,
    dataset=None,
    dispersion_units=units.AA,
    graft_lusso_template=True,
    torch_device=utilities.set_device(),
    gmm_n_components=10,
):
    # Set up the paths
    training_set_dir = Path(local_path / "SDSS_DR16Q")

    training_set_fname = (
        local_path
        / model_params["input_dataset_parent_dir"]
        / model_params["input_dataset_name"]
    )

    if dataset is None:
        logger.info("Loading dataset.")
        dataset = SpecDataset(
            training_set_fname,
            subsample=1,
            replace_nan=True,
            replace_val=0,
            device=torch_device,
        )

    dispersion = dataset.dispersion * dispersion_units

    logger.info("Loading model.")
    batch_size = 128  # does not really matter
    model, save_dir, model_fname, train_model, timestamp = load.load_model(
        model_type=InfoSpecVAE,
        root_dir=training_set_dir,
        dispersion=dataset.dispersion,
        parameters=model_params,
        device=utilities.set_device(),
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # and set the model to evaluation mode - needs to stay like this for the
    #  entire time anyway
    model.eval()

    # generate GMM to sample from posterior

    # Sample sources from the VAE
    with torch.no_grad():
        logger.info("Sampling from model.")
        gmm, _ = utilities.create_latent_space_gmm(
            model,
            data_loader,
            n_components=gmm_n_components,
        )

        # some weird results, not sure
        #  for now work with SDSS spectrum for quality control, I know the mag output
        # This does not work, changed based on what we discussed on 2024/04/19
        spectra = (
            model.sample(grid.grid_data.shape[0], gmm=gmm)[0].cpu().detach().numpy()
        )

        # graft spectrum in the blue part
        if graft_lusso_template:
            logger.info("Grafting Lusso+15 template.")
            spectra, dispersion = graft_lusso_15(spectra, dispersion)

        return spectra, dispersion


## ============================================================================= ##


def generate_grid(params: dict = default_grid_params) -> SpectrumGrid:
    M1450_range = params["M1450_range"]
    redshift_range = params["redshift_range"]

    n_M1450_bins = params["n_M1450_bins"]
    n_redshift_bins = params["n_redshift_bins"]
    n_per_bin = params["n_per_bin"]

    grid = SpectrumGrid(
        M1450_range, redshift_range, n_M1450_bins, n_redshift_bins, n_per_bin
    )

    # sample the grid
    if params["sample"] == "uniform":
        logger.info("Sampling grid uniformly.")
        grid.uniform_sampling()
    elif params["sample"] == "hist2d":
        logger.info("Sampling grid using 2D histogram.")
        grid.hist2d_sampling(**params["hist2d_sampling_params"])
    elif params["sample"] == "lf":
        logger.info("Sampling grid using luminosity function.")
        grid.lf_sampling(**params["lf_sampling_params"])
    else:
        logger.error("Invalid sampling method. Valid options are {'lf', 'uniform'}")
        raise NotImplementedError
    return grid


## ============================================================================= ##


def redden_sampled_spectra(
    dispersion,
    spectra,
    reddening_model,
    r_v,
    b_bounds,
    rng=default_rng,
    dust_map_path=local_path / "dust_maps",
):
    # dispersion is a common array, equivalent to the Euclid filters
    n_spec = spectra.shape[0]

    b_bounds = b_bounds.to(units.deg).value

    # assign random coordinates to each object, not too close to the galactic plane
    l = rng.uniform(0.0, 360.0, size=n_spec) * units.deg  # noqa: E741
    b = utilities.limited_uniform(-90, 90, b_bounds, rng, size=n_spec) * units.deg

    # make coordindates for dust de-reddening
    sky_coords = SkyCoord(
        l,
        b,
        frame="galactic",
    )

    # deredden these objects - this is the dust map we use
    deredden.setup_dust_maps(dust_map_path)
    csfd = CSFDQuery()
    # 0.86 comes from Schlafly+2010 (2011? depends on who you ask)
    #  and is not included in the CSFD map
    logger.info("Computing reddening values...")
    a_v = csfd(sky_coords).astype(float) * r_v * 0.86
    a_lambda = reddening_model(
        dispersion * np.ones(a_v.shape[0])[..., None],
        r_v=r_v,
        a_v=a_v,
    )
    logger.info("Reddening spectra...")
    ne.redden(a_lambda, spectra, inplace=True)

    return spectra


## ============================================================================= ##


def generate_photometry_from_VAE(
    cosmology,
    filters,
    grid_params=default_grid_params,
    igm="simqso",
    dataset=None,
    graft_lusso_template=True,
    torch_device=None,
    redden=True,
    reddening_model=ne.Go23,
    r_v=3.1,
):
    # Sample redshift and M1450 absolute magnitudes
    # For now this is a uniform redshift luminosity (M1450) distribution
    # Set up the quasar grid
    # TODO: normalise to J band as well (using the lambda_ref I guess!)
    # TODO: pass kwargs around
    grid = generate_grid(params=grid_params)

    # sample from VAE
    # dispersion is either propagate through the function, or read from the specdatafile
    spectra, dispersion = sample_from_VAE(
        grid,
        dataset=dataset,
        graft_lusso_template=graft_lusso_template,
        torch_device=torch_device,
    )

    # compute quantities that are needed to scale the spectra
    compute_scale_factor(dispersion, spectra, grid.grid_data, cosmology)

    # scale spectra to match reference
    scaled_spectra = scale_VAE_spectra(spectra, grid.grid_data["scale"])

    # resample spectra based on new grid
    resampled_spectra, new_dispersion = resample_on_wavelength_grid(
        filters,
        dispersion,
        grid.grid_data["redshift"].to_numpy(),
        scaled_spectra,
    )

    if redden:
        # TODO: Check the actual Euclid limit
        redden_sampled_spectra(
            new_dispersion,
            resampled_spectra,
            reddening_model,
            r_v,
            b_min=15 * units.deg,
        )

    # Compute IGM contribution
    if igm == "zero":
        resampled_spectra_applied_IGM = compute_apply_IGM_zero(
            new_dispersion, grid, resampled_spectra
        )
    elif igm == "simqso":
        resampled_spectra_applied_IGM = compute_apply_IGM_simqso(
            new_dispersion, grid, resampled_spectra
        )
    else:
        raise ValueError("Specify IGM mode. Valid modes: {zero, simqso}")

    # return new magnitudes
    return pd.concat(
        (
            grid.grid_data,
            filters.get_ab_magnitudes(
                resampled_spectra_applied_IGM, new_dispersion
            ).to_pandas(),
        ),
        axis=1,
    )


## ============================================================================= ##


def AB_to_flux(mag, flag=None):
    # outputs to f_nu, Jansky!
    mag_copy = mag[:]

    if np.any(mag == flag):
        mag_copy[mag == flag] = np.nan

    return 3631 * 10 ** (mag_copy / (-2.5)) * units.Jy


## ============================================================================= ##


def flux_to_AB(flux, flag=None):
    # requires flux in Jy!
    flux_copy = dc(flux)

    if np.any(flux <= 0) and not flag:
        raise RuntimeError(
            "Negative flux detected, allow it by setting `flag` to any value."
        )
    elif np.any(flux <= 0) and flag:
        flux_copy[flux_copy <= 0] = np.nan
        mags = -2.5 * np.log10(flux_copy / (3631 * units.Jansky))
        mags[np.isnan(mags)] = flag
    else:
        mags = -2.5 * np.log10(flux_copy / (3631 * units.Jansky))

    return mags


## ============================================================================= ##


@units.quantity_input
def sample_error(
    flux: units.Jansky, error_function: units.Jansky, name, rng=default_rng
) -> units.Jansky:
    if not (flux.unit.is_equivalent, error_function.unit):
        logger.warning(
            "Flux and error function have to be in the equivalent units! Aborting."
        )
        return None

    # convert the error function to the flux units
    # local variable, no need to convert back
    error_function = error_function.to(flux.unit)

    mu = np.interp(
        flux.value,
        error_function[name + "_flux_grid"].value,
        error_function[name + "_mu"].value,
        left=error_function[name + "_mu"].value[0],
        right=error_function[name + "_mu"].value[-1],
    )
    sigma = np.interp(
        flux.value,
        error_function[name + "_flux_grid"].value,
        error_function[name + "_sigma"].value,
        left=error_function[name + "_sigma"].value[0],
        right=error_function[name + "_sigma"].value[-1],
    )

    if np.any(sigma <= 0):
        logger.warning(
            f"{np.sum(sigma <= 0)}/{len(sigma)} sampled errors are negative, setting them to 0."
        )
        sigma[sigma <= 0] = -sigma[sigma <= 0]

    # sampled errors cannot and should not be negative!
    return np.abs(rng.normal(loc=mu, scale=sigma)) * flux.unit


## ============================================================================= ##


@units.quantity_input
def perturb_photometry(
    flux: units.Jansky, sampled_error: units.Jansky, rng=default_rng
) -> units.Jansky:
    # is this the best way to go about this?
    # in principle I could have negative values, those would just be failure
    # in the extraction, so as a first approximation this would work fine?
    # Note the units! Has to match the error function!
    # Probably only an issue for VIS, maybe Y?
    if not (flux.unit.is_equivalent, sampled_error.unit):
        print(
            "[Warning] Flux and error function have to be in the equivalent units! Aborting."
        )
        return None

    # convert the error function to the flux units
    # local variable, no need to convert back
    sampled_error = sampled_error.to(flux.unit)

    return rng.normal(loc=flux.value, scale=sampled_error.value) * flux.unit


## ============================================================================= ##


@units.quantity_input
def generate_perturbed_magnitudes(
    flux: units.Jansky,
    error_function: units.Jansky,
    name,
    rng=default_rng,
    flag=None,
):
    sampled_errors = sample_error(flux, error_function, name, rng=rng).to(units.Jansky)
    perturbed_flux = perturb_photometry(flux, sampled_errors, rng=rng).to(units.Jansky)
    return (
        flux_to_AB(perturbed_flux, flag=flag),
        get_AB_mag_error(perturbed_flux, sampled_errors),
        perturbed_flux,
        perturbed_flux / sampled_errors,
    )


## ============================================================================= ##


@units.quantity_input
def get_AB_mag_error(flux: units.Jansky, flux_err: units.Jansky) -> units.Jansky:
    return 1.08574 * flux_err / flux * flux.unit


## ============================================================================= ##
## ===========================  End of FG functions  =========================== ##
## ============================================================================= ##


# TODO: Make it so this generates some data with some default parameters!
# if __name__ == "__main__":
#     # Load Euclid filters
#     # TODO: Load the updated filers, not these. These are wrong...
#     euclid_filters = filters.load_filters("Euclid-*")

#     # generate photometry using standard cosmology (P18)
#     from astropy.cosmology import Planck18 as cosmology

#     generate_photometry_from_VAE(cosmology, filters)


## ============================================================================= ##
## ================================  Deprecated  =============================== ##
## ============================================================================= ##
