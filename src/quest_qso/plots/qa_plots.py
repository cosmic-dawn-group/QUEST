import json
import logging
import os
import random
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter1d
from torch.nn.functional import mse_loss

from quest_qso import LOCAL_PATH as local_path
from quest_qso import all_logging_disabled
from quest_qso.utils import utilities

logger = logging.getLogger(__name__)

default_qa_dir = local_path / "QA"

# TODO: check that the QA plots are using the correct input data
#  for the reconstruction!


## ========================================================================= ##


def plot_latent_space_dims(
    model,
    n_samples,
    z=None,
    dataloader=None,
    ls_range=None,
    percentiles=[1.0, 99.0],
    ncols=2,
    save=False,
    save_dir=default_qa_dir,
    save_name="latent_space_variance.png",
    dpi=200,
):
    """Visualize the impact of the latent space dimensions on the generated
     spectra.

    :param model:
    :param num_samples:
    :param ls_range:
    :param save:
    :param save_dir:
    :param save_name:
    :param dpi:
    :return:
    """
    save_dir = Path(save_dir)

    # I need either of these, with the dataloader being the preferred option
    assert (z is not None) or (dataloader is not None) or (ls_range is not None)
    if z is not None:
        logger.info("Using the provided `z` to compute the `ls_range`.")
        logger.info("Ignoring the provided dataloader and `ls_range`.")
        ls_range, dataloader = None, None
    elif dataloader is not None:
        logger.info("Using the provided dataloader to compute the `ls_range`.")
        logger.info("Ignoring the provided `ls_range`.")
        ls_range = None
    else:
        logger.info("Using user-provided `ls_range`.")

    # model to eval mode, in principle not needed, but to be safe
    model.eval()

    # Basic setup
    n = model.latent_dim
    nrows = int(np.ceil(n / ncols))

    _figsize = (10, 5 * (1 + model.latent_dim / 20))
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=_figsize,
        layout="constrained",
        sharex=True,
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
    )

    # if we have a dataloader, then we can compute things with percentiles
    #  which I think is a good idea
    if z is None and dataloader is not None:
        z = model.latent_space_posterior(dataloader)["z"]

    axes = axs.flatten()
    for idx in range(model.latent_dim):
        ax = axes[idx]
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

        if idx < model.latent_dim - 2:
            ax.tick_params(axis="x", which="both", bottom=False)

        # Sample from the latent space - this is now decently gaussian
        #  if the training went well, so if we don't pass a ls range
        #  we compute it on the fly as the 0.025 and 0.975 quantiles
        if ls_range is None:
            assert percentiles is not None
            logger.info(
                f"`ls_range` not provided, computing as {percentiles} "
                "percentiles of the relevant dimension."
            )
            ls_range = np.percentile(z, percentiles)

        # recompute z using the actual range now
        # not using med_z should be fine as dimensions are now approximately
        #  gaussian
        samples, _ = model.sample_latent_space_dimensions(n_samples, idx, ls_range)

        # Generate color scale, should be between 0 and 1!
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples))

        for jdx in range(n_samples):
            ax.plot(
                model.dispersion,
                samples[jdx, :],
                color=colors[jdx],
                lw=1.0,
            )

        # only do this once, everything is the same size
        if idx == 0:
            text_pos_x, text_pos_y = 0.9, 0.9

        text = ax.text(
            text_pos_x,
            text_pos_y,
            f"Dimension {idx}",
            transform=ax.transAxes,
        )

        # probably not the best solution, oh well
        bbox_text = text.get_window_extent(fig.canvas.get_renderer()).bounds
        bbox_axis = ax.get_window_extent(fig.canvas.get_renderer()).bounds
        text_out_x, text_out_y = True, True
        # ^ assumes the text is outside and check at least once

        while idx == 0 and (text_out_x or text_out_y):
            if bbox_text[0] + bbox_text[2] > bbox_axis[0] + bbox_axis[2]:
                if text in ax.get_children():
                    text.remove()
                text_pos_x *= 0.99
            else:
                text_out_x = False

            if bbox_text[1] + bbox_text[3] > bbox_axis[1] + bbox_axis[3]:
                if text in ax.get_children():
                    text.remove()
                text_pos_y *= 0.99
            else:
                text_out_y = False

            # retry drawing the text
            text = ax.text(
                text_pos_x,
                text_pos_y,
                f"Dimension {idx}",
                transform=ax.transAxes,
            )

            # get the new bounding boxes
            bbox_text = text.get_window_extent(fig.canvas.get_renderer()).bounds
            bbox_axis = ax.get_window_extent(fig.canvas.get_renderer()).bounds

    fig.supxlabel(r"Wavelength [$\AA$]")
    fig.supylabel(r"Flux [Arbitrary Units]")

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


## ========================================================================= ##


def plot_latent_space_corner(
    model,
    dataloader,
    random_mask_flag=False,  # TODO: still don't know if I want to use True or False here
    coverage_mask_flag=False,
    save=False,
    save_dir=default_qa_dir,
    save_name="latent_space_variance.png",
    dpi=200,
    figsize=(12, 12),
    **kwargs,
):
    """Visualize the impact of the latent space dimensions on the generated
     spectra.

    :param model:
    :param num_samples:
    :param ls_range:
    :param save:
    :param save_dir:
    :param save_name:
    :param dpi:
    :return:
    """
    save_dir = Path(save_dir)

    model.eval()
    z = model.latent_space_posterior(
        dataloader,
        random_mask_flag=random_mask_flag,
        coverage_mask_flag=coverage_mask_flag,
    )["z"]

    fig = plt.figure(figsize=figsize)
    labels = [f"Dim. {i + 1}" for i in range(z.shape[1])]
    _ = corner.corner(
        z,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        fig=fig,
        **kwargs,
    )

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


## ========================================================================= ##


def plot_loss_histories(
    train_loss_history: pd.DataFrame,
    valid_loss_history: pd.DataFrame,
    exclude_cols=[],
    save=False,
    save_dir=default_qa_dir,
    save_name="loss_histories.png",
    dpi=200,
):
    """Plot the train and validation loss history

    :param loss_history:
    :param save:
    :param save_dir:
    :param save_name:
    :param dpi:
    :return:
    """
    assert not all([col in exclude_cols for col in train_loss_history.columns]), (
        "`train_loss_history.columns` and `exclude_cols` are the same, provide at least one column to plot."
    )
    save_dir = Path(save_dir)

    # get rid of zeros if I stopped the training early
    inds = np.where(train_loss_history.total_loss != 0)[0][:-1]
    train_loss_history = train_loss_history.iloc[inds, :]
    valid_loss_history = valid_loss_history.iloc[inds, :]

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1]},
    )

    # top plot
    for col in train_loss_history.columns:
        if col in exclude_cols:
            logger.info(f"Excluding loss term `{col}` from plot.")
            continue

        axs[0].plot(train_loss_history[col], label=col)

    # middle plot
    axs[1].plot(train_loss_history["reconstruction_loss"], label="Rec. loss")
    axs[1].plot(valid_loss_history["val_loss"], label="Val. loss")

    # bottom plot
    axs[2].plot(
        train_loss_history["reconstruction_loss"] / valid_loss_history["val_loss"],
        label="Rec. loss / Val. loss",
    )
    axs[2].set_ylabel("Train / Valid")

    # fig lvl
    fig.suptitle("Loss histories")
    fig.supxlabel("Epoch")

    for ax in axs[:2]:
        ax.tick_params(bottom=False)
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.semilogy()

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


## ========================================================================= ##


def plot_catalog_hist(
    df,
    col,
    bins=100,
    save=False,
    save_dir=default_qa_dir,
    save_name="cat_histogram.pdf",
    dpi=200,
):
    """Plot the histogram of a column in the catalog

    :param df:
    :param col:
    :param bins:
    :param save:
    :param save_dir:
    :param save_name:
    :param dpi:
    :return:
    """

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 1, 1)

    ax.hist(df[col], bins=bins)

    ax.title.set_text("Histogram of {}".format(col))
    ax.set_xlabel(col)
    ax.set_ylabel("Counts")

    if save:
        plt.savefig(os.path.join(save_dir, save_name), dpi=dpi)
        plt.close()
    else:
        plt.show()


## ========================================================================= ##


def _plot_spectra(
    ax,
    dispersion,
    orig_input,
    masked_input,
    predicted,
    quantile_up=None,
    quantile_down=None,
):
    ax.step(
        dispersion,
        orig_input,
        c="k",
        label="Orig. input",
        lw=1.5,
    )

    ax.step(
        dispersion,
        masked_input,
        c="b",
        label="Masked input",
        ls="--",
        lw=0.5,
    )

    ax.step(
        dispersion,
        predicted,
        c="r",
        label="Reconstruction",
    )

    if quantile_up is not None and quantile_down is not None:
        ax.fill_between(
            dispersion,
            quantile_down,
            quantile_up,
            color="r",
            alpha=0.3,
            label="Reconstruction 16-84 percentile",
        )


## ========================================================================= ##


def _make_single_spec_canvas(fig, gs_outer):
    """Helper function. Populates a gridspec with a subgrid with the desired layout.

    :param fig: Parent figure on which to draw the gridspec on.
    :type fig: matplotlib.figure.Figure
    :param gs_outer: Outer gridspec.
    :type gs_outer: _type_
    :return: _description_
    :rtype: _type_
    """
    gs = GridSpecFromSubplotSpec(
        2,
        1,
        height_ratios=[4, 1],
        subplot_spec=gs_outer,
        hspace=0.0,
    )
    ax_step = fig.add_subplot(gs[0, 0])
    ax_resid = fig.add_subplot(gs[1, 0], sharex=ax_step)

    ax_step.tick_params(
        axis="both",
        which="both",
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False,
    )

    ax_resid.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
    )

    return ax_step, ax_resid


## ========================================================================= ##


def _plot_residuals(ax, dispersion, orig_input, predicted):
    origin_valid_inds = np.where(orig_input > 0)
    diff = (predicted - orig_input)[origin_valid_inds]
    ax.plot(
        dispersion[origin_valid_inds],
        diff,
        c="mediumorchid",
        label="Diff. input, rec.",
        lw=1.0,
    )

    ax.plot(
        dispersion[origin_valid_inds],
        gaussian_filter1d(diff, 11),
        c="k",
        label="Diff. input, rec. (smooth)",
    )

    ax.axhline(0.0, c="lightgrey", zorder=-1)
    ax.set_ylim(np.nanmin(diff) * 0.95, np.nanmax(diff) * 1.05)


## ========================================================================= ##


def plot_reconstructed_spectrum(
    dispersion,
    orig_input,
    masked_input,
    predicted,
    quantile_up=None,
    quantile_down=None,
    fig=None,
    gs=None,
    gs_idx=None,
    inset=None,
    save=False,
    show=None,
    save_dir="/tmp",
    save_name="reconstructed_spectrum.png",
):
    standalone = False  # do I need to create a figure or did we pass it?
    save_dir = Path(save_dir)

    # bit of a waste maybe, but to keep the function signature consistent
    if fig is None or gs is None:
        fig = plt.figure(figsize=(6, 6 / 1.61))  # , layout="constrained")
        gs = fig.add_gridspec(1, 1)
        gs_idx = [0, 0]
        standalone = True

    gs = gs[gs_idx[0], gs_idx[1]]
    ax_step, ax_resid = _make_single_spec_canvas(fig, gs)

    _plot_spectra(
        ax_step,
        dispersion,
        orig_input,
        masked_input,
        predicted,
        quantile_up=quantile_up,
        quantile_down=quantile_down,
    )

    # TODO: Do I want the masked input here?
    _plot_residuals(ax_resid, dispersion, orig_input, predicted)

    fig.supxlabel(r"Wavelength [$\AA$]")
    fig.supylabel(r"Flux density [A.U.]")

    ax_step.set_ylim(
        bottom=-0.1 * np.nanmedian(orig_input),
        top=np.nanpercentile(orig_input, 99) * 1.4,
    )

    if inset is not None:
        inds_inset = np.where((dispersion > inset[0]) & (dispersion < inset[1]))[0]

        axin = ax_step.inset_axes([0.5, 0.5, 0.4, 0.4], xlim=[*inset])
        _plot_spectra(
            axin,
            dispersion[inds_inset],
            orig_input[inds_inset],
            masked_input[inds_inset],
            predicted[inds_inset],
        )

        axin.set_xlabel(r"Wavelength [$\AA$]")
        axin.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

    if standalone:
        if not save and show is None:
            show = True

        ax_step.legend(ncols=2)

        if save:
            plt.savefig(save_dir / save_name, dpi=200)
            plt.close()
        if show:
            plt.show()

    return fig, ax_step, ax_resid


## ========================================================================= ##


def plot_reconstructed_spectra(
    dispersion,
    orig_input,
    masked_input,
    predicted,
    quantile_down=None,
    quantile_up=None,
    spec_idxs=None,
    figsize=None,
    num_samples=4,
    ncols=2,
    save=False,
    save_dir=default_qa_dir,
    save_name="reconstructed_spectra.png",
    extra_text=None,
    dpi=200,
):
    nrows = int(np.ceil(num_samples / ncols))
    figsize = figsize if figsize is not None else (12, 2 * nrows)

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(nrows, ncols)

    for idx in range(num_samples):
        # get the spectrum, or just go in order if no indeces are provided
        if spec_idxs is not None:
            sidx = spec_idxs[idx]
        else:
            sidx = idx

        _, ax_step, _ = plot_reconstructed_spectrum(
            dispersion,
            orig_input.cpu().detach()[sidx, :],
            masked_input.cpu().detach()[sidx, : len(dispersion)],
            predicted.cpu().detach()[sidx, :],
            quantile_up=quantile_up.cpu().detach()[sidx, :]
            if quantile_up is not None
            else None,
            quantile_down=quantile_down.cpu().detach()[sidx, :]
            if quantile_down is not None
            else None,
            fig=fig,
            gs=gs,
            gs_idx=[idx // ncols, idx % ncols],
        )

        if extra_text is not None:
            ax_step.text(
                0.05,
                0.9,
                extra_text[idx],
                transform=ax_step.transAxes,
            )

    # make legend on the last plot
    ax_step.legend(ncols=2)

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


## ========================================================================= ##


def reconstruct_and_plot_spectra(
    model,
    dataloader,
    num_samples=4,
    ncols=2,
    spec_idxs=None,
    flux_autofit=False,
    random_mask_flag=False,
    coverage_mask_flag=False,
    save=False,
    save_dir=default_qa_dir,
    save_name="reconstructed_spectra.png",
    save_input_data=False,
    save_input_data_dir=default_qa_dir,
    save_input_data_name="reconstructed_and_input_datadump.npz",
    dpi=200,
):
    model.eval()

    # Prepare data - this will always load the same spectra I think
    # also there should be a choice of whether I want to use the
    # smoothed spectra or not, otherwise might as well remove
    # everything
    # Get all the data from the dataloader, from which we'll choose randomly
    #  if we don't ask for specific indices
    x = torch.vstack(list(dataloader))
    if spec_idxs is None:
        logger.info("No specific indices provided, selecting random spectra.")
        spec_idxs = random.sample(range(x.shape[0]), num_samples)
        logger.info(f"Selected spectra: {spec_idxs}")
    else:
        logger.info(f"Using user-provided indeces: {spec_idxs}")

    # use prepare input to avoid code duplication
    (
        stacked_input,
        _,
        _,
        _,
        flux,
        ivar,
        coverage_mask,
        random_mask,
    ) = utilities.prepare_input(
        x,
        flux_autofit=flux_autofit,
        random_mask_flag=random_mask_flag,
        coverage_mask_flag=coverage_mask_flag,
        device=model.device,
    )

    predicted_ = model.reconstruct(
        stacked_input[spec_idxs, :],
        n_samples=100,
        compute_median=False,
    )

    median_, perc_16, perc_84 = predicted_.quantile(
        torch.tensor([0.5, 0.16, 0.84], device=model.device), dim=0
    ).detach()

    if save_input_data:
        np.savez(
            Path(save_input_data_dir) / save_input_data_name,
            **{
                "dispersion": model.dispersion,
                "masked_input": stacked_input.cpu().detach().numpy(),
                "predicted": predicted_.cpu().detach().numpy(),
            },
        )

    plot_reconstructed_spectra(
        model.dispersion,
        flux,
        stacked_input,
        median_,
        quantile_down=perc_16,
        quantile_up=perc_84,
        num_samples=num_samples,
        ncols=ncols,
        save=save,
        save_dir=save_dir,
        save_name=save_name,
        dpi=dpi,
    )


## ========================================================================= ##


def plot_clean_spectrum(
    disp,
    flux,
    ivar,
    flux_clean,
    mask,
    save=False,
    save_dir=default_qa_dir,
    note=None,
    save_name="smoothed_spectrum.pdf",
    dpi=200,
):
    """Plot the smoothed spectrum"""

    # Create the save directory if it does not exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # QA plot for smoothed spectrum
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.step(disp, mask, color="0.5", label=r"$Mask$", where="mid")
    ax.step(
        disp,
        flux,
        c=(0.25, 0.25, 0.25),
        label=r"$\rm{Observed\ spectrum}$",
        where="mid",
    )
    ax.step(
        disp,
        1 / np.sqrt(ivar),
        c="r",
        linewidth=1,
        label=r"$\rm{Flux\ error}$",
        where="mid",
    )
    ax.step(
        disp[mask],
        flux_clean[mask],
        color="c",
        label=r"$\rm{Cleaned\ spectrum}$",
        where="mid",
    )
    ax.set_ylabel(
        r"${\rm flux\ [erg\ s}$" + r"$^{-1} {\rm cm} ^{-2}\mathrm{\AA}$" + r"${\rm ]}$"
    )
    ax.set_xlabel(r"${\rm rest\ wavelength\ [}$" r"$\mathrm{\AA}$" r"${\rm ]}$")

    ax.set_title("Cleaned Spectrum {}".format(note))

    if len(disp[mask]) > 0:
        ax.set_xlim([disp[0], disp[-1]])
        ax.set_ylim(
            bottom=-0.1 * np.nanmedian(flux[mask]),
            top=np.nanpercentile(flux[mask], 99) * 1.4,
        )

    ax.legend()

    if save:
        plt.savefig(os.path.join(save_dir, save_name), dpi=dpi)
    else:
        plt.show()

    plt.close()


## ========================================================================= ##


# TODO: Review this function
def example_spectra_dataloader(
    n_examples,
    dispersion,
    dataloader,
    full_dataloader=False,
    flux_autofit=False,
    random_mask_flag=False,
    coverage_mask_flag=False,
    device=utilities.set_device(),
    save=False,
    save_dir=default_qa_dir,
    save_name="Example_input.png",
):
    if full_dataloader:
        x = torch.vstack([*dataloader])
    else:
        x = next(iter(dataloader))

    (
        stacked_input,
        _,
        _,
        _,
        flux,
        ivar,
        coverage_mask,
        random_mask,
    ) = utilities.prepare_input(
        x,
        flux_autofit=flux_autofit,
        random_mask_flag=random_mask_flag,
        coverage_mask_flag=coverage_mask_flag,
        device=device,
    )

    # go to numpy arrays
    masked_input = stacked_input.cpu().detach().numpy()
    orig_input = flux.cpu().detach().numpy()
    coverage_mask = coverage_mask.cpu().detach().numpy()

    # remove the converage mask from the masked input
    masked_input = masked_input[:, : len(dispersion)]

    # select N spectra at random
    inds = random.sample(list(range(masked_input.shape[0])), n_examples)

    rows, cols = utilities.alt_almost_square(n_examples)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4, cols * 2 / 1.61),
        layout="constrained",
        sharex=True,
    )

    for n, ax in enumerate(axs.flatten()):
        orig_input[~coverage_mask.astype(bool)] = np.nan
        masked_input[~coverage_mask.astype(bool)] = np.nan

        try:
            ax.step(dispersion, orig_input[inds[n], :], c="k", label="Original")
            ax.step(
                dispersion,
                masked_input[inds[n], :],
                c="b",
                label="Masked Input Spectrum",
                ls="--",
            )

            ax.set_xlabel(r"Wavelength [$\AA$]")
            ax.set_ylabel(r"Flux density [A.U.]")
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
            )

            ax.set_ylim(
                bottom=-0.1 * np.nanmedian(orig_input),
                top=np.nanpercentile(orig_input, 99) * 1.4,
            )
        except IndexError:
            ax.remove()

    if save:
        plt.savefig(save_dir / save_name, dpi=200)
    else:
        plt.show()

    plt.close()


## ========================================================================= ##


def plot_reconstructed_spectrum_bh_mass(
    dispersion,
    orig_input,
    predicted,
    ax=None,
    save=False,
    save_dir=default_qa_dir,
    save_name="reconstructed_spectrum.png",
    inset=None,
):
    ax_was_none = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8 / 1.61 / 1.5), layout="constrained")
        ax_was_none = True

    ax.step(
        dispersion,
        orig_input,
        c="k",
        label="Input spectrum",
    )

    ax.step(
        dispersion,
        predicted,
        c="r",
        label="Reconstructed spectrum",
    )

    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"Flux density [A.U.]")
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
    )

    ax.set_ylim(
        bottom=-0.1 * np.nanmedian(orig_input),
        top=np.nanpercentile(orig_input, 99) * 1.4,
    )

    if inset is not None:
        inds_inset = np.where((dispersion > inset[0]) & (dispersion < inset[1]))[0]
        dispersion_inset = dispersion[inds_inset]
        orig_input_inset = orig_input[inds_inset]
        predicted_inset = predicted[inds_inset]

        axin = ax.inset_axes([0.5, 0.6, 0.47, 0.35], xlim=[*inset])
        axin.step(
            dispersion_inset,
            orig_input_inset,
            c="k",
            label="Input spectrum",
        )
        axin.step(
            dispersion_inset,
            predicted_inset,
            c="r",
            label="Reconstructed Spectrum",
        )
        axin.set_xlabel(r"Wavelength [$\AA$]")
        axin.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

    if ax_was_none:
        leg = ax.legend(loc="upper left")
        leg.get_frame().set_alpha(1.0)

        if save:
            plt.savefig(save_dir / save_name, dpi=200)
        else:
            plt.show()

        plt.close()


## ========================================================================= ##


def plot_loss_over_wavelength(
    dispersion,
    loss,
    ylabel,
    fig=None,
    ax=None,
    save=False,
    save_dir=default_qa_dir,
    save_name="loss_over_wavelength.png",
    dpi=200,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            2,
            2,
            figsize=(10, 5),
            layout="constrained",
            sharex=True,
        )

    with all_logging_disabled():
        for i in [0, 1]:
            ax[i, 0].step(dispersion, np.nanmedian(loss, axis=0))
            ax[i, 0].fill_between(
                dispersion,
                *np.nanpercentile(loss, [16, 84], axis=0),
                alpha=0.3,
            )

            ax[i, 1].step(dispersion, np.nanmean(loss, axis=0))
            ax[i, 1].fill_between(
                dispersion,
                *np.nanpercentile(loss, [16, 84], axis=0),
                alpha=0.3,
            )

    for ax_ in ax[0, :]:
        ax_.semilogy()

    ax[0, 0].set_title(r"Median $\pm 1\sigma$")
    ax[0, 1].set_title(r"Mean $\pm 1\sigma$")

    fig.supxlabel(r"Wavelength [$\AA$]")
    fig.supylabel(ylabel)

    fig.align_labels()

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


## ========================================================================= ##


def plot_loss_histogram(
    masked_loss,
    masked_loss_name,  # either MSE or Chi2
    fig=None,
    ax=None,
    save=False,
    save_dir=default_qa_dir,
    save_name="loss_hist.png",
    dpi=200,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            2, 2, figsize=(10, 5), layout="constrained", sharex="col"
        )

    for n, fnc in enumerate([np.nanmedian, np.nanmean]):
        ax[0, n].hist(
            fnc(masked_loss, axis=0),
            bins=25,
            alpha=0.7,
            edgecolor="C0",
            facecolor="C0",
        )
        ax[1, n].hist(
            fnc(masked_loss, axis=0),
            bins=25,
            alpha=0.7,
            edgecolor="C0",
            facecolor="C0",
            log=True,
        )

    ax[0, 0].set_ylabel("Counts")
    ax[1, 0].set_ylabel("Log(counts)")
    ax[1, 0].set_xlabel("Median")
    ax[1, 1].set_xlabel("Mean")

    fig.suptitle("Histograms of [median|mean] per wavelength")
    fig.align_ylabels()

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


## ========================================================================= ##


def compute_write_one_to_one_statistics(
    loss_masked,
    loss_name,
    prefix,
    json_save_dir=default_qa_dir,
):
    percentiles = (84, 95, 99)

    # JSON file to be
    json_content = {
        "nanmean_" + prefix: float(np.nanmean(loss_masked)),
        "nanmedian_" + prefix: float(np.nanmedian(loss_masked)),
        "percentile_" + prefix: {
            percentiles[n]: float(i)
            for n, i in enumerate(np.nanpercentile(loss_masked, percentiles))
        },
    }

    # Only dump json at this stage
    with open(json_save_dir / f"{prefix}_stat_{loss_name}.json", "w") as f:
        json.dump(json_content, f, indent=4)


## ========================================================================= ##


def loss_statistics(
    model,
    dataloader,
    flux_autofit=False,
    random_mask_flag=False,
    coverage_mask_flag=False,
    save=False,
    save_dir=default_qa_dir,
    json_save_dir=default_qa_dir,
    dpi=200,
):
    x = torch.vstack(list(dataloader))

    (
        stacked_input,
        _,
        _,
        _,
        flux,
        ivar,
        coverage_mask,
        random_mask,
    ) = utilities.prepare_input(
        x,
        flux_autofit=flux_autofit,
        random_mask_flag=random_mask_flag,
        coverage_mask_flag=coverage_mask_flag,
        device=model.device,
    )

    predicted_ = model.reconstruct(
        stacked_input,
        n_samples=100,
        compute_median=False,
    )
    median_ = predicted_.quantile(0.5, dim=0).detach()

    # ======================================================================== #

    # compute losses, no reduction as I will make it myself
    #  only account for region that did exists in the original spectrum
    #  these are the things I care about
    # This returns the chi square already zero-ed out by the coverage mask!
    chi_sq_masked = (
        utilities.chisq_stat(
            median_,
            flux,
            ivar,
            coverage_mask,
        )
        .cpu()
        .detach()
        .numpy()
    )
    chi_sq_masked[chi_sq_masked == 0.0] = np.nan

    mse = mse_loss(median_, flux, reduction="none")
    mse_masked = (mse * coverage_mask).cpu().detach().numpy()
    mse_masked[mse_masked == 0.0] = np.nan

    # ======================================================================== #

    # Plot all the things for both the mse and the chi square
    for masked_loss, masked_loss_name in zip(
        [mse_masked, chi_sq_masked],
        ["MSE", "χ²"],
    ):
        logger.info(f"Plotting loss statistics for {masked_loss_name}.")
        plot_loss_over_wavelength(
            model.dispersion,
            masked_loss,
            masked_loss_name,
            save=save,
            save_dir=save_dir,
            save_name="one-to-one-rec_" + masked_loss_name,
            dpi=dpi,
        )

        compute_write_one_to_one_statistics(
            masked_loss,
            masked_loss_name,
            "one-to-one-rec",
            json_save_dir,
        )

        plot_loss_histogram(
            masked_loss,
            masked_loss_name,  # either MSE or Chi2
            save=save,
            save_dir=save_dir,
            save_name="one-to-one-rec_" + masked_loss_name + "_hist.png",
            dpi=dpi,
        )


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


def compute_write_sampled_aggregate_stats(
    dispersion,
    input_spectra,
    trained_model,
    gmm,
    ivar=None,
    normed=False,
    json_save_dir=default_qa_dir,
    wave_limits=[1550, 4000],
    percentiles=[16, 84],
):
    # compute stats for the input spectra and the sampled spectra
    input_spectra[input_spectra == 0] = np.nan
    input_ = {
        "centre_line": np.nanpercentile(
            input_spectra,
            50,
            axis=0,
            weights=np.log1p(ivar) if ivar is not None else None,
            method="inverted_cdf",
        ),
        "lower_line": np.nanpercentile(
            input_spectra,
            percentiles[0],
            axis=0,
            weights=np.log1p(ivar) if ivar is not None else None,
            method="inverted_cdf",
        ),
        "upper_line": np.nanpercentile(
            input_spectra,
            percentiles[1],
            axis=0,
            weights=np.log1p(ivar) if ivar is not None else None,
            method="inverted_cdf",
        ),
    }

    sampled_spectra = trained_model.sample(10000, gmm=gmm)[0].cpu().detach().numpy()
    sampled = {
        "centre_line": np.median(sampled_spectra, axis=0),
        "lower_line": np.percentile(sampled_spectra, percentiles[0], axis=0),
        "upper_line": np.percentile(sampled_spectra, percentiles[1], axis=0),
    }

    if normed:
        den = np.nanmedian(input_["centre_line"], axis=0)
        input_["centre_line"] /= den
        input_["lower_line"] /= den
        input_["upper_line"] /= den

        sampled["centre_line"] /= den
        sampled["lower_line"] /= den
        sampled["upper_line"] /= den

    # total number of pixels we are considering
    total_blue = np.sum((dispersion < wave_limits[0]))
    total_red = np.sum((dispersion > wave_limits[1]))

    # compute the number of pixels above and below a given percentile on either side of the
    #  wavelength range
    num_below_blue = np.sum(
        (dispersion < wave_limits[0]) & (sampled["lower_line"] < input_["lower_line"])
    )
    num_below_red = np.sum(
        (dispersion > wave_limits[1]) & (sampled["lower_line"] < input_["lower_line"])
    )

    num_above_blue = np.sum(
        (dispersion < wave_limits[0]) & (sampled["upper_line"] > input_["upper_line"])
    )
    num_above_red = np.sum(
        (dispersion > wave_limits[1]) & (sampled["upper_line"] > input_["upper_line"])
    )

    # compare the 16 to 84 percentiles for the sampled and input spectra using the MSE
    MSE_blue_16 = np.sum(
        (
            sampled["lower_line"][dispersion < wave_limits[0]]
            - input_["lower_line"][dispersion < wave_limits[0]]
        )
        ** 2
    )
    MSE_blue_84 = np.sum(
        (
            sampled["upper_line"][dispersion < wave_limits[0]]
            - input_["upper_line"][dispersion < wave_limits[0]]
        )
        ** 2
    )
    MSE_red_16 = np.sum(
        (
            sampled["lower_line"][dispersion > wave_limits[1]]
            - input_["lower_line"][dispersion > wave_limits[1]]
        )
        ** 2
    )
    MSE_red_84 = np.sum(
        (
            sampled["upper_line"][dispersion > wave_limits[1]]
            - input_["upper_line"][dispersion > wave_limits[1]]
        )
        ** 2
    )

    MSE_rb_16 = np.sum(
        (
            sampled["lower_line"][
                (dispersion < wave_limits[0]) | (dispersion > wave_limits[1])
            ]
            - input_["lower_line"][
                (dispersion < wave_limits[0]) | (dispersion > wave_limits[1])
            ]
        )
        ** 2
    )
    MSE_rb_84 = np.sum(
        (
            sampled["upper_line"][
                (dispersion < wave_limits[0]) | (dispersion > wave_limits[1])
            ]
            - input_["upper_line"][
                (dispersion < wave_limits[0]) | (dispersion > wave_limits[1])
            ]
        )
        ** 2
    )

    out = {
        "total_red": int(total_red),
        "total_blue": int(total_blue),
        "num_below_red": int(num_below_red),
        "num_below_blue": int(num_below_blue),
        "num_above_red": int(num_above_red),
        "num_above_blue": int(num_above_blue),
        "perc_below_red": float(num_below_red / total_red),
        "perc_below_blue": float(num_below_blue / total_blue),
        "perc_above_red": float(num_above_red / total_red),
        "perc_above_blue": float(num_above_blue / total_blue),
        "perc_below": float(
            (num_below_red + num_below_blue) / (total_red + total_blue)
        ),
        "perc_above": float(
            (num_above_red + num_above_blue) / (total_red + total_blue)
        ),
        "compare_median_MSE": float(
            np.sum((sampled["centre_line"] - input_["centre_line"]) ** 2)
        ),
        "compare_median_χ²": float(
            np.sum(
                (
                    (sampled["centre_line"] - input_["centre_line"])
                    / (np.abs(input_["upper_line"] - input_["lower_line"]) / 2)
                )
                ** 2
            )
        ),
        "MSE_red_16": float(MSE_red_16),
        "MSE_red_84": float(MSE_red_84),
        "MSE_blue_16": float(MSE_blue_16),
        "MSE_blue_84": float(MSE_blue_84),
        "MSE_rb_16": float(MSE_rb_16),
        "MSE_rb_84": float(MSE_rb_84),
    }

    json_fname = (
        "sampled_spectra_stats.json"
        if not normed
        else "sampled_spectra_normed_stats.json"
    )
    json_fname = json_save_dir / json_fname
    with open(json_fname, "w") as f:
        json.dump(out, f, indent=4)

    return out


## ========================================================================= ##


def _compare_sampled_input_top_panel(
    ax,
    dispersion,
    input_,
    sampled,
    percentiles=[16, 84],
):
    ax.plot(
        dispersion,
        input_["centre_line"],
        label="Median input spectra",
        color="black",
    )
    ax.plot(
        dispersion,
        input_["lower_line"],
        color="black",
        ls="--",
    )
    ax.plot(
        dispersion,
        input_["upper_line"],
        color="black",
        ls="--",
    )

    ax.plot(
        dispersion,
        sampled["centre_line"],
        label="Median sampled spectra",
        color="red",
    )
    ax.fill_between(
        dispersion,
        sampled["lower_line"],
        sampled["upper_line"],
        alpha=0.3,
        label=f"{percentiles[0]}th-{percentiles[1]}th percentile",
        color="red",
    )
    ax.legend()
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Flux density [Arbitrary]")
    ax.axhline(0, color="lightgrey", ls="-.", zorder=-10)


## ========================================================================= ##


def _compare_sampled_input_bottom_panel(
    ax,
    dispersion,
    input_,
    sampled,
    percentiles,
):
    ax.plot(
        dispersion,
        input_["centre_line"] / input_["centre_line"],
        label="Median input spectra",
        color="black",
    )
    ax.plot(
        dispersion,
        input_["lower_line"] / input_["centre_line"],
        color="black",
        ls="--",
    )
    ax.plot(
        dispersion,
        input_["upper_line"] / input_["centre_line"],
        color="black",
        ls="--",
    )

    ax.plot(
        dispersion,
        sampled["centre_line"] / input_["centre_line"],
        label="Median sampled spectra",
        color="red",
    )
    ax.fill_between(
        dispersion,
        sampled["lower_line"] / input_["centre_line"],
        sampled["upper_line"] / input_["centre_line"],
        alpha=0.3,
        label=f"{percentiles[0]}th-{percentiles[1]}th percentile",
        color="red",
    )
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Flux density [Arbitrary]")
    ax.set_ylim(
        -0.15, np.nanpercentile(sampled["upper_line"] / input_["centre_line"], 99) * 1.1
    )
    ax.axhline(0, color="lightgrey", ls="-.", zorder=-10)


## ========================================================================= ##


def compare_sampled_input(
    dispersion,
    input_spectra,
    trained_model,
    gmm,
    ivar=None,
    percentiles=[16, 84],
    show=True,
    save=False,
    save_dir=None,
    save_name=None,
    dpi=200,
):
    save_dir = Path(save_dir)

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 10 / 1.61),
        layout="constrained",
        sharex=True,
    )

    sampled_spectra = trained_model.sample(10000, gmm=gmm)[0].cpu().detach().numpy()
    sampled = {
        "centre_line": np.median(sampled_spectra, axis=0),
        "lower_line": np.percentile(sampled_spectra, percentiles[0], axis=0),
        "upper_line": np.percentile(sampled_spectra, percentiles[1], axis=0),
    }

    input_spectra[input_spectra == 0] = np.nan
    input_ = {
        "centre_line": np.nanpercentile(
            input_spectra,
            50,
            axis=0,
            weights=np.log1p(ivar)
            if ivar is not None
            else None,  # do I want the log1p of the ivar?
            method="inverted_cdf",
        ),
        "lower_line": np.nanpercentile(
            input_spectra,
            percentiles[0],
            axis=0,
            weights=np.log1p(ivar) if ivar is not None else None,
            method="inverted_cdf",
        ),
        "upper_line": np.nanpercentile(
            input_spectra,
            percentiles[1],
            axis=0,
            weights=np.log1p(ivar) if ivar is not None else None,
            method="inverted_cdf",
        ),
    }

    _compare_sampled_input_top_panel(
        ax[0], dispersion, input_, sampled, percentiles=percentiles
    )
    _compare_sampled_input_bottom_panel(
        ax[1], dispersion, input_, sampled, percentiles=percentiles
    )

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()

    if show:
        plt.show()
    else:
        plt.close()


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


def all_qa_plots(
    trained_model,
    parameters,
    train_loss_history,
    valid_loss_history,
    train_set,
    valid_set,
    save,
    fig_save_dir,
    dpi=200,
):
    trained_model.eval()

    # make save dirs
    fig_save_dir = Path(fig_save_dir)
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    json_save_dir = fig_save_dir.parent / "jsons"
    json_save_dir.mkdir(parents=True, exist_ok=True)

    # This should reset both dataset as they just keep a reference of the original
    #  SpecDataset - Probably a lot of of a hack...
    train_set.dataset.reset_scaling()

    # Plot reconstructed spectra
    # Only used to make the latent space distribution plots
    train_dataloder = torch.utils.data.DataLoader(
        train_set,
        batch_size=parameters.default_batch_size,
        shuffle=False,
    )

    # only for qa purposes, it mildly bothers me that I am using the same validation dataset...
    # TODO: maybe take DESI EDR spectra as comparison as well?
    qa_dataloader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=parameters.default_batch_size,
        shuffle=False,
    )

    # directly fit the GMM, no need to do this multiple times
    logger.info("Fitting GMM to latent space.")
    gmm, qa_z = utilities.create_latent_space_gmm(
        trained_model,
        qa_dataloader,
        n_components=10,
    )

    # Latent space dimenstions and how they influence the spectra
    logger.info("Plotting latent space dimensions.")
    plot_latent_space_dims(
        trained_model,
        100,
        dataloader=train_dataloder,
        save=save,
        save_dir=fig_save_dir,
        save_name="latent_space_dims.png",
        dpi=dpi,
    )

    # loss histories
    logger.info("Plotting loss histories.")
    plot_loss_histories(
        train_loss_history,
        valid_loss_history,
        save=save,
        save_dir=fig_save_dir,
        save_name="loss_histories.png",
        dpi=dpi,
    )

    # Latent space distributions
    logger.info("Plotting latent space dimensions distributions.")
    plot_latent_space_corner(
        trained_model,
        dataloader=train_dataloder,
        save=save,
        save_dir=fig_save_dir,
        save_name="latent_space_dims_corner.png",
        random_mask_flag=False,
        coverage_mask_flag=False,
    )

    # Plot the input spectra and the corresponding reconstructions
    logger.info("Plotting example spectra.")
    # index of the spectra to plot
    spec_idxs = [0, 1, 2, 3, 4, 5]

    reconstruct_and_plot_spectra(
        trained_model,
        qa_dataloader,
        spec_idxs=spec_idxs,
        num_samples=len(spec_idxs),
        save=save,
        save_dir=fig_save_dir,
        save_name="example_spec.png",
        random_mask_flag=False,
        coverage_mask_flag=False,
        dpi=dpi,
    )

    # print and save some statistics about how we did on this round
    # names are generated automatically based on MSE and X2 stats
    logger.info("Computing and plotting loss statistics.")
    loss_statistics(
        trained_model,
        qa_dataloader,
        save=save,
        save_dir=fig_save_dir,
        json_save_dir=json_save_dir,
        dpi=dpi,
    )

    # statistics related to sampling several spectra and comparing with the
    #  input spectra (median and percentile)
    logger.info("Comparing samples to input data.")
    logger.info("Plotting sampled spectra vs input spectra.")
    compare_sampled_input(
        trained_model.dispersion,
        # what a beast of a line
        valid_set.dataset.input.data[:, 0].clone().detach().cpu().numpy(),
        trained_model,
        gmm,
        ivar=None,  # valid_set.dataset.input.data[:, 4].clone().detach().cpu().numpy(),
        show=False,
        save=save,
        save_dir=fig_save_dir,
        save_name="sampled_vs_input.png",
        dpi=dpi,
    )

    logger.info("Computing aggregate statistics.")
    compute_write_sampled_aggregate_stats(
        trained_model.dispersion,
        valid_set.dataset.input.data[:, 0].cpu().detach().numpy(),
        trained_model,
        gmm,
        ivar=None,  # valid_set.dataset.input.data[:, 4].cpu().detach().numpy(),
        json_save_dir=json_save_dir,
    )

    compute_write_sampled_aggregate_stats(
        trained_model.dispersion,
        valid_set.dataset.input.data[:, 0].cpu().detach().numpy(),
        trained_model,
        gmm,
        ivar=None,  # valid_set.dataset.input.data[:, 4].cpu().detach().numpy(),
        normed=True,
        json_save_dir=json_save_dir,
    )
