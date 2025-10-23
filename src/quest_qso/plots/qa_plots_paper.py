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

from quest_qso import LOCAL_PATH as local_path
from quest_qso import all_logging_disabled
from quest_qso.utils import utilities

# Thanks https://stackoverflow.com/a/44312585
#  for custom legend handlers

logger = logging.getLogger(__name__)

default_qa_dir = local_path / "QA"

# TODO: check that the QA plots are using the correct input data
#  for the reconstruction!

## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


class Handler(object):
    def __init__(self, line_kwargs, rect_kwargs, no_line=False):
        self.no_line = no_line
        self.line_kwargs = line_kwargs
        self.rect_kwargs = rect_kwargs

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = plt.Rectangle(
            [x0, y0],
            width,
            height,
            **self.rect_kwargs,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)

        if not self.no_line:
            line = plt.Line2D(
                [x0, x0 + width],
                [y0 + height / 2, y0 + height / 2],
                **self.line_kwargs,
                transform=handlebox.get_transform(),
            )
            handlebox.add_artist(line)

        return patch


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


def plot_latent_space_dims(
    fig,
    axs,
    model,
    n_samples,
    n_dims_to_plot,
    z=None,
    dataloader=None,
    ls_range=None,
    percentiles=[1.0, 99.0],
    save=False,
    save_dir=default_qa_dir,
    save_name="latent_space_variance.png",
    dpi=200,
    cmap=None,
    mask_=None,
    ylim_func=np.max,
    add_ld_text=False,
    text_alignment="left",
    sort_by_max_variation=True,
    plot_baseline=False,
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
    if save:
        save_dir = Path(save_dir)

    if cmap is None:
        cmap = plt.cm.viridis

    if mask_ is None:
        mask_ = np.ones_like(model.dispersion.max(), dtype=bool)

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

    # if we have a dataloader, then we can compute things with percentiles
    #  which I think is a good idea
    if z is None and dataloader is not None:
        z = model.latent_space_posterior(dataloader)["z"]

    # if we want to add the baseline, decode the fully zero latent space
    if plot_baseline:
        baseline = (
            model.decode(
                torch.zeros(
                    1,
                    model.latent_dim,
                    dtype=torch.float32,
                    device=model.device,
                )
            )
            .detach()
            .cpu()
            .numpy()
        )[0]

    # we only want to plot the fist N dimensions, so we first compute the variations
    deltas, samples_dict = [], {}
    for ld_n in range(model.latent_dim):
        ls_range = np.percentile(z, percentiles)
        samples, _ = model.sample_latent_space_dimensions(n_samples, ld_n, ls_range)

        delta = np.median(np.abs(samples[0, :] - samples[-1, :]))
        deltas.append(delta)
        samples_dict[ld_n] = samples

    # Make this general so I can pass the indexes directly instead of
    #  computing them below
    try:
        iter(n_dims_to_plot)
        sorting_inds = n_dims_to_plot
    except TypeError:
        logger.warning(
            "`n_dims_to_plot` is not iterable, assuming it's an integer and computing the indexes. "
            "If you want to pass specific indeces, provide an iterable (e.g., a list or a numpy array). "
            f"Values in use: {n_dims_to_plot}, {sort_by_max_variation}"
        )

        # get the first N in order or variations
        if sort_by_max_variation:
            sorting_inds = np.argsort(deltas)[::-1][:n_dims_to_plot]
        else:
            sorting_inds = np.arange(n_dims_to_plot)

    colors = cmap(np.linspace(0, 1, n_samples))
    for n, idx in enumerate(sorting_inds):
        ax = axs[n]

        for jdx in range(n_samples):
            ax.plot(
                model.dispersion,
                samples_dict[idx][jdx, :] / samples_dict[idx].max(),
                color=colors[jdx],
                lw=0.5,
            )

        if plot_baseline:
            ax.plot(
                model.dispersion,
                baseline / samples_dict[idx].max(),
                color="k",
                lw=0.5,
                ls=":",
                label="Baseline",
            )

        ax.set_ylim(
            (samples_dict[idx][:, mask_] / samples_dict[idx].max()).min() * 0.5,
            ylim_func(samples_dict[idx][:, mask_] / samples_dict[idx].max()) * 1.1,
        )

        if add_ld_text:
            ax.text(
                0.99 if text_alignment == "right" else 0.01,
                0.80,
                f"Latent dim. {idx + 1}",
                transform=ax.transAxes,
                fontsize=8,
                ha=text_alignment,
            )

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
        plt.close()


## ========================================================================= ##


def plot_latent_space_corner(
    fig,
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

    labels = [f"Dim. {i + 1}" for i in range(z.shape[1])]
    _ = corner.corner(
        z,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        fig=fig,
        hist_kwargs=dict(linewidth=0.75),
        **kwargs,
    )

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)
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


def _compare_sampled_input_top_panel(
    ax,
    dispersion,
    input_,
    sampled,
    median_color,
    shade_color,
    percentiles=[16, 84],
    inds=None,
    delta=0,
    show_x_labels=False,
    hide_xticks=True,
    make_legend=True,
    hide_input=False,
    kwargs_sampled_median={},
    kwargs_shaded_area={},
):
    if inds is None:
        inds = np.ones_like(dispersion, dtype=bool)

    if not hide_input:
        ax.plot(
            dispersion[inds],
            input_["centre_line"][inds] + delta,
            label="Input",
            color="black",
            lw=0.75,
        )
        ax.plot(
            dispersion[inds],
            input_["lower_line"][inds] + delta,
            color="black",
            ls="--",
            lw=0.75,
        )
        ax.plot(
            dispersion[inds],
            input_["upper_line"][inds] + delta,
            color="black",
            ls="--",
            lw=0.75,
            label=f"{percentiles[0]}$"
            r"^{\rm th}$-"
            f"{percentiles[1]}"
            r"$^{\rm th}$ perc.",
        )

    ax.plot(
        dispersion[inds],
        sampled["centre_line"][inds] + delta,
        label="Sampled",
        color=median_color,
        **kwargs_sampled_median,
    )

    ax.fill_between(
        dispersion[inds],
        sampled["lower_line"][inds] + delta,
        sampled["upper_line"][inds] + delta,
        alpha=0.3,
        label=f"{percentiles[0]}$"
        r"^{\rm th}$-"
        f"{percentiles[1]}"
        r"$^{\rm th}$ perc.",
        color=shade_color,
        **kwargs_shaded_area,
    )

    if make_legend:
        ax.legend(ncol=2, fontsize=9, loc="upper right", frameon=False)

    ax.axhline(0, color="lightgrey", ls="-.", zorder=-10, lw=0.75)
    if show_x_labels:
        ax.set_xlabel(r"Wavelength [$\AA$]")

    if hide_xticks:
        ax.tick_params(bottom=False, labelbottom=False)

    ax.set_ylabel("Flux den. [A. U.]")


## ========================================================================= ##


def _compare_sampled_input_bottom_panel(
    ax,
    dispersion,
    input_,
    sampled,
    percentiles,
    inds=None,
):
    if inds is None:
        inds = np.ones_like(dispersion, dtype=bool)

    # this is centre / centre (both input)
    #  which is always one
    ax.axhline(
        1,
        color="lightgrey",
        zorder=-10,
        lw=0.75,
        ls="-.",
    )
    ax.plot(
        dispersion[inds],
        input_["lower_line"][inds] / input_["centre_line"][inds],
        color="black",
        ls="--",
        lw=0.75,
    )
    ax.plot(
        dispersion[inds],
        input_["upper_line"][inds] / input_["centre_line"][inds],
        color="black",
        ls="--",
        lw=0.75,
    )

    ax.plot(
        dispersion[inds],
        sampled["centre_line"][inds] / input_["centre_line"][inds],
        label="Median sampled spectra",
        color="red",
        lw=0.75,
    )
    ax.fill_between(
        dispersion[inds],
        sampled["lower_line"][inds] / input_["centre_line"][inds],
        sampled["upper_line"][inds] / input_["centre_line"][inds],
        alpha=0.3,
        label=f"{percentiles[0]}th-{percentiles[1]}th percentile",
        color="red",
        lw=0.5,
    )
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Ratios")
    # -.015, np.nanpercentile(sampled["upper_line"] / input_["centre_line"], 99) * 1.1
    # ax.axhline(0, color="lightgrey", ls="-.", zorder=-10)


## ========================================================================= ##


def compare_sampled_input_paper(
    dispersion,
    input_spectra,
    trained_model,
    gmm,
    ivar=None,
    percentiles=[16, 84],
    show=True,
    save=False,
    close=True,
    save_dir=None,
    save_name=None,
    dpi=200,
    fig=None,
    ax=None,
    data_lims=None,
    delta=0,
    hide_xticks=False,
    show_x_labels=True,
    make_legend=True,
    main_color="red",
    median_color=None,
    shade_color=None,
    hide_input=False,
    kwargs_shaded_area={},
    kwargs_sampled_median={},
):
    assert main_color is not None or (
        median_color is not None and shade_color is not None
    )

    if median_color is None:
        median_color = main_color
    if shade_color is None:
        shade_color = main_color

    if save:
        save_dir = Path(save_dir)

    if fig is None and ax is None:
        fig, ax = plt.subplots(
            2,
            1,
            figsize=(10, 10 / 1.61),
            layout="constrained",
            sharex=True,
        )
    else:
        if ax is not None:
            logger.info("Passed an existing `ax` instance, ignoring `dpi` argument.")
        else:
            logger.error(
                "Received figure but no axis, cannot continue. "
                "If providing a figure, please pass also a tuple of axis using the `ax` argument."
            )
            raise RuntimeError("Need two axis to work on!")

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

    if data_lims is not None:
        inds = np.where((dispersion > data_lims[0]) & (dispersion < data_lims[1]))[0]
    else:
        inds = np.ones_like(dispersion, dtype=bool)

    _compare_sampled_input_top_panel(
        ax,
        dispersion,
        input_,
        sampled,
        median_color,
        shade_color,
        percentiles=percentiles,
        inds=inds,
        delta=delta,
        hide_xticks=hide_xticks,
        show_x_labels=show_x_labels,
        make_legend=make_legend,
        hide_input=hide_input,
        kwargs_shaded_area=kwargs_shaded_area,
        kwargs_sampled_median=kwargs_sampled_median,
    )

    if save:
        plt.savefig(save_dir / save_name, dpi=dpi)

    if show:
        plt.show()
    else:
        if close:
            plt.close()
