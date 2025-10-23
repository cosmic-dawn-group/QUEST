import logging

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import ndimage, stats
from scipy.stats import gaussian_kde
from tqdm import tqdm

# =========================================================================== #

# xlims mimicking Daming's ones
xlim_dict = {0: (-0.02, 0.06), 1: (0.1, 1.3), 2: (0.4, 2.8)}

# kde grid
kde_grid_dict = {
    f"{k_i}{k_j}": (v_i, v_j)
    for k_i, v_i in xlim_dict.items()
    for k_j, v_j in xlim_dict.items()
}

# =========================================================================== #


logger = logging.getLogger(__name__)

# =========================================================================== #


def binned_nanmedian(data):
    inds = np.where(np.isfinite(data))[0]
    if len(inds) != 0:
        1
        # TODO: refix this now that I have the correct logging system
        # root_logger.warn_once(logger, "Input contains np.nan")
    return np.nanmedian(data)


# =========================================================================== #


def calculate_redshift_median_colors(
    df,
    passband_a,
    passband_b,
    redsh_bins=20,
    statistic=binned_nanmedian,
    redshift_col="redshift",
):
    """Calculate the median color by redshift bin.

    :param df:
    :param passband_a:
    :param passband_b:
    :param redsh_bins:
    :param statistic:
    :return:
    """

    redsh = df[redshift_col]

    if isinstance(redsh_bins, int):
        nbins = redsh_bins
    elif isinstance(redsh_bins, list) or isinstance(redsh_bins, np.ndarray):
        nbins = len(redsh_bins) - 1
    else:
        raise ValueError(
            "Type of redsh_bins not understood. Allowed types are int and list."
        )

    redsh_binmiddle = np.zeros(nbins)

    color = df[passband_a] - df[passband_b]

    med_color, bin_edges, _ = stats.binned_statistic(
        redsh, color, bins=redsh_bins, statistic=statistic
    )

    if isinstance(redsh_bins, int):
        redsh_binmiddle = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    elif isinstance(redsh_bins, list) or isinstance(redsh_bins, np.ndarray):
        redsh_binmiddle = (np.array(redsh_bins[:-1]) + np.array(redsh_bins[1:])) / 2.0

    return med_color, redsh_binmiddle


# =========================================================================== #


def get_redshift_tracks(
    df, colnames, redsh_bins=20, statistic=binned_nanmedian, redshift_col="redshift"
):
    med_color_a, redsh_binmiddle = calculate_redshift_median_colors(
        df,
        colnames[0],
        colnames[1],
        redsh_bins=redsh_bins,
        statistic=statistic,
        redshift_col=redshift_col,
    )
    med_color_b, redsh_binmiddle = calculate_redshift_median_colors(
        df,
        colnames[2],
        colnames[3],
        redsh_bins=redsh_bins,
        statistic=statistic,
        redshift_col=redshift_col,
    )

    return med_color_a, med_color_b, redsh_binmiddle


# =========================================================================== #


def z_track(
    color_a,
    color_b,
    tl,
    redsh_binmiddle,
    ax=None,
    label="Median color",
    mult=0.15,
    figsize=(6, 6 / 1.61),
    track_color="k",
    add_annotations=True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # connecting line
    ax.plot(
        color_a,
        color_b,
        track_color,
        label=label,
        lw=2.0,
        path_effects=[pe.withStroke(linewidth=3, foreground="lightgrey")],
    )

    # points with labels
    if tl is not None and isinstance(tl, int):
        ax.plot(
            color_a[::tl],
            color_b[::tl],
            ms=5,
            linestyle="none",
            marker="o",
            markeredgecolor="lightgrey",
            markerfacecolor=track_color,
        )
        redsh_labels = [r"${:.2f}$".format(x) for x in redsh_binmiddle[::tl]]

        for _n, jdx in enumerate(range(len(redsh_labels))):
            if not add_annotations:
                continue
            ax.annotate(
                redsh_labels[jdx],
                (color_a[tl * jdx], color_b[tl * jdx] + (-1) ** _n * mult),
                fontsize=11,
                textcoords="offset points",
                ha="center",
                va="center",
                color="k",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="lightgrey")],
            )


# ======================================================================================== #


# simply plots the color of the selected objects in a figure
def color(df, m_names, label_dict, ax=None, figsize=(6, 6 / 1.61), **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlabel(
        f"{label_dict[m_names[0]]} - {label_dict[m_names[1]]}",
    )

    ax.set_ylabel(
        f"{label_dict[m_names[2]]} - {label_dict[m_names[3]]}",
    )

    sc = ax.scatter(
        df[m_names[0]] - df[m_names[1]], df[m_names[2]] - df[m_names[3]], **kwargs
    )

    return sc


# ======================================================================================== #


# colors and redshift track
def color_z_track(
    data,
    columns,
    labels_dict,
    fig=None,
    ax=None,
    hst=None,
    cs_labels=True,
    figsize=(6, 6 / 1.61),
    tl=1,
    mult=0.125,
    stddev=3,
    levels=None,
    redshift_bins=20,
    flag=None,
    redshift_col="redshift",
    track_color="k",
    add_cbar=True,
    add_annotations=True,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if flag:
        data = data.replace(99.0, np.nan)

    scatter_plot = color(data, columns, labels_dict, ax=ax, **kwargs)
    tracks = get_redshift_tracks(
        data, columns, redsh_bins=redshift_bins, redshift_col=redshift_col
    )

    z_track(
        tracks[0],
        tracks[1],
        tl,
        tracks[2],
        ax=ax,
        label="Redshift track",
        mult=mult,
        track_color=track_color,
        add_annotations=add_annotations,
    )

    if add_cbar:
        cbar = fig.colorbar(scatter_plot)
        cbar.set_label("Redshift")
        cbar.solids.set(alpha=1)

    if hst is not None:
        min = np.nanmin(hst[0][hst[0] > 0])
        cs = ax.contour(
            hst[1][:-1],
            hst[2][:-1],
            ndimage.gaussian_filter(hst[0].T, stddev),
            levels=np.logspace(np.log10(min), np.log10(np.nanmax(hst[0])), 10)
            if levels is None
            else levels,
            colors="lightgrey",
            zorder=-3,
        )
        ax.axvline(
            ax.get_xlim()[0] - 1,
            0,
            -100,
            label="Contaminants",
            color="lightgrey",
            lw=3,
        )
        if cs_labels:
            ax.clabel(cs)

    leg = ax.legend()
    leg.legend_handles[0].set_alpha(1)
    leg.legend_handles[0].set_edgecolor("k")
    leg.legend_handles[0]._sizes = [10]


# ======================================================================================== #


def color_histogram(color, z, thresholds, edgecolors, **kwargs):
    fig, ax = plt.subplots()

    # shift the range a tiny bit so that the histogram are understandable
    bins = kwargs.get("bins", 10)
    range_ = kwargs.pop("range", None)
    range_i = []

    if range_ is not None:
        delta = (range_[-1] - range_[0]) / bins
        for i in range(len(thresholds)):
            range_i.append(
                [
                    range_[0] + delta / len(thresholds) * i,
                    range_[-1] + delta / len(thresholds) * i,
                ]
            )
    else:
        for i in range(len(thresholds)):
            range_i.append(range_)

    for n, th in enumerate(thresholds):
        inds = np.where(np.isnan(color))
        if len(inds[0]) > 0:
            logger.warning("Found NaN while plotting, check your data!")
        ax.hist(
            color[z > th],
            facecolor="none",
            edgecolor=edgecolors[n],
            range=range_i[n],
            histtype="stepfilled",
            **kwargs,
        )

    return fig, ax


# ======================================================================================== #


def make_corner_canvas(
    n=3, tight_layout=False, subplots_adjust=True, figsize=(8, 8), **kwargs
):
    if subplots_adjust and tight_layout:
        logger.warning("Use either `tight_layout` or `subplots_adjust`!")
        return

    # default spacing
    if len(kwargs.keys()) == 0 and subplots_adjust:
        kwargs = {"wspace": 0.075, "hspace": 0.075}

    # kwargs are passed to tight_layout or subplots_adjust, depending on what you use
    # warning! this assumes a square grid!
    fig, ax = plt.subplots(n, n, figsize=figsize, sharex="col")

    # remove axis off upper diagonal, not useful
    for i in range(n):
        for j in range(n):
            if j > i:
                fig.delaxes(ax[i, j])
            else:
                # rotate labels
                ax[i, j].tick_params(axis="both", labelrotation=45)

    if tight_layout:
        fig.tight_layout(**kwargs)

    if subplots_adjust:
        fig.subplots_adjust(**kwargs)

    return fig, ax


# ======================================================================================== #


# Honestly no idea why I wrote this.
#  I really should have added some comments somewhere
#  It was made for the presentation at EAS
# It is most likely something related to figuring out which points
#  are not part of the 'side' of the last contour
def split_array(arr):
    i = 0
    idxs = []
    _out = {}

    for n in range(len(arr)):
        prev = False if n == 0 else arr[n - 1]

        current_val = arr[n]
        if current_val & prev:
            idxs.append(n)
        else:
            _out[i] = idxs
            idxs = []
            i += 1

    # loop over the keys and remove the useless arrays
    i = 0
    out = {}
    for key in _out.keys():
        if len(_out[key]) > 1:
            out[i] = _out[key]
            i += 1
    return out


# =========================================================================== #


def kde_plot(
    ax,
    x,
    y,
    xgrid,
    ygrid,
    tol=None,
    cnt=True,
    cntf=True,
    kwargs_contour={},
    kwargs_contourf={},
    kwargs_scatter={},
    set_i=1,
    print_set_i_keys=True,
    denorm=False,
):
    # see also /Users/francesco/repo/misc/plot_spectra.py
    #   for the original implementation
    X, Y, Z = make_scipy_kde(x=x, y=y, xgrid=xgrid, ygrid=ygrid, tol=tol)

    if denorm:
        Z *= len(x)

    # initialise variable for the rest of the function
    cc, ccf, data = None, None, np.array([x, y])

    if cnt:
        cc = ax.contour(X, Y, Z, **kwargs_contour)
    if cntf:
        ccf = ax.contourf(X, Y, Z, **kwargs_contourf)

    inside = np.full_like(data[0], False, dtype=bool)
    if ccf is not None:
        for p in ccf.get_paths():
            inside |= p.contains_points(data.T)
    elif ccf is None and cc is not None:
        for p in cc.get_paths():
            inside |= p.contains_points(data.T)
    elif ccf is None and cc is None:
        raise ValueError("Need either contour of contourf")

    # add last contour an points outside
    if cnt and ccf is not None:
        # this some hacky way of adding the last contour
        # one needs to manually adjust the indexes most likely
        vertices_outer = ccf.get_paths()[0]._vertices
        vals, counts = np.unique(vertices_outer[:, 0], return_counts=True)
        most_common_val = vals[np.argmax(counts)]

        inds = split_array(vertices_outer[:, 0] != most_common_val)
        if print_set_i_keys:
            print(f"Current keys: {inds.keys()}")
        vertices = vertices_outer[inds[set_i], :]

        # this produces the final contour
        ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            c=kwargs_contour.get("colors", None),
            lw=kwargs_contour.get("linewidths", 3),
        )

    # Finally, add the last points as a scatter
    ax.scatter(data[0, :][~inside], data[1, :][~inside], **kwargs_scatter)
    return cc, ccf


# ======================================================================================== #


def make_hist_kde(x, y, smooth=1, **kwargs):
    # note: this is not a true kde, but more a 2d histogram with gaussian smoothing
    # make a grid out of the input, j indicates how many points to include
    counts, xx, yy = np.histogram2d(x, y, density=True, **kwargs)
    img = ndimage.gaussian_filter(counts, smooth)
    img = ma.masked_array(img, mask=[counts == 0])

    return xx[:-1], yy[:-1], img.T


# ======================================================================================== #


def make_scipy_kde(
    x,
    y,
    xgrid=[0, 1],
    ygrid=[0, 1],
    Nx=100j,
    Ny=100j,
    tol=None,
    **kwargs,
):
    # the input to the kde should be something
    #  with shape (2, N)
    # x and y should be normal 1D arrays
    data = np.array([x, y])
    kde = gaussian_kde(data, **kwargs)
    X, Y = np.mgrid[xgrid[0] : xgrid[1] : Nx, ygrid[0] : ygrid[1] : Ny]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)
    if tol is not None:
        Z[Z < tol] = np.nan

    return X, Y, Z


# ======================================================================================== #


def make_kde(
    x,
    y,
    method="scipy",
    **kwargs,
):
    if method not in ["scipy", "kde", "hist"]:
        raise ValueError("Valid options are `scipy`, `kde`, `hist`")
    if method == "hist":
        return make_hist_kde(x, y, **kwargs)
    elif method == "scipy" or method == "kde":
        return make_scipy_kde(x, y, **kwargs)


# ======================================================================================== #


def precompute_kde_corner_corner(
    fluxes,
    flux_names,
    flux_ref_name,
    kde_grid_dict=kde_grid_dict,
    kde_kwargs={},
):
    out = {}

    fluxes = fluxes[[*flux_names, flux_ref_name]]

    # columns pair to plot later on
    inds = np.arange(len(flux_names))
    pairs = [(i, j) for i in inds for j in inds]

    logger.info("Computing kde for all flux ratios.")
    for i, j in tqdm(pairs):
        if i != j:
            xy_grid = kde_grid_dict[f"{j}{i}"]
            out[f"{j}{i}"] = make_kde(
                fluxes[flux_names[j]] / fluxes[flux_ref_name],
                fluxes[flux_names[i]] / fluxes[flux_ref_name],
                xgrid=xy_grid[0],
                ygrid=xy_grid[1],
                **kde_kwargs,
            )

    return out


# ======================================================================================== #


def color_corner(
    fluxes,
    flux_names,
    flux_ref_name,
    fig=None,
    ax=None,
    label_names=None,
    label_ref_name=None,
    threshold=1000,
    contour_input={},
    xlim_dict=xlim_dict,
    kde_grid_dict=kde_grid_dict,
    canvas_kwargs={},
    hist_kwargs={"density": True},
    scatter_kwargs={},
    kde_kwargs={},
    contour_kwargs={},
):
    # make figure if needed
    if ax is None:
        fig, ax = make_corner_canvas(**canvas_kwargs)

    # fluxes should be a pd.DataFrame from which I select flux_names
    # ref is what divides everything else
    fluxes = fluxes[[*flux_names, flux_ref_name]]

    # is labels are not provided, uses the names in the dataframe
    if label_names is None:
        label_names = flux_names

    if label_ref_name is None:
        label_ref_name = flux_ref_name

    # columns pair to plot later on
    inds = np.arange(len(flux_names))
    pairs = [(i, j) for i in inds for j in inds]

    # fill the corner plot with actual data
    for i, j in pairs:
        # diagonal, histograms
        if i == j:
            ax[i, j].tick_params(top=True, labeltop=True)
            ax[i, j].hist(
                fluxes[flux_names[i]] / fluxes[flux_ref_name],
                range=xlim_dict[i],
                **hist_kwargs,
            )
        else:
            # threshold switches from points to kde for contours
            # if too many points, make a kde
            if fluxes[flux_ref_name].shape[0] > threshold:
                # for histogram
                xy_grid = kde_grid_dict[f"{j}{i}"]
                kde_kwargs["range"] = xy_grid

                contour_input_ = contour_input.get(f"{j}{i}", None)
                if contour_input_ is None:
                    contour_input_ = make_kde(
                        fluxes[flux_names[j]] / fluxes[flux_ref_name],
                        fluxes[flux_names[i]] / fluxes[flux_ref_name],
                        # xgrid=xy_grid[0],
                        # ygrid=xy_grid[1],
                        **kde_kwargs,
                    )

                ax[i, j].contour(*contour_input_, **contour_kwargs)
            else:
                ax[i, j].scatter(
                    fluxes[flux_names[j]] / fluxes[flux_ref_name],
                    fluxes[flux_names[i]] / fluxes[flux_ref_name],
                    **scatter_kwargs,
                )

        # fix inner labels
        if i < len(flux_names) - 1:
            if i != j:
                ax[i, j].tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
            if i == j:
                ax[i, j].tick_params(
                    axis="x", which="both", bottom=False, labelbottom=False
                )
        else:
            ax[i, j].set_xlabel(f"{label_names[j]}/{label_ref_name}")

        if j > 0:
            ax[i, j].tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )
        else:
            ax[i, j].set_ylabel(f"{label_names[i]}/{label_ref_name}")

    # for each col make the axis limit a bit wider than the limits of the histogram
    for i_ in inds:
        ax_lim = ax[i_, i_].get_xlim()
        extent = ax_lim[1] - ax_lim[0]
        ax[i_, i_].set_xlim(ax_lim[0] - 0.01 * extent, ax_lim[1] + 0.01 * extent)

    # make legend
    ax[-1, -1].legend()

    # return fig and ax, if available, to work on later on
    return fig, ax
