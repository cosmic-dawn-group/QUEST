"""Utility helpers to create and inspect photometric filter responses.

This module provides convenience functions used across the package to fetch
filter transmission curves (primarily from the SVO Filter Profile Service),
cache them locally, adapt them for use with ``speclite``, and plot/filter
collections for quick inspection.

Key functions
- ``add_zero_extrema(filter_array)``: ensures a response starts/ends at zero
    transmission (required by ``speclite`` in some cases).
- ``get_SVO_response(facility, instr, band, path)``: download a filter
    response from SVO (or load from a local cache). Returns an (N,2) array of
    wavelength [Angstrom] and transmission.
- ``generate_custom_filter(facility, instr, band, group, path)``: create a
    ``speclite.filters.FilterResponse`` from an SVO response and attach metadata.
- ``get_response(filter)``: extract wavelength/response arrays from a
    ``speclite`` filter object (helper for plotting).
- ``plot_filters(filters, ...)``: plot a sequence of filters with optional
    annotations (effective wavelength, model coverage lines, etc.).

Caching
- Responses fetched from SVO are saved under ``local_path`` (controlled by
    the environment variable ``QUEST_LOCALPATH``) so subsequent runs avoid
    network calls.

Usage examples
>>> from quest_qso.photometry import custom_filters as cf
>>> resp = cf.get_SVO_response('SDSS', 'sdss', 'g')
>>> flt = cf.generate_custom_filter('SDSS', 'sdss', 'g', 'sdss_group')

"""

from io import StringIO

import astropy.units as au
import matplotlib.pyplot as plt
import numpy as np
import requests
import speclite.filters as sfilters
from numpy.typing import NDArray

from quest_qso import LOCAL_PATH as local_path
from quest_qso.utils import utilities


# the SVO function wants zero at the end, so fix the filters to match that
def add_zero_extrema(current_filter: NDArray) -> NDArray:
    """Ensure a filter response starts and ends with zero transmission.

    speclite expects filter response curves to include boundary points with
    zero transmission. Many SVO-provided transmission files omit explicit
    zero endpoints; this helper adds a single zero-valued sample one bin
    width before the first wavelength and one after the last wavelength.

    Parameters
    ----------
    current_filter
        An (N, 2) array-like where the first column is wavelength (same units
        convention as the source file, typically Angstroms) and the second
        column is the dimensionless transmission in [0, 1]. N must be >= 2.

    Returns
    -------
    numpy.ndarray
        An (N+2, 2) ndarray with one extra row prepended and appended where
        the transmission values are zero. The wavelength spacing used for the
        new endpoints is the first delta between input wavelengths.

    Raises
    ------
    IndexError
        If `current_filter` has fewer than 2 wavelength samples.
    """
    delta = np.diff(current_filter[:, 0])[0]
    return np.concatenate(
        (
            [[current_filter[:, 0][0] - delta, 0.0]],
            current_filter,
            [[current_filter[:, 0][-1] + delta, 0.0]],
        )
    )

# =====


# download filtes from SVO, or grab them from cache folder
def get_SVO_response(facility: str, instr: str, band: str, path=local_path) -> NDArray:
    """Fetch a filter transmission curve from the SVO Filter Profile Service.

    This function first checks for a cached file at ``path / facility / '{instr}.{band}.dat'``
    and loads it if present. Otherwise it requests the ASCII response from the
    SVO FPS, parses it with numpy, ensures zero-valued endpoints (via
    ``add_zero_extrema``) and saves the cached copy for future use.

    Parameters
    ----------
    facility
        SVO facility identifier (for example 'SDSS' or other facility names as
        used by the SVO database).
    instr
        Instrument identifier within the facility (string).
    band
        Band identifier (string) as used by SVO.
    path
        Base cache directory (Path-like). Defaults to the module-level
        ``local_path``. The function will create ``path/facility`` if needed.

    Returns
    -------
    numpy.ndarray
        An (N, 2) array where column 0 is wavelength (float) and column 1 is
        the dimensionless transmission (float in [0, 1]). Wavelength units are
        those provided by the SVO file (typically Angstroms). The returned
        array is guaranteed to start and end with zero transmission values.

    Raises
    ------
    requests.HTTPError
        If the HTTP request to SVO fails (non-2xx status).  numpy.loadtxt may
        raise a ValueError if the ASCII file is malformed.
    """
    filter_path = path / f"{facility}"
    filter_path.mkdir(parents=True, exist_ok=True)

    if (filter_path / f"{instr}.{band}.dat").is_file():
        utilities.logger.info(
            f"Response exists, loading file from file in {filter_path}."
        )
        out = np.loadtxt(
            filter_path / f"{instr}.{band}.dat",
            delimiter=" ",
        )

        return (
            out if (out[:, 1][0] == 0 and out[:, 1][-1] == 0) else add_zero_extrema(out)
        )

    url = f"http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id={facility}/{instr}.{band}"
    req = requests.get(url)

    # let numpy figure out the delimiter itself, as the SVO files are not consistent
    #  HSC-z for example has a double space, which creates issues
    nploaded = np.loadtxt(StringIO(req.content.decode(req.encoding)))
    out = add_zero_extrema(nploaded)
    np.savetxt(filter_path / f"{instr}.{band}.dat", out)
    return out

# =====


# generate new filter set using the SVO utilities
def generate_custom_filter(
    facility: str, instr: str, band: str, group: str, path=local_path / "custom_filters"
):
    """Create a ``speclite.filters.FilterResponse`` from an SVO response.

    The returned object is suitable for inclusion in a ``speclite.filters.FilterList``
    and provides metadata in the ``meta`` dict. The wavelength passed to
    ``FilterResponse`` is converted to an ``astropy.units.Quantity`` (Angstrom).

    Parameters
    ----------
    facility, instr, band, group
        Same identifiers used for ``get_SVO_response``. ``group`` is saved in
        the filter metadata under the key ``group_name`` and ``band_name``.
    path
        Cache directory used by ``get_SVO_response`` (Path-like).

    Returns
    -------
    speclite.filters.FilterResponse
        A FilterResponse object with attributes ``wavelength`` (Quantity) and
        ``response`` (ndarray). Use this object directly or include it in
        a ``speclite.filters.FilterList``.
    """

    SVO_Euclid_filter_data = get_SVO_response(facility, instr, band, path=path)

    SVO_Euclid_filter = sfilters.FilterResponse(
        wavelength=SVO_Euclid_filter_data[:, 0] * au.Angstrom,  # required afaik
        response=SVO_Euclid_filter_data[:, 1],
        meta=dict(group_name=group, band_name=band),
    )
    return SVO_Euclid_filter

# =====


def get_response(_filter):
    """Extract wavelength and response arrays from a ``speclite`` filter.

    Parameters
    ----------
    _filter
        A ``speclite.filters.FilterResponse`` (or another object exposing
        ``_wavelength`` and ``_response`` attributes). The function is a thin
        helper used by plotting utilities.

    Returns
    -------
    tuple
        A pair ``(wavelength, response)`` where ``wavelength`` is an ndarray
        (or Quantity) and ``response`` is a 1-D ndarray of transmission
        values in [0,1].
    """
    return (_filter._wavelength, _filter._response)

# =====


def plot_filters(
    filters,
    fig=None,
    ax=None,
    figsize=(10, 10 / 1.61),
    model_end=None,
    legend=True,
    add_filter_name=False,
    add_effective_wavelength=False,
):
    """Visualize a sequence of filter responses.

    This helper plots each filter in ``filters._responses`` (speclite stores
    responses in this attribute) and optionally annotates effective
    wavelengths or a vertical line showing the redshifted coverage of a
    rest-frame feature (``model_end``).

    Parameters
    ----------
    filters
        A ``speclite.filters.FilterList`` or similar object exposing
        ``_responses`` and ``get_ab_magnitudes``.
    fig, ax
        Optional matplotlib Figure and Axes to draw into. If not provided,
        a new figure is created using ``figsize``.
    figsize
        Figure size used when creating a new figure.
    model_end
        Optional rest-frame wavelength (float, Angstrom) that will be drawn
        at a set of example redshifts to illustrate spectral coverage.
    legend, add_filter_name, add_effective_wavelength
        Boolean flags controlling plot annotations.

    Returns
    -------
    (fig, ax)
        The matplotlib Figure and Axes objects used for plotting.

    Notes
    -----
    This function accesses ``filters._responses`` which is an internal
    attribute of speclite filter objects; the internal layout is stable but
    could change in future speclite versions.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="compressed")

    # set the upper y-axis
    upper_yax = 0.0

    for n, _filter in enumerate(filters._responses):
        filter_name = _filter.name.split("-")[1].upper()
        facility_name = _filter.name.split("-")[0]
        if "SVO" in facility_name:
            facility_name = "".join(facility_name.split("_SVO"))

        ax.plot(
            *get_response(_filter),
            alpha=0.7,
            label=facility_name + " " + filter_name,
        )
        ax.fill_between(*get_response(_filter), alpha=0.4)
        if add_effective_wavelength:
            eff_w = _filter.effective_wavelength.to(au.Angstrom).value
            ax.axvline(
                eff_w,
                color="grey",
                linestyle=":",
                alpha=0.7,
            )

        upper_yax = np.max((upper_yax, get_response(_filter)[1].max()))

    if model_end is not None and isinstance(model_end, (int, float)):
        xlims = ax.get_xlim()
        for z_ in np.arange(0.5, 12, 1):
            if model_end * (1 + z_) < xlims[0] or model_end * (1 + z_) > xlims[1]:
                continue

            ax.axvline(
                model_end * (1 + z_),
                color="red",
                linestyle="--",
            )
            ax.text(
                model_end * (1 + z_) + (xlims[1] - xlims[0]) / 500,
                upper_yax * 0.125 * -1,
                f"{z_:.1f}",
                rotation=90,
                color="red",
            )

        ax.axvline(
            -10000,
            color="red",
            linestyle="--",
            label="Red coverage limit at z",
        )
        ax.set_xlim(*xlims)

    ax.axhline(0.0, color="lightgrey", zorder=-2)

    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Filter transmission")
    if legend:
        ax.legend(loc="upper right", ncol=n // 2 + 1)

    ax.set_ylim(-upper_yax * 0.05, upper_yax * 1.1)

    # optinally, add the filter name
    if add_filter_name:
        for n, _filter in enumerate(filters._responses):
            filter_name = _filter.name
            eff_w = _filter.effective_wavelength.to(au.Angstrom).value

            _, resp_ = get_response(_filter)
            max_pos = np.argmax(resp_)
            ax.text(
                eff_w,
                resp_[max_pos],
                filter_name,
                ha="center",
                va="bottom",
            )

    return fig, ax
