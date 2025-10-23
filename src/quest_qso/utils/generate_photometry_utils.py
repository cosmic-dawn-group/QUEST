import argparse
import random

import matplotlib.pyplot as plt

from quest_qso import LOCAL_PATH
from quest_qso.photometry import qa_plots as qa_plots

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def plot_example_spectra(
    N_examples,
    spectra,
    dispersion,
    fname,
    ylabel="Flux Density",  # better to always specify!
    dir=None,
    make_qa=True,
):
    """
    Plot a few example spectra from the VAE model.
    """
    if not make_qa:
        return

    inds = random.sample(list(range(spectra.shape[0])), N_examples)

    fig, axs = plt.subplots(
        N_examples // 2,
        2,
        figsize=(12, N_examples / 1.61),
        layout="compressed",
        sharex=True,
    )
    for n, ax in enumerate(axs.flatten()):
        ax.step(dispersion, spectra[inds[n]])

    for ax in axs.flatten():
        ax.label_outer()

    fig.supxlabel(r"Wavelength [$\AA$]")
    fig.supylabel(ylabel)

    if dir is not None:
        dir = LOCAL_PATH / "QA"
        dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(dir / fname, bbox_inches="tight")


# =========================================================================== #


# get bands for output name
def get_bands(e):
    if "SVO" in e:
        _ = e.split("_")[1].split("SVO-")
        return e.split("_")[0] + "_" + _[1]
    else:
        return e.replace("-", "_")

# =========================================================================== #


def merge_bands(cols):
    e = set([get_bands(i) for i in cols])

    out = {}
    for _ in e:
        facility, band = _.split("_")
        if facility not in out:
            out[facility] = []
        out[facility].append(band)

    for k in out.keys():
        out[k] = sorted(out[k])

    return "_".join([k + "_" + "".join(v) for k, v in out.items()])


# =========================================================================== #


def parse_config():
    parser = argparse.ArgumentParser(description="Generate photometry configuration")

    parser.add_argument(
        "-p",
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained VAE model",
    )

    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Clear intermediate products (helpful if running on laptop or a machine with limited memory)",
    )

    parser.add_argument(
        "--make-qa-plots",
        action="store_true",
        help="Make (very basic) QA plots of the generated spectra",
    )

    parser.add_argument(
        "--n-examples",
        type=int,
        default=12,
        help="Number of spectra to plot in QA examples, defaults to 12",
    )

    parser.add_argument(
        "--n-per-bin",
        type=int,
        default=1,
        help="Number of object in each `M_1450`-`z` grid bin",
    )

    parser.add_argument(
        "--n-z",
        type=int,
        default=60,
        help="Number of redshift bins in generating the grid, defaults to 60",
    )

    parser.add_argument(
        "--low-z-lim",
        type=int,
        default=6,
        help="Lower redshift limit in generating the grid, defaults to 6",
    )

    parser.add_argument(
        "--high-z-lim",
        type=int,
        default=9,
        help="Upper redshift limit in generating the grid, defaults to 9",
    )

    parser.add_argument(
        "--n-m1450",
        type=int,
        default=120,
        help="Number of `M_1450` bins in generating the grid, defaults to 120",
    )
    parser.add_argument(
        "--faint-m1450-lim",
        type=int,
        default=-21,
        help="Lower `M_1450` limit in generating the grid, defaults to -21",
    )
    parser.add_argument(
        "--bright-m1450-lim",
        type=int,
        default=-28,
        help="Upper `M_1450` limit in generating the grid, defaults to -28",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="uniform",
        help="Sampling mode, either `uniform` or `lf`, defaults to `uniform`",
    )
    parser.add_argument(
        "--luminosity-function",
        type=str,
        default=None,
        help="Luminosity function from `atelier.lumfun`, defaults to None (i.e., uniform sampling)",
    )
    parser.add_argument(
        "--sky-area",
        type=float,
        default=14500,
        help="Sky area to sample for the Luminosity Function sampling, defaults to 14500 (i.e., Euclid wide survey)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility, defaults to 42",
    )

    args = parser.parse_args()

    return args
