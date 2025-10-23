import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_binned_confidence_int(
    to_bin_x,
    to_bin_y,
    error_data=None,
    ax=None,
    figsize=(8, 8 / 1.61),
    x_log_scale=False,
    y_log_scale=False,
    x_label=None,
    y_label=None,
    **kwargs,
):
    """Plots the 2d histogram based on `to_bin_x` and `to_bin_y`, and overplots
    the (binned) error function and confidence intervals. kwargs are passed to
    plt.hist2d.

    :param to_bin_x: _description_
    :type to_bin_x: _type_
    :param to_bin_y: _description_
    :type to_bin_y: _type_
    :param error_data: _description_
    :type error_data: _type_
    :param ax: _description_, defaults to None
    :type ax: _type_, optional
    :param figsize: _description_, defaults to (8, 8 / 1.61)
    :type figsize: tuple, optional
    :param x_log_scale: _description_, defaults to False
    :type x_log_scale: bool, optional
    :param y_log_scale: _description_, defaults to False
    :type y_log_scale: bool, optional
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    safe = ~(np.isnan(to_bin_x) | np.isnan(to_bin_y))
    data_x = to_bin_x[safe]
    data_y = to_bin_y[safe]

    ax.hist2d(data_x, data_y, **kwargs)
    if error_data:
        ax.scatter(error_data[:, 0], error_data[:, 1], c="k", s=1)
        ax.scatter(error_data[:, 0], error_data[:, 2], c="lightgrey", s=1)
        ax.scatter(error_data[:, 0], error_data[:, 3], c="lightgrey", s=1)

    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


## ============================================================================= ##


def plot_binned_confidence_int(
    error_data,
    ax=None,
    figsize=(8, 8 / 1.61),
    x_log_scale=False,
    y_log_scale=False,
    x_label=None,
    y_label=None,
):
    """Plots the 2d histogram based on `to_bin_y` and `to_bin_y`, and overplots
    the (binned) error function and confidence intervals.

    :param to_bin_x: _description_
    :type to_bin_x: _type_
    :param to_bin_y: _description_
    :type to_bin_y: _type_
    :param error_data: _description_
    :type error_data: _type_
    :param ax: _description_, defaults to None
    :type ax: _type_, optional
    :param figsize: _description_, defaults to (8, 8 / 1.61)
    :type figsize: tuple, optional
    :param x_log_scale: _description_, defaults to False
    :type x_log_scale: bool, optional
    :param y_log_scale: _description_, defaults to False
    :type y_log_scale: bool, optional
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.scatter(error_data[:, 0], error_data[:, 1], c="k", s=1)
    ax.scatter(error_data[:, 0], error_data[:, 2], c="lightgrey", s=1)
    ax.scatter(error_data[:, 0], error_data[:, 3], c="lightgrey", s=1)

    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


## ============================================================================= ##


def plot_error_functions(
    error_function,
    ax=None,
    figsize=(8, 8 / 1.61),
    x_log_scale=False,
    y_log_scale=False,
):
    """Plots the error function, oveplotting the 1-sigma confidence interval.

    :param error_function: _description_
    :type error_function: _type_
    :param ax: _description_, defaults to None
    :type ax: _type_, optional
    :param figsize: _description_, defaults to (8, 8 / 1.61)
    :type figsize: tuple, optional
    :param x_log_scale: _description_, defaults to False
    :type x_log_scale: bool, optional
    :param y_log_scale: _description_, defaults to False
    :type y_log_scale: bool, optional
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(error_function[0], error_function[1], c="k")

    ax.plot(error_function[0], error_function[1] - error_function[2], c="C0")
    ax.plot(error_function[0], error_function[1] + error_function[2], c="C0")

    ax.fill_between(
        error_function[0],
        error_function[1] - error_function[2],
        error_function[1] + error_function[2],
        alpha=0.5,
        label=r"1 $\sigma$",
    )

    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")
