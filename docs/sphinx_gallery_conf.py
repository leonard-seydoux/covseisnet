"""Sphinx extension to change the default settings of Matplotlib."""

import matplotlib

GREY = "0.3"


def reset_matplotlib(*args):
    """Resetter function to be imported by Sphinx-Gallery.

    The natural format of Matplotlib is PNG with white background. This
    function changes the default settings of Matplotlib to make the plots more
    harmonious in dark more. The changes are the following: lines are thinner,
    background is transparent, and any black color is replaced by a light grey
    so it can either be seen on a white or dark background.


    Note that the arguments of Sphinx-Gallery are not used in this function.
    """
    # Font
    matplotlib.rcParams["font.sans-serif"] = [
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Helvetica Neue",
        "Arial",
        "Apple Color Emoji",
        "Segoe UI Emoji",
        "Segoe UI Symbol",
    ]

    # Lines
    matplotlib.rcParams["lines.linewidth"] = 0.7

    # Colors
    matplotlib.rcParams["axes.facecolor"] = (1, 1, 1, 0)
    matplotlib.rcParams["figure.facecolor"] = "none"
    matplotlib.rcParams["axes.edgecolor"] = GREY
    matplotlib.rcParams["axes.labelcolor"] = GREY
    matplotlib.rcParams["text.color"] = GREY
    matplotlib.rcParams["xtick.color"] = GREY
    matplotlib.rcParams["ytick.color"] = GREY
    matplotlib.rcParams["axes.titlecolor"] = GREY
    matplotlib.rcParams["grid.color"] = GREY
    matplotlib.rcParams["grid.alpha"] = 0.3
