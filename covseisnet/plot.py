"""
This module contains functions to simplify the examples in the documentation, 
mostly plotting functions, but also to provide basic tools to quickly visualize
data and results from this package.
"""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from obspy.core.trace import Trace
from scipy.fft import rfft, rfftfreq

import covseisnet as csn
from .signal import ShortTimeFourierTransform


def make_axis_symmetric(ax: Axes, axis: str = "both") -> None:
    """Make the axis of a plot symmetric.

    Given an axis, this function will set the limits of the axis to be symmetric
    around zero. This is useful to have a better visual representation of data
    that is symmetric around zero.

    Arguments
    ---------
    ax : :class:`~matplotlib.axes.Axes`
        The axis to modify.
    axis : str
        The axis to modify. Can be "both", "x", or "y".

    Examples
    --------

    Create a simple plot and make the x-axis symmetric:

    .. plot::

        import covseisnet as csn
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.linspace(-10, 10, 100)
        y = np.random.randn(100)

        fig, ax = plt.subplots(ncols=2, figsize=(6, 3))
        ax[0].plot(x, y)
        ax[1].plot(x, y)
        ax[0].grid()
        ax[1].grid()
        csn.plot.make_axis_symmetric(ax[1], axis="both")
        ax[0].set_title("Original axis")
        ax[1].set_title("Symmetric axis")

    """
    if axis in ["both", "x"]:
        xlim = ax.get_xlim()
        xabs = np.abs(xlim)
        ax.set_xlim(-max(xabs), max(xabs))

    if axis in ["both", "y"]:
        ylim = ax.get_ylim()
        yabs = np.abs(ylim)
        ax.set_ylim(-max(yabs), max(yabs))


def trace_and_spectrum(trace: Trace) -> list:
    """Plot a trace and its spectrum side by side.

    The spectrum is calculated with the :func:`scipy.fft.rfft` function, which
    assumes that the trace is real-valued and therefore only returns the
    positive frequencies.

    Arguments
    ---------
    trace : :class:`~obspy.core.trace.Trace`
        The trace to plot.

    Returns
    -------
    ax : :class:`~matplotlib.axes.Axes`

    Examples
    --------

    Read a stream and plot the first trace and its spectrum:

    .. plot::

        import covseisnet as csn
        stream = csn.read()
        trace = stream[0]
        csn.plot.trace_and_spectrum(trace)

    """
    # Create figure
    _, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(6, 3))

    # Extract data
    times = trace.times()
    waveform = trace.data

    # Calculate spectrum
    spectrum = np.array(rfft(waveform))
    frequencies = rfftfreq(len(waveform), trace.stats.delta)

    # Plot trace
    ax[0].plot(times, waveform)
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid()

    # Plot spectrum
    ax[1].loglog(frequencies, abs(spectrum))
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Spectrum")
    ax[1].grid()

    return ax


def trace_and_spectrogram(
    trace: Trace, f_min: None | float = None, **kwargs: Any
) -> list:
    """Plot a trace and its spectrogram.

    This function is deliberately simple and does not allow to customize the
    spectrogram plot. For more advanced plotting, you should consider creating
    a derived function.

    Arguments
    ---------
    trace : :class:`~obspy.core.trace.Trace`
        The trace to plot.
    f_min : None or float
        The minimum frequency to display. Frequencies below this value will be
        removed from the spectrogram.
    **kwargs
        Additional arguments to pass to :func:`~covseisnet.stream.calculate_spectrogram`.

    Examples
    --------

    Read a stream and plot the first trace and its spectrogram:

    .. plot::

        import covseisnet as csn
        stream = csn.read()
        trace = stream[0]
        csn.plot.trace_and_spectrogram(trace, window_duration=1)

    """
    # Create figure
    _, ax = plt.subplots(nrows=2, constrained_layout=True, sharex=True)

    # Calculate spectrogram
    stft = ShortTimeFourierTransform(
        sampling_rate=trace.stats.sampling_rate,
        **kwargs,
    )

    # Extract spectra
    spectra_times, frequencies, spectra = stft.transform(trace)

    # Remove zero frequencies for display
    if f_min is not None:
        n = np.abs(frequencies - f_min).argmin()
    else:
        n = 1
    frequencies = frequencies[n:]
    spectra = spectra[n:]

    # Calculate spectrogram
    spectrogram = np.log10(np.abs(spectra) + 1e-10)

    # Plot trace
    ax[0].plot(trace.times("matplotlib"), trace.data)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Trace")
    ax[0].grid()
    xlim = ax[0].get_xlim()

    # Plot spectrogram
    mappable = ax[1].pcolormesh(
        spectra_times,
        frequencies,
        spectrogram,
        shading="nearest",
        rasterized=True,
    )
    ax[1].grid()
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_title("Spectrogram")
    ax[1].set_xlim(xlim)
    csn.plot.dateticks(ax[1])

    # Colorbar
    colorbar = plt.colorbar(mappable, ax=ax[1])
    colorbar.set_label("Spectral energy (dBA)")

    return ax


def coherence(times, frequencies, coherence, f_min=None, ax=None, **kwargs):
    """Plot a coherence matrix.

    This function is deliberately simple and does not allow to customize the
    coherence plot. For more advanced plotting, you should consider creating
    a derived function.

    Arguments
    ---------
    times : :class:`~numpy.ndarray`
        The time axis of the coherence matrix.
    frequencies : :class:`~numpy.ndarray`
        The frequency axis of the coherence matrix.
    coherence : :class:`~numpy.ndarray`
        The coherence matrix.
    f_min : float, optional
        The minimum frequency to display. Frequencies below this value will be
        removed from the coherence matrix.
    ax : :class:`~matplotlib.axes.Axes`, optional
        The axis to plot on. If not provided, a new figure will be created.
    **kwargs
        Additional arguments passed to the pcolormesh method. By default, the
        shading is set to "nearest" and the colormap is set to "magma_r".

    Returns
    -------
    ax : :class:`~matplotlib.axes.Axes`
    """
    # Create figure
    if ax is None:
        ax = plt.gca()

    # Remove low frequencies
    if f_min is not None:
        n = np.abs(frequencies - f_min).argmin()
        frequencies = frequencies[n:]
        coherence = coherence[:, n:]

    # Show coherence
    kwargs.setdefault("shading", "nearest")
    kwargs.setdefault("cmap", "magma_r")
    mappable = ax.pcolormesh(times, frequencies, coherence.T, **kwargs)

    # Frequency axis
    ax.set_yscale("log")
    ax.set_ylim(frequencies[1], frequencies[-1])

    # Colorbar
    plt.colorbar(mappable, ax=ax).set_label("Spectral width")

    return ax


def stream_and_coherence(
    stream: csn.stream.NetworkStream,
    times: np.ndarray,
    frequencies: np.ndarray,
    coherence: np.ndarray,
    f_min: float | None = None,
    trace_factor: float = 0.1,
    **kwargs: dict,
) -> list[Axes]:
    """Plot a stream of traces and the coherence matrix.

    This function is deliberately simple and does not allow to customize the
    coherence plot. For more advanced plotting, you should consider creating
    a derived function.

    Arguments
    ---------
    stream : :class:`~obspy.core.stream.Stream`
        The stream to plot.
    times : :class:`~numpy.ndarray`
        The time axis of the coherence matrix.
    frequencies : :class:`~numpy.ndarray`
        The frequency axis of the coherence matrix.
    coherence : :class:`~numpy.ndarray`
        The coherence matrix.
    f_min : float, optional
        The minimum frequency to display. Frequencies below this value will be
        removed from the coherence matrix.
    trace_factor : float, optional
        The factor to multiply the traces by for display.
    **kwargs
        Additional arguments passed to the pcolormesh method.
    """
    # Create figure
    _, ax = plt.subplots(nrows=2, constrained_layout=True, sharex=True)

    # Show traces
    trace_times = stream.times(type="matplotlib")
    for index, trace in enumerate(stream):
        waveform = trace.data * trace_factor
        ax[0].plot(trace_times, waveform + index, color="C0", lw=0.3)

    # Labels
    stations = [trace.stats.station for trace in stream]
    ax[0].set_title("Normalized seismograms")
    ax[0].grid()
    ax[0].set_yticks(range(len(stations)), labels=stations, fontsize="small")
    ax[0].set_ylabel("Normalized amplitude")
    ax[0].set_ylim(-1, len(stations))
    xlim = ax[0].get_xlim()

    # Plot coherence
    csn.plot.coherence(
        times, frequencies, coherence, f_min=f_min, ax=ax[1], **kwargs
    )
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_title("Spatial coherence")

    # Date formatter
    dateticks(ax[1])
    ax[1].set_xlim(xlim)

    return ax


def covariance_matrix_modulus_and_spectrum(
    covariance: csn.covariance.CovarianceMatrix,
) -> Axes:
    """Plot the modulus of a covariance matrix and its spectrum.

    This function plots the modulus of the covariance matrix and its
    eigenvalues in a single figure. The eigenvalues are normalized to sum to
    1.


    Arguments
    ---------
    covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
        The covariance matrix to plot.

    Returns
    -------
    ax : :class:`~matplotlib.axes.Axes`

    Examples
    --------

    Create a covariance matrix and plot its modulus and spectrum:

    .. plot::

        import covseisnet as csn
        import numpy as np
        np.random.seed(0)
        c = np.random.randn(3, 3)
        c = (c @ c.T) / 0.5
        c = csn.covariance.CovarianceMatrix(c)
        c.stats = [{"station": station} for station in "ABC"]
        csn.plot.covariance_matrix_modulus_and_spectrum(c)
    """
    # Normalize covariance
    covariance = covariance / np.max(np.abs(covariance))

    # Calculate eigenvalues
    eigenvalues = covariance.eigenvalues(norm=np.sum)

    # Create figure
    _, ax = plt.subplots(ncols=2, figsize=(8, 2.7), constrained_layout=True)

    # Plot covariance matrix
    mappable = ax[0].matshow(np.abs(covariance), cmap="viridis", vmin=0)

    # Coherence
    spectral_width = covariance.coherence(kind="spectral_width")
    entropy = covariance.coherence(kind="entropy")
    diversity = covariance.coherence(kind="diversity")

    # Labels
    if covariance.stats is None:
        stations = [f"Station {i}" for i in range(covariance.shape[0])]
    else:
        stations = [stat["station"] for stat in covariance.stats]
    xticks = range(covariance.shape[0])
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(stations, rotation=90, fontsize="small")
    ax[0].set_yticks(xticks, labels=stations, fontsize="small")
    ax[0].xaxis.set_ticks_position("bottom")
    ax[0].set_xlabel(r"Channel $i$")
    ax[0].set_ylabel(r"Channel $j$")
    ax[0].set_title("Covariance matrix")
    plt.colorbar(mappable).set_label(r"Covariance modulus $|\mathbf{C}|$")

    # Plot eigenvalues
    eigenindex = np.arange(covariance.shape[0])
    ax[1].plot(eigenindex, eigenvalues, marker="o")
    ax[1].set_ylim(bottom=0, top=1)
    ax[1].set_xticks(eigenindex)
    ax[1].set_xlabel(r"Eigenvalue index ($n$)")
    ax[1].set_ylabel(r"Eigenvalue ($\lambda_n$)")
    ax[1].set_title("Eigenspectrum")
    ax[1].grid()

    # Annotations
    ax[1].axvline(spectral_width, color="C1", label="Spectral width")
    ax[1].axvline(entropy, color="C2", label="Entropy")
    ax[1].axvline(diversity, color="C3", label="Diversity")
    ax[1].legend(loc="upper left", frameon=False, bbox_to_anchor=(1, 1))

    return ax


def dateticks(ax: Axes, locator: mdates.DateLocator | None = None) -> None:
    """Set date ticks on the x-axis of a plot.

    Arguments
    ---------
    ax : :class:`~matplotlib.axes.Axes`
        The axis to modify.
    locator : :class:`~matplotlib.dates.DateLocator`
        The locator to use for the date ticks. This can be an instance of
        :class:`~matplotlib.dates.AutoDateLocator`,
        :class:`~matplotlib.dates.DayLocator`, etc. Check the documentation
        for more information.
    formatter : :class:`~matplotlib.dates.DateFormatter`
        The formatter to use for the date ticks. This can be an instance of
        :class:`~matplotlib.dates.ConciseDateFormatter` for example. Check the
        documentation for more information.

    Examples
    --------
    Create a simple plot with date ticks:

    .. plot::

        import covseisnet as csn
        import numpy as np
        import matplotlib.pyplot as plt

        stream = csn.read()
        trace = stream[0]

        fig, ax = plt.subplots(nrows=2, figsize=(6, 3), constrained_layout=True)

        ax[0].plot(trace.times(), trace.data)
        ax[0].set_title("Time series with times in seconds")
        ax[0].grid()
        ax[0].set_xlabel("Time (seconds)")

        ax[1].plot(trace.times("matplotlib"), trace.data)
        ax[1].set_title("Time series with times in datetime")
        ax[1].grid()
        csn.plot.dateticks(ax[1])
    """
    xticks = locator or mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter
    xticklabels = formatter(xticks)
    ax.xaxis.set_major_locator(xticks)
    ax.xaxis.set_major_formatter(xticklabels)


def grid3d(
    grid,
    profile_coordinates=None,
    receiver_coordinates=None,
    label=None,
    **kwargs,
) -> dict:
    """Plot a three-dimensional grid of data.

    This function plots the data of a three-dimensional grid in three
    different views: xy, xz, and zy. The grid data is plotted as a contour
    plot. In addition, the function can plot receiver coordinates, and a
    profile line can be highlighted.

    Arguments
    ---------
    grid : :class:`~covseisnet.grid.Grid3D` or derived class
        The grid to plot.
    profile_coordinates : tuple, optional
        The coordinates of the profile line to highlight. The profile line
        will be highlighted with dashed lines.
    receiver_coordinates : tuple, optional
        The coordinates of the receivers to plot. The receivers will be
        plotted as black squares.
    label : str, optional
        The label of the colorbar.
    **kwargs
        Additional arguments passed to the contourf method.

    Returns
    -------
    ax : dict
        A dictionary with the axes of the plot. The keys are "xy", "xz", "zy",
        "cb", and ".". This is the output of the
        :func:`~matplotlib.pyplot.subplot_mosaic` function.

    Examples
    --------
    Create a simple grid and plot it:

    .. plot::

        import covseisnet as csn
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a random grid
        grid = csn.spatial.Regular3DGrid(extent=(-10, 10, -10, 10, 0, 10), shape=(10, 10, 10))
        grid[:] = grid.mesh[0] + grid.mesh[1] + grid.mesh[2]

        # Show the grid
        csn.plot.grid3d(
            grid,
            profile_coordinates=(0, 0, 0),
            receiver_coordinates=(0, 0, 0),
            label="Amplitude (dB)"
        )
        plt.show()

    """
    # Set limits
    if grid.lon is None or grid.lat is None or grid.depth is None:
        return {"empty": plt.gca()}

    # Create mosaic plot
    _, ax = plt.subplot_mosaic(
        figsize=(5, 5),
        mosaic=[["xy", "zy"], ["xz", "."], ["xz", "cb"], ["xz", "."]],
        gridspec_kw={
            "hspace": 0.6,
            "wspace": 0.2,
            "width_ratios": [1, 0.5],
            "height_ratios": [1, 0, 0.05, 0.1],
        },
    )

    # Default values for profile coordinates
    if profile_coordinates is None:
        if hasattr(grid, "receiver_coordinates"):
            if isinstance(grid.receiver_coordinates[0], (list, tuple)):
                profile_coordinates = grid.receiver_coordinates[0]
            else:
                profile_coordinates = grid.receiver_coordinates
        else:
            i, j, k = np.unravel_index(grid.argmax(), grid.shape)
            profile_coordinates = profile_coordinates or (
                grid.lon[i],
                grid.lat[j],
                grid.depth[k],
            )

    # Find out profile index
    profile_index = (
        np.argmin(np.abs(grid.lon - profile_coordinates[0])),
        np.argmin(np.abs(grid.lat - profile_coordinates[1])),
        np.argmin(np.abs(grid.depth - profile_coordinates[2])),
    )

    # Default values for contour
    if isinstance(grid, csn.travel_times.DifferentialTravelTimes):
        vmax = max(np.abs(grid.min()), np.abs(grid.max()))
        vmin = -vmax
        kwargs.setdefault("cmap", "RdBu")
    else:
        vmax = np.max(grid)
        vmin = np.min(grid)

    # Contour levels
    contour_levels = np.linspace(vmin, vmax, kwargs.pop("levels", 20))

    # Assign default values
    kwargs.setdefault("vmin", vmin)
    kwargs.setdefault("vmax", vmax)
    kwargs.setdefault("levels", contour_levels)

    # Plotting xy
    mappable = ax["xy"].contourf(
        grid.lon, grid.lat, grid[:, :, profile_index[2]].T, **kwargs
    )
    ax["xz"].contourf(
        grid.lon, grid.depth, grid[:, profile_index[1], :].T, **kwargs
    )
    ax["zy"].contourf(
        grid.depth, grid.lat, grid[profile_index[0], :, :], **kwargs
    )
    ax["xy"].axvline(profile_coordinates[0], dashes=[8, 3], color="k", lw=0.5)
    ax["xy"].axhline(profile_coordinates[1], dashes=[8, 3], color="k", lw=0.5)
    ax["zy"].axvline(profile_coordinates[2], dashes=[8, 3], color="k", lw=0.5)
    ax["xz"].axhline(profile_coordinates[2], dashes=[8, 3], color="k", lw=0.5)
    ax["zy"].axhline(profile_coordinates[1], dashes=[8, 3], color="k", lw=0.5)
    ax["xz"].axvline(profile_coordinates[0], dashes=[8, 3], color="k", lw=0.5)
    ax["xz"].invert_yaxis()
    ax["xy"].set_ylabel("Latitude (º)")
    ax["xz"].set_xlabel("Longitude (º)")
    ax["xz"].set_ylabel("Depth (km)")
    ax["zy"].set_xlabel("Depth (km)")
    ax["zy"].yaxis.tick_right()
    ax["zy"].yaxis.set_label_position("right")
    ax["zy"].set_yticks(ax["xy"].get_yticks())
    ax["zy"].set_yticklabels(ax["xy"].get_yticklabels())
    cb = plt.colorbar(
        mappable, cax=ax["cb"], orientation="horizontal", shrink=0.5
    )
    cb.ax.xaxis.set_major_locator(MaxNLocator(4))
    if label is not None:
        cb.set_label(label)

    ax["xy"].set_xlim(grid.lon.min(), grid.lon.max())
    ax["xy"].set_ylim(grid.lat.min(), grid.lat.max())
    ax["xz"].set_xlim(grid.lon.min(), grid.lon.max())
    ax["zy"].set_ylim(grid.lat.min(), grid.lat.max())
    ax["zy"].set_xlim(grid.depth.min(), grid.depth.max())
    ax["xz"].set_ylim(grid.depth.max(), grid.depth.min())

    # If receiver coordinates are provided, plot them
    if receiver_coordinates is None:
        if hasattr(grid, "receiver_coordinates"):
            if grid.receiver_coordinates is None:
                return ax
            if not isinstance(grid.receiver_coordinates[0], (list, tuple)):
                receiver_coordinates = (grid.receiver_coordinates,)
            else:
                receiver_coordinates = grid.receiver_coordinates
        else:
            return ax
    else:
        if not isinstance(receiver_coordinates[0], (list, tuple)):
            receiver_coordinates = (receiver_coordinates,)

    for receiver in receiver_coordinates:
        if not len(receiver) == 3:
            raise ValueError(
                "Receiver coordinates must be a list or tuple of length 3."
            )
        ax["xy"].plot(*receiver[:2], "ks", clip_on=False)
        ax["xz"].plot(receiver[0], receiver[2], "kv", clip_on=False)
        ax["zy"].plot(receiver[2], receiver[1], "k>", clip_on=False)

    return ax
