"""
This module contains functions to simplify the examples in the documentation, 
mostly plotting functions, but also to provide basic tools to quickly visualize
data and results from this package.
"""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.ticker import Formatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obspy
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
        csn.plot.trace_and_spectrogram(trace, window_duration_sec=1)

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
        Additional arguments passed to the pcolormesh method.

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
    mappable = ax.pcolormesh(
        times,
        frequencies,
        coherence.T,
        shading="nearest",
        cmap="magma_r",
        **kwargs,
    )

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
        ax[0].plot(trace_times, waveform + index, color="k", lw=0.3)

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
        c.set_stations(["A", "B", "C"])
        csn.plot.covariance_matrix_modulus_and_spectrum(c)
    """
    # Normalize covariance
    covariance = covariance / np.max(np.abs(covariance))

    # Calculate eigenvalues
    eigenvalues = covariance.eigenvalues(norm=np.sum)

    # Create figure
    _, ax = plt.subplots(ncols=2, figsize=(8, 2.7), constrained_layout=True)

    # Plot covariance matrix
    mappable = ax[0].matshow(np.abs(covariance), cmap="cividis", vmin=0)

    # Coherence
    spectral_width = covariance.coherence(kind="spectral_width")
    entropy = covariance.coherence(kind="entropy")
    diversity = covariance.coherence(kind="diversity")

    # Labels
    stations = [stat.station for stat in covariance.stats]
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
