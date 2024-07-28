"""
This module contains functions to simplify the examples in the documentation, 
mostly plotting functions, but also to provide basic tools to quickly visualize
data and results from this package.
"""

import time


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obspy
from scipy.fft import rfft, rfftfreq

import covseisnet as csn
from .signal import ShortTimeFourierTransform


def utc2datetime(utc_array: np.ndarray) -> np.ndarray:
    """Convert an array of UTC times to datetime objects.

    Arguments
    ---------
    utc_array : :class:`~numpy.ndarray`
        The array of UTC times.

    Returns
    -------
    :class:`~numpy.ndarray`
        The array of datetime objects.
    """
    return list(map(lambda t: t.datetime, utc_array))

    # return np.array([t.datetime for t in utc_array])


def make_axis_symmetric(ax: plt.Axes, axis: str = "both") -> None:
    """Make the axis of a plot symmetric.

    Arguments
    ---------
    ax : :class:`~matplotlib.axes.Axes`
        The axis to modify.
    axis : str
        The axis to modify. Can be "both", "x", or "y".
    """
    if axis in ["both", "x"]:
        xlim = ax.get_xlim()
        xabs = np.abs(xlim)
        ax.set_xlim(-max(xabs), max(xabs))

    if axis in ["both", "y"]:
        ylim = ax.get_ylim()
        yabs = np.abs(ylim)
        ax.set_ylim(-max(yabs), max(yabs))


def trace_and_spectrum(trace: obspy.core.trace.Trace) -> None:
    """Plot a trace and its spectrum side by side.

    The spectrum is calculated with the :func:`scipy.fft.rfft` function, which
    assumes that the trace is real-valued and therefore only returns the
    positive frequencies.

    Arguments
    ---------
    trace : :class:`~obspy.core.trace.Trace`
        The trace to plot.
    """
    # Create figure
    _, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(6, 3))

    # Extract data
    times = trace.times()
    waveform = trace.data

    # Calculate spectrum
    spectrum = rfft(waveform)
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


def trace_and_spectrogram(
    trace: obspy.core.trace.Trace, **kwargs: dict
) -> None:
    """Plot a trace and its spectrogram.

    This function is deliberately simple and does not allow to customize the
    spectrogram plot. For more advanced plotting, you should consider creating
    a derived function.

    Arguments
    ---------
    trace : :class:`~obspy.core.trace.Trace`
        The trace to plot.
    **kwargs
        Additional arguments to pass to :func:`~covseisnet.stream.calculate_spectrogram`.
    """
    # Create figure
    _, ax = plt.subplots(nrows=2, constrained_layout=True, sharex=True)

    # Calculate spectrogram
    kwargs["sampling_rate"] = trace.stats.sampling_rate
    stft = ShortTimeFourierTransform(**kwargs)
    spectra_times, frequencies, spectra = stft.transform(trace)
    spectrogram = np.log10(np.abs(spectra) + 1e-10)

    # Remove zero frequencies for display
    frequencies = frequencies[1:]
    spectra = spectra[1:]

    # Turn times into datetime objects
    trace_times = utc2datetime(trace.times("utcdatetime"))
    spectra_times = utc2datetime(spectra_times)

    # Plot trace
    ax[0].plot(trace_times, trace.data)
    ax[0].grid()
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Trace")
    xlim = ax[0].get_xlim()
    make_axis_symmetric(ax[0], axis="y")

    # Plot spectrogram
    mappable = ax[1].pcolormesh(
        spectra_times,
        frequencies,
        spectrogram,
        shading="nearest",
    )
    ax[1].grid()
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_title("Spectrogram")
    ax[1].set_xlim(xlim)

    # Colorbar
    colorbar = plt.colorbar(mappable, ax=ax[1])
    colorbar.set_label("Spectral energy (dBA)")

    plt.savefig("spectrogram.png")


def stream_and_coherence(
    stream: csn.stream.NetworkStream,
    times: np.ndarray,
    frequencies: np.ndarray,
    coherence: np.ndarray,
    **kwargs: dict,
) -> None:
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
    **kwargs
        Additional arguments passed to the pcolormesh method.
    """
    # Create figure
    fig, ax = plt.subplots(nrows=2, constrained_layout=True, sharex=True)

    # Show traces
    tic = time.time()
    trace_times = stream.times(type="matplotlib")
    print(f"Elapsed time: {time.time() - tic:.2f} s")
    global_max = np.max([np.max(np.abs(trace.data)) for trace in stream])
    for index, trace in enumerate(stream):
        waveform = trace.data
        waveform /= global_max
        ax[0].plot(trace_times, waveform + index, color="k", lw=0.5)
    # Labels
    stations = stream.stations
    ax[0].set_title("Normalized seismograms")
    ax[0].grid()
    ax[0].set_yticks(range(len(stations)), labels=stations, fontsize="small")
    ax[0].set_ylabel("Normalized amplitude")
    xlim = ax[0].get_xlim()

    # Get time
    times = utc2datetime(times)
    # d_times = times[1] - times[0]
    # times = np.concatenate([times, [times[-1] + 2 * d_times]])

    # Add one frequency to frequencies for pcolormesh
    # df = frequencies[1] - frequencies[0]
    # frequencies = np.concatenate([frequencies, [frequencies[-1] + df]])

    # Remove zero frequency
    n = 6
    frequencies = frequencies[n:]
    coherence = coherence[:, n:]

    # Show coherence
    tic = time.time()
    mappable = ax[1].pcolormesh(
        times,
        frequencies,
        coherence.T,
        shading="nearest",
        cmap="magma_r",
        # vmin=0,
        **kwargs,
    )
    print(f"Elapsed time: {time.time() - tic:.2f} s")

    # Frequency axis
    ax[1].set_yscale("log")
    ax[1].set_ylim(frequencies[1], frequencies[-1] / 2)
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_title("Spatial coherence")

    # Colorbar
    plt.colorbar(mappable, ax=ax[1]).set_label("Coherence")

    # Date formatter
    xticks = mdates.AutoDateLocator()
    xticklabels = mdates.ConciseDateFormatter(xticks)
    ax[1].xaxis.set_major_locator(xticks)
    ax[1].xaxis.set_major_formatter(xticklabels)
    ax[1].set_xlim(xlim)

    fig.savefig("coherence.png", dpi=300)


def covariance_matrix_modulus_and_spectrum(
    covariance: csn.covariance.CovarianceMatrix,
) -> None:
    """Plot the modulus of a covariance matrix and its spectrum.

    Arguments
    ---------
    covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
        The covariance matrix to plot.
    """
    # Normalize covariance
    covariance = covariance / np.max(np.abs(covariance))

    # Calculate eigenvalues
    eigenvalues = covariance.eigenvalues(norm=np.sum)

    # Create figure
    _, ax = plt.subplots(ncols=2, figsize=(6, 2.7), constrained_layout=True)

    # Plot covariance matrix
    mappable = ax[0].matshow(np.abs(covariance), cmap="cividis", vmin=0)

    # Coherence
    coherence = covariance.coherence()

    # Labels
    xticks = range(covariance.shape[0])
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(covariance.stations, rotation=90, fontsize="small")
    ax[0].set_yticks(xticks, labels=covariance.stations, fontsize="small")
    ax[0].xaxis.set_ticks_position("bottom")
    ax[0].set_xlabel(r"Channel $i$")
    ax[0].set_ylabel(r"Channel $j$")
    ax[0].set_title("Covariance matrix")
    plt.colorbar(mappable).set_label(r"Covariance modulus $|\mathbf{C}|$")

    # Plot eigenvalues
    eigenindex = np.arange(covariance.shape[0]) + 1
    ax[1].plot(eigenindex, eigenvalues, marker="o")
    ax[1].axvline(coherence, c="C1", label=f"Width: {coherence:.1f}")
    ax[1].legend(loc="upper right", frameon=False)
    ax[1].set_ylim(bottom=0, top=1)
    ax[1].set_xticks(eigenindex)
    ax[1].set_xlabel(r"Eigenvalue index ($n$)")
    ax[1].set_ylabel(r"Eigenvalue ($\lambda_n$)")
    ax[1].set_title("Eigenspectrum")
    ax[1].grid()
