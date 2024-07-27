"""
This module contains functions to simplify the examples in the documentation, 
mostly plotting functions, but also to provide basic tools to quickly visualize
data and results from this package.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obspy
from scipy.fft import rfft, rfftfreq

import covseisnet as csn


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
    times, frequencies, spectrogram = csn.calculate_spectrogram(
        trace, **kwargs
    )

    # Remove zero frequencies
    frequencies = frequencies[1:]
    spectrogram = spectrogram[1:, :]

    # Make sure the spectrogram is in dB
    spectrogram = 20 * np.log10(spectrogram)

    # Plot trace
    ax[0].plot(trace.times(), trace.data)
    ax[0].set_ylabel("Amplitude")
    ax[0].grid()

    # Plot spectrogram
    mappable = ax[1].pcolormesh(times, frequencies, spectrogram)
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].grid()
    ax[1].set_yscale("log")

    # Colorbar
    colorbar = plt.colorbar(mappable, ax=ax[1])
    colorbar.set_label("Spectral energy (dB)")


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
    global_max = np.max([np.max(np.abs(trace.data)) for trace in stream])
    for index, trace in enumerate(stream):
        waveform = trace.data
        waveform /= global_max
        ax[0].plot(
            trace.times("matplotlib"), waveform + index, color="k", lw=0.5
        )

    # Get times in matplotlib format
    starttimes = stream[0].stats.starttime.datetime
    times = mdates.date2num(starttimes) + times / 86400.0

    # Show coherence
    mappable = ax[1].pcolormesh(
        times,
        frequencies,
        coherence.T,
        cmap="magma_r",
        vmin=0,
        **kwargs,
    )

    # Labels
    stations = stream.stations
    ax[0].set_title("Normalized seismograms")
    ax[0].grid()
    ax[0].set_yticks(range(len(stations)), labels=stations, fontsize="small")
    ax[0].set_ylabel("Normalized amplitude")

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
