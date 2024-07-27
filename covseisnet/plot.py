"""Utilities to simplify the examples.

This module contains functions to simplify the examples in the documentation.

Made by Leonard Seydoux in 2024.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

import covseisnet as csn


def trace_and_spectrum(trace):
    """Plot a trace and its spectrum.

    Arguments
    ---------
    trace : obspy.Trace
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


def trace_and_spectrogram(trace, **kwargs):
    """Plot a trace and its spectrogram.

    This function is deliberately simple and does not allow to customize the
    spectrogram plot. For more advanced plotting, you should consider creating
    a derived function.

    Arguments
    ---------
    trace : obspy.Trace
        The trace to plot.
    **kwargs
        Additional arguments to pass to :func:`~covseisnet.calculate_spectrogram`.
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
