"""Utilities to simplify the examples.

This module contains functions to simplify the examples in the documentation.

Made by Leonard Seydoux in 2024.
"""

import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(6, 3))

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


def trace_and_spectrogram(trace):
    """Plot a trace and its spectrogram.

    Arguments
    ---------
    trace : obspy.Trace
        The trace to plot.
    """
    # Create figure
    fig, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(6, 3))

    # Extract data
    times = trace.times()
    waveform = trace.data

    # Calculate spectrogram
    f, t, Sxx = csn.stream.spectrogram(
        waveform, fs=1 / trace.stats.delta, nperseg=256
    )

    # Plot trace
    ax[0].plot(times, waveform)
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid()

    # Plot spectrogram
    ax[1].pcolormesh(t, f, Sxx, shading="gouraud")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].grid()
