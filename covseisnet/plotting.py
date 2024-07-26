"""Utilities to simplify the examples."""

import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq


def plot_trace_and_spectrum(trace):
    """Plot a trace and its spectrum."""
    # Create figure
    _, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(6, 3))

    # Plot trace
    times = trace.times()
    waveform = trace.data
    ax[0].plot(times, waveform)
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid()

    # Calculate spectrum
    spectrum = rfft(waveform)
    frequencies = rfftfreq(len(waveform), trace.stats.delta)
    ax[1].loglog(frequencies, abs(spectrum))
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Spectrum")
    ax[1].grid()
