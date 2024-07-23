"""
Single-station covariance matrix
================================

Interchannel spectral covariance matrix calculated from the example trace
available in obspy.

We consider the continuous seismograms at the station RJOB from the BW seismic
network. The data is available in the obspy example data, readable without any
argument. In this case, the covariance matrix is calculated between the three
channels of the station RJOB, and has three components. 

The following example use a Fourier estimation window of 1 second and is
estimated over 5 consecutive windows.
"""

import matplotlib.pyplot as plt
import numpy as np

import covseisnet as csn

# Read example stream
stream = csn.read()

# Pre-process stream
stream.detrend("linear")
stream.taper(max_percentage=0.05)
stream.filter("highpass", freq=2)


# Get channels
channels = [trace.stats.channel for trace in stream]

# Calculate covariance matrix
times, frequencies, covariances = csn.calculate_covariance_matrix(
    stream,
    window_duration_sec=1.0,
    average=4,
)

# Calculate coherence
coherence = covariances.coherence()

# show covariance from first window and first frequency
covariance_show = np.random.rand(3, 3)

# Show
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)

# Show trace
for index, trace in enumerate(stream):
    waveform = trace.data
    waveform /= np.max(np.abs(waveform))
    ax[0].plot(trace.times(), waveform + index, label=trace.stats.channel)
mappable = ax[1].pcolormesh(
    times, frequencies, coherence.T, cmap="magma_r", vmin=0, vmax=1
)

# Labels
ax[0].grid()
ax[0].set_ylabel("Amplitude (counts)")
ax[1].set_yscale("log")
ax[1].set_ylim(frequencies[1], frequencies[-1] / 2)
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Frequency (Hz)")
plt.colorbar(mappable).set_label("Covariance matrix\nspectral width")

# mappable = ax.matshow(covariance_show, vmin=0, vmax=1, cmap="RdPu")

# Axes
# channels = [trace.stats.channel for trace in stream]
# ax.set_xticks(range(len(stream)), labels=channels)
# ax.set_yticks(range(len(stream)), labels=channels)

# # Labels
# ax.set_title("Single-station channel-wise covariance")
# plt.colorbar(mappable).set_label("Covariance modulus")
