"""
Single-station coherence
========================

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

# %%
# Read and pre-process stream
# ---------------------------
#
# The stream is read and pre-processed by detrending, tapering, and highpass
# filtering. Several other pre-processing methods are available in the
# :class:`~covseisnet.stream.NetworkStream` class.

# Read example stream
stream = csn.read()

# Pre-process stream
stream.detrend("linear")
stream.taper(max_percentage=0.05)
stream.filter("highpass", freq=2)

# Get channels
channels = [trace.stats.channel for trace in stream]

# %%
# Calculate the covariance matrix
# -------------------------------
#
# The covariance matrix is calculated using the method :func:`~covseisnet.covariance.calculate_covariance_matrix`. The method returns the times, frequencies, and covariances of the covariance matrix. Among the parameters of the method, the window duration and the number of windows are important to consider. The window duration is the length of the Fourier estimation window in seconds, and the number of windows is the number of windows to average to estimate the covariance matrix. We can then visualize the covariance matrix at a given time and frequency, and its corresponding eigenvalues.

times, frequencies, covariances = csn.calculate_covariance_matrix(
    stream,
    window_duration_sec=1.0,
    average=5,
)

# Show covariance from first window and first frequency
covariance = covariances[0, 0]
covariance /= np.max(np.abs(covariance))

# Calculate eigenvalues
eigenvalues = covariance.eigenvalues(norm=sum)

# Show
fig, ax = plt.subplots(ncols=2, figsize=(6, 2.7), constrained_layout=True)
mappable = ax[0].matshow(np.abs(covariance), cmap="GnBu", vmin=0)
ax[0].set_xticks(range(len(stream)), labels=channels)
ax[0].set_yticks(range(len(stream)), labels=channels)
ax[0].xaxis.set_ticks_position("bottom")
ax[0].set_xlabel(r"Channel $i$")
ax[0].set_ylabel(r"Channel $j$")
ax[0].set_title("Covariance matrix")
ax[1].plot(eigenvalues, marker="o")
ax[1].set_ylim(bottom=0, top=1)
ax[1].set_xticks(range(len(eigenvalues)))
ax[1].set_xlabel(r"Eigenvalue index ($n$)")
ax[1].set_ylabel(r"Eigenvalue ($\lambda_n$)")
ax[1].set_title("Eigenspectrum")
ax[1].grid()
plt.colorbar(mappable).set_label("Covariance modulus")

# %%
# Calculate coherence
# -------------------
#
# We here extract the coherence from the covariance matrix. The coherence is
# calculated using the method
# :func:`~covseisnet.covariance.CovarianceMatrix.coherence`. It can either
# measure the spectral width of the eigenvalue distribution at each frequency,
# or with applying the formula of the Neumann entropy.

# Calculate coherence
coherence = covariances.coherence()


# Show
# sphinx_gallery_thumbnail_number = 2
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)

# Show traces
for index, trace in enumerate(stream):
    waveform = trace.data
    waveform /= np.max(np.abs(waveform)) * 2
    ax[0].plot(trace.times(), waveform + index, color="k")

# Show coherence
mappable = ax[1].pcolormesh(
    times,
    frequencies,
    coherence.T,
    cmap="magma_r",
    vmin=0,
    vmax=1,
)

# Labels
ax[0].set_title("Normalized seismograms")
ax[0].grid()
ax[0].set_yticks(range(len(stream)), labels=channels)
ax[0].set_ylabel("Normalized amplitude")
ax[1].set_title("Coherence")
ax[1].set_yscale("log")
ax[1].set_ylim(frequencies[1], frequencies[-1] / 2)
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Frequency (Hz)")
plt.colorbar(mappable).set_label("Covariance matrix\nspectral width")

# %%
# More about this result in the papers associated with the package, presented
# in the home of this documentation.
