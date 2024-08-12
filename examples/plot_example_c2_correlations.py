"""
Cross-correlation
=================

Pairwise cross-correlation in time domain.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import covseisnet as csn


# sphinx_gallery_thumbnail_number = 2

# %%
# Read and pre-process stream
# ---------------------------

# Path to the example stream
filepath_waveforms = (
    "/Users/seydoux/Github/covseisnet/data/usarray_example.mseed"
)

# Read example stream
stream = csn.read(filepath_waveforms)
stream.detrend("linear")

stream.filter("highpass", freq=0.001)
stream.synchronize()
stream.time_normalize(method="smooth", smooth_length=61)
stream.plot()

stream.assign_coordinates(
    "/Users/seydoux/Documents/Work/testing_covseisnet/inv"
)

# %%
# Covariance matrix
# -----------------
#
# The covariance matrix is calculated using the method
# :func:`~covseisnet.covariance.calculate_covariance_matrix`. The method
# returns the times, frequencies, and covariances of the covariance matrix.
# Among the parameters of the method, the window duration and the number of
# windows are important to consider. The window duration is the length of the
# Fourier estimation window in seconds, and the number of windows is the
# number of windows to average to estimate the covariance matrix.
#
# We can then visualize the covariance matrix at a given time and frequency,
# and its corresponding eigenvalues.

# Calculate covariance matrix
times, frequencies, covariances = csn.calculate_covariance_matrix(
    stream, window_duration=500, average=2000, whiten="window"
)

# %%
# Spectral width
# --------------
#
# We here extract the coherence from the covariance matrix. The coherence is
# calculated using the method
# :func:`~covseisnet.covariance.CovarianceMatrix.coherence`. It can either
# measure the spectral width of the eigenvalue distribution at each frequency,
# or with applying the formula of the Neumann entropy.


frequency_band = 1 / 70, 1 / 20

# Calculate coherence
coherence = covariances.coherence(kind="spectral_width")

# Show
ax = csn.plot.stream_and_coherence(
    stream, times, frequencies, coherence, trace_factor=1e-1
)

# Indicate frequency band
ax[1].axhspan(*frequency_band, facecolor="none", edgecolor="w", clip_on=False)

# %%
# Pairwise cross-correlation
# --------------------------


# Calculate cross-correlation
lags, pairs, cross_correlation = csn.calculate_cross_correlation_matrix(
    covariances
)

# Get inter-station distance
pairs = np.array(pairs)
distances = np.array(csn.pairwise_distances(cross_correlation.stats))

# Bandpass filter
cross_correlation = cross_correlation.mean(axis=1)
cross_correlation.bandpass(frequency_band)
cross_correlation = cross_correlation.taper()

# Plot
fig, ax = plt.subplots()
for i_pair, pair in enumerate(pairs):
    cc = cross_correlation[i_pair] / abs(cross_correlation[i_pair]).max() * 30
    ax.plot(lags, cc + distances[i_pair], color="C0", lw=0.8)

# Plot some velocity
ax.axline((0, 0), slope=3.5, color="C1", label="3.5 km/s")
ax.axline((0, 0), slope=-3.5, color="C1")

ax.legend()
ax.grid()
ax.set_title(
    f"Cross-correlation functions between {1 / frequency_band[0]} and {1 / frequency_band[1]} seconds"
)
ax.set_xlabel("Lag time (s)")
ax.set_ylabel("Pairwise distance (km)")
