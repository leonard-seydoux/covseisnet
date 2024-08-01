"""
Cross-correlation
=================

Pairwise cross-correlation in time domain.
"""

import os

import matplotlib.pyplot as plt

import covseisnet as csn


# sphinx_gallery_thumbnail_number = 2

# %%
# Read and pre-process stream
# ---------------------------
#
# The stream is read and pre-processed by detrending, tapering, and highpass
# filtering. Several other pre-processing methods are available in the
# :class:`~covseisnet.stream.NetworkStream` classs. The stream is then whitened
# using the method :func:`~covseisnet.stream.NetworkStream.whiten`. The method
# requires a window duration in seconds and a smooth length to smooth the
# spectral whitening.

# Path to the example stream
filepath_waveforms = "../data/undervolc_example.mseed"

# Download stream if not available
if not os.path.exists(filepath_waveforms):
    csn.data.download_undervolc_data()

# Read example stream
stream = csn.read(filepath_waveforms)
stream = stream.select(station="UV1*")
stream.filter("highpass", freq=0.1)
stream.normalize(method="smooth", smooth_length=1001)
stream.taper(max_percentage=0.01)


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
    stream, window_duration_sec=20, average=20, whiten="slice"
)

# Show covariance from sample window and frequency
t_index = 65
f_index = 100
csn.plot.covariance_matrix_modulus_and_spectrum(covariances[t_index, f_index])

# %%
# Spectral width
# --------------
#
# We here extract the coherence from the covariance matrix. The coherence is
# calculated using the method
# :func:`~covseisnet.covariance.CovarianceMatrix.coherence`. It can either
# measure the spectral width of the eigenvalue distribution at each frequency,
# or with applying the formula of the Neumann entropy.

frequency_band = 0.3, 0.7

# Calculate coherence
coherence = covariances.coherence(kind="spectral_width")

# Show
ax = csn.plot.stream_and_coherence(stream, times, frequencies, coherence)

# Indicate frequency band
ax[1].axhspan(*frequency_band, facecolor="none", edgecolor="k", clip_on=False)

# Save
ax[1].figure.savefig("coherence")

# %%
# Pairwise cross-correlation
# --------------------------

# Calculate cross-correlation
lags, pairs, cross_correlation = csn.calculate_cross_correlation_matrix(
    covariances
)

# Stack
cross_correlation = cross_correlation.mean(axis=1)
# Bandpass filter
cross_correlation.bandpass(frequency_band)
cross_correlation = cross_correlation.taper()
envelope = cross_correlation.envelope()
envelope_smooth = envelope.smooth(51)


# Get a given pair
i_pair = 3
pair = pairs[i_pair]
cross_correlation = cross_correlation[i_pair]
envelope = envelope[i_pair]
envelope_smooth = envelope_smooth[i_pair]

# Plot
fig, ax = plt.subplots()
ax.plot(lags, cross_correlation)
ax.plot(lags, envelope)
ax.plot(lags, envelope_smooth)
ax.grid()
ax.set_title(f"Cross-correlation between {pair}")
ax.set_xlabel("Lag time (s)")
fig.savefig("cross_correlation")
