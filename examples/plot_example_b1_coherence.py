"""
Spatial coherence
=================

Spatial coherence on the Piton de la Fournaise volcano.

We here reproduce a part of the result published in :footcite:`seydoux_detecting_2016`.


The following example use a Fourier estimation window of 1 second and is
estimated over 5 consecutive windows.
"""

import os

import covseisnet as csn


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
stream.filter("highpass", freq=0.5)
stream.normalize(method="smooth", smooth_length=1001)
stream.taper(max_percentage=0.01)


# %%
# Covariance matrix
# -----------------
#
# The covariance matrix is calculated using the method :func:`~covseisnet.covariance.calculate_covariance_matrix`. The method returns the times, frequencies, and covariances of the covariance matrix. Among the parameters of the method, the window duration and the number of windows are important to consider. The window duration is the length of the Fourier estimation window in seconds, and the number of windows is the number of windows to average to estimate the covariance matrix. We can then visualize the covariance matrix at a given time and frequency, and its corresponding eigenvalues.

# Calculate covariance matrix
times, frequencies, covariances = csn.calculate_covariance_matrix(
    stream, window_duration_sec=20, average=20, whiten="slice"
)

# Show covariance from sample window and frequency
csn.plot.covariance_matrix_modulus_and_spectrum(covariances[42, 42])

# %%
# Spectral width
# --------------
#
# We here extract the coherence from the covariance matrix. The coherence is
# calculated using the method
# :func:`~covseisnet.covariance.CovarianceMatrix.coherence`. It can either
# measure the spectral width of the eigenvalue distribution at each frequency,
# or with applying the formula of the Neumann entropy.

# Calculate coherence
coherence = covariances.coherence(kind="spectral_width")

# Show
# sphinx_gallery_thumbnail_number = 2
csn.plot.stream_and_coherence(stream, times, frequencies, coherence, f_min=0.5)


# %%
# More about this result in the papers associated with the package, presented
# in the home of this documentation.
#
# .. footbibliography::
