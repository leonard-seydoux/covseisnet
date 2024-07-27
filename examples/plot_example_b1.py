"""
Spatial coherence
=================

Spatial coherence on the Piton de la Fournaise volcano.

We here reproduce a part of the result published in :footcite:`seydoux_detecting_2016`.


The following example use a Fourier estimation window of 1 second and is
estimated over 5 consecutive windows.
"""

import numpy as np

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

# Read example stream
stream = csn.read("../data/unvervolc_sample.mseed")
stream = stream.select(station="UV1*")
# starttime = stream[0].stats.starttime
# endtime = starttime + 1200
# stream.cut(starttime, endtime)

# Get channels
channels = [trace.stats.channel for trace in stream]

# Pre-process stream
stream.detrend("linear")
stream.filter("highpass", freq=0.5)
# stream.taper(max_percentage=0.05)
# stream.whiten(window_duration_sec=50)

# %%
# Covariance matrix
# -----------------
#
# The covariance matrix is calculated using the method :func:`~covseisnet.covariance.calculate_covariance_matrix`. The method returns the times, frequencies, and covariances of the covariance matrix. Among the parameters of the method, the window duration and the number of windows are important to consider. The window duration is the length of the Fourier estimation window in seconds, and the number of windows is the number of windows to average to estimate the covariance matrix. We can then visualize the covariance matrix at a given time and frequency, and its corresponding eigenvalues.

times, frequencies, covariances = csn.calculate_covariance_matrix(
    stream, window_duration_sec=20, average=14
)

# Show covariance from first window and first frequency
t_index = 1
f_index = np.abs(frequencies - 1).argmin()
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

# Calculate coherence
coherence = covariances.coherence()

# Show
# sphinx_gallery_thumbnail_number = 2
csn.plot.stream_and_coherence(stream, times, frequencies, coherence)

# %%
# More about this result in the papers associated with the package, presented
# in the home of this documentation.
#
# .. footbibliography::
