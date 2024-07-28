"""
Compare pre-processing
======================

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
starttime = stream[0].stats.starttime
endtime = starttime + 3600
stream.cut(starttime, endtime)

# Get channels
channels = [trace.stats.channel for trace in stream]

# Pre-process stream
stream.detrend("linear")
stream.filter("highpass", freq=0.5)
# stream.taper(max_percentage=0.05)

# %%
# Covariance matrix
# -----------------
#
# The covariance matrix is calculated using the method :func:`~covseisnet.covariance.calculate_covariance_matrix`. The method returns the times, frequencies, and covariances of the covariance matrix. Among the parameters of the method, the window duration and the number of windows are important to consider. The window duration is the length of the Fourier estimation window in seconds, and the number of windows is the number of windows to average to estimate the covariance matrix. We can then visualize the covariance matrix at a given time and frequency, and its corresponding eigenvalues.


# No pre-processing
case_1 = "Original"
stream_1 = stream.copy()

# Pre-process stream with temporal normalization
case_2 = "Temporal normalization"
stream_2 = stream.copy()
stream_2.normalize(smooth_length=11)

# Pre-process stream with whitening
case_3 = "Whitening"
stream_3 = stream.copy()
stream_3.whiten(window_duration_sec=150, smooth_length=11)

# Pre-process stream with whitening and temporal normalization
case_4 = "Whitening and temporal normalization"
stream_4 = stream.copy()
stream_4.normalize(smooth_length=11)
stream_4.whiten(window_duration_sec=150, smooth_length=11)


# Calculate covariance matrix
for stream, case in zip(
    [stream_1, stream_2, stream_3, stream_4], [case_1, case_2, case_3, case_4]
):
    times, frequencies, covariances = csn.calculate_covariance_matrix(
        stream, window_duration_sec=20, average=15
    )

    # Calculate coherence
    coherence = covariances.coherence()

    # Show
    # sphinx_gallery_thumbnail_number = 2
    csn.plot.stream_and_coherence(stream, times, frequencies, coherence)
