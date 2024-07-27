"""
Spectral whitening
==================

This example show the effect of spectral whitening on a stream of traces.
The stream is read from the obspy example data, and the whitening is performed
with the method :func:`~covseisnet.stream.NetworkStream.whiten`. The method
applies a Fourier transform to the traces, divides the spectrum of the traces
by the modulus of the spectrum (or a smooth version of it), and then applies the
inverse Fourier transform to the traces.
"""

# sphinx_gallery_thumbnail_number = 2

import covseisnet as csn

WINDOW_DURATION = 2

# %%
# Read the example stream (shipped with ObsPy)
# --------------------------------------------
#
# The stream is read from the obspy example data, and is available without any
# argument. The stream is then plotted to visualize the traces.

stream = csn.read()

# Plot trace and corresponding spectrum
csn.plot.trace_and_spectrogram(stream[0], window_duration_sec=WINDOW_DURATION)

# %%
# Spectral whitening without smoothing
# ------------------------------------
#
# The spectral whitening is applied to the stream using the method
# :func:`~covseisnet.stream.NetworkStream.whiten`. The method applies a
# Fourier transform to the traces, divides the spectrum of the traces by the
# modulus of the spectrum, and then applies the inverse Fourier transform to
# the traces.


whitened_stream = stream.copy()
whitened_stream.whiten(window_duration_sec=WINDOW_DURATION, smooth_length=0)

# Plot whitened trace and corresponding spectrum
csn.plot.trace_and_spectrogram(
    whitened_stream[0],
    window_duration_sec=WINDOW_DURATION,
)

# %%
# Spectral whitening with smoothing
# ---------------------------------
#
# The spectral whitening is applied to the stream using the method
# :func:`~covseisnet.stream.NetworkStream.whiten`. The method applies a Fourier
# transform to the traces, divides the spectrum of the traces by a smooth
# version of the modulus of the spectrum, and then applies the inverse Fourier
# transform. The smoothing is performed with a Savitzky-Golay filter, with a
# window length of 31 frequency bins.

whitened_stream = stream.copy()
whitened_stream.whiten(
    window_duration_sec=WINDOW_DURATION,
    smooth_length=31,
)

# Plot whitened trace and corresponding spectrum
csn.plot.trace_and_spectrogram(
    whitened_stream[0],
    window_duration_sec=WINDOW_DURATION,
)
