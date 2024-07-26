"""
Stream whitening
================

This example show the effect of spectral whitening on a stream of traces.
The stream is read from the obspy example data, and the whitening is performed
with the method :func:`~covseisnet.stream.NetworkStream.whiten`. The method
applies a Fourier transform to the traces, divides the spectrum of the traces
by the modulus of the spectrum (or a smooth version of it), and then applies the
inverse Fourier transform to the traces.
"""

import matplotlib.pyplot as plt

import covseisnet as csn

# %%
# Read the example stream (shipped with ObsPy)
# --------------------------------------------
#
# The stream is read from the obspy example data, and is available without any
# argument. The stream is then plotted to visualize the traces.

stream = csn.read()

# Plot trace and corresponding spectrum
csn.plotting.plot_trace_and_spectrum(stream[0])

# %%
# Spectral whitening on a small window
# ------------------------------------
#
# The spectral whitening is applied to the stream using the method
# :func:`~covseisnet.stream.NetworkStream.whiten`. The method applies a Fourier
# transform to the traces, divides the spectrum of the traces by the modulus of
# the spectrum (or a smooth version of it), and then applies the inverse Fourier
# transform to the traces. The whit

whitened_stream = stream.copy()
whitened_stream.whiten(window_duration_sec=2)

# Plot whitened trace and corresponding spectrum
csn.plotting.plot_trace_and_spectrum(whitened_stream[0])

# %%
# Spectral whitening on the entire signal
# ---------------------------------------
#
# The spectral whitening is applied to the stream using the method
# :func:`~covseisnet.stream.NetworkStream.whiten`. The method applies a Fourier
# transform to the traces, divides the spectrum of the traces by the modulus of
# the spectrum (or a smooth version of it), and then applies the inverse Fourier
# transform to the traces. The whit

whitened_stream = stream.copy()
whitened_stream.whiten(window_duration_sec=20)

# Plot whitened trace and corresponding spectrum
csn.plotting.plot_trace_and_spectrum(whitened_stream[0])
