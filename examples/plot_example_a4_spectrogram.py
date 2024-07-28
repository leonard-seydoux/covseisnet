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


# %%
# Read waveforms
# --------------
#
# This section reads an example stream of seismic data, which is shipped with
# ObsPy. The stream contains three traces.

# Read the example stream (shipped with ObsPy)
stream = csn.read("/Users/seydoux/Desktop/undervolc_20.mseed")
stream = stream.select(station="UV05")

# Extract the first trace, and preprocess it
trace = stream[0]
trace.filter("highpass", freq=0.4)

# Plot trace and corresponding spectrum
ax = csn.plot.trace_and_spectrogram(
    stream[0], window_duration_sec=1000, f_min=0.5
)
ax[0].figure.savefig("trace_and_spectrogram.png")
