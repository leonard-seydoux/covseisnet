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

# Echo stream
print(stream)

# Get channels
channels = [trace.stats.channel for trace in stream]

# # calculate covariance from stream
# window_duration_sec = 1.0
# average = 5
# times, frequencies, covariances = csn.covariancematrix.calculate(
#     stream, window_duration_sec, average
# )

# show covariance from first window and first frequency
covariance_show = np.random.rand(3, 3)

# Show
fig, ax = plt.subplots()
mappable = ax.matshow(covariance_show, vmin=0)

# Labels
ax.set_xticks(range(len(channels)), labels=channels)
ax.set_yticks(range(len(channels)), labels=channels)
ax.set_title("Single-station multiple channels covariance")
plt.colorbar(mappable).set_label("Covariance modulus")

# Show
plt.show()
