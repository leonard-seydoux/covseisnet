"""
Traces synchronization
======================

This example demonstrates how to synchronize a stream of traces using the
:func:`~covseisnet.stream.NetworkStream.synchronize` method. This method finds
the latest start time and the earliest end time among the traces in the
stream, and interpolates the traces between these times with a common sampling
interval. More information about the method can be found in the documentation.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import covseisnet as csn

# %%
# Read waveforms
# --------------
#
# This section reads an example stream of seismic data, which is shipped with
# ObsPy. The stream contains three traces, which are highpass at a very high
# frequency to see more details in the synchronization.

# Read the example stream (shipped with ObsPy)
stream = csn.read()

# Highpass filter the stream to better see the sync in high frequencies
stream.filter("highpass", freq=30)

# Print the original stream
print(stream)

# %%
# Desynchronize
# -------------
#
# This first section desynchronizes the traces in the stream, in order to
# demonstrate the synchronization method from the example stream. The traces
# are shifted in time by different amounts, and a small number of samples are
# collected for visualization.

# Collect a reference start time and sampling interval
starttime = stream[0].stats.starttime
sampling_interval = stream[0].stats.delta

# Desynchronize the traces
stream[1].stats.starttime = starttime + sampling_interval * 1.3
stream[2].stats.starttime = starttime + sampling_interval * 0.6

# Collect a small number of samples for visualization
sart_sample = 1000
n_samples = 20
for trace in stream:
    trace.data = trace.data[sart_sample : sart_sample + n_samples]

# Print the desynchronized stream
print(stream)

# %%
# Synchronize
# -----------
#
# We now synchronize the traces in the stream using the
# :func:`~covseisnet.stream.NetworkStream.synchronize` method. The method
# finds the latest start time and the earliest end time among the traces in
# the stream, and aligns the traces to these times with interpolation.

# Synchronize the traces
stream_sync = stream.copy()
stream_sync.synchronize()

# Print the synchronized stream
print(stream_sync)

# %%
# Compare traces
# --------------
#
# The synchronized traces are plotted alongside the original traces to compare
# the effect of the synchronization method. Note that several interpolation
# methods are available in the synchronization method. Chech the documentation
# for more information.

# Create figure
fig, axes = plt.subplots(
    3,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

# Loop over traces
for trace, synced, ax in zip(stream, stream_sync, axes):

    # Plot traces
    ax.plot(trace.times("matplotlib"), trace.data, ".-", label="Original")
    ax.plot(synced.times("matplotlib"), synced.data, ".-", label="Synced")

    # Local settings
    ax.grid()
    ax.set_title(trace.id, size="medium", weight="normal")

# Labels
axes[0].legend(loc="upper right")
axes[1].set_ylabel("Amplitude (counts)")

# Date formatting
xticks = mdates.AutoDateLocator()
xticklabels = mdates.ConciseDateFormatter(xticks)
axes[2].xaxis.set_major_locator(xticks)
axes[2].xaxis.set_major_formatter(xticklabels)