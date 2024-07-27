"""
Temporal normalization
======================

This example shows how to perform a temporal normalization of traces with
different methods. 
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

# %%
# Apply temporal normalizations
# -----------------------------

normalization_methods = ["onebit", "smooth"]
streams_normalized = []
for method in normalization_methods:

    # Apply the normalization
    normalized_stream = stream.copy()
    normalized_stream.normalize(method=method)

    # Append to the list of normalized streams
    streams_normalized.append(normalized_stream)

# %%
# Compare the original and normalized traces
# -------------------------------------------

# Concatenate the original stream with the normalized streams
all_streams = [stream] + streams_normalized
all_titles = ["Original"] + normalization_methods

# Create gigure
fig, axes = plt.subplots(
    nrows=len(all_streams), sharex=True, constrained_layout=True
)

# Loop over the streams
for ax, traces, title in zip(axes, all_streams, all_titles):
    ax.plot(traces[0].times(), traces[0].data, label=traces[0].id)
    ax.set_title(title.title())
    ax.grid()
    ax.set_ylabel("Amplitude")

axes[-1].set_xlabel("Time (seconds)")
