"""
Temporal normalization
======================

This example shows how to perform a temporal normalization of traces with
different methods. 
"""

import matplotlib.pyplot as plt

import covseisnet as csn

# %%
# Read waveforms
# --------------
#
# This section reads an example stream of seismic data, which is shipped with
# ObsPy. The stream contains three traces.

# Read the example stream (shipped with ObsPy)
stream = csn.read()

# Highpass filter the stream to better see the sync in high frequencies
stream.filter("highpass", freq=1.0)

# Print the original stream
print(stream)

# %%
# Temporal normalization
# ----------------------
#
# This section normalizes the traces in the stream with different methods. The
# methods are applied to the stream, and the normalized traces are stored in a
# list. See\ :footcite:`bensen2007processing` for more information on the normalization

# Initialize the list of normalized streams
normalization_methods = ["onebit", "smooth"]
normalized_streams = []

# Loop over the normalization methods
for method in normalization_methods:

    # Apply the normalization
    normalized_stream = stream.copy()
    normalized_stream.normalize(method=method)

    # Append to the list of normalized streams
    normalized_streams.append(normalized_stream)

# %%
# Comparison
# ----------
#
# This section compares the original stream with the normalized streams. The
# traces are plotted in a figure, where the original stream is plotted first,
# and the normalized streams are plotted below.

# Concatenate the original stream with the normalized streams
all_streams = [stream] + normalized_streams
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

# Set the x-axis label
axes[-1].set_xlabel("Time (seconds)")

# Show the figure
plt.show()

# %%
# References
# ----------
#
# .. footbibliography::
