"""Test of the ArrayStream class."""

import covseisnet as csn

# Create an instance of the NetworkStream class
stream = csn.read()
assert isinstance(stream, csn.NetworkStream)

# Check that there is only one station in the stream
stations = stream.stations
assert len(stations) == 1

# Check that there are three channels in the stream
channels = stream.channels
assert len(channels) == 3
