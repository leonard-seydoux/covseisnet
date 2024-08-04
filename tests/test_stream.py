"""Test of the ArrayStream class."""

import pytest

from covseisnet import NetworkStream


def test_stream_instance():
    """Test that the read function returns a NetworkStream instance."""
    stream = NetworkStream.read()
    assert isinstance(stream, NetworkStream)


# # Create an instance of the NetworkStream class
# stream = read()
# assert isinstance(stream, NetworkStream)

# # Check that there is only one station in the stream
# stations = stream.stations
# assert len(stations) == 1

# # Check that there are three channels in the stream
# channels = stream.channels
# assert len(channels) == 3
