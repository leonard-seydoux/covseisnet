"""Test of the ArrayStream class."""

import pytest

from covseisnet import NetworkStream


def test_stream_instance():
    """Read class method returns an instance of the NetworkStream class."""
    stream = NetworkStream.read()
    assert isinstance(stream, NetworkStream)


def test_stream_station_property():
    """The station property returns a list of stations."""
    stream = NetworkStream.read()
    assert len(stream.ids) == len(stream)


def test_cut_with_endtime():
    """Test trimming the stream with an endtime."""
    stream = NetworkStream.read()
    stream.cut("2009-08-24 00:20:05", "2009-08-24 00:20:12")
    assert stream[0].stats.npts == 701


def test_cut_with_duration():
    """Test trimming the stream with an endtime."""
    stream = NetworkStream.read()
    stream.cut("2009-08-24 00:20:05", duration_sec=7)
    assert stream[0].stats.npts == 701


# # Check that there is only one station in the stream
# stations = stream.stations
# assert len(stations) == 1

# # Check that there are three channels in the stream
# channels = stream.channels
# assert len(channels) == 3
