"""Test of the ArrayStream class."""

import obspy

from covseisnet import NetworkStream
from covseisnet.stream import get_minimal_time_extent


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
    assert stream.stats.npts == 701


def test_cut_with_duration():
    """Test trimming the stream with an endtime."""
    stream = NetworkStream.read()
    stream.cut("2009-08-24 00:20:05", duration=7)
    assert stream.stats.npts == 701


def test_sampling_rate():
    """Test the sample rate of the stream."""
    stream = NetworkStream.read()
    assert stream.stats.sampling_rate == 100.0


def test_minimimal_extent():
    """Test the minimal extent of the stream."""
    stream = obspy.read()
    start, end = get_minimal_time_extent(stream)
    assert start == stream[0].stats.starttime
    assert end == stream[0].stats.endtime
