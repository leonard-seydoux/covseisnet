"""Test of the ArrayStream class."""

from covseisnet import read
from covseisnet import NetworkStream


def test_network_stream_instance():
    """Read class method returns an instance of the NetworkStream class."""
    stream = NetworkStream.read()
    assert isinstance(stream, NetworkStream)


def test_stats_attribute():
    """The station property returns a list of stations."""
    stream = NetworkStream.read()
    assert len(stream) == len([trace.stats for trace in stream])


def test_cut_with_endtime():
    """Test trimming the stream with an endtime."""
    stream = NetworkStream.read()
    stream.cut("2009-08-24 00:20:05", "2009-08-24 00:20:12")
    assert stream.npts == 701


def test_cut_with_duration():
    """Test trimming the stream with an endtime."""
    stream = NetworkStream.read()
    stream.cut("2009-08-24 00:20:05", duration=7)
    assert stream.npts == 701


def test_minimimal_extent():
    """Test the minimal extent of the stream.

    This test should not fail because the traces are synced in the example.
    """
    stream = NetworkStream.read()
    start, end = stream.time_extent()
    assert start == stream[0].stats.starttime
    assert end == stream[0].stats.endtime


def test_read_with_function():
    """Test the read function with a function."""
    stream_1 = NetworkStream.read()
    stream_2 = read()
    assert stream_1 == stream_2


def test_whiten_method():
    """Test the whiten method."""
    stream = NetworkStream.read()
    stream.whiten(window_duration=2)


def test_whiten_method_smooth():
    """Test the whiten method with smoothing."""
    stream = NetworkStream.read()
    stream.whiten(2, smooth_length=5, smooth_order=3, epsilon=1e-5)


def test_time_normalize_method():
    """Test the time_normalize method."""
    stream = NetworkStream.read()
    stream.time_normalize()


def test_time_normalize_method_smooth():
    """Test the time_normalize method."""
    stream = NetworkStream.read()
    stream.time_normalize(method="smooth", smooth_length=5, smooth_order=3)


def test_synced():
    """Test if the synced flag works."""
    stream = NetworkStream.read()
    assert stream.synced
    stream[0].stats.starttime += 1
    assert not stream.synced


def test_synchronize():
    stream = NetworkStream.read()
    assert stream.synced
    stream[0].stats.starttime += 1
    assert not stream.synced
    stream.synchronize()
    assert stream.synced


def test_equal_length():
    stream = NetworkStream.read()
    assert stream.equal_length


def test_equal_length_false():
    stream = NetworkStream.read()
    stream[0].trim(stream[0].stats.starttime, stream[0].stats.starttime + 10)
    assert not stream.equal_length


def test_equal_rates():
    stream = NetworkStream.read()
    assert stream.equal_rates
    stream[0].resample(99.0)
    assert not stream.equal_rates


def test_sampling_rate():
    stream = NetworkStream.read()
    assert stream.sampling_rate == 100.0


def test_npts():
    stream = NetworkStream.read()
    assert stream.npts == 3000


def test_stats():
    stream = NetworkStream.read()
    assert stream.stats() == stream[0].stats
    assert stream.stats(key="sampling_rate") == stream[0].stats.sampling_rate
