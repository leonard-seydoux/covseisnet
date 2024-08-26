"""Test of the ArrayStream class."""

import pickle
import pytest

import covseisnet as csn
import numpy as np


def test_correlation_matrix_instance():
    """Check various instances of the CovarianceMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    assert isinstance(correlation, csn.CrossCorrelationMatrix)


def test_pickle_persistance():
    """Tests on the pickle persistance of the CovarianceMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    corr_read = pickle.loads(pickle.dumps(correlation))
    assert np.allclose(correlation.data, corr_read.data)
    assert hasattr(corr_read, "stats")
    assert hasattr(corr_read, "stft")


def test_taper():
    """Test on the taper method of the CrossCorrelationMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    correlation = correlation.taper()
    assert hasattr(correlation, "stats")
    assert correlation[0, 0, 0] == 0.0


def test_bandpass():
    """Test on the bandpass method of the CrossCorrelationMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    correlation = correlation.bandpass(
        frequency_band=(0.1, 0.3), filter_order=2
    )
    assert hasattr(correlation, "stats")


def test_bandpass_error():
    """Test bandpass out of bounds error."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    with pytest.raises(ValueError):
        correlation.bandpass(frequency_band=(30, 50))


def test_sampling_rate():
    """Test on the sampling_rate method of the CrossCorrelationMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    assert correlation.sampling_rate == stream[0].stats.sampling_rate


def test_envelope():
    """Test on the envelope method of the CrossCorrelationMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    correlation = correlation.envelope()
    correlation = correlation.taper()
    assert hasattr(correlation, "stats")
    assert correlation[0, 0, 0] == 0.0


def test_smooth():
    """Test on the smooth method of the CrossCorrelationMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    correlation = correlation.smooth(sigma=2)
    assert hasattr(correlation, "stats")


def test_flat():
    """Test on the flat method."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    lags, pairs, correlation = csn.calculate_cross_correlation_matrix(
        covariance
    )
    if correlation.stats is not None:
        assert correlation.flat().shape == (
            len(pairs) * correlation.shape[1],
            len(lags),
        )
