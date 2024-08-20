"""Test of the ArrayStream class."""

import covseisnet as csn


def test_correlation_matrix_instance():
    """Check various instances of the CovarianceMatrix class."""
    stream = csn.NetworkStream.read()
    *_, covariance = csn.calculate_covariance_matrix(
        stream, window_duration=2, average=10
    )
    *_, correlation = csn.calculate_cross_correlation_matrix(covariance)
    assert isinstance(correlation, csn.CrossCorrelationMatrix)
