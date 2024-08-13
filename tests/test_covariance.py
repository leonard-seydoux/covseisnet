"""Test of the ArrayStream class."""

import numpy as np

import covseisnet as csn


def test_covariance_matrix_instance():
    """Check various instances of the CovarianceMatrix class."""
    x = np.eye(3)
    covariance = csn.CovarianceMatrix(x)
    sub_covariance = covariance[1]
    assert covariance.shape == (3, 3)
    assert isinstance(covariance, csn.CovarianceMatrix)
    assert isinstance(sub_covariance, csn.CovarianceMatrix)


def test_covariance_matrix_operations():
    x = np.eye(3) + 1j * np.eye(3).T
    covariance = csn.CovarianceMatrix(x)
    assert isinstance(covariance.sum(axis=1), csn.CovarianceMatrix)
    assert isinstance(covariance.mean(axis=0), csn.CovarianceMatrix)


def test_covariance_matrix_stats():
    # Calculate covariance
    stream = csn.read()
    times, frequencies, covariances = csn.calculate_covariance_matrix(
        stream, window_duration=5, average=5
    )
    # Assertions
    assert covariances.shape == (len(times), len(frequencies), 3, 3)
    assert covariances.stats == [trace.stats for trace in stream]
    # Look if sliced covariances have also stats
    assert covariances[0].stats == covariances.stats


def test_flat():
    # Calculate covariance
    stream = csn.read()
    *_, covariances = csn.calculate_covariance_matrix(
        stream, window_duration=5, average=5
    )
    # Assertions
    assert covariances.flat().shape == (
        covariances.shape[0] * covariances.shape[1],
        len(stream),
        len(stream),
    )


def test_triu():
    # Calculate covariance
    stream = csn.read()
    *_, covariances = csn.calculate_covariance_matrix(
        stream, window_duration=5, average=5
    )

    # Assertions
    assert covariances.triu().shape == (
        covariances.shape[0],
        covariances.shape[1],
        6,
    )
