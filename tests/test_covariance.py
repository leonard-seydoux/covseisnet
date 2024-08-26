"""Test of the ArrayStream class."""

import pickle
import pytest

import numpy as np

import covseisnet as csn


def test_covariance_matrix_construction_errors():
    """Not enought dimensions."""
    with pytest.raises(ValueError):
        csn.CovarianceMatrix(np.array(1))
    with pytest.raises(ValueError):
        csn.CovarianceMatrix(np.array([1, 2]))
    with pytest.raises(ValueError):
        csn.CovarianceMatrix(np.random.randn(3, 3, 3, 3, 3))


def test_covariance_matrix_instance():
    """Check various instances of the CovarianceMatrix class."""
    x = np.eye(3)
    covariance = csn.CovarianceMatrix(x)
    sub_covariance = covariance[1]
    assert covariance.shape == (3, 3)
    assert isinstance(covariance, csn.CovarianceMatrix)
    assert isinstance(sub_covariance, csn.CovarianceMatrix)


def test_covariance_matrix_operations():
    """Check some operations of the CovarianceMatrix class."""
    # Build a complex Hermitian matrix
    x = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    x = x @ x.T.conj()
    covariance = csn.CovarianceMatrix(x)
    assert isinstance(covariance.sum(axis=1), csn.CovarianceMatrix)
    assert isinstance(covariance.mean(axis=0), csn.CovarianceMatrix)


def test_covariance_matrix_stats():
    """Tests on the stats attribute of the CovarianceMatrix class."""
    # Calculate covariance
    stream = csn.read()
    times, frequencies, covariances = csn.calculate_covariance_matrix(
        stream,
        window_duration=5,
        average=5,
    )

    # Assertions
    assert covariances.shape == (len(times), len(frequencies), 3, 3)
    assert covariances.stats == [trace.stats for trace in stream]

    # Look if sliced covariances have also stats
    assert covariances[0].stats == covariances.stats


def test_flat():
    """Tests on the flat attribute of the CovarianceMatrix class."""
    # Calculate covariance
    stream = csn.read()
    *_, covariances = csn.calculate_covariance_matrix(
        stream, window_duration=5, average=5
    )

    # Assertions
    assert covariances.flat.shape == (
        covariances.shape[0] * covariances.shape[1],
        len(stream),
        len(stream),
    )


def test_triu():
    """Tests on the triu method of the CovarianceMatrix class."""
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


def test_eigenvalues():
    """Tests on the eigenvalues method of the CovarianceMatrix class."""
    cov = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    cov = np.array([cov @ cov.T.conj() for cov in cov])
    cov = csn.CovarianceMatrix(cov)
    cov.eigenvalues()


def test_eigenvectors():
    """Tests on the eigenvectors method of the CovarianceMatrix class."""
    cov = np.zeros((3, 5, 10, 10), dtype=np.complex128)
    for i in range(3):
        for j in range(5):
            cov[i, j] = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
            cov[i, j] = cov[i, j] @ cov[i, j].T.conj()
    cov = csn.CovarianceMatrix(cov)
    cov_r = cov.eigenvectors(
        rank=slice(0, cov.shape[-1] + 1), return_covariance=True
    )
    assert isinstance(cov_r, csn.CovarianceMatrix)
    assert np.allclose(cov_r, cov)


def test_pickle_persistance():
    """Tests on the pickle persistance of the CovarianceMatrix class."""
    cov = np.zeros((3, 5, 10, 10), dtype=np.complex128)
    for i in range(3):
        for j in range(5):
            cov[i, j] = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
            cov[i, j] = cov[i, j] @ cov[i, j].T.conj()
    cov = csn.CovarianceMatrix(
        cov, stats=[{"station": station} for station in "ABCDE"]
    )

    # Save and load
    cov = pickle.loads(pickle.dumps(cov))
    cov.coherence()

    # Assertions
    assert hasattr(cov, "stats")
