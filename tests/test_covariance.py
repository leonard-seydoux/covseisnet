"""Test of the ArrayStream class."""

import pickle
import pytest

import numpy as np

import covseisnet as csn


def test_fuzzy_covariance_matrix():
    """Check if the fuzzy covariance matrix is calculated correctly."""
    with pytest.raises(ValueError):
        csn.CovarianceMatrix(np.array(1))


def test_covariance_matrix_instance():
    """Check various instances of the CovarianceMatrix class."""
    x = np.eye(3)
    covariance = csn.CovarianceMatrix(x)
    sub_covariance = covariance[1]
    assert covariance.shape == (3, 3)
    assert isinstance(covariance, csn.CovarianceMatrix)
    assert isinstance(sub_covariance, csn.CovarianceMatrix)


def test_covariance_matrix_operations():
    # Build a complex Hermitian matrix
    x = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    x = x @ x.T.conj()
    covariance = csn.CovarianceMatrix(x)
    assert isinstance(covariance.sum(axis=1), csn.CovarianceMatrix)
    assert isinstance(covariance.mean(axis=0), csn.CovarianceMatrix)


def test_covariance_matrix_stats():
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
    cov = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    cov = np.array([cov @ cov.T.conj() for cov in cov])
    cov = csn.CovarianceMatrix(cov)
    cov.eigenvalues()


def test_eigenvectors():
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
    cov = np.zeros((3, 5, 10, 10), dtype=np.complex128)
    for i in range(3):
        for j in range(5):
            cov[i, j] = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
            cov[i, j] = cov[i, j] @ cov[i, j].T.conj()
    cov = csn.CovarianceMatrix(
        cov, stats=[{"station": station} for station in "ABCDE"]
    )

    # Save
    # with open("cov.pkl", "wb") as f:
    #     pickle.dump(cov, f)
    cov = pickle.loads(pickle.dumps(cov))
    cov.coherence()

    # Load
    # with open("cov.pkl", "rb") as f:
    #     cov = pickle.load(f)

    # Print
    print(cov.stats)
