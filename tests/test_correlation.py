"""Test of the ArrayStream class."""

import pickle

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
