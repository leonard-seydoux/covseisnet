"""Covseisnet package initialization.

This module initializes the covseisnet package by importing the main objects
and methods of the package. The objects are the classes that represent the
main objects of the package, such as the covariance matrix, cross-correlation
matrix, network stream, and short-time Fourier transform. The methods are the
functions that perform the main operations of the package, such as calculating
the covariance matrix, cross-correlation matrix, and reading the stream.

Made by Leonard Seydoux in 2024.
"""

__all__ = [
    "CovarianceMatrix",
    "CrossCorrelationMatrix",
    "NetworkStream",
    "ShortTimeFourierTransform",
    "calculate_covariance_matrix",
    "calculate_cross_correlation_matrix",
    "pairwise_great_circle_distances_from_stats",
    "read",
    "plot",
    "signal",
    "data",
    "spatial",
]

# High-level objects
from .covariance import CovarianceMatrix
from .correlation import CrossCorrelationMatrix
from .stream import NetworkStream
from .signal import ShortTimeFourierTransform
from . import spatial
from . import velocity
from . import travel_times
from . import backprojection

# High-level methods
from .covariance import calculate_covariance_matrix
from .correlation import calculate_cross_correlation_matrix
from .stream import read
from .spatial import pairwise_great_circle_distances_from_stats

# Modules
from . import signal
from . import plot
from . import data
