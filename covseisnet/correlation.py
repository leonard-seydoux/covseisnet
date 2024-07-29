"""Correlation matrix in time domain."""

import numpy as np

from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d

from .covariance import CovarianceMatrix, get_twosided_covariance


class CorrelationMatrix(np.ndarray):
    r"""Correlation Matrix.

    This class is a subclass of :class:`numpy.ndarray`.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def nwin(self):
        """Returns the number of windows in the correlation matrix.

        Returns
        -------
        int
            The number of windows in the correlation matrix.

        """
        return self.shape[0]

    def hilbert_envelope(self, **kwargs):
        """Apply the Hilbert transform to the correlation matrix. Uses
        :func:`~scipy.signal.hilbert`

        """
        return np.abs(hilbert(self, axis=0, **kwargs)).view(CorrelationMatrix)

    def smooth(self, sigma, **kwargs):
        """Apply a 1-D Gaussian filter to the correlation matrix. Uses
        :func:`~scipy.ndimage.gaussian_filter1d`.

        Parameters
        ----------

        sigma: float
            Standard deviation for Gaussian kernel

        """
        return gaussian_filter1d(self, sigma, axis=0, **kwargs).view(
            CorrelationMatrix
        )

    def bandpass(self, low_cut, high_cut, sampling_rate, **kwargs):
        """Apply a Butterworth bandpass filter to the correlation matrix. Uses
        :func:`~scipy.signal.butter` and :func:`~scipy.signal.filtfilt`.

        Parameters
        ----------

        low_cut: float
            Pass band low corner frequency.

        high_cut: float
            Pass band high corner frequency.

        sampling_rate: float
            Sampling rate in Hz.

        """
        # calculate the Nyquist frequency
        nyquist = 0.5 * sampling_rate

        # design filter
        order = 4
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = butter(order, [low, high], btype="band")

        filtered = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                filtered[i, j] = filtfilt(b, a, self[i, j], **kwargs)
        return filtered.view(CorrelationMatrix)


def calculate_cross_correlation_matrix(covariance_matrix):
    """Extract correlation in time domain from the given covariance matrix.

    Parameters
    ----------
    covariance_matrix: :class:`~covseisnet.covariance.CovarianceMatrix`
        The covariance matrix.

    Returns
    -------
    :class:`~numpy.ndarray`
        The lag time between stations.
    :class:`~covseisnet.correlation.CorrelationMatrix`
        The correlation matrix with shape ``(n_pairs, n_windows, n_lags)``.

    """
    # Two-sided covariance matrix
    covariance_matrix_twosided = get_twosided_covariance(covariance_matrix)

    # Extract upper triangular
    covariance_triu = covariance_matrix_twosided.triu(k=1)

    # Change axes position to have the pairs first
    covariance_triu = covariance_triu.transpose(2, 0, 1)

    # Get transform parameters
    stft = covariance_matrix.stft
    n_samples_in = len(stft.win)
    sampling_rate = stft.fs
    n_lags = 2 * n_samples_in - 1

    # Inverse Fourier transform
    correlation = np.fft.fftshift(
        np.fft.ifft(covariance_triu, n=n_lags), axes=-1
    ).real

    # Calculate lags
    lag_max = (n_lags - 1) // 2 / sampling_rate
    lags = np.linspace(-lag_max, lag_max, n_lags)

    return lags, correlation.view(CorrelationMatrix)
