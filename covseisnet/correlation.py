"""Pairwise cross-correlation in time domain."""

import numpy as np

from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d

from .covariance import CovarianceMatrix, get_twosided_covariance


class PairwiseCrossCorrelation(np.ndarray):
    r"""Correlation Matrix.

    This class is a subclass of :class:`numpy.ndarray`.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def set_sampling_rate(self, sampling_rate):
        """Set the sampling rate of the correlation.

        Arguments
        ---------
        sampling_rate: float
            The sampling rate in Hz.
        """
        self.sampling_rate = sampling_rate

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
        return np.abs(hilbert(self, axis=0, **kwargs)).view(
            PairwiseCrossCorrelation
        )

    def smooth(self, sigma, **kwargs):
        """Apply a 1-D Gaussian filter to the correlation matrix. Uses
        :func:`~scipy.ndimage.gaussian_filter1d`.

        Parameters
        ----------

        sigma: float
            Standard deviation for Gaussian kernel

        """
        return gaussian_filter1d(self, sigma, axis=0, **kwargs).view(
            PairwiseCrossCorrelation
        )

    def flat(self):
        r"""Flatten the first dimensions.

        The shape of the pairwise cross-correlation is at maximum 3D, with the
        first dimension being the number of pairs, the second dimension the
        number of windows, and the third dimension the number of lags. This
        method flattens the first two dimensions to have a 2D array with the
        pairs and windows in the first dimension and the lags in the second
        dimension. This method also works for smaller dimensions.

        Returns
        -------
        :class:`np.ndarray`
            The flattened pairwise cross-correlation.s
        """
        return self.reshape(-1, self.shape[-1])

    def bandpass(
        self, frequency_band: tuple | list, filter_order: int = 4, **kwargs
    ):
        """Bandpass filter the correlation functions.

        Apply a Butterworth bandpass filter to the correlation functions. Uses
        :func:`~scipy.signal.butter` and :func:`~scipy.signal.filtfilt` to
        avoid phase shift.

        Parameters
        ----------
        frequency_band: tuple
            The frequency band to filter in Hz.
        **kwargs: dict, optional
            Additional keyword arguments to pass to :func:`~
            scipy.signal.filtfilt`.
        """
        # Turn frequencies into normalized frequencies
        nyquist = 0.5 * self.sampling_rate
        normalized_frequency_band = [f / nyquist for f in frequency_band]

        # Extract filter
        butter_coefficients = butter(
            filter_order,
            normalized_frequency_band,
            btype="band",
        )

        # Apply filter
        correlation_flat = self.flat()
        correlation_flat = filtfilt(
            *butter_coefficients, correlation_flat, **kwargs
        )
        correlation_filtered = correlation_flat.reshape(self.shape)
        return correlation_filtered.view(PairwiseCrossCorrelation)


def calculate_cross_correlation(
    covariance_matrix: CovarianceMatrix,
) -> tuple[np.ndarray, PairwiseCrossCorrelation]:
    """Extract correlation in time domain from the given covariance matrix.

    This method calculates the correlation in the time domain from the given
    covariance matrix. The covariance matrix is expected to be obtained from
    the method :func:`~covseisnet.covariance.calculate_covariance_matrix`,
    in the :class:`~covseisnet.covariance.CovarianceMatrix` class. The method
    relies.

    Parameters
    ----------
    covariance_matrix: :class:`~covseisnet.covariance.CovarianceMatrix`
        The covariance matrix.

    Returns
    -------
    :class:`~numpy.ndarray`
        The lag time between stations.
    :class:`~covseisnet.correlation.PairwiseCrossCorrelation`
        The correlation matrix with shape ``(n_pairs, n_windows, n_lags)``.

    """
    # Two-sided covariance matrix
    covariance_matrix_twosided = get_twosided_covariance(covariance_matrix)

    # Extract upper triangular
    covariance_triu = covariance_matrix_twosided.triu(k=1)

    # Extract pairs names from the combination of stations
    stations = covariance_matrix.stations
    n_stations = len(stations)
    pairs = []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            pairs.append(f"{stations[i]} - {stations[j]}")

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

    # Turn into PairwiseCrossCorrelation
    correlation = correlation.view(PairwiseCrossCorrelation)
    correlation.set_sampling_rate(sampling_rate)

    return lags, pairs, correlation
