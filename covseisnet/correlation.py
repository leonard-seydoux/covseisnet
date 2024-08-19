"""Cross-correlation matrix in time domain."""

import numpy as np

from obspy.core.trace import Stats
from scipy.signal import hilbert, windows
from scipy.ndimage import gaussian_filter1d

from .covariance import CovarianceMatrix
from .signal import bandpass_filter, hilbert_envelope


class CrossCorrelationMatrix(np.ndarray):
    r"""
    This class is a subclass of :class:`numpy.ndarray`. It is used to store
    the correlation matrix in the time domain. The cross-correlation is
    defined as the inverse Fourier transform of the covariance. Given a
    covariance matrix :math:`C_{ij}(\omega)`, the correlation matrix
    :math:`R_{ij}(\tau)` is defined as:

    .. math::

        R_{ij}(\tau) = \mathcal{F}^{-1} \{ C_{ij}(\omega) \}

    where :math:`\mathcal{F}^{-1}` is the inverse Fourier transform,
    :math:`\tau` is the lag time, and :math:`i` and :math:`j` are the station
    indices. Note that the correlation matrix is a symmetric matrix, with the
    diagonal elements being the auto-correlation. Therefore, we do not store
    the lower triangular part of the matrix.

    The correlation matrix is stored as a 3D array with the first dimension
    being the number of pairs, the second dimension the number of windows, and
    the third dimension the number of lags. The correlation matrix can be
    visualized as a 2D array with the pairs and windows in the first dimension
    and the lags in the second dimension with the method :meth:`~flat`.
    """

    def __new__(cls, input_array: np.ndarray):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array, dtype=complex).view(cls)

        # Add the new attribute to the created instance. Here, only the stats
        # attribute is added. Let's try to keep it that way.
        if obj.shape:
            default_stats = list([Stats() for _ in range(obj.shape[-1])])
        else:
            default_stats = list([Stats()])
        obj._stats = default_stats
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # Check if the object is None
        if obj is None:
            return
        # Copy the attributes from the input object
        if obj.shape:
            default_stats = list([Stats() for _ in range(obj.shape[-1])])
        else:
            default_stats = list([Stats()])
        stats = getattr(obj, "_stats", default_stats)
        self._stats = getattr(obj, "_stats", stats)

    @property
    def sampling_rate(self) -> float:
        """Return the sampling rate."""
        return self.stats[0].sampling_rate

    @property
    def stats(self) -> list[Stats]:
        """Return the stats."""
        return self._stats

    def set_stats(self, stats: list[Stats] | list):
        """Set the stats.

        Arguments
        ---------
        stats: list of :class:`~obspy.core.trace.Stats`
            The list of stats for each trace.
        """
        if not isinstance(stats[0], Stats):
            stats = [Stats(stat) for stat in stats]
        self._stats = stats

    def envelope(self, **kwargs) -> "CrossCorrelationMatrix":
        r"""Hilbert envelope of the correlation matrix.

        The Hilbert envelope is calculated using the Hilbert transform of the
        pairwise cross-correlation:

        .. math::

            E_{ij}(\tau) = | \mathcal{H} R_{ij}(\tau) | = | R_{ij}(\tau) + i \mathcal{H} R_{ij}(\tau)  |

        where :math:`\mathcal{H}` is the Hilbert transform, and :math:`i` is
        the imaginary unit. The Hilbert envelope is the absolute value of the
        Hilbert transform of the correlation matrix.


        Arguments
        ---------
        **kwargs: dict
            Additional arguments to pass to :func:`~scipy.signal.hilbert`. By
            default, the axis is set to the last axis (lags).

        Returns
        -------
        :class:`~covseisnet.correlation.CrossCorrelationMatrix`
            The Hilbert envelope of the correlation matrix.
        """
        # Default axis is the last axis
        kwargs.setdefault("axis", -1)

        # Return a view of the Hilbert envelope
        correlation_envelopes = hilbert_envelope(self, **kwargs)
        correlation_envelopes = correlation_envelopes.view(
            CrossCorrelationMatrix
        )

        # Copy attributes
        correlation_envelopes.__dict__.update(self.__dict__)
        return correlation_envelopes

    def taper(self, max_percentage: float = 0.1) -> "CrossCorrelationMatrix":
        r"""Taper the correlation matrix.

        Taper the correlation matrix with the given taper. The taper is
        applied to the last axis (lags) of the correlation matrix. The tapered
        correlation matrix is defined as:

        .. math::

            R'_{ij}(\tau) = w_T{\tau} R_{ij}(\tau)

        where :math:`w_T` is the taper of maximum duration :math:`T`.

        Arguments
        ---------
        max_percentage: float
            The maximum percentage of the taper. The taper is a Tukey window
            with the given percentage of the window duration. Default is 0.1.

        Returns
        -------
        :class:`~covseisnet.correlation.CrossCorrelationMatrix`
            The tapered correlation matrix.
        """
        # Apply taper
        correlation_tapered = self * windows.tukey(
            self.shape[-1], max_percentage
        )

        # Return a view of the tapered correlation matrix
        correlation_tapered = correlation_tapered.view(CrossCorrelationMatrix)

        # Copy attributes
        correlation_tapered.__dict__.update(self.__dict__)
        return correlation_tapered

    def smooth(self, sigma: float = 1, **kwargs) -> "CrossCorrelationMatrix":
        r"""Use a Gaussian kernel to smooth the correlation matrix.

        This function is usually applied to the envelope of the correlation
        matrix to smooth the envelope. The smoothing is done using a Gaussian
        kernel with a standard deviation :math:`\sigma`. The smoothing is done
        along the last axis (lags). The smoothed correlation matrix is
        :math:`R'` is defined as:

        .. math::

            R'_{ij}(\tau) = G_{\sigma} * R_{ij}(\tau)

        where :math:`G_{\sigma}` is the Gaussian kernel with standard
        deviation :math:`\sigma`.

        Arguments
        ---------
        sigma: float
            Standard deviation of the Gaussian kernel. The larger the value,
            the smoother the correlation matrix. Default is 1.
        **kwargs: dict
            Additional arguments passed to
            :func:`~scipy.ndimage.gaussian_filter1d`.

        Returns
        -------
        :class:`~covseisnet.correlation.CrossCorrelationMatrix`
            The smoothed cross-correlation matrix.
        """
        # Smooth the correlation matrix
        correlation_smoothed = gaussian_filter1d(self, sigma=sigma, **kwargs)

        # Return a view of the smoothed correlation matrix
        correlation_smoothed = correlation_smoothed.view(
            CrossCorrelationMatrix
        )

        # Copy attributes
        correlation_smoothed.__dict__.update(self.__dict__)
        return correlation_smoothed

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

    def bandpass(self, frequency_band: tuple | list, filter_order: int = 4):
        r"""Bandpass filter the correlation functions.

        Apply a Butterworth bandpass filter to the correlation functions. Uses
        :func:`~scipy.signal.butter` and :func:`~scipy.signal.filtfilt` to
        avoid phase shift.

        Parameters
        ----------
        frequency_band: tuple
            The frequency band to filter in Hz.
        filter_order: int, optional
            The order of the Butterworth filter.
        """
        # Flatten the correlation functions
        correlation_flat = self.flat()

        # Apply bandpass filter
        correlation_filtered = bandpass_filter(
            correlation_flat,
            self.stats[0].sampling_rate,
            frequency_band,
            filter_order,
        )

        # Reshape the correlation functions
        correlation_filtered = correlation_filtered.reshape(self.shape)

        # Update self array
        self[:] = correlation_filtered


def calculate_cross_correlation_matrix(
    covariance_matrix: CovarianceMatrix,
    include_autocorrelation: bool = True,
) -> tuple[np.ndarray, list, CrossCorrelationMatrix]:
    r"""Extract correlation in time domain from the given covariance matrix.

    This method calculates the correlation in the time domain from the given
    covariance matrix. The covariance matrix is expected to be obtained from
    the method :func:`~covseisnet.covariance.calculate_covariance_matrix`,
    in the :class:`~covseisnet.covariance.CovarianceMatrix` class. The method
    relies.

    Parameters
    ----------
    covariance_matrix: :class:`~covseisnet.covariance.CovarianceMatrix`
        The covariance matrix.
    include_autocorrelation: bool, optional
        Include the auto-correlation in the correlation matrix. Default is
        True.


    Returns
    -------
    :class:`~numpy.ndarray`
        The lag time between stations.
    :class:`~covseisnet.correlation.CrossCorrelationMatrix`
        The correlations with shape ``(n_pairs, n_windows, n_lags)``.

    """
    # Two-sided covariance matrix
    covariance_matrix_twosided = covariance_matrix.twosided()

    # Extract upper triangular
    distante_to_diagonal = 0 if include_autocorrelation else 1
    covariance_triu = covariance_matrix_twosided.triu(k=distante_to_diagonal)

    # Extract pairs names from the combination of stations
    stations = [stat.station for stat in covariance_matrix.stats]
    n_stations = len(stations)
    pairs = []
    for i in range(n_stations):
        for j in range(i + distante_to_diagonal, n_stations):
            pairs.append(f"{stations[i]} - {stations[j]}")

    # Change axes position to have the pairs first
    covariance_triu = covariance_triu.transpose(2, 0, 1)

    # Get transform parameters
    stft = covariance_matrix.stft
    if stft is None:
        raise ValueError(
            "STFT parameters are needed to calculate correlation."
        )
    n_samples_in = len(stft.win)
    sampling_rate = stft.fs
    # n_lags = 2 * n_samples_in - 1

    # Inverse Fourier transform
    correlation = np.fft.fftshift(np.fft.ifft(covariance_triu), axes=-1).real

    # Calculate lags
    # lag_max = (n_lags - 1) // 2 / sampling_rate
    n_lags = n_samples_in
    lag_max = n_samples_in / sampling_rate / 2
    lags = np.linspace(-lag_max, lag_max, n_lags)

    # Turn into CrossCorrelationMatrix
    correlation = correlation.view(CrossCorrelationMatrix)
    correlation.set_stats(covariance_matrix.stats)

    return lags, pairs, correlation
