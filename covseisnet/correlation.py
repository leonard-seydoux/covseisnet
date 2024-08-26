"""Cross-correlation matrix in time domain."""

import numpy as np

from obspy.core.trace import Stats
from scipy.signal import windows, correlation_lags
from scipy.ndimage import gaussian_filter1d

from .covariance import CovarianceMatrix
from .signal import bandpass_filter, hilbert_envelope
from .signal import ShortTimeFourierTransform


class CrossCorrelationMatrix(np.ndarray):
    r"""

    .. rubric:: Mathematical definition

    This class is used to store the correlation matrix in the time domain. The
    cross-correlation is defined as the inverse Fourier transform of the
    covariance. Given a covariance matrix :math:`C_{ij}(f)`, the correlation
    matrix :math:`R_{ij}(\tau)` is defined as:

    .. math::

        R_{ij}(\tau) = \mathcal{F}^{-1} \{ C_{ij}(f) \}

    where :math:`\mathcal{F}^{-1}` is the inverse Fourier transform,
    :math:`\tau` is the lag time, and :math:`i` and :math:`j` are the station
    indices. Note that the correlation matrix is a symmetric matrix, with the
    diagonal elements being the auto-correlation. Therefore, we do not store
    the lower triangular part of the matrix.

    .. rubric:: Practical implementation

    The correlation matrix is stored as a 3D array with the first dimension
    being the number of pairs, the second dimension the number of windows, and
    the third dimension the number of lags. The correlation matrix can be
    visualized as a 2D array with the pairs and windows in the first dimension
    and the lags in the second dimension with the method :meth:`~flat`.

    Tip
    ---

    The correlation matrix is usually calculated from the covariance matrix
    using the method :func:`~calculate_cross_correlation_matrix`. The method
    returns a :class:`~covseisnet.correlation.CrossCorrelationMatrix` object
    with adequate attributes.
    """

    def __new__(
        cls,
        input_array: np.ndarray,
        stats: list[Stats] | None = None,
        stft: ShortTimeFourierTransform | None = None,
    ) -> "CrossCorrelationMatrix":
        r"""Note
        ----

        If for some reason you need to create a
        :class:`~covseisnet.covariance.CrossCorrelationMatrix` object from a
        numpy array, you have the several solutions provided by the `numpy
        subclassing mechanism
        <https://numpy.org/doc/stable/user/basics.subclassing.html#subclassing-ndarray>`_.
        We recommend using the contructor of the class, which allows to pass
        the stats and stft attributes, as in the following example:

        >>> import numpy as np
        >>> from covseisnet.correlation import CrossCorrelationMatrix
        >>> correlation = CrossCorrelationMatrix(
        ...     np.random.rand(10, 5, 100),
        ...     stats=None,
        ...     stft=None,
        ... )
        """
        # Enforce the input array to be a complex-valued numpy array.
        input_array = np.asarray(input_array)

        # Cast the input array into CovarianceMatrix and add the stats and
        # stft attributes.
        obj = input_array.view(cls)
        obj.stats = stats
        obj.stft = stft
        return obj

    def __array_finalize__(self, obj):
        r"""Finalize the array.

        This method is essential to the subclassing mechanism of numpy. It
        guarantees that the attributes of the object are correctly set when
        slicing or reducing the array. The method is called after the
        creation of the object, and is used to set the attributes of the
        object to the attributes of the input array.
        """
        if obj is None:
            return
        self.stats = getattr(obj, "stats", None)
        self.stft = getattr(obj, "stft", None)

    def __reduce__(self):
        r"""Reduce the object.

        This method is used to preserve the attributes and methods of the object
        over pickling and unpickling.
        """
        # Get the reduction tuple from the base ndarray
        pickled_state = super().__reduce__()

        # Combine the ndarray state with the additional attributes saved in
        # the __dict__ attribute.
        return (
            pickled_state[0],
            pickled_state[1],
            (pickled_state[2], self.__dict__),
        )

    def __setstate__(self, state):
        r"""Set the state of the object.

        This method is used to set the state of the object after pickling and
        unpickling.
        """
        # Extract the ndarray part of the state and the additional attributes
        ndarray_state, attributes = state

        # Set the ndarray part of the state
        super().__setstate__(ndarray_state)

        # Set the additional attributes
        self.__dict__.update(attributes)

    @property
    def sampling_rate(self) -> float:
        r"""Return the sampling rate in Hz.

        The sampling rate is extracted from the first stats in the list of
        stats extracted from the covariance matrix. Note that if the
        covariance matrix was calculated with the
        :func:`~covseisnet.covariance.calculate_covariance_matrix`, all streams
        must be synchronized and have the same sampling rate.

        Returns
        -------
        float
            The sampling rate in Hz.

        Raises
        ------
        ValueError
            If the stats attribute is not set.
        """
        if self.stats is None:
            raise ValueError("Stats are needed to get the sampling rate.")
        return self.stats[0].sampling_rate

    def envelope(self, **kwargs) -> "CrossCorrelationMatrix":
        r"""Hilbert envelope of the correlation matrix.

        The Hilbert envelope is calculated using the Hilbert transform of the
        pairwise cross-correlation:

        .. math::

            E_{ij}(\tau) = | R_{ij}(\tau) + i \mathcal{H} R_{ij}(\tau)  |

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
        kwargs.setdefault("axis", -1)
        return CrossCorrelationMatrix(
            hilbert_envelope(self, **kwargs),
            stats=self.stats,
            stft=self.stft,
        )

    def taper(self, max_percentage: float = 0.1) -> "CrossCorrelationMatrix":
        r"""Taper the correlation matrix.

        Taper the correlation matrix with the given taper. The taper is
        applied to the last axis (lags) of the correlation matrix. The tapered
        correlation matrix is defined as:

        .. math::

            R'_{ij}(\tau) = w_T(\tau) R_{ij}(\tau)

        where :math:`w_T(\tau)` is the taper of maximum duration :math:`T`.

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
        return CrossCorrelationMatrix(
            self * windows.tukey(self.shape[-1], max_percentage),
            stats=self.stats,
            stft=self.stft,
        )

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
        return CrossCorrelationMatrix(
            gaussian_filter1d(self, sigma=sigma, **kwargs),
            stats=self.stats,
            stft=self.stft,
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
        self, frequency_band: tuple | list, filter_order: int = 4
    ) -> "CrossCorrelationMatrix":
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

        Returns
        -------
        :class:`~covseisnet.correlation.CrossCorrelationMatrix`
            The bandpass filtered correlation functions.
        """
        # Flatten the correlation functions
        correlation_flat = self.flat()

        # Apply bandpass filter
        correlation_filtered = bandpass_filter(
            correlation_flat,
            self.sampling_rate,
            frequency_band,
            filter_order,
        )

        # Reshape the correlation functions
        correlation_filtered = correlation_filtered.reshape(self.shape)

        # Update self array
        return CrossCorrelationMatrix(
            correlation_filtered, stats=self.stats, stft=self.stft
        )


def calculate_cross_correlation_matrix(
    covariance_matrix: CovarianceMatrix,
    include_autocorrelation: bool = True,
) -> tuple[np.ndarray, list, CrossCorrelationMatrix]:
    r"""Extract correlation in time domain from the given covariance matrix.

    This method calculates the correlation in the time domain from the given
    covariance matrix. The covariance matrix is expected to be obtained from
    the method :func:`~covseisnet.covariance.calculate_covariance_matrix`, in
    the :class:`~covseisnet.covariance.CovarianceMatrix` class.

    The method first duplicate the covariance matrix to have a two-sided
    covariance matrix and perform the inverse Fourier transform to obtain the
    correlation matrix. The attributes of the covariance matrix are used to
    calculate the lags and pairs names of the correlation matrix.

    For more details, see the
    :class:`~covseisnet.correlation.CrossCorrelationMatrix` class, in the
    rubric **Mathematical definition**.

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
    list
        The list of pairs.
    :class:`~covseisnet.correlation.CrossCorrelationMatrix`
        The correlations with shape ``(n_pairs, n_windows, n_lags)``.

    """
    # Two-sided covariance matrix
    covariance_matrix_twosided = covariance_matrix.twosided()

    # Extract upper triangular
    distante_to_diagonal = 0 if include_autocorrelation else 1
    covariance_triu = covariance_matrix_twosided.triu(k=distante_to_diagonal)

    # Extract pairs names from the combination of stations
    if covariance_matrix.stats is not None:
        stations = [stat.station for stat in covariance_matrix.stats]
        pairs = []
        for i in range(len(stations)):
            for j in range(i + distante_to_diagonal, len(stations)):
                pairs.append((stations[i], stations[j]))
    else:
        pairs = [f"pair {i}" for i in range(covariance_triu.shape[0])]

    # Change axes position to have the pairs first
    covariance_triu = covariance_triu.transpose(2, 0, 1)

    # Get transform parameters
    if covariance_matrix.stft is None:
        raise ValueError(
            "STFT parameters are needed to calculate correlation."
        )
    n_samples = len(covariance_matrix.stft.win)
    sampling_rate = covariance_matrix.stft.fs

    # Inverse Fourier transform
    correlation = np.fft.fftshift(np.fft.ifft(covariance_triu), axes=-1).real

    # Calculate lags
    lags = correlation_lags(n_samples, n_samples, mode="same") / sampling_rate

    # Turn into CrossCorrelationMatrix
    correlation = CrossCorrelationMatrix(
        correlation, stats=covariance_matrix.stats, stft=covariance_matrix.stft
    )

    return lags, pairs, correlation
