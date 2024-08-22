"""Spectral analysis and covariance matrix calculation."""

from typing import Any

import numpy as np
from numpy.linalg import eigvalsh, eigh
from obspy.core.trace import Stats

from .stream import NetworkStream
from . import signal


class CovarianceMatrix(np.ndarray):
    r"""
    This class is a subclass of :class:`numpy.ndarray`, which means it
    inherits all the methods of standard numpy arrays, along with extra
    methods tailored for array processing. Any numpy method or function
    applied to an instance of :class:`~covseisnet.covariance.CovarianceMatrix`
    will return an instance of
    :class:`~covseisnet.covariance.CovarianceMatrix`.

    Let's consider a continuous set of network seismograms
    :math:`\{u_i(t)\}_{i=1\dots N}` with :math:`N` traces. The Fourier
    transform of the traces is defined as :math:`u_i(f)`, with :math:`f` the
    frequency. The spectral covariance matrix is defined as

    .. math::

        C_{ij}(f) = \sum_{m=1}^M u_{i}(t \in \tau_m, f) u_{j}^*(t \in \tau_m,
        f)

    where :math:`M` is the number of windows used to estimate the covariance,
    :math:`^*` is the complex conjugate, and :math:`\tau_m` is the time window
    of index :math:`m`. The covariance matrix is a complex-valued matrix of
    shape :math:`N \times N`. Depending on the averaging size and frequency
    content, the covariance matrix can have a different shape:

    - ``(n_traces, n_traces)`` if a single frequency and time sample is given.

    - ``(n_frequencies, n_traces, n_traces)`` if only one time frame is given,
      for ``n_frequencies`` frequencies, which depends on the window size and
      sampling rate.

    - ``(n_times, n_frequencies, n_traces, n_traces)`` if multiple time frames
      are given.

    All the methods defined in the the
    :class:`~covseisnet.covariance.CovarianceMatrix` class are performed on
    the flattened array with the
    :meth:`~covseisnet.covariance.CovarianceMatrix.flat` method, which allow
    to obtain as many :math:`N \times N` covariance matrices as time and
    frequency samples.

    Note
    ----

    The :class:`~covseisnet.covariance.CovarianceMatrix` class is not meant to
    be instantiated directly. It should be obtained from the output of the
    :func:`~covseisnet.covariance.calculate_covariance_matrix` function.

    If you want to create a :class:`~covseisnet.covariance.CovarianceMatrix`
    object from a numpy array, you can use the :meth:`~numpy.ndarray.view`
    method:

    >>> import covseisnet as cn
    >>> import numpy as np
    >>> c = np.zeros((4, 4)).view(cn.CovarianceMatrix)
    >>> c
    CovarianceMatrix([[ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.]])
    """

    def __new__(cls, input_array: np.ndarray):
        """Create a new instance of CovarianceMatrix."""
        # Force input array to be an array
        input_array = np.asarray(input_array, dtype=complex)

        # Check that the input array had at least two dimensions
        if input_array.ndim < 2:
            raise ValueError(
                "Input array must have at least two dimensions, "
                f"got {input_array.ndim}."
            )

        # Check that the two last dimensions are Hermitian
        if not are_two_last_dimensions_hermitian(input_array):
            raise ValueError(
                "The two last dimensions of the input array must be Hermitian."
            )

        # Cast the input array to the CovarianceMatrix class
        obj = input_array.view(cls)

        # Add the default attributes (empty stats and no STFT)
        obj._stats = list([Stats() for _ in range(obj.shape[-1])])

        return obj

    def __array_finalize__(self, obj):
        # Return if the object is None
        if obj is None:
            return
        default_stats = list([Stats() for _ in range(obj.shape[-1])])
        self._stats = getattr(obj, "_stats", default_stats)
        self.stft = getattr(obj, "stft", None)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if (
            isinstance(result, np.ndarray)
            and result.ndim >= 2
            and are_two_last_dimensions_hermitian(result)
        ):
            result = result.view(CovarianceMatrix)
            result._stats = self._stats
        return result

    def __reduce__(self):
        # Get the reduction tuple from the base ndarray
        pickled_state = super().__reduce__()
        # Combine the ndarray state with the additional attributes
        return (
            pickled_state[0],
            pickled_state[1],
            (pickled_state[2], self.__dict__),
        )

    def __setstate__(self, state):
        # Extract the ndarray part of the state and the additional attributes
        ndarray_state, attributes = state
        # Set the ndarray part of the state
        super().__setstate__(ndarray_state)
        # Set the additional attributes
        self.__dict__.update(attributes)

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

    @property
    def stats(self) -> list[Stats]:
        """Return the stats."""
        return self._stats

    def set_stft(self, stft):
        """Set the ShortTimeFourierTransform instance for further processing.

        Arguments
        ---------
        stft: :class:`~covseisnet.signal.ShortTimeFourierTransform`
            The ShortTimeFourierTransform instance.
        """
        self.stft = stft

    def coherence(self, kind="spectral_width", epsilon=1e-10):
        r"""Covariance-based coherence estimation.

        The measured is performed onto all the covariance matrices from the
        eigenvalues obtained with the method
        :meth:`~covseisnet.covariancematrix.CovarianceMatrix.eigenvalues`. For
        a given matrix :math:`N \times N` matrix :math:`M` with eigenvalues
        :math:`\mathbf{\lambda} = \lambda_i` where :math:`i=1\ldots n`. The
        coherence is obtained as :math:`F(\lambda)`, with :math:`F` being
        defined by the `kind` parameter.

        - The spectral width is obtained with setting
          ``kind="spectral_width"`` , and returns the width :math:`\sigma` of
          the eigenvalue distribution (obtained at each time and frequency)
          such as

          .. math::

              \sigma = \frac{\sum_{i=0}^n i \lambda_i}{\sum_{i=0}^n \lambda_i}


        - The entropy is obtained with setting ``kind="entropy"``, and returns
          the entropy :math:`h` of the eigenvalue distribution (obtained at
          each time and frequency) such as

          .. math::

              h = - \sum_{i=0}^n i \lambda_i \log(\lambda_i + \epsilon)

        - The Shanon diversity index is obtained with setting
          ``kind="diversity"``, and returns the diversity index :math:`D` of
          the eigenvalue distribution (obtained at each time and frequency)
          such as the exponential of the entropy:

          .. math::

              D = \exp(h + \epsilon)


        Keyword arguments
        -----------------
        kind: str, optional
            The type of coherence, may be "spectral_width" (default),
            "entropy", or "diversity".
        epsilon: float, optional
            The regularization parameter for the logarithm.

        Returns
        -------
        :class:`numpy.ndarray`
            The coherence of maximal shape ``(n_times, n_frequencies)``,
            depending on the input covariance matrix shape.

        """
        if kind in ["spectral_width", "entropy", "diversity"]:
            eigenvalues = self.eigenvalues(norm=np.sum)
        else:
            message = "{} is not an available option for kind."
            raise ValueError(message.format(kind))
        if kind == "spectral_width":
            return signal.width(eigenvalues, axis=-1)
        elif kind == "entropy":
            return signal.entropy(self.eigenvalues(norm=np.sum), axis=-1)
        elif kind == "diversity":
            return signal.diversity(self.eigenvalues(norm=np.sum), axis=-1)
        else:
            message = "{} is not an available option for kind."
            raise ValueError(message.format(kind))

    def eigenvalues(self, norm=np.max):
        r"""Eigenvalue decomposition.

        Given and Hermitian matrix :math:`C` of shape :math:`N \times N`, the
        eigenvalue decomposition is defined as

        .. math::

            C = U D U^\dagger

        where :math:`U` is the unitary matrix of eigenvectors (which are not
        calculated here, see
        :meth:`~covseisnet.covariance.CovarianceMatrix.eigenvectors` for this
        purpose), and :math:`D` is the diagonal matrix of eigenvalues, as

        .. math::

            D = \pmatrix{\lambda_1 & 0 & \cdots & 0 \\
                            0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots &
                            \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_N}

        with :math:`\lambda_i` the eigenvalues. For obvious reasons, the
        eigenvalues are returned as a vector such as 

        .. math::

            \lambda = \pmatrix{\lambda_1 \\ \lambda_2 \\ \vdots \\ \lambda_N}
        
        The eigenvalues are sorted in decreasing order. The eigenvalues are
        normalized by the maximum eigenvalue by default, but can be normalized
        by any function provided by numpy. Since the matrix :math:`C` is
        Hermitian by definition, the eigenvalues are real- and
        positive-valued. Also, the eigenvectors are orthogonal and normalized.

        The eigenvalue decomposition is performed onto the two last dimensions
        of the :class:`~covseisnet.covariance.CovarianceMatrix` object. The
        function used for eigenvalue decomposition is
        :func:`numpy.linalg.eigvalsh`. It assumes that the input matrix is 2D
        and hermitian, so the decomposition is performed onto the lower
        triangular part in order to save time.

        Arguments
        ---------
        norm : function, optional
            The function used to normalize the eigenvalues. Can be
            :func:`numpy.max`, (default), any other functions provided by
            numpy, or a custom function. Note that the function must accept
            the ``axis`` keyword argument.

        Returns
        -------
        :class:`numpy.ndarray`
            The eigenvalues of maximal shape ``(n_times, n_frequencies,
            n_stations)``.

        Notes
        -----

        The eigenvalue decomposition is performed onto the two last dimensions
        of the :class:`~covseisnet.covariance.CovarianceMatrix` object. The
        matrices are first flattened with the
        :meth:`~covseisnet.covariance.CovarianceMatrix.flat` method, so the
        eigenvalues are calculated for each time and frequency sample. The
        eigenvalues are sorted in decreasing order, and normalized by the
        maximum eigenvalue by default, before being reshaped to the original
        shape of the covariance matrix. This maximizes the performance of the
        eigenvalue decomposition.


        See also
        --------
        :meth:`~covseisnet.covariance.CovarianceMatrix.eigenvectors`
        :func:`numpy.linalg.eigvalsh`

        Examples
        --------
        Calculate the eigenvalues of the example covariance matrix:

        >>> import covseisnet as cn
        >>> import numpy as np
        >>> c = np.arange(8).reshape((2, 2, 2)).view(cn.CovarianceMatrix)
        >>> c
            CovarianceMatrix([[[0, 1],
                               [2, 3]],
                             [[4, 5],
                              [6, 7]]])  
        >>> c.eigenvalues()
            array([[1.        , 0.25      ],
                   [1.        , 0.05859465]])
        """
        # Flatten the array
        matrices = self.flat()

        # Parallel computation of eigenvalues
        eigenvalues = eigvalsh(matrices)

        # Sort and normalize
        eigenvalues = np.sort(np.abs(eigenvalues), axis=-1)[:, ::-1]
        eigenvalues /= norm(eigenvalues, axis=-1, keepdims=True)

        return eigenvalues.reshape(self.shape[:-1])

    def eigenvectors(self, rank=0, covariance=False):
        """Extract eigenvectors of given rank.

        The function used for extracting eigenvectors is
        :func:`scipy.linalg.eigh`. It assumes that the input matrix is 2D
        and hermitian. The decomposition is performed onto the lower triangular
        part.

        Keyword arguments
        -----------------
        rank : int, optional
            Eigenvector rank, 0 by default (first eigenvector).

        covariance: int, optional
            Outer-product of eigenvectors of rank ``rank``.

        Returns
        -------
        :class:`numpy.ndarray`
            The complex-valued eigenvector array of shape
            ``(n_times, n_freq, n_sta)`` if the parameter ``covariance`` is
            ``False``, else ``(n_times, n_freq, n_sta, n_sta)``.


        Todo
        ----
        Implement a new option on the ``rank`` in order to accept a list, so
        the filtered covariance can be obtained from multiple eigenvector
        ranks. This should be defined together with a ``normalize`` boolean
        keyword argument in order to take into account the eigenvalues or not,
        and therefore the isotropization of the covariance matrix would be
        here defined fully (so far, the user has to define a loop in the
        main script).

        """
        # Initialization
        matrices = self.flat()
        eigenvectors = np.zeros(
            (matrices.shape[0], matrices.shape[-1]), dtype=complex
        )

        # Calculation over submatrices
        for i, m in enumerate(matrices):
            l, v = eigh(m)
            isort = np.argsort(np.abs(l))[::-1]
            eigenvectors[i] = v[isort][rank]

        if covariance:
            ec = np.zeros(self.shape, dtype=complex)
            ec = ec.view(CovarianceMatrix)
            ec = ec.flat()
            for i in range(eigenvectors.shape[0]):
                ec[i] = eigenvectors[i, :, None] * np.conj(eigenvectors[i])
            ec = ec.reshape(self.shape)
            return ec.view(CovarianceMatrix)
        else:
            return eigenvectors.reshape(self.shape[:-1])

    def flat(self):
        r"""Covariance matrices with flatten first dimensions.

        The shape of the covariance matrix depend on the number of time
        windows and frequencies. The method
        :meth:`~covseisnet.covariance.CovarianceMatrix.flat` allows to obtain
        as many :math:`N \times N` covariance matrices as time and frequency
        samples.

        Returns
        -------
        :class:`np.ndarray`
            The covariance matrices in a shape ``(a * b, n, n)``.

        Example
        -------
        >>> import covseisnet as cn
        >>> import numpy as np
        >>> c = np.arange(16).reshape((2, 2, 2, 2)).view(cn.CovarianceMatrix)
        >>> c.shape
            (2, 2, 2, 2)
        >>> c.flat().shape
            (4, 2, 2)
        """
        return self.reshape(-1, *self.shape[-2:])

    def triu(self, **kwargs):
        """Extract upper triangular on flatten array.

        This method is useful when calculating the cross-correlation matrix
        associated with the covariance matrix. Indeed, since the covariance
        matrix is Hermitian, the cross-correlation matrix is symmetric, so
        there is no need to calculate the lower triangular part.

        The method :meth:`~covseisnet.covariance.CovarianceMatrix.triu` is
        applied to the flattened array, then reshaped to the original shape of
        the covariance matrix. The last dimension of the returned matrix is
        the number of upper triangular elements of the covariance matrix.

        Arguments
        ---------
        **kwargs: dict, optional
            Keyword arguments passed to the :func:`numpy.triu_indices`
            function.

        Returns
        -------
        :class:`~covseisnet.covariance.CovarianceMatrix`
            The upper triangular part of the covariance matrix, with a maximum
            shape of ``(n_times, n_frequencies, n_traces * (n_traces + 1) //
            2)``.


        Example
        -------

        >>> import covseisnet as cn
        >>> import numpy as np
        >>> c = np.arange(8).reshape((2, 2, 2)).view(cn.CovarianceMatrix)
        >>> c
            CovarianceMatrix([[[0, 1],
                              [2, 3]],
                             [[4, 5],
                              [6, 7]]])
        >>> c.triu()
            CovarianceMatrix([[0, 1, 3],
                              [4, 5, 7]])

        """
        trii, trij = np.triu_indices(self.shape[-1], **kwargs)
        return self[..., trii, trij]

    def twosided(self, axis: int = 1) -> "CovarianceMatrix":
        """Get the full covariance spectrum.

        Given that the covariance matrix is Hermitian, the full covariance
        matrix can be obtained by filling the negative frequencies with the
        complex conjugate of the positive frequencies. The function
        :func:`~covseisnet.covariance.get_twosided_covariance` performs this
        operation.

        The frequency axis is assumed to be the second axis of the covariance
        matrix. The function returns a new covariance matrix with the negative
        frequencies filled with the complex conjugate of the positive
        frequencies.

        Arguments
        ---------
        axis: int, optional
            The frequency axis of the covariance matrix. Default is 1.

        Returns
        -------
        :class:`~covseisnet.covariance.CovarianceMatrix`
            The full covariance matrix.
        """
        # Get number of samples that were used to calculate the covariance matrix
        stft = self.stft
        if stft is None:
            raise ValueError("ShortTimeFourierTransform instance not found.")

        # Get number of samples in the window
        n_samples_in = len(stft.win)

        # Find out output shape
        input_shape = self.shape
        output_shape = list(input_shape)
        output_shape[axis] = n_samples_in

        # Initialize full covariance matrix with negative frequencies
        covariance_matrix_full = np.zeros(output_shape, dtype=np.complex128)

        # Fill negative frequencies
        covariance_matrix_full[:, : n_samples_in // 2 + 1] = self
        covariance_matrix_full[:, n_samples_in // 2 + 1 :] = np.conj(
            self[:, -2:0:-1]
        )

        # Return full covariance matrix
        covariance_matrix_full = covariance_matrix_full.view(CovarianceMatrix)
        covariance_matrix_full.__dict__.update(self.__dict__)
        return covariance_matrix_full

    @property
    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """Check if the covariance matrix is Hermitian.

        Arguments
        ---------
        tol: float, optional
            The tolerance for the comparison.

        Returns
        -------
        bool
            True if the covariance matrix is Hermitian, False otherwise.
        """
        return np.allclose(self, np.conj(self).swapaxes(-2, -1), atol=tol)


def calculate_covariance_matrix(
    stream: NetworkStream,
    window_duration: float,
    average: int,
    average_step: int | None = None,
    whiten: str = "none",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, CovarianceMatrix]:
    r"""Calculate covariance matrix.

    The covariance matrix is calculated from the Fourier transform of the
    input stream. The covariance matrix is calculated for each time window and
    frequency, and averaged over a given number of windows.

    Given a stream of :math:`N` traces :math:`\{u_i(t)\}_{i=1\dots N}` with
    :math:`N` traces, the Fourier transform of the traces is defined as
    :math:`u_i(f)`, with :math:`f` the frequency. The spectral covariance
    matrix is defined as

    .. math::

        C_{ij}(f) = \sum_{m=1}^M u_{i}(t \in \tau_m, f) u_{j}^*(t \in \tau_m,
        f)

    where :math:`M` is the number of windows used to estimate the covariance,
    :math:`^*` is the complex conjugate, and :math:`\tau_m` is the time window
    of index :math:`m`. The covariance matrix is a complex-valued matrix of
    shape :math:`N \times N`. Depending on the averaging size and frequency
    content, the covariance matrix can have a different shape. Please refer to
    the :class:`~covseisnet.covariance.CovarianceMatrix` class for more
    information. You can also find more information on the covariance matrix
    in the paper of :footcite:`seydoux_detecting_2016`.

    The whitening parameter can be used to normalize the covariance matrix.
    The parameter can be set to "none" (default), "slice", or "window". The
    "none" option does not apply any whitening to the covariance matrix. The
    "slice" option normalizes the spectra :math:`u_i(t \in \tau_m, f)` by the
    mean of the absolute value of the spectra within the same group of time
    windows :math:`\{\tau_m\}_{m=1\dots M}`, so that

    .. math::

        u_i(t \in \tau_m, f) = \frac{u_i(t \in \tau_m, f)}{\sum_{i=1}^N |u_i(t
        \in \tau_m, f)|}

    The "window" option normalizes the spectra :math:`u_i(t \in \tau_m, f)` by
    the absolute value of the spectra within the same time window :math:`\tau_m`
    so that

    .. math::

        u_i(t \in \tau_m, f) = \frac{u_i(t \in \tau_m, f)}{|u_i(t \in \tau_m,
        f)|}

    These additional whitening methods can be used in addition to the
    :meth:`~covseisnet.stream.NetworkStream.whiten` method to further whiten
    the covariance matrix.

    Arguments
    ---------
    stream: :class:`~covseisnet.stream.NetworkStream`
        The input data stream.
    average: int
        The number of window used to estimate the sample covariance.
    average_step: int, optional
        The sliding window step for covariance matrix calculation (in number
        of windows).
    whiten: str, optional
        The type of whitening applied to the covariance matrix. Can be
        "none" (default), "slice", or "window". This parameter can be used in
        addition to the :meth:`~covseisnet.stream.NetworkStream.whiten` method
        to further whiten the covariance matrix.
    **kwargs: dict, optional
        Additional keyword arguments passed to the
        :class:`~covseisnet.signal.ShortTimeFourierTransform` class.

    Returns
    -------
    :class:`numpy.ndarray`
        The time vector of the beginning of each covariance window.
    :class:`numpy.ndarray`
        The frequency vector.
    :class:`~covseisnet.covariance.CovarianceMatrix`
        The complex covariance matrix, with a shape depending on the number of
        time windows and frequencies, maximum shape ``(n_times, n_frequencies,
        n_traces, n_traces)``.


    Example
    -------
    Calculate the covariance matrix of the example stream with 1 second
    windows averaged over 5 windows:

    >>> import covseisnet as csn
    >>> stream = csn.read()
    >>> t, f, c = csn.calculate_covariance_matrix(stream, window_duration_sec=1., average=5)
    >>> print(c.shape)
        (27, 51, 3, 3)

    References
    ----------
    .. footbibliography::

    """
    # Assert stream is synchronized
    assert stream.synced, "Stream is not synchronized."

    # Calculate spectrogram
    short_time_fourier_transform = signal.ShortTimeFourierTransform(
        window_duration=window_duration,
        sampling_rate=stream.sampling_rate,
        **kwargs,
    )

    # Extract spectra
    spectra_times, frequencies, spectra = (
        short_time_fourier_transform.map_transform(stream)
    )

    # Check whiten parameter
    if whiten.lower() not in ["none", "slice", "window"]:
        message = "{} is not an available option for whiten."
        raise ValueError(message.format(whiten))

    # Remove modulus
    if whiten.lower() == "window":
        spectra /= np.abs(spectra) + 1e-5

    # Parametrization
    step = average // 2 if average_step is None else average * average_step
    n_traces, n_frequencies, n_times = spectra.shape

    # Times of the covariance matrix
    indices = range(0, n_times - average + 1, step)
    covariance_n_times = len(indices)
    covariance_shape = (covariance_n_times, n_frequencies, n_traces, n_traces)

    # Initialization
    covariance_times = []
    covariances = np.zeros(covariance_shape, dtype=complex)

    # Compute with Einstein convention
    for i, index in enumerate(indices):
        # Slice
        selection = slice(index, index + average)
        spectra_slice = spectra[..., selection]

        # Whiten
        if whiten.lower() == "slice":
            spectra_slice /= np.mean(
                np.abs(spectra_slice),
                axis=-1,
                keepdims=True,
            )

        # Covariance
        covariances[i] = np.einsum(
            "ift,jft -> fij", spectra_slice, np.conj(spectra_slice)
        )

        # Center time
        duration = spectra_times[selection][-1] - spectra_times[selection][0]
        covariance_times.append(spectra_times[selection][0] + duration / 2)

    # Set covariance matrix
    covariances = covariances.view(CovarianceMatrix)

    # Turn times into array
    covariance_times = np.array(covariance_times)

    # Add metadata
    covariances.set_stats([trace.stats for trace in stream])
    covariances.set_stft(short_time_fourier_transform)

    return covariance_times, frequencies, covariances


def are_two_last_dimensions_hermitian(matrix: np.ndarray) -> bool:
    """Check if the two last dimensions of a matrix are Hermitian.

    Arguments
    ---------
    matrix: :class:`numpy.ndarray`
        The input matrix.

    Returns
    -------
    bool
        True if the two last dimensions are Hermitian, False otherwise.
    """
    if matrix.shape[-1] != matrix.shape[-2]:
        return False
    return np.allclose(matrix, np.conj(matrix).swapaxes(-2, -1))
