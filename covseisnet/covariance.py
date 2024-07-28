"""Spectral analysis and covariance matrix calculation."""

import numpy as np
from numpy.linalg import eigvalsh, eigh

from .stream import NetworkStream
from .signal import ShortTimeFourierTransform


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

    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype=complex).view(cls)
        obj.stations = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stations = getattr(obj, "stations", None)

    def __getitem__(self, index):
        result = super(CovarianceMatrix, self).__getitem__(index)
        if isinstance(result, np.ndarray):
            result = result.view(CovarianceMatrix)
            result.stations = self.stations
        return result

    def set_stations(self, stations):
        """Set the station names.

        Arguments
        ---------
        stations: list
            The list of station names.
        """
        self.stations = stations

    def coherence(self, kind="spectral_width", epsilon=1e-10):
        r"""Covariance-based coherence estimation.

        The measured is performed onto all the covariance matrices from
        the eigenvalues obtained with the method
        :meth:`~covseisnet.covariancematrix.CovarianceMatrix.eigenvalues`.
        For a given matrix :math:`N \times N` matrix :math:`M` with
        eigenvalues :math:`\mathbf{\lambda} = \lambda_i` where
        :math:`i=1\ldots n`. The coherence is obtained as :math:`F(\lambda)`,
        with :math:`F` being defined by the `kind` parameter.

        - The spectral width is obtained with setting ``kind='spectral_width'``
          , and returns the width :math:`\sigma` of the
          eigenvalue distribution such as

          .. math::

              \sigma = \frac{\sum_{i=0}^n i \lambda_i}{\sum_{i=0}^n \lambda_i}


        - The entropy is obtained with setting ``kind='entropy'``, and returns
          the entropy :math:`h` of the eigenvalue distribution such as

          .. math::

              h = - \sum_{i=0}^n i \lambda_i \log(\lambda_i + \epsilon)


        Keyword arguments
        -----------------
        kind: str, optional
            The type of coherence, may be "spectral_width" (default) or
            "entropy".

        epsilon: float, optional
            The regularization parameter for log-entropy calculation. Default
            to ``1e-10``.

        Returns
        -------

        :class:`numpy.ndarray`
            The spectral width of maximal shape ``(n_times, n_frequencies)``.

        """
        if kind == "spectral_width":
            eigenvalues = self.eigenvalues(norm=np.sum)
            indices = np.arange(self.shape[-1])
            return np.multiply(eigenvalues, indices).sum(axis=-1)
        elif kind == "entropy":
            eigenvalues = self.eigenvalues(norm=np.sum)
            log_eigenvalues = np.log(eigenvalues + epsilon)
            return -np.sum(eigenvalues * log_eigenvalues, axis=-1)
        else:
            message = "{} is not an available option for kind."
            raise ValueError(message.format(kind))

    def eigenvalues(self, norm=np.max):
        """Eigenvalue decomposition.

        The eigenvalue decomposition is performed onto the two last dimensions
        of the :class:`~covseisnet.covariancematrix.CovarianceMatrix` object.
        The function used for eigenvalue decomposition is
        :func:`scipy.linalg.eigvalsh`. It assumes that the input matrix is 2D
        and hermitian. The decomposition is performed onto the lower triangular
        part in order to save time.

        Keyword arguments
        -----------------
        norm : function, optional
            The function used to normalize the eigenvalues. Can be :func:`max`,
            (default), any other functions.

        Returns
        -------
        :class:`numpy.ndarray`
            The eigenvalues of maximal shape ``(n_times, n_freq, n_sta)``.

        """
        # Flatten the array
        matrices = self.flat()

        # Parallel computation of eigenvalues
        eigs = eigvalsh(matrices)

        # Sort and normalize
        eigs = np.sort(np.abs(eigs), axis=-1)[:, ::-1]
        eigs /= norm(eigs, axis=-1, keepdims=True)

        # Original shape
        eigevalue_shape = self.shape[:-1]
        return eigs.reshape(eigevalue_shape)

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
        """Covariance matrices with flatten first dimensions.

        Returns
        -------

        :class:`np.ndarray`
            The covariance matrices in a shape ``(a * b, n, n)``.
        """
        return self.reshape(-1, *self.shape[-2:])

    def triu(self, **kwargs):
        """Extract upper triangular on flatten array.

        Keyword arguments
        -----------------
        **kwargs: dict, optional
            The keyword arguments passed to the :func:`numpy.triu` function.


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


def calculate_covariance_matrix(
    stream: NetworkStream,
    average: int,
    average_step: int | None = None,
    **kwargs: dict,
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

    Arguments
    ---------
    stream: :class:`~covseisnet.stream.NetworkStream`
        The input data stream.
    average: int
        The number of window used to estimate the sample covariance.
    average_step: int, optional
        The sliding window step for covariance matrix calculation (in number
        of windows).
    **kwargs: dict, optional
        Additional keyword arguments passed to the
        :func:`~covseisnet.stream.calculate_short_time_spectra` function.

    Returns
    -------
    :class:`numpy.ndarray`
        The time vector of the beginning of each covariance window.
    :class:`numpy.ndarray`
        The frequency vector.
    :class:`covseisnet.covariance.CovarianceMatrix`
        The complex covariance matrix, with a shape depending on the number of
        time windows and frequencies, maximum shape ``(n_times, n_frequencies,
        n_traces, n_traces)``.


    Example
    -------
    Calculate the covariance matrix of the example stream with 1 second
    windows averaged over 5 windows:

    >>> import covseisnet as csn
    >>> stream = csn.read()
    >>> t, f, c = csn.covariance.calculate_covariance_matrix(stream, 1., 5)
    >>> print(c.shape)
        (28, 199, 3, 3)


    References
    ----------
    .. footbibliography::

    """
    # Calculate spectrogram
    kwargs["sampling_rate"] = stream.sampling_rate
    stft = ShortTimeFourierTransform(**kwargs)

    # Extract spectra
    spectra_times, frequencies, spectra = stft.map_transform(stream)

    # Remove modulus
    # spectra /= np.abs(spectra) + 1e-5

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
        spectra_slice /= np.mean(np.abs(spectra_slice), axis=-1, keepdims=True)
        covariances[i] = np.einsum(
            "ift,jft -> fij", spectra_slice, np.conj(spectra_slice)
        )

        # Center time
        duration = spectra_times[selection][-1] - spectra_times[selection][0]
        # print(spectra_times[selection][0])
        # print(spectra_times[selection][-1])
        covariance_times.append(spectra_times[selection][0] + duration / 2)

    # Add stations
    covariances = covariances.view(CovarianceMatrix)
    covariances.set_stations(stream.stations)

    return covariance_times, frequencies, covariances
