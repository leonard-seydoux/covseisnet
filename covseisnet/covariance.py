"""Calculation and analysis of the network covariance matrix."""

from typing import Callable

import numpy as np
from numpy.linalg import eigvalsh, eigh
from scipy.linalg import ishermitian
from obspy.core.trace import Stats

from .stream import NetworkStream
from . import signal


class CovarianceMatrix(np.ndarray):
    r"""
    This class inherits all the methods of standard numpy arrays, along with
    extra methods tailored for array processing. Any numpy method or function
    applied to an instance of CovarianceMatrix will return an instance of
    CovarianceMatrix.

    .. rubric:: Mathematical definition

    We consider the set of network seismograms :math:`\{v_i(t)\}_{i=1}^N` with
    :math:`N` traces, all recorded on the same time samples (synchronized).
    For the sake of simplicity, we use the continuous time notation. We first
    calculate the short-time Fourier transform of each trace within a sliding
    window of duration :math:`T`, and center time :math:`t_m` (with the index
    :math:`m` the time window index), defined on the time interval
    :math:`\tau_m = [t_m - T/2, t_m + T/2]`. The short-time Fourier transform
    of the traces is defined as

    .. math::

        u_{i,m}(f) = \mathcal{F} v_i(t \in \tau_m)

    The spectral covariance matrix is defined as the average of the outer
    product of the short-time Fourier transform of the traces over a set of
    time windows :math:`\{\tau_m\}_{m=1}^M`, with :math:`M` the number of
    windows used to estimate the covariance. This lead to the definition of
    the spectral covariance matrix :math:`C_{ij}(f)` as

    .. math::

        C_{ij}(f) = \sum_{m=1}^M u_{i,m}(f) u_{j,m}^*(f)

    where :math:`M` is the number of windows used to estimate the covariance,
    :math:`^*` is the complex conjugate, and :math:`\tau_m` is the time window
    of index :math:`m`. The covariance matrix is a complex-valued matrix of
    shape :math:`N \times N`.

    .. rubric:: Practical implementation

    The short-time Fourier transform is calculated with the help of the
    :class:`~covseisnet.signal.ShortTimeFourierTransform` class. Both the
    calculation of the short-time Fourier transform and the covariance matrix
    can run in parallel depending on the available resources. Depending on the
    averaging size and frequency content, the covariance matrix can have a
    different shape:

    - Shape is ``(n_traces, n_traces)`` if a single frequency and time window
      is given. This configuration is also obtained when slicing or reducing a
      covariance matrix object.

    - Shape is ``(n_frequencies, n_traces, n_traces)`` if only one time frame
      is given, for ``n_frequencies`` frequencies, which depends on the window
      size and sampling rate.

    - Shape is ``(n_times, n_frequencies, n_traces, n_traces)`` if multiple
      time frames are given.

    All the methods defined in the the
    :class:`~covseisnet.covariance.CovarianceMatrix` class are performed on
    the flattened array with the
    :meth:`~covseisnet.covariance.CovarianceMatrix.flat` method, which allow
    to obtain as many :math:`N \times N` covariance matrices as time and
    frequency samples.

    Tip
    ---

    The :class:`~covseisnet.covariance.CovarianceMatrix` class is rarely meant
    to be instantiated directly. It should most often be obtained from the
    output of the :func:`~covseisnet.covariance.calculate_covariance_matrix`
    function as in the following example:

    >>> import covseisnet as csn
    >>> import numpy as np
    >>> stream = csn.NetworkStream()
    >>> time, frequency, covariance = csn.calculate_covariance_matrix(
    ...     stream, window_duration=10.0, average=10
    ... )
    >>> print(type(covariance))
    <class 'covseisnet.covariance.CovarianceMatrix'>


    """

    def __new__(
        cls,
        input_array: "np.ndarray | CovarianceMatrix | list",
        stats: list[Stats] | list[dict] | None = None,
        stft: signal.ShortTimeFourierTransform | None = None,
    ) -> "CovarianceMatrix":
        r"""
        Note
        ----

        If for some reason you need to create a
        :class:`~covseisnet.covariance.CovarianceMatrix` object from a numpy
        array, you have the several solutions provided by the `numpy
        subclassing mechanism
        <https://numpy.org/doc/stable/user/basics.subclassing.html#subclassing-ndarray>`_.
        We recommend using the contructor of the class, which allows to pass
        the stats and stft attributes, as in the following example:

        >>> import covseisnet as csn
        >>> import numpy as np
        >>> covariance = csn.CovarianceMatrix(
        ...     np.zeros((4, 4)), stats=None, stft=None
        ... )
        >>> covariance
        CovarianceMatrix([[ 0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.]])

        In that case, the covariance matrix object passes through several
        tests such as the Hermitian property of the two last dimensions, and
        the number of dimensions of the input array. If the input array does
        not pass the tests, a ValueError is raised. The input array can be any
        array-like object, as it is passed to the :func:`numpy.asarray`
        function, which also allow to turn the input array into a
        complex-valued numpy array. The input array is then cast into a
        CovarianceMatrix object with the `view casting mechanism of numpy
        <https://numpy.org/doc/stable/user/basics.subclassing.html#view-casting>`_,
        and the attributes stats and stft are added to the object. You can
        also directly employ the view method of the numpy array, but you will
        have to set the stats and stft attributes manually.
        """
        # Enforce the input array to be a complex-valued numpy array.
        input_array = np.asarray(input_array, dtype=complex)

        # Check that the input array had at least two dimensions. This is
        # not necessarily the case after slicing a covariance matrix for
        # convenience, so this case is only checked when creating the object.
        if (ndim := input_array.ndim) < 2:
            raise ValueError(f"Input array must be at least 2D, got {ndim}D.")

        # Check that the two last dimensions are Hermitian. Again, this does
        # not need to be checked after slicing the array, but at the creation
        # of the object.
        for index in np.ndindex(input_array.shape[:-2]):
            if not ishermitian(input_array[index], rtol=1e-4):
                raise ValueError("Input last dimensions must be Hermitian.")

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

    def coherence(self, kind="spectral_width", epsilon=1e-10):
        r"""Covariance-based coherence estimation.

        The coherence is obtained from each covariance matrix eigenvalue
        distribution, calculated with the
        :meth:`~covseisnet.covariance.CovarianceMatrix.eigenvalues` method.
        For a given matrix :math:`\mathbf{C} \in \mathbb{C}^{N \times N}`, we
        obtain the eigenvalues :math:`\boldsymbol{\lambda} =
        \{\lambda_i\}_{i=1\ldots N}`, with :math:`\lambda_i \in [0, 1]`
        normalized by the sum of the eigenvalues. The coherence is obtained
        from :math:`c = f(\boldsymbol{\lambda})`, with :math:`f` being defined
        by the ``kind`` parameter:

        - The spectral width is obtained with setting
          ``kind="spectral_width"`` , and returns the width :math:`\sigma` of
          the eigenvalue distribution (obtained at each time and frequency)
          such as

          .. math::

              \sigma = \sum_{i=0}^n i \lambda_i


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

        In each case, :math:`\epsilon` is a regularization parameter to avoid
        the logarithm of zero. The coherence is calculated for each time and
        frequency sample (if any), and the result is a coherence matrix of
        maximal shape ``(n_times, n_frequencies)``. The :math:`\log` and
        :math:`\exp` functions are the natural logarithm and exponential.


        Arguments
        ---------
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

        Raises
        ------
        ValueError
            If the covariance matrix is not Hermitian.

        See also
        --------
        :meth:`~covseisnet.covariance.CovarianceMatrix.eigenvalues`
        """
        # Check that self is still Hermitian
        if not self.is_hermitian:
            raise ValueError("Covariance matrix is not Hermitian.")
        # Calculate coherence
        match kind:
            case "spectral_width":
                eigenvalues = self.eigenvalues(norm=np.sum)
                return signal.width(eigenvalues, axis=-1)

            case "entropy":
                eigenvalues = self.eigenvalues(norm=np.sum)
                return signal.entropy(eigenvalues, axis=-1)

            case "diversity":
                eigenvalues = self.eigenvalues(norm=np.sum)
                return signal.diversity(eigenvalues, axis=-1)

            case _:
                raise NotImplementedError(f"{kind} coherence not implemented.")

    def eigenvalues(self, norm: Callable = np.max) -> np.ndarray:
        r"""Eigenvalue decomposition.

        Given and Hermitian matrix :math:`\mathbf{C} \in \mathbb{C}^{N \times
        N}`, the eigenvalue decomposition is defined as

        .. math::

            \mathbf{C} = \mathbf{U D U}^\dagger

        where :math:`\mathbf{U} \in \mathbb{C}^{N \times N}` is the unitary
        matrix of eigenvectors (which are not calculated here, see
        :meth:`~covseisnet.covariance.CovarianceMatrix.eigenvectors` for this
        purpose), :math:`\mathbf{U}^\dagger` is the conjugate transpose of
        :math:`\mathbf{U}`, and :math:`\mathbf{D} \in \mathbb{R}^{N \times N}`
        is a diagonal matrix of eigenvalues, as

        .. math::

            \mathbf{D} = \pmatrix{\lambda_1 & 0 & \cdots & 0 \\
                            0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots &
                            \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_N}

        with :math:`\lambda_i` the eigenvalues. For convenience, the
        eigenvalues are returned as a vector such as 

        .. math::

            \boldsymbol\lambda = \pmatrix{\lambda_1, \lambda_2, \ldots,
            \lambda_N}
        
        The eigenvalues are sorted in decreasing order and normalized by the
        maximum eigenvalue by default, but can be normalized by any function
        provided by numpy. Since the matrix :math:`\mathbf{C}` is Hermitian by
        definition, the eigenvalues are real- and positive-valued. Also, the
        eigenvectors are orthogonal and normalized.

        The eigenvalue decomposition is performed onto the two last dimensions
        of the :class:`~covseisnet.covariance.CovarianceMatrix` object. The
        function used for eigenvalue decomposition is
        :func:`numpy.linalg.eigvalsh`. It sassumes that the input matrix is 2D
        and hermitian, so the decomposition is performed onto the lower
        triangular part to save time.

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
        >>> cov = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
        >>> cov = np.array([cov @ cov.T.conj() for cov in cov])
        >>> cov = csn.CovarianceMatrix(cov)
        >>> cov.eigenvalues()
            array([[1.        , 0.14577012, 0.02521345],
                   [1.        , 0.13510247, 0.00369051],
                   [1.        , 0.22129766, 0.0148769 ]])
        """
        # Check that self is still Hermitian
        if not self.is_hermitian:
            raise ValueError("Covariance matrix is not Hermitian.")

        # Parallel computation of eigenvalues.
        eigenvalues = eigvalsh(self.flat)

        # Sort and normalize. Note that the eigenvalues are supposed to be
        # real-valued, so the absolute value is taken (Hermitian matrix).
        eigenvalues = np.sort(np.abs(eigenvalues), axis=-1)[:, ::-1]
        eigenvalues /= norm(eigenvalues, axis=-1, keepdims=True)

        return eigenvalues.reshape(self.shape[:-1])

    def eigenvectors(
        self,
        rank: int | tuple | slice | None = None,
        return_covariance: bool = False,
        weights: np.ndarray | None = None,
    ) -> "np.ndarray | CovarianceMatrix":
        r"""Extract eigenvectors of given rank.

        Given a Hermitian matrix :math:`\mathbf{C} \in \mathbb{C}^{N \times
        N}`, the eigenvector decomposition is defined as

        .. math::

            \mathbf{C} = \mathbf{U D U}^\dagger

        where :math:`\mathbf{U} \in \mathbb{C}^{N \times N}` is the unitary
        matrix of eigenvectors, :math:`\mathbf{U}^\dagger` is the conjugate
        transpose of :math:`\mathbf{U}`, and :math:`\mathbf{D} \in
        \mathbb{R}^{N \times N}` is a diagonal matrix of eigenvalues. The
        eigenvectors are normalized and orthogonal, and the eigenvalues are
        real- and positive-valued. The eigenvectors are returned as a matrix

        .. math::

            \mathbf{U} = \pmatrix{\mathbf{u}_1, \mathbf{u}_2, \ldots,
            \mathbf{u}_R}

        with :math:`\mathbf{v}_i` the eigenvectors, and :math:`R` the maximum
        rank of the eigenvectors. The eigenvectors are sorted in decreasing
        order of the eigenvalues, and the eigenvectors are normalized. The
        function used for extracting eigenvectors is
        :func:`scipy.linalg.eigh`. It assumes that the input matrix is 2D and
        hermitian, so the decomposition is performed onto the lower triangular
        part.

        If the ``covariance`` parameter is set to False (default), the
        eigenvectors are returned as a matrix of shape ``(n_times,
        n_frequencies, n_stations, rank)`` if the parameter ``covariance`` is
        ``False``, else ``(n_times, n_frequencies, n_stations, n_stations)``,
        resulting from the outer product of the eigenvectors.

        If the ``covariance`` parameter is set to True, the return value is a
        covariance matrix object with the outer product of the eigenvectors
        multiplied by the given weights :math:`\mathbf{w} \in \mathbb{R}^{R}`:

        .. math::

            \tilde{\mathbf{C}} = \sum_{i=1}^R w_i \mathbf{u}_i
            \mathbf{u}_i^\dagger

        The weights are the eigenvalues of the covariance matrix if
        ``weights`` is None (default), else the weights are the input weights.
        The weights are used to scale the eigenvectors before the outer
        product. In particular, if the weights are zeros and ones, this
        function can be used to apply spatial whitening to the covariance
        matrix.

        Arguments
        ---------
        rank : int, tuple, slice, or None, optional
            The rank of the eigenvectors. If None, all eigenvectors are
            returned. If a tuple or slice, the eigenvectors are sliced
            according to the tuple or slice. If an integer is given, the
            eigenvectors are sliced according to the integer. Default is None.
        return_covariance: bool, optional
            If True, the outer product of the eigenvectors is returned as a
            covariance matrix object. Default is False.
        weights: :class:`numpy.ndarray`, optional
            The weights used to scale the eigenvectors before the outer
            product. If None, the eigenvalues are used as weights. Default is
            None.

        Returns
        -------
        :class:`covseisnet.covariance.CovarianceMatrix`
            The complex-valued eigenvector array of shape ``(n_times,
            n_frequencies, n_stations, rank)`` if ``covariance`` is ``False``,
            else ``(n_times, n_frequencies, n_stations, n_stations)``.

        Raises
        ------
        ValueError
            If the covariance matrix is not Hermitian.

        See also
        --------
        :meth:`~covseisnet.covariance.CovarianceMatrix.eigenvalues`
        :func:`scipy.linalg.eigh`

        """
        # Check that self is still Hermitian.
        if not self.is_hermitian:
            raise ValueError("Covariance matrix is not Hermitian.")

        # Calculate eigenvectors.
        eigenvalues, eigenvectors = eigh(self.flat)

        # Sort according to eigenvalues.
        isort = np.argsort(np.abs(eigenvalues)[:, ::-1])[:, ::-1]
        eigenvalues = np.take_along_axis(np.abs(eigenvalues), isort, axis=1)
        eigenvectors = np.take_along_axis(
            eigenvectors, isort[:, :, np.newaxis], axis=1
        )

        # Select according to rank.
        if rank is not None:
            if isinstance(rank, int):
                rank = (rank,)
            eigenvectors = eigenvectors[..., rank]
            eigenvalues = eigenvalues[..., rank]

        # Perform outer product if covariance is True
        if return_covariance is True:
            if weights is None:
                weights = eigenvalues
            reconstructed = np.zeros_like(self.flat)
            for i, vectors in enumerate(eigenvectors):
                weight = np.diag(weights[i])
                reconstructed[i] += vectors @ weight @ np.conj(vectors.T)

            # Reshape to original shape
            return CovarianceMatrix(
                reconstructed.reshape(self.shape),
                stats=self.stats,
                stft=self.stft,
            )

        else:
            return eigenvectors.reshape(
                (*self.shape[:-1], eigenvectors.shape[-1])
            )

    @property
    def flat(self):
        r"""Covariance matrices with flatten first dimensions.

        The shape of the covariance matrix depend on the number of time
        windows and frequencies. The property
        :attr:`~covseisnet.covariance.CovarianceMatrix.flat` allows to obtain
        as many :math:`N \times N` covariance matrices as time and frequency
        samples.

        Returns
        -------
        :class:`np.ndarray`
            The covariance matrices in a shape ``(n_times * n_frequencies,
            n_traces, n_traces)``.

        Example
        -------
        >>> import covseisnet as csn
        >>> import numpy as np
        >>> cov = np.zeros((5, 4, 2, 2))
        >>> cov = csn.CovarianceMatrix(cov)
        >>> cov.shape
            (5, 4, 2, 2)
        >>> c.flat.shape
            (20, 2, 2)
        """
        return self.reshape(-1, *self.shape[-2:])

    def triu(self, **kwargs):
        r"""Extract upper triangular matrices.

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

        >>> import covseisnet as csn
        >>> import numpy as np
        >>> cov = np.zeros((5, 4, 2, 2))
        >>> cov = csn.CovarianceMatrix(cov)
        >>> cov.triu().shape
            (5, 4, 3)

        """
        trii, trij = np.triu_indices(self.shape[-1], **kwargs)
        return self[..., trii, trij]

    def twosided(self, axis: int = -3) -> "CovarianceMatrix":
        r"""Get the full covariance spectrum.

        Given that the covariance matrix is Hermitian, the full covariance
        matrix can be obtained by filling the negative frequencies with the
        complex conjugate of the positive frequencies. The method
        :meth:`~covseisnet.covariance.CovarianceMatrix.twosided` performs this
        operation.

        The frequency axis is assumed to be the second axis of the covariance
        matrix. The function returns a new covariance matrix with the negative
        frequencies filled with the complex conjugate of the positive
        frequencies.

        Arguments
        ---------
        axis: int, optional
            The frequency axis of the covariance matrix. Default is -3.

        Returns
        -------
        :class:`~covseisnet.covariance.CovarianceMatrix`
            The full covariance matrix.
        """
        # Get number of samples used to calculate the covariance matrix
        if self.stft is None:
            raise ValueError("ShortTimeFourierTransform instance not found.")

        # Get number of samples in the window
        n_samples = len(self.stft.win)

        # Find out output shape
        input_shape = self.shape
        output_shape = list(input_shape)
        output_shape[axis] = n_samples

        # Initialize full covariance matrix with negative frequencies
        output = np.zeros(output_shape, dtype=np.complex128)

        # Fill negative frequencies
        n_oneside = n_samples // 2 + 1
        output[:, :n_oneside] = self
        output[:, n_oneside:] = np.conj(self[:, -2:0:-1])

        return CovarianceMatrix(output, stats=self.stats, stft=self.stft)

    @property
    def is_hermitian(self, tol: float = 1e-10) -> bool:
        r"""Check if the covariance matrix is Hermitian.

        Given a covariance matrix :math:`\mathbf{C} \in \mathbb{C}^{N \times
        N}`, the matrix is Hermitian if the matrix is equal to its conjugate
        transpose, that is

        .. math::

            \mathbf{C} = \mathbf{C}^\dagger

        to some extent provided by the ``tol`` parameter. This check is
        performed on all the covariance matrices of the object, e.g., for each
        time and frequency sample if any.

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
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, CovarianceMatrix]:
    r"""Calculate covariance matrix.

    The covariance matrix is calculated from the Fourier transform of the
    input stream. The covariance matrix is calculated for each time window and
    frequency, and averaged over a given number of windows. Please refer to
    the :class:`~covseisnet.covariance.CovarianceMatrix` class, rubric
    **Mathematical definition** for a detailed explanation of the covariance
    matrix. You may also refer to the paper :footcite:`seydoux_detecting_2016`
    for deeper insights.

    .. rubric:: Covariance-level whitening

    The whitening parameter can be used to normalize the covariance matrix
    directly, in addition to the trace-level whitening implementation in the
    :meth:`covseisnet.stream.NetworkStream.whiten` method. The parameter can
    be set to "none" (default), "slice", or "window". The "none" option does
    not apply any whitening to the covariance matrix. The "slice" option
    normalizes the spectra :math:`u_{i,m}(f)` by the mean of the absolute
    value of the spectra within the same group of time windows
    :math:`\{\tau_m\}_{m=1}^M`, so that

    .. math::

        u_{i,m}(f) = \frac{u_{i,m}(f)}{\sum_{i=1}^M |u_{i,m}(f)|}

    The "window" option normalizes the spectra :math:`u_{i,m}(f)` by the
    absolute value of the spectra within the same time window :math:`\tau_m`
    so that

    .. math::

        u_{i,m}(f) = \frac{u_{i,m}(f)}{|u_{i,m}(f)|}

    These additional whitening methods can be used in addition to the
    trave-level :meth:`~covseisnet.stream.NetworkStream.whiten` method to
    further whiten the covariance matrix.

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
        The type of whitening applied to the covariance matrix. Can be "none"
        (default), "slice", or "window". This parameter can be used in
        addition to the :meth:`~covseisnet.stream.NetworkStream.whiten` method
        to further whiten the covariance matrix.
    **kwargs: dict, optional
        Additional keyword arguments passed to the
        :class:`~covseisnet.signal.ShortTimeFourierTransform` class.

    Returns
    -------
    :class:`numpy.ndarray`
        The time vector of the beginning of each covariance window, expressed
        in matplotlib time.
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
    >>> t, f, c = csn.calculate_covariance_matrix(
    ...     stream, window_duration_sec=1., average=5
    ... )
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
    covariance_shape = (len(indices), n_frequencies, n_traces, n_traces)

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
                np.abs(spectra_slice), axis=-1, keepdims=True
            )

        # Covariance
        covariances[i] = np.einsum(
            "ift,jft -> fij", spectra_slice, np.conj(spectra_slice)
        )

        # Center time
        duration = spectra_times[selection][-1] - spectra_times[selection][0]
        covariance_times.append(spectra_times[selection][0] + duration / 2)

    # Set covariance matrix
    covariances = CovarianceMatrix(
        input_array=covariances,
        stft=short_time_fourier_transform,
        stats=[trace.stats for trace in stream],
    )

    # Turn times into array
    covariance_times = np.array(covariance_times)

    return covariance_times, frequencies, covariances
