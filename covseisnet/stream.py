"""
The package **covseisnet** provides a comprehensive toolkit for analyzing
seismic data recorded on seismic networks. To achieve this, we offer a set of
classes and methods designed for reading, pre-processing, and analyzing
seismic data from these networks.

The workflow involves working on traces that have been synchronized and
pre-processed using consistent methods. To facilitate this, we introduce the
concept of a :class:`~covseisnet.stream.NetworkStream`, a subclass of the
ObsPy :class:`~obspy.core.stream.Stream` object. The
:class:`~covseisnet.stream.NetworkStream` object retains all the methods of
the Stream object while adding specialized methods for the pre-processing and
synchronization of traces.
"""

from functools import partial

import numpy as np
import obspy
from scipy import signal, stats


class NetworkStream(obspy.Stream):
    """
    Subclass of the ObsPy :class:`~obspy.core.stream.Stream` tailored for
    managing continuous data from seismic networks. The class is designed to
    handle multiple traces from different stations, and provides additional
    methods for pre-processing and synchronization of the traces. It also
    provide network-wide methods such as the calculation of the common time
    vector of the traces.

    .. rubric:: _`Attributes`

    - :attr:`~covseisnet.stream.NetworkStream.are_ready_to_process` — check if
      traces are ready to be processed.

    - :attr:`~covseisnet.stream.NetworkStream.are_time_vectors_equal` — check
      if traces are sampled on the same time vector.

    - :attr:`~covseisnet.stream.NetworkStream.are_sampling_rates_equal` —
      check if all traces have the same sampling rate.

    - :attr:`~covseisnet.stream.NetworkStream.are_npts_equal` — check if all
      traces have the same number

    - :attr:`~covseisnet.stream.NetworkStream.stations` — list of unique
      station

    - :attr:`~covseisnet.stream.NetworkStream.channels` — list of unique
      channel

    - :attr:`~covseisnet.stream.NetworkStream.sampling_rate` — sampling rate
      of the traces.

    .. rubric:: _`Methods`

    - :meth:`~covseisnet.stream.NetworkStream.cut()` — trim stream between
      given start and end times.

    - :meth:`~covseisnet.stream.NetworkStream.times()` — common time vector of
      the NetworkStream.

    - :meth:`~covseisnet.stream.NetworkStream.synchronize()` — synchronize the
      traces into the same times with interpolation.

    - :meth:`~covseisnet.stream.NetworkStream.whiten()` — whiten traces in the
      spectral domain.

    - :meth:`~covseisnet.stream.NetworkStream.normalize()` — normalize the
      traces in the temporal domain.



    .. note::

        This class is not meant to be instantiated directly. Instead, it is
        returned by the :func:`~covseisnet.stream.read` function when reading
        seismic data, as shown in the example below.

        .. doctest::

            >>> import covseisnet as csn
            >>> stream = csn.read()
            >>> print(type(stream))
            <class 'covseisnet.stream.NetworkStream'>
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self, **kwargs):
        """Print the NetworkStream object.

        This method prints the NetworkStream object in a human-readable
        format. The methods ressembles the ObsPy Stream object, but with
        additional information about the number of traces and stations in the
        stream. By default, the method prints all traces in the stream.

        Arguments
        ---------
        **kwargs: dict, optional
            A way to handle legacy arguments. No argument is used in this method.
        """
        # Get longest trace id index among traces
        if self.traces:
            longest_id = self and max(len(tr.id) for tr in self) or 0
        else:
            longest_id = 0

        # Get number of traces and stations
        n_traces = len(self.traces)
        n_stations = len(self.stations)

        # Synced flag
        synced_flag = "synced" if self.are_time_vectors_equal else "not synced"

        # Initialize output string
        out = f"NetworkStream of {n_traces} traces from {n_stations} stations ({synced_flag}):\n"

        # Print all traces
        out = out + "\n".join([trace.__str__(longest_id) for trace in self])

        return out

    def cut(
        self,
        starttime: str | obspy.UTCDateTime,
        endtime: str | obspy.UTCDateTime,
        **kwargs: dict,
    ):
        """Trim traces between start and end date times.

        This function is a wrapper to the ObsPy
        :meth:`~obspy.core.stream.Stream.trim` method, but supports string
        format for the start and end times. The function uses the ObsPy
        :class:`~obspy.core.utcdatetime.UTCDateTime` function in order to
        parse the start and end times.

        Arguments
        ---------

        starttime : str or :class:`~obspy.core.utcdatetime.UTCDateTime`
            The start date time.
        endtime : str or :class:`~obspy.core.utcdatetime.UTCDateTime`
            The end date time.
        **kwargs: dict, optional
            Arguments passed to the :meth:`~obspy.core.stream.Stream.trim` method.

        Example
        -------

        This example shows how to cut a stream between two given times. The
        stream is first read from the example data, and then cut between two
        given times.

        >>> import covseisnet as csn
        >>> stream = csn.read()
        >>> stream.cut("2009-08-24 00:20:05", "2009-08-24 00:20:12")
        >>> print(stream)
        Network Stream of 3 traces from 1 stations:
        BW.RJOB..EHZ | 2009-08-24T00:20:05.000000Z... | 100.0 Hz, 701 samples
        W.RJOB..EHN  | 2009-08-24T00:20:05.000000Z... | 100.0 Hz, 701 samples
        BW.RJOB..EHE | 2009-08-24T00:20:05.000000Z... | 100.0 Hz, 701 samples

        """
        # Convert start and end times to UTCDateTime
        starttime = obspy.UTCDateTime(starttime)
        endtime = obspy.UTCDateTime(endtime)

        # Trim stream
        self.trim(starttime, endtime, **kwargs)

    def times(self, **kwargs: dict) -> np.ndarray:
        """Common time vector.

        Because the :class:`~covseisnet.stream.NetworkStream` handles traces
        sampled on the same time vector, this function only returns the times
        of the first trace with the Trace method
        :meth:`~obspy.core.trace.Trace.times` if the traces are synchronized.

        Arguments
        ---------
        **kwargs: dict, optional
            Arguments passed to the method
            :meth:`~obspy.core.trace.Trace.times`. For instance, passing
            ``type="matplotlib"`` allows to recover matplotlib timestamps
            instead of seconds from the start of the trace (default).

        Returns
        -------
        numpy.ndarray
            The timestamps.

        Raises
        ------
        AssertionError
            If the traces are not synchronized.

        See Also
        --------
        :meth:`~obspy.core.trace.Trace.times`


        .. tip::

            By default, the method returns the times in seconds since the
            start of the trace. In order to extract times in matplotlib
            format, you can set the ``type`` parameter of the
            :meth:`~obspy.core.trace.Trace.times` method such as

            >>> import covseisnet as csn
            >>> stream = csn.read()
            >>> stream.times(type="matplotlib")
            array([14480.01392361, 14480.01392373, 14480.01392384, ...,
            14480.01427049, 14480.0142706 , 14480.01427072])

        """
        # Check if the traces are synchronized
        assert (
            self.are_time_vectors_equal
        ), "Traces are not synced, check the `synchronize` method."

        # Return the times of the first trace
        return self[0].times(**kwargs)

    def synchronize(
        self,
        interpolation_method: str = "linear",
        **kwargs: dict,
    ) -> None:
        """Synchronize seismic traces with interpolation.

        This method synchronizes the seismic traces in the stream by
        interpolating the traces to a common time vector. The method uses the
        largest start time and the smallest end time of the traces to
        interpolate all traces to the same time vector with the ObsPy
        method :meth:`~obspy.core.trace.Trace.interpolate`.

        Arguments
        ---------
        method: str, default
            Interpolation method. Default to "linear".
        **kwargs: dict, optional
            Additional keyword arguments are passed to the
            :meth:`~obspy.core.trace.Trace.interpolate` method. Check the ObsPy
            documentation for more details on the available options.
        """
        # Find out the largest start time and the smallest end time
        largest_starttime = max([trace.stats.starttime for trace in self])
        smallest_endtime = min([trace.stats.endtime for trace in self])
        duration = smallest_endtime - largest_starttime
        npts = int(duration * self[0].stats.sampling_rate) + 1

        # Update kwargs
        kwargs.setdefault("method", interpolation_method)
        kwargs.setdefault("npts", npts)
        kwargs.setdefault("starttime", largest_starttime)
        kwargs.setdefault("endtime", smallest_endtime)
        kwargs.setdefault("sampling_rate", self.sampling_rate)

        # Interpolate all traces
        for trace in self:
            trace.interpolate(**kwargs)

    @property
    def are_ready_to_process(self) -> bool:
        """Check if traces are ready to be processed.

        This method checks if the traces are ready to be processed. This is
        useful to ensure that the traces are synchronized before performing any
        operation that requires the traces to be sampled on the same time vector.

        Returns
        -------
        bool
            True if all traces are ready
        """
        # Assert sampling rate
        assert (
            self.are_sampling_rates_equal
        ), "Traces have different sampling rates."

        # Assert number of samples
        assert self.are_npts_equal, "Traces have different number of samples."

        # Check if all traces are ready
        return self.are_time_vectors_equal

    @property
    def are_time_vectors_equal(self) -> bool:
        """Check if traces are sampled on the same time vector.

        This method checks if all traces are sampled on the same time vector.
        This is useful to ensure that the traces are synchronized before
        performing any operation that requires the traces to be sampled on the
        same time vector.

        Returns
        -------
        bool
            True if all traces are sampled on the same time vector, False
            otherwise.
        """
        # Assert sampling rate
        assert (
            self.are_sampling_rates_equal
        ), "Traces have different sampling rates."

        # Assert number of samples
        assert self.are_npts_equal, "Traces have different number of samples."

        # Collect time vectors. We use the matplotlib format for comparison of
        # the absolute values of the time vectors.
        time_vectors = [trace.times(type="matplotlib") for trace in self]

        # Check if all time vectors are the same (only the first is enough)
        for time_vector in time_vectors:
            if not np.allclose(time_vector, time_vectors[0], rtol=0):
                return False
        return True

    @property
    def are_sampling_rates_equal(self) -> bool:
        """Check if all traces have the same sampling rate.

        This method checks if all traces have the same sampling rate. This is
        useful to ensure that the traces are synchronized before performing any
        operation that requires the traces to be sampled on the same time vector.

        Returns
        -------
        bool
            True if all traces have the same sampling rate, False otherwise.
        """
        # Collect sampling rates
        sampling_rates = [trace.stats.sampling_rate for trace in self]

        # Check if all sampling rates are the same (only the first is enough)
        if len(set(sampling_rates)) > 1:
            return False
        return True

    @property
    def are_npts_equal(self) -> bool:
        """Check if all traces have the same number of samples.

        This method checks if all traces have the same number of samples. This
        is useful to ensure that the traces are synchronized before performing
        any operation that requires the traces to be sampled on the same time
        vector.

        Returns
        -------
        bool
            True if all traces have the same number of samples, False otherwise.
        """
        # Collect number of samples
        npts = [trace.stats.npts for trace in self]

        # Check if all number of samples are the same (only the first is enough)
        if len(set(npts)) > 1:
            return False
        return True

    @property
    def stations(self) -> set[str]:
        """List of unique station names.

        This property is also available directly from looping over the traces
        and accessing the :attr:`~obspy.core.trace.Trace.stats.station` attribute.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.stations
        {'RJOB'}

        """
        return set([trace.stats.station for trace in self.traces])

    @property
    def channels(self) -> set[str]:
        """List of unique channel names.

        This property is also available directly from looping over the traces
        and accessing the :attr:`~obspy.core.trace.Trace.stats.channel` attribute.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.channels
        {'EHZ', 'EHN', 'EHE'}
        """
        return set([trace.stats.channel for trace in self.traces])

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the traces.

        This property is also available directly from looping over the traces
        and accessing the :attr:`~obspy.core.trace.Trace.stats.sampling_rate` attribute.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.sampling_rate
        100.0
        """
        # Assert sampling rate
        assert (
            self.are_sampling_rates_equal
        ), "Traces have different sampling rates."

        # Return the sampling rate of the first trace
        return self[0].stats.sampling_rate

    def whiten(
        self,
        method: str = "onebit",
        smooth_length: int = 11,
        smooth_order: int = 1,
        epsilon: float = 1e-10,
        **kwargs: dict,
    ):
        r"""Whiten traces in the spectral domain.

        The action of whitening a seismic trace is to normalize the trace in
        the spectral domain. Typically, the spectrum becomes flat after
        whitening, resembling white noise. This strategy is often used to
        remove the influence of time-localized signal and diminish the site
        effects from a seismic station to another. Any local source is also
        drastically reduced thanks to the whitening process.

        Arguments
        ---------
        method: str, optional
            Must be one of "onebit" (default), or "smooth".

            - ``"onebit"`` divides the spectrum by its modulus, and keeps the
              phase. It uses the :func:`modulus_division` function.

            - ``"smooth"`` divides the spectrum by a smoothed version of its
              modulus. The smoothing is performed with the Savitzky-Golay
              filter, and the order and length of the filter are set by the
              ``smooth_length`` and ``smooth_order`` parameters. It makes use
              of the :func:`smooth_modulus_division` function.

        smooth_length: int, optional
            The length of the Savitzky-Golay filter for smoothing the
            spectrum. This parameter is only used if the ``method`` parameter
            is set to "smooth". Default to 11 points in frequency.
        smooth_order: int, optional
            The order of the Savitzky-Golay filter for smoothing the spectrum.
            This parameter is only used if the ``method`` parameter is set to
            "smooth". Default to 1, which corresponds to a linear filter.
        epsilon: float, optional
            Regularization parameter in division, set to ``1e-10`` by default.
        **kwargs: dict, optional
            Additional keyword arguments passed to the :func:`self.stft`
            method.

        """
        # Instanciate ShortTimeFFT object
        kwargs.setdefault("sampling_rate", self.sampling_rate)
        transform = stft(**kwargs)

        # Assert that the transform is invertible
        assert transform.invertible, "The transform is not invertible."

        # Define the whitening method
        if method == "onebit":
            whiten_method = partial(
                modulus_division,
                epsilon=epsilon,
            )
        elif method == "smooth":
            whiten_method = partial(
                smooth_modulus_division,
                smooth=smooth_length,
                order=smooth_order,
                epsilon=epsilon,
            )
        else:
            raise ValueError("Unknown method {}".format(method))

        # Loop over traces
        for trace in self:

            # Calculate the Short-Time Fourier Transform
            waveform = trace.data
            spectrum = transform.stft(waveform)

            # Whiten the spectrum
            spectrum = whiten_method(spectrum)

            # Inverse Short-Time Fourier Transform
            waveform = transform.istft(spectrum)
            trace.data = waveform

    def normalize(
        self, method="onebit", smooth_length=11, smooth_order=1, epsilon=1e-10
    ):
        r"""Normalize the seismic traces in temporal domain.
        Considering :math:`x_i(t)` being the seismic trace :math:`x_i(t)`, the
        normalized trace :math:`\tilde{x}_i(t)` is obtained with

        .. math::

            \tilde{x}_i(t) = \frac{x_i(t)}{Fx_i(t) + \epsilon}

        where :math:`Fx` is a characteristic of the trace :math:`x` that
        depends on the ``method`` argument, and :math:`\epsilon > 0` is a
        regularization value to avoid division by 0, set by the ``epsilon``
        keyword argument.

        Keyword arguments
        -----------------
        method : str, optional
            Must be one of "onebit" (default), "mad", or "smooth".

            - "onebit" compress the seismic trace into a series of 0 and 1. In
              this case, :math:`F` is defined as :math:`Fx(t) = |x(t)|`.

            - "mad" normalize each trace by its median absolute deviation. In
              this case, :math:`F` delivers a scalar value defined as
              :math:`Fx(t) = \text{MAD}x(t) = \text{median}(|x(t) - \langle
              x(t)\rangle|)`, where :math:`\langle x(t)\rangle)` is the
              signal's average.

            - "smooth" normalize each trace by a smooth version of its
              envelope. In this case, :math:`F` is obtained from the signal's
              Hilbert envelope.

        smooth_length: int, optional
            If the ``method`` keyword argument is set to "smooth", the
            normalization is performed with the smoothed trace envelopes,
            calculated over a sliding window of `smooth_length` samples.
        smooth_order: int, optional
            If the ``method`` keyword argument is set to "smooth", the
            normalization is performed with the smoothed trace envelopes. The
            smoothing order is set by the ``smooth_order`` parameter.
        epsilon: float, optional
            Regularization parameter in division, set to ``1e-10`` by default.

        """
        if method == "onebit":
            for trace in self:
                trace.data = trace.data / (np.abs(trace.data) + epsilon)
        elif method == "smooth":
            for trace in self:
                envelope = np.abs(signal.hilbert(trace.data))
                smooth_envelope = signal.savgol_filter(
                    envelope, smooth_length, smooth_order
                )
                trace.data = trace.data / (smooth_envelope + epsilon)
        else:
            raise ValueError("Unknown method {}".format(method))


def read(pathname_or_url=None, **kwargs):
    """Read seismic waveforms files into an NetworkStream object.

    This function uses the :func:`obspy.core.stream.read` function to read
    the streams. A detailed list of arguments and options are available at
    https://docs.obspy.org. This function opens either one or multiple
    waveform files given via file name or URL using the ``pathname_or_url``
    attribute. The format of the waveform file will be automatically detected
    if not given. See the `Supported Formats` section in
    the :func:`obspy.core.stream.read` function.

    This function returns an :class:`~covseisnet.arraystream.ArrayStream` object, an
    object directly inherited from the :class:`obspy.core.stream.Stream`
    object.

    Keyword arguments
    -----------------
    pathname_or_url: str or io.BytesIO or None
        String containing a file name or a URL or a open file-like object.
        Wildcards are allowed for a file name. If this attribute is omitted,
        an example :class:`~covseisnet.arraystream.ArrayStream` object will be
        returned.

    Other parameters
    ----------------
    **kwargs: dict
        Other parameters are passed to the :func:`obspy.core.stream.read`
        directly.

    Returns
    -------
    :class:`~covseisnet.arraystream.ArrayStream`
        An :class:`~covseisnet.arraystream.ArrayStream` object.

    Example
    -------

    In most cases a filename is specified as the only argument to
    :func:`obspy.core.stream.read`. For a quick start you may omit all
    arguments and ObsPy will create and return a basic example seismogram.
    Further usages of this function can be seen in the ObsPy documentation.

    >>> import covseisnet as cn
    >>> stream = cn.arraystream.read()
    >>> print(stream)
    3 Trace(s) in Stream:
    BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples

    .. rubric:: _`Further Examples`

    Example waveform files may be retrieved via https://examples.obspy.org.
    """
    stream = obspy.read(pathname_or_url, **kwargs)
    stream = NetworkStream(stream)
    return stream


def normalize(
    stream, method="onebit", smooth_length=11, smooth_order=1, epsilon=1e-10
):
    r"""Normalize the seismic traces in temporal domain.

    Considering :math:`x_i(t)` being the seismic trace :math:`x_i(t)`, the
    normalized trace :math:`\tilde{x}_i(t)` is obtained with

    .. math::
        \tilde{x}_i(t) = \frac{x_i(t)}{Fx_i(t) + \epsilon}

    where :math:`Fx` is a characteristic of the trace :math:`x` that
    depends on the ``method`` argument, and :math:`\epsilon > 0` is a
    regularization value to avoid division by 0, set by the ``epsilon``
    keyword argument.

    Keyword arguments
    -----------------
    method : str, optional
        Must be one of "onebit" (default), "mad", or "smooth".

        - "onebit" compress the seismic trace into a series of 0 and 1.
          In this case, :math:`F` is defined as :math:`Fx(t) = |x(t)|`.

        - "mad" normalize each trace by its median absolute deviation.
          In this case, :math:`F` delivers a scalar value defined as
          :math:`Fx(t) = \text{MAD}x(t) =
          \text{median}(|x(t) - \langle x(t)\rangle|)`, where
          :math:`\langle x(t)\rangle)` is the signal's average.

        - "smooth" normalize each trace by a smooth version of its
          envelope. In this case, :math:`F` is obtained from the
          signal's Hilbert envelope.

    smooth_length: int, optional
        If the ``method`` keyword argument is set to "smooth", the
        normalization is performed with the smoothed trace envelopes,
        calculated over a sliding window of `smooth_length` samples.


    smooth_order: int, optional
        If the ``method`` keyword argument is set to "smooth", the
        normalization is performed with the smoothed trace envelopes.
        The smoothing order is set by the ``smooth_order`` parameter.


    epsilon: float, optional
        Regularization parameter in division, set to ``1e-10`` by default.

    """
    if method == "onebit":
        for trace in stream:
            trace.data = trace.data / (np.abs(trace.data) + epsilon)

    elif method == "smooth":
        for trace in stream:
            trace_env_smooth = signal.savgol_filter(
                np.abs(trace.data), smooth_length, smooth_order
            )
            trace.data = trace.data / (trace_env_smooth + epsilon)

    elif method == "mad":
        for trace in stream:
            trace.data = trace.data / (
                stats.median_absolute_deviation(trace.data) + epsilon
            )

    else:
        raise ValueError("Unknown method {}".format(method))


def modulus_division(x, epsilon=1e-10):
    r"""Modulus division of a complex number.

    Given a complex number (or complex-valued array) :math:`x = a e^{i\phi}`,
    where :math:`a` is the modulus and :math:`\phi` the phase, the function
    returns the unit-modulus complex number such as

    .. math::

              \tilde{x} = e^{i\phi} = \frac{x}{|x| + \epsilon}

    This method normalizes the input complex number by dividing it by its modulus
    plus a small epsilon value to prevent division by zero, effectively scaling
    the modulus to 1 while preserving the phase.

    Arguments
    ---------
    x: :class:`np.ndarray`
        The complex-valued data to extract the unit complex number from.
    epsilon: float, optional
        A small value added to the modulus to avoid division by zero. Default is 1e-10.

    Returns
    -------
    :class:`np.ndarray`
        The unit complex number with the same phase as the input data.
    """
    return x / (np.abs(x) + epsilon)


def smooth_modulus_division(x, smooth=None, order=None, epsilon=1e-10):
    r"""Modulus division of a complex number with smoothing.

    Given a complex array :math:`x[n] = a[n] e^{i\phi[n]}`, where :math:`a[n]`
    is the modulus and :math:`\phi[n]` the phase, the function returns the
    normalized complex array :math:`\tilde{x}[n]` such as

    .. math::

        \tilde{x}[n] = \frac{x[n]}{s(a)[n] + \epsilon}

    where :math:`s(a)[n]` is a smoothed version of the modulus array
    :math:`a[n]`. The smoothing function is performed with the Savitzky-Golay
    filter, and the order and length of the filter are set by the ``smooth``
    and ``order`` parameters.].

    Arguments
    ---------
    x: :class:`numpy.ndarray`
        The spectra to detrend. Must be of shape ``(n_frequencies, n_times)``.
    smooth: int
        Smoothing window size in points.
    order: int
        Smoothing order. Check the scipy function
        :func:`~scipy.signal.savgol_filter` function for more details.

    Keyword arguments
    -----------------
    epsilon: float, optional
        A regularizer for avoiding zero division.

    Returns
    -------
    The spectrum divided by the smooth modulus spectrum.
    """
    smooth_modulus = signal.savgol_filter(np.abs(x), smooth, order, axis=0)
    return x / (smooth_modulus + epsilon)


def stft(
    window_duration_sec: float = 2.0,
    window_step_sec: None | float = None,
    window_function: str = "hann",
    sampling_rate: None | float = 1.0,
    **kwargs,
):
    r"""Short-Time Fourier Transform instance.

    This function creates a Short-Time Fourier Transform instance for
    transforming seismic traces into the spectral domain. The transformation
    is performed with the :func:`scipy.signal.ShortTimeFFT` function. This
    instance can both calculate the forward and inverse Short-Time Fourier
    Transform, and other useful methods.

    Arguments
    ---------
    window_duration_sec: float, optional
        The duration of the Fourier window in seconds. Default to 2 seconds.
    window_step_sec: float, optional
        The step between two consecutive windows in seconds. If not provided,
        the step is set to half the window size.
    window_function: str, optional
        The window function to use for the Short-Time Fourier Transform.
        Default to "hann".
    sampling_rate: float, optional
        The sampling rate of the seismic traces. Default to 1.0 Hz.
    **kwargs: dict, optional
        Additional keyword arguments are passed to the
        :func:`scipy.signal.ShortTimeFFT` class constructor. Check the SciPy
        documentation for more details on the available options.

    Returns
    -------
    :class:`scipy.signal.ShortTimeFFT`
        A Short-Time Fourier Transform instance.
    """
    # Get window
    window_size = int(window_duration_sec * sampling_rate)
    window = signal.windows.get_window(window_function, window_size)

    # Define hop
    if window_step_sec is not None:
        hop = int(window_step_sec * sampling_rate)
    else:
        hop = window_size // 2

    # Set sampling rate
    kwargs.setdefault("fs", sampling_rate)

    return signal.ShortTimeFFT(window, hop, **kwargs)


def calculate_spectrogram(trace, **kwargs):
    """Calculate the spectrogram of a seismic traces.

    The spectrogram is calculated with the :func:`covseisnet.stream.stft`
    function, and the Short-Time Fourier Transform is applied to the trace.

    Arguments
    ---------
    trace: :class:`~obspy.core.trace.Trace`
        The seismic trace to calculate the spectrogram from.
    **kwargs: dict, optional
        Additional keyword arguments are passed to the
        :func:`covseisnet.stream.stft` function. Check the function
        documentation for more details on the available options.
    """
    # Instanciate ShortTimeFFT object
    kwargs.setdefault("sampling_rate", trace.stats.sampling_rate)
    transform = stft(**kwargs)

    # Calculate the Short-Time Fourier Transform
    waveform = trace.data
    short_time_fft = transform.stft(waveform)
    spectrogram = np.abs(short_time_fft)

    # Get metadata
    time = transform.t(trace.stats.npts)
    frequencies = transform.f

    return time, frequencies, spectrogram
