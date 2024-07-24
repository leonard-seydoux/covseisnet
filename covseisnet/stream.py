"""Read and pre-process network seismic data.

The core idea of Covseisnet is to provide a set of tools for the analysis of
seismic data from networks. To that end, we provide a set of classes and
methods for reading, pre-processing, and analyzing seismic data from networks.
The essential idea is to work on traces that have been synchronized and
pre-processed with the same methods. In order to do so, we build the concept of
a NetworkStream, a subclass of the ObsPy Stream object that offers the same 
methods of the Stream object, but with additional methods for pre-processing and
synchronization of the traces.
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

    def __str__(self, extended=False):
        """Print the NetworkStream object.

        This method prints the NetworkStream object in a human-readable
        format. The methods ressembles the ObsPy Stream object, but with
        additional information about the number of traces and stations in the
        stream. By default, the method prints all traces in the stream.
        """
        # Get longest trace id index among traces
        if self.traces:
            longest_id = self and max(len(tr.id) for tr in self) or 0
        else:
            longest_id = 0

        # Get number of traces and stations
        n_traces = len(self.traces)
        n_stations = len(self.stations)

        # Initialize output string
        out = f"Network Stream of {n_traces} traces from {n_stations} stations:\n"

        # Print all traces
        out = out + "\n".join([trace.__str__(longest_id) for trace in self])

        return out

    def cut(
        self,
        starttime: str | obspy.UTCDateTime,
        endtime: str | obspy.UTCDateTime,
        **kwargs: dict,
    ):
        """Cut (trim) stream between given start and end times.

        This function is a wrapper to the ObsPy
        :meth:`~obspy.core.stream.Stream.trim` method, but works directly with
        datetimes in :class:`str` format. The function uses the native ObsPy
        :class:`~obspy.core.utcdatetime.UTCDateTime` class in order to convert
        the datetimes from :class:`str` into
        :class:`obspy.core.utcdatetime.UTCDateTime` format.

        Arguments
        ---------

        starttime : str or :class:`~obspy.core.utcdatetime.UTCDateTime`
            The start date time.
        endtime : str or :class:`~obspy.core.utcdatetime.UTCDateTime`
            The end date time.
        **kwargs: dict, optional
            Additional keyword arguments passed to the
            :meth:`~obspy.core.stream.Stream.trim` method of ObsPy. Check the
            ObsPy documentation for more details on the available options.

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

    def times(
        self,
        **kwargs: dict,
    ):
        """Common time vector of the NetworkStream.

        Because the :class:`~covseisnet.stream.NetworkStream` handles traces
        sampled on the same time vector, this function only returns the times
        of the first trace with the Trace method
        :meth:`~obspy.core.trace.Trace.times` if the traces are synchronized.

        Arguments
        ---------
        **kwargs: dict, optional
            Additional keyword arguments are directly passed to the Trace
            method :meth:`~obspy.core.trace.Trace.times`. For instance, passing
            ``type="matplotlib"`` allows to recover matplotlib timestamps
            provided by the :func:`matplotlib.dates.date2num` function and thus
            enables the use of date labels in plots.

        Returns
        -------
        :class:`numpy.ndarray`
            An array of timestamps in a :class:`numpy.ndarray` or in a
            :class:`list`.


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
        self.assert_synchronized
        return self[0].times(**kwargs)

    def synchronize(
        self,
        start="2010-01-01T00:00:00.00",
        duration_sec=24 * 3600,
        method="linear",
    ):
        """Synchronize seismic traces into the same times.

        This function uses the
        :meth:`obspy.core.stream.Stream.trim` method in order to cut all
        traces of the Stream object to given start and end time with
        padding if necessary, and the
        :meth:`obspy.core.trace.Trace.interpolate` method in order to
        synchronize the traces. Then data are linearly interpolated.

        Keyword arguments
        -----------------
        start: str, default
            Start date of the interpolation.
            Default to "2010-01-01T00:00:00.00".

        duration_sec: int, default
            Duration of the data traces in second. Default to 86,400 seconds,
            the total number of seconds on a single day.

        method: str, default
            Interpolation method. Default to "linear".
        """
        # Duplicate stream
        stream_i = self.copy()
        start = obspy.UTCDateTime(start)
        end = start + duration_sec
        sampling = self[0].stats.sampling_rate
        npts = int(sampling * duration_sec)
        stream_i.trim(start, end, pad=True, fill_value=0, nearest_sample=False)

        for tr, tr_i in zip(self, stream_i):
            t = tr.times() / duration_sec + tr.stats.starttime.matplotlib_date
            shift = tr_i.stats.starttime - start
            tr_i.interpolate(
                sampling, method, start=start, npts=npts, time_shift=-shift
            )
            ti = (
                tr_i.times() / duration_sec
                + tr_i.stats.starttime.matplotlib_date
            )
            tr_i.data = np.interp(ti, t, tr.data)

        return stream_i

    @property
    def assert_synchronized(self) -> None:
        """Check if all traces are sampled on the same time vector.

        This method checks if all traces are sampled on the same time vector.
        This is useful to ensure that the traces are synchronized before
        performing any operation that requires the traces to be sampled on the
        same time vector.

        Raises
        ------
        ValueError
            If the traces are not synchronized.
        """
        # Collect time vectors
        time_vectors = [trace.times() for trace in self]

        # Loop over all combinations of time vectors
        for t1 in time_vectors:
            for t2 in time_vectors:
                if len(t1) != len(t2) or not np.allclose(t1, t2):
                    raise ValueError(
                        "The traces are not synchronized.\n"
                        + "Please check the `synchronize` method."
                    )

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

    def preprocess(self, domain="spectral", **kwargs):
        r"""Pre-process each trace in temporal or spectral domain."""
        kwargs.setdefault("epsilon", 1e-10)
        if domain == "spectral":
            whiten(self, **kwargs)
        elif domain == "temporal":
            normalize(self, **kwargs)
        else:
            raise ValueError(
                "Invalid preprocessing domain {} - please specify 'spectral' or 'temporal'".format(
                    domain
                )
            )
        pass


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


def whiten(
    stream,
    method="onebit",
    window_duration_sec=2,
    smooth_length=11,
    smooth_order=1,
    epsilon=1e-10,
):
    r"""Normalize in the spectral domain."""
    if method == "onebit":
        whiten_method = phase
    elif method == "smooth":
        whiten_method = partial(
            detrend_spectrum,
            smooth=smooth_length,
            order=smooth_order,
            epsilon=epsilon,
        )
    else:
        raise ValueError("Unknown method {}".format(method))
    r"""Whiten traces in the spectral domain."""
    fft_size = int(window_duration_sec * stream[0].stats.sampling_rate)
    for index, trace in enumerate(stream):
        data = trace.data
        _, _, data_fft = signal.stft(data, nperseg=fft_size)
        data_fft = whiten_method(data_fft)
        _, data = signal.istft(data_fft, nperseg=fft_size)
        trace.data = data
    pass


def detrend_spectrum(x, smooth=None, order=None, epsilon=1e-10):
    r"""Smooth modulus spectrum.

    Arugments
    ---------
    x: :class:`np.ndarray`
        The spectra to detrend. Must be of shape `(n_frequencies, n_times)`.

    smooth: int
        Smoothing window size in points.

    order: int
        Smoothing order. Please check the :func:`savitzky_golay` function
        for more details.

    Keyword arguments
    -----------------
    epsilon: float, optional
        A regularizer for avoiding zero division.

    Returns
    -------
    The spectrum divided by the smooth modulus spectrum.
    """
    n_frequencies, n_times = x.shape
    for t in range(n_times):
        x_smooth = signal.savgol_filter(np.abs(x[:, t]), smooth, order)
        x[:, t] /= x_smooth + epsilon
    return x


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


def phase(x):
    r"""Complex phase extraction.

    Given a complex number (or complex-valued array)
    :math:`x = r e^{\imath \phi}`, where :math:`r` is the complex modulus
    and :math:`phi` the complex phase, the function returns the unitary-modulus
    complex number such as

    .. math::

              \tilde{x} = e^{\imath \phi}

    Arguments
    ---------
    x: :class:`np.ndarray`
        The complex-valued data to extract the complex phase from.
    """
    return np.exp(1j * np.angle(x))
