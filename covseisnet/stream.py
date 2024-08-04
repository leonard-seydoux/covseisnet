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
from typing import Any

import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import AttribDict, Stats, Trace

from . import signal


class NetworkStats(AttribDict):
    r"""Header information of a :class:`~NetworkStream` object.

    A ``NetworkStats`` object contains a subset of header information (also
    known as metadata) of the :class:`~obspy.core.trace.Trace` objects of a
    :class:`~NetworkStream` object. Those headers may be accessed or modified
    either in the dictionary style or directly via a corresponding attribute.
    There are various default attributes which are required by every waveform
    import and export modules within ObsPy such as :mod:`obspy.io.mseed`.

    Arguments
    ---------
    header: dict or :class:`~obspy.core.trace.Stats`, optional
        Dictionary containing meta information of the
        :class:`~obspy.core.trace.Trace` objects.

    Example
    -------

    >>> from covseisnet import NetworkStats
    >>> stats = NetworkStats()
    >>> print(stats)
                starttime: 1970-01-01T00:00:00.000000Z
                  endtime: 1970-01-01T00:00:00.000000Z
            sampling_rate: 1.0
                    delta: 1.0
                     npts: 0

    .. rubric:: _`Default Attributes`

    ``sampling_rate`` : float, optional
        Sampling rate in Hertz (default to 1.0).
    ``delta`` : float, optional
        Sample distance in seconds (default to 1.0).
    ``npts`` : int, optional
        Number of sample points per trace (default to 0).
    ``starttime`` : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the first data sample given in UTC (default value is
        "1970-01-01T00:00:00.0Z").
    ``endtime`` : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the last data sample given in UTC (default value is
        "1970-01-01T00:00:00.0Z").

    .. rubric:: Notes

    (1) The attributes ``sampling_rate`` and ``delta`` are linked to each
        other. If one of the attributes is modified the other will be
        recalculated.

        >>> from covseisnet import NetworkStats
        >>> stats = NetworkStats()
        >>> stats.sampling_rate
        1.0
        >>> stats.delta = 0.005
        >>> stats.sampling_rate
        200.0

    (2) The attributes ``starttime``, ``npts``, ``sampling_rate`` and
        ``delta`` are monitored and used to automatically calculate the
        ``endtime``.

        >>> from covseisnet import NetworkStats
        >>> stats = Stats()
        >>> stats.npts = 60
        >>> stats.delta = 1.0
        >>> stats.starttime = UTCDateTime(2009, 1, 1, 12, 0, 0)
        >>> stats.endtime
        2009-01-01T12:00:59.000000Z
        >>> stats.delta = 0.5
        >>> stats.endtime
        2009-01-01T12:00:29.500000Z

    (3) The attribute ``endtime`` is read only and can not be modified.

        >>> stats = Stats()
        >>> stats.endtime = UTCDateTime(2009, 1, 1, 12, 0, 0)
        Traceback (most recent call last):
        ...
        AttributeError: Attribute "endtime" in Stats object is read only!
        >>> stats['endtime'] = UTCDateTime(2009, 1, 1, 12, 0, 0)
        Traceback (most recent call last):
        ...
        AttributeError: Attribute "endtime" in Stats object is read only!

    (4)
        The attribute ``npts`` will be automatically updated from the
        :class:`~obspy.core.trace.Trace` object.

        >>> trace = Trace()
        >>> trace.stats.npts
        0
        >>> trace.data = np.array([1, 2, 3, 4])
        >>> trace.stats.npts
        4

    (5)
        The attribute ``component`` can be used to get or set the component,
        i.e. the last character of the ``channel`` attribute.

        >>> stats = Stats()
        >>> stats.channel = 'HHZ'
        >>> stats.component  # doctest: +SKIP
        'Z'
        >>> stats.component = 'L'
        >>> stats.channel  # doctest: +SKIP
        'HHL'

    """

    # Immutable keys
    readonly = ["endtime"]

    # Default values
    defaults = {
        "sampling_rate": 1.0,
        "delta": 1.0,
        "starttime": UTCDateTime(0),
        "endtime": UTCDateTime(0),
        "npts": 0,
    }

    # Keys which need to refresh derived values
    _refresh_keys = {
        "delta",
        "sampling_rate",
        "starttime",
        "npts",
    }

    def __init__(self, header: dict = {}):
        super().__init__(header)

    def __setitem__(self, key, value):
        if key in self._refresh_keys:
            # Ensure correct data type
            if key == "delta":
                key = "sampling_rate"
                try:
                    value = 1.0 / float(value)
                except ZeroDivisionError:
                    value = 0.0
            elif key == "sampling_rate":
                value = float(value)
            elif key == "starttime":
                value = UTCDateTime(value)
            elif key == "npts":
                if not isinstance(value, int):
                    value = int(value)
            # Set current key
            super(NetworkStats, self).__setitem__(key, value)

            # Set derived value: delta
            try:
                delta = 1.0 / float(self.sampling_rate)
            except ZeroDivisionError:
                delta = 0
            self.__dict__["delta"] = delta

            # Set derived value: endtime
            if self.npts == 0:
                timediff = 0
            else:
                timediff = float(self.npts - 1) * delta
            self.__dict__["endtime"] = self.starttime + timediff
            return

    # Set the __setattr__ method to the __setitem__ method
    __setattr__ = __setitem__

    def __getitem__(self, key, default=None):
        return super(NetworkStats, self).__getitem__(key, default)

    def __str__(self):
        """Return better readable string representation of Stats object."""
        priorized_keys = [
            "starttime",
            "endtime",
            "sampling_rate",
            "delta",
            "npts",
        ]
        return self._pretty_str(priorized_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unneeded entries
        state.pop("delta", None)
        state.pop("endtime", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # trigger refreshing
        self.__setitem__("sampling_rate", state["sampling_rate"])


class NetworkStream(Stream):
    """
    Subclass of the ObsPy :class:`~obspy.core.stream.Stream` tailored for
    managing continuous data from seismic networks. The class is designed to
    handle multiple traces from different stations, and provides additional
    methods for pre-processing and synchronization of the traces. It also
    provide network-wide methods such as the calculation of the common time
    vector of the traces.

    .. rubric:: _`Attributes`

    - :attr:`~covseisnet.stream.NetworkStream.is_ready_to_process` — check if
      stream is ready to be processed.

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
        n_stations = len(self.ids)

        # Synced flag
        synced_flag = "synced" if self.are_time_vectors_equal else "not synced"

        # Initialize output string
        out = f"NetworkStream of {n_traces} traces from {n_stations} stations ({synced_flag}):\n"

        # Print all traces
        out = out + "\n".join([trace.__str__(longest_id) for trace in self])

        return out

    def __getitem__(self, index) -> Trace | Stream:
        return super().__getitem__(index)

    @property
    def stats(self) -> NetworkStats:
        """Stats dictionary of the first trace.

        This property is also available directly from looping over the traces
        and accessing the :attr:`stats` attribute.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.stats.sampling_rate
        100.0
        """
        # Extract stats from the first trace
        if not (stats := getattr(self[0], "stats")):
            raise ValueError("Stats dictionary is not defined.")

        # Meta
        stats["networks"] = [trace.stats.network for trace in self]
        stats["stations"] = [trace.stats.station for trace in self]
        stats["ids"] = [trace.id for trace in self]
        stats["channels"] = [trace.stats.channel for trace in self]
        return NetworkStats(stats)

    @property
    def all_stats(self) -> list[Stats]:
        """Stats dictionary of the first trace.

        This property is also available directly from looping over the traces
        and accessing the :attr:`stats` attribute.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.stats.sampling_rate
        100.0
        """
        stats = [getattr(trace, "stats") for trace in self]
        if any(not stat for stat in stats):
            raise ValueError("Stats dictionary is not defined.")
        return stats

    @classmethod
    def read(cls, pathname_or_url=None, **kwargs) -> "NetworkStream":
        """Read seismic waveforms files into an NetworkStream object.

        This function uses the :func:`obspy.core.stream.read` function to read
        the streams. A detailed list of arguments and options are available in
        the documentation. This function opens either one or multiple waveform
        files given via file name or URL using the ``pathname_or_url``
        attribute. The format of the waveform file will be automatically
        detected if not given. See the `Supported Formats` section in the
        :func:`obspy.core.stream.read` function.

        This function returns an :class:`~covseisnet.stream.NetworkStream`
        object which directly inherits from the :class:`obspy.core
        .stream.Stream` object.

        Arguments
        ---------
        pathname_or_url: str or io.BytesIO or None
            String containing a file name or a URL or a open file-like object.
            Wildcards are allowed for a file name. If this attribute is
            omitted, an example :class:`~covseisnet.stream.NetworkStream`
            object will be returned.
        **kwargs: dict, optional
            Other parameters are passed to the :func:`obspy.core.stream.read`
            directly.

        Returns
        -------
        :class:`~covseisnet.stream.NetworkStream`
            The seismic waveforms.


        Example
        -------
        >>> import covseisnet as csn
        >>> stream = csn.NetworkStream.read()
        >>> print(stream)
        Network Stream of 3 traces from 1 stations (synced):
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z... | 100.0 Hz, 3000 samples

        See Also
        --------
        :func:`~obspy.core.stream.read`
        """
        return cls(obspy.read(pathname_or_url, **kwargs))

    def cut(
        self,
        starttime: str | UTCDateTime,
        endtime: str | UTCDateTime | None = None,
        duration: float | None = None,
        **kwargs: Any,
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
        duration : float, optional
            The duration of the trace in seconds. If set, the end time is
            calculated as ``starttime + duration``. This parameter is
            ignored if the ``endtime`` parameter is set.
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
        Network Stream of 3 traces from 1 stations (synced):
        BW.RJOB..EHZ | 2009-08-24T00:20:05.000000Z... | 100.0 Hz, 701 samples
        W.RJOB..EHN  | 2009-08-24T00:20:05.000000Z... | 100.0 Hz, 701 samples
        BW.RJOB..EHE | 2009-08-24T00:20:05.000000Z... | 100.0 Hz, 701 samples

        See Also
        --------
        :meth:`~obspy.core.stream.Stream.trim`
        """
        # Convert start and end times to UTCDateTime
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime or starttime + duration)

        # Trim stream
        self.trim(starttime, endtime, **kwargs)

    def times(self, *args, **kwargs: dict) -> np.ndarray:
        """Common time vector.

        Because the :class:`~covseisnet.stream.NetworkStream` handles traces
        sampled on the same time vector, this function only returns the times
        of the first trace with the Trace method
        :meth:`~obspy.core.trace.Trace.times` if the traces are synchronized.

        Arguments
        ---------
        *args: tuple
            Arguments passed to the method
            :meth:`~obspy.core.trace.Trace.times`. For instance, passing
            ``"matplotlib"`` allows to recover matplotlib timestamps instead
            of seconds from the start of the trace (default).
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


        .. tip::

            By default, the method returns the times in seconds since the
            start of the trace. In order to extract times in matplotlib
            format, you can set the ``type`` parameter of the
            :meth:`~obspy.core.trace.Trace.times` method such as

            >>> import covseisnet as csn
            >>> stream = csn.read()
            >>> stream.times("matplotlib")
            array([14480.01392361, 14480.01392373, 14480.01392384, ...,
            14480.01427049, 14480.0142706 , 14480.01427072])


        See Also
        --------
        :meth:`~obspy.core.trace.Trace.times`
        """
        # Check if the traces are synchronized
        assert (
            self.are_time_vectors_equal
        ), "Traces are not synced, check the `synchronize` method."

        # Get the first trace
        trace = self[0]
        assert isinstance(trace, Trace), "Trace is not of type Trace."

        # Return the times of the first trace
        return trace.times(*args, **kwargs)

    def synchronize(
        self,
        interpolation_method: str = "linear",
        **kwargs: Any,
    ) -> None:
        """Synchronize seismic traces with interpolation.

        This method synchronizes the seismic traces in the stream by
        interpolating the traces to a common time vector. The method uses the
        largest start time and the smallest end time of the traces to
        interpolate all traces to the same time vector with the ObsPy method
        :meth:`~obspy.core.trace.Trace.interpolate`.

        Arguments
        ---------
        method: str, default
            Interpolation method. Default to ``"linear"``.
        **kwargs: dict, optional
            Additional keyword arguments passed to the
            :meth:`~obspy.core.trace.Trace.interpolate` method. Check the
            ObsPy documentation for more details on the available options.

        See Also
        --------
        :meth:`~obspy.core.trace.Trace.interpolate`
        """
        # Return if the traces are already synchronized
        if self.are_time_vectors_equal:
            return

        # Find out the largest start time and the smallest end time
        time_extent = get_minimal_time_extent(self)
        duration = time_extent[-1] - time_extent[0]

        # Extract the number of samples from the first trace
        trace = self[0]
        stats = getattr(trace, "stats", None)
        sampling_rate = getattr(stats, "sampling_rate", None)
        if not sampling_rate:
            raise ValueError("Sampling rate is not defined.")
        npts = int(duration * sampling_rate) + 1

        # Update kwargs
        kwargs.setdefault("method", interpolation_method)
        kwargs.setdefault("npts", npts)
        kwargs.setdefault("starttime", time_extent[0])
        kwargs.setdefault("endtime", time_extent[-1])
        kwargs.setdefault("sampling_rate", self.sampling_rate)

        # Interpolate all traces
        for trace in self:
            trace.interpolate(**kwargs)

    def whiten(
        self,
        smooth_length: int = 0,
        smooth_order: int = 1,
        epsilon: float = 1e-10,
        **kwargs: Any,
    ) -> None:
        r"""Whiten traces in the spectral domain.

        The action of whitening a seismic trace is to normalize the trace in
        the spectral domain. Typically, the spectrum becomes flat after
        whitening, resembling white noise. This strategy is often used to
        remove the influence of time-localized signal and diminish the site
        effects from a seismic station to another. Any local source is also
        drastically reduced thanks to the whitening process.

        The following description is applied to every trace in the stream. For
        the sake of simplicity, we consider a single trace :math:`x(t)`. Note
        that the method is applied in every window of a short-time Fourier
        transform of the trace, namely :math:`s(t, \omega)` before applying
        the inverse short-time Fourier transform to obtain the whitened
        seismogram :math:`\hat x(t)`. We here nore :math:`x(\omega)` the
        spectrum of the trace within a given window. For more information on
        the short-time Fourier transform, see the
        :class:`~covseisnet.signal.ShortTimeFourierTransform` class
        documentation.

        We define the whitening process as

        .. math::

            \hat x(\omega) = \frac{x(\omega)}{\mathcal{S}x(\omega) +
            \epsilon},

        where :math:`\mathcal{S}` is a smoothing operator applied to the
        spectrum :math:`x(\omega)`, and :math:`\epsilon` is a regularization
        parameter to avoid division by zero. The smoothing operator is defined
        by the ``smooth_length`` parameter. We distinguish two cases:

        - If the ``smooth_length`` parameter is set to 0, the operator
          :math:`\mathcal{S}` is defined as :math:`\mathcal{S}x(\omega) =
          |x(\omega)|`, and therefore

          .. math::

              \hat x(\omega) = \frac{x(\omega)}{|x(\omega)| + \epsilon}
              \approx e^{i\phi}.

          In this case, the method calls the
          :func:`~covseisnet.signal.modulus_division`.

        - If the ``smooth_length`` parameter is set to a value greater than 0,
          the operator :math:`\mathcal{S}` is defined as a `Savitzky-Golay
          filter
          <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_ with
          parameters set by ``smooth_length`` and ``smooth_order``. This
          allows to introduce less artifacts in the whitening process. In this
          case, the method calls the
          :func:`~covseisnet.signal.smooth_modulus_division`.

        Arguments
        ---------
        smooth_length: int, optional
            The length of the Savitzky-Golay filter for smoothing the
            spectrum. If set to 0, the spectrum is not smoothed (default).
        smooth_order: int, optional
            The order of the Savitzky-Golay filter for smoothing the spectrum.
            This parameter is only used if ``smooth_length`` is greater than
            0.
        epsilon: float, optional
            Regularization parameter in division, set to 1e-10 by default.
        **kwargs: dict, optional
            Additional keyword arguments passed to the covseisnet
            :func:`~covseisnet.signal.ShortTimeFourierTransform` class
            constructor. The main parameter is the ``window_duration_sec``
            parameter, which defines the window duration of the short-time
            Fourier transform.

        """
        # Automatically set the sampling rate from self
        kwargs.setdefault("sampling_rate", self.stats.sampling_rate)

        # Short-Time Fourier Transform instance
        stft_instance = signal.ShortTimeFourierTransform(**kwargs)

        # Assert that the transform is invertible
        assert stft_instance.invertible, "The transform is not invertible."

        # Define the whitening method
        if smooth_length == 0:
            whiten_method = partial(
                signal.modulus_division,
                epsilon=epsilon,
            )
        elif smooth_length > 0:
            whiten_method = partial(
                signal.smooth_modulus_division,
                smooth=smooth_length,
                order=smooth_order,
                epsilon=epsilon,
            )
        else:
            raise ValueError(f"Incorrect smooth_length value: {smooth_length}")

        # Loop over traces
        for trace in self:

            # Calculate the Short-Time Fourier Transform
            waveform = trace.data
            spectrum = stft_instance.stft(waveform)

            # Whiten the spectrum
            spectrum = whiten_method(spectrum)

            # Inverse Short-Time Fourier Transform
            waveform = stft_instance.istft(spectrum)

            # Truncate the waveform and replace the trace data
            waveform = waveform[: trace.stats.npts]
            trace.data = waveform

    def normalize(
        self, method="onebit", smooth_length=11, smooth_order=1, epsilon=1e-10
    ) -> None:
        r"""Normalize the seismic traces in temporal domain.

        Considering the seismic trace :math:`x(t)`, the normalized trace
        :math:`\hat x(t)` is obtained with

        .. math::

            \hat x(t) = \frac{x(t)}{\mathcal{A}x(t) + \epsilon}

        where :math:`A` is an operator applied to the trace :math:`x(t)`, and
        :math:`\epsilon > 0` is a regularization value to avoid division by 0.
        The operator :math:`\mathcal{A}` is defined by the ``method``
        parameter. We distinguish two cases:

        - If the ``method`` parameter is set to ``"onebit"``, the operator
          :math:`\mathcal{A}` is defined as :math:`\mathcal{A}x(t) = |x(t)|`,
          and therefore

          .. math::

            \hat x(t) = \frac{x(t)}{|x(t)| + \epsilon} \approx
            \text{sign}(x(t)).

          In this case, the method calls the
          :func:`~covseisnet.signal.modulus_division`.

        - If the ``method`` parameter is set to ``"smooth"``, the operator
          :math:`\mathcal{A}` is defined as a `Savitzky-Golay filter
          <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_
          applied to the Hilbert envelope of the trace. The Savitzky-Golay
          filter is defined by the ``smooth_length`` and ``smooth_order``
          parameters. This allows to introduce less artifacts in the
          normalization process. In this case, the method calls the
          :func:`~covseisnet.signal.smooth_envelope_division`.

        Arguments
        ---------
        method : str, optional
            Must be one of ``"onebit"`` (default) or ``"smooth"``.

            - ``"onebit"``: compress the seismic trace into a series of -1 and
              1.

            - ``"smooth"``: normalize each trace by a smooth version of its
              envelope.

        smooth_length: int, optional
            If the ``method`` keyword argument is set to ``"smooth"``, the
            normalization is performed with the smoothed trace envelopes,
            calculated over a sliding window of ``smooth_length`` samples.
        smooth_order: int, optional
            If the ``method`` keyword argument is set to ``"smooth"``, the
            normalization is performed with the smoothed trace envelopes. The
            smoothing order is set by the ``smooth_order`` parameter.
        epsilon: float, optional
            Regularization parameter in division, set to 1e-10 by default.

        """
        if method == "onebit":
            for trace in self:
                trace.data = signal.modulus_division(
                    trace.data,
                    epsilon=epsilon,
                )
        elif method == "smooth":
            for trace in self:
                trace.data = signal.smooth_envelope_division(
                    trace.data,
                    smooth_length,
                    smooth_order,
                    epsilon,
                )
        else:
            raise ValueError(f"Unknown method {method}")

    @property
    def is_ready_to_process(self) -> bool:
        """Check if traces are ready to be processed.

        This method checks if the traces are ready to be processed. This is
        useful to ensure that the traces are synchronized before performing any
        operation that requires the traces to be sampled on the same time vector.

        This property performs the following checks in order:

        - :attr:`~covseisnet.stream.NetworkStream.are_sampling_rates_equal`

        - :attr:`~covseisnet.stream.NetworkStream.are_npts_equal`

        - :attr:`~covseisnet.stream.NetworkStream.are_time_vectors_equal`

        Returns
        -------
        bool
            True if all the checks are passed, False otherwise.
        """
        checks = (
            self.are_sampling_rates_equal,
            self.are_npts_equal,
            self.are_time_vectors_equal,
        )
        return all(checks)

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

        Raises
        ------
        AssertionError
            If the traces have different sampling rates or number of samples.
        """
        # Assert sampling rate
        if not self.are_sampling_rates_equal:
            return False

        # Assert number of samples
        if not self.are_npts_equal:
            return False

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
    def ids(self) -> list[str]:
        """List of unique trace ids.

        This property is also available directly from looping over the traces
        and accessing the :attr:`id` attribute.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.ids
        ['BW.RJOB..EHE', 'BW.RJOB..EHN', 'BW.RJOB..EHZ']
        """
        return [trace.id for trace in self.traces]

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the traces in Hz.

        This property is also available directly from looping over the traces
        and accessing the :attr:`stats.sampling_rate` attribute.

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
        return self.stats["sampling_rate"]

    @property
    def npts(self) -> int:
        """Number of samples of the traces.

        This property is also available directly from looping over the traces
        and accessing the :attr:`stats.npts` attribute.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.npts
        3000
        """
        # Assert number of samples
        assert self.are_npts_equal, "Traces have different number of samples."

        # Return the number of samples of the first trace
        return self.stats.npts


def get_minimal_time_extent(
    stream: NetworkStream | Stream,
) -> tuple[UTCDateTime, UTCDateTime]:
    """Get the minimal time extent of traces in a stream.

    This function returns the minimal start and end times of the traces in the
    stream. This is useful when synchronizing the traces to the same time
    vector. The start time is defined as the maximum start time of the traces,
    and the end time is defined as the minimum end time of the traces.

    Arguments
    ---------
    stream: :class:`~covseisnet.stream.NetworkStream` or :class:`~obspy.core.stream.Stream`
        The stream object.

    Returns
    -------
    tuple of :class:`~obspy.core.utcdatetime.UTCDateTime`
        The minimal start and end times of the traces.

    Example
    -------
    >>> import covseisnet as csn
    >>> stream = csn.read()
    >>> get_minimal_time_extent(stream)
    (2009-08-24T00:20:03.000000Z, 2009-08-24T00:20:32.990000Z)
    """
    return (
        max(trace.stats.starttime for trace in stream),
        min(trace.stats.endtime for trace in stream),
    )


def read(pathname_or_url=None, **kwargs) -> NetworkStream:
    """Read seismic waveforms files into an NetworkStream object.

    This function uses the :func:`obspy.core.stream.read` function to read the
    streams. A detailed list of arguments and options are available in the
    documentation. This function opens either one or multiple waveform files
    given via file name or URL using the ``pathname_or_url`` attribute. The
    format of the waveform file will be automatically detected if not given.
    See the `Supported Formats` section in the :func:`obspy.core.stream.read`
    function.

    This function returns an :class:`~covseisnet.stream.NetworkStream` object
    which directly inherits from the :class:`obspy.core.stream.Stream` object.

    Arguments
    ---------
    pathname_or_url: str or io.BytesIO or None
        String containing a file name or a URL or a open file-like object.
        Wildcards are allowed for a file name. If this attribute is omitted,
        an example :class:`~covseisnet.stream.NetworkStream` object will be
        returned.
    **kwargs: dict, optional
        Other parameters are passed to the :func:`obspy.core.stream.read`
        directly.

    Returns
    -------
    :class:`~covseisnet.stream.NetworkStream`
        The seismic waveforms.

    Example
    -------

    For a quick start you may omit all arguments and ObsPy will load a basic
    example seismogram. Further usages of this function can be seen in the
    ObsPy documentation.

    >>> import covseisnet as csn
    >>> stream = csn.read()
    >>> print(stream)
    Network Stream of 3 traces from 1 stations (synced):
    BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z... | 100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z... | 100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z... | 100.0 Hz, 3000 samples

    See Also
    --------
    :func:`~obspy.core.stream.read`
    """
    return NetworkStream.read(pathname_or_url, **kwargs)
