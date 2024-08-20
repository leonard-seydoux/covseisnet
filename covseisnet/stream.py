"""
In order to ease the analysis of network seismic data, we offer a set of
classes and methods designed for reading, pre-processing, and analyzing
seismic data from these networks. The workflow involves working on traces that
have been synchronized and pre-processed using consistent methods. To
facilitate this, we introduce the concept of a
:class:`~covseisnet.stream.NetworkStream`, a subclass of the ObsPy
:class:`~obspy.core.stream.Stream` object, which retains all the methods of
the Stream object while adding specialized methods for the pre-processing and
synchronization of traces.
"""

from io import BytesIO
from functools import partial
from typing import Any

import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import Stats, Trace
from obspy.core.inventory import Inventory

from . import signal


class NetworkStream(Stream):
    """
    Subclass of the ObsPy :class:`~obspy.core.stream.Stream` tailored for
    managing continuous data from seismic networks. The class is designed to
    handle multiple traces from different stations, and provides additional
    methods for pre-processing and synchronization of the traces. It also
    provide network-wide methods such as the calculation of the common time
    vector of the traces.

    .. note::

        The following list of methods and attributes are the ones that are
        specific to the :class:`~covseisnet.stream.NetworkStream` object. The
        full list of methods and attributes of the ObsPy
        :class:`~obspy.core.stream.Stream` object are available in the ObsPy
        documentation.

    .. rubric:: Boolean attributes

    - :attr:`~covseisnet.stream.NetworkStream.synced` — traces share the same
      time vector.

    - :attr:`~covseisnet.stream.NetworkStream.equal_rates` — traces have the
      same sampling rate.

    - :attr:`~covseisnet.stream.NetworkStream.equal_length` — traces have the
      same number of points.

    .. rubric:: Numeric attributes

    - :attr:`~covseisnet.stream.NetworkStream.sampling_rate` — common traces
      sampling rate.

    - :attr:`~covseisnet.stream.NetworkStream.npts` — common traces number of
      samples.

    - :attr:`~covseisnet.stream.NetworkStream.coordinates` — geographical
      coordinates of the stations.

    .. rubric:: Methods

    - :meth:`~covseisnet.stream.NetworkStream.stats()` — stats dictionaries of
      traces.

    - :meth:`~covseisnet.stream.NetworkStream.read()` — read seismic waveforms
      files into a :class:`~covseisnet.stream.NetworkStream` object.

    - :meth:`~covseisnet.stream.NetworkStream.times()` — common traces time
      vector.

    - :meth:`~covseisnet.stream.NetworkStream.time_extent()` — minimal time
      extent of traces in a stream.

    - :meth:`~covseisnet.stream.NetworkStream.cut()` — trim stream between
      given start and end times with string format.

    - :meth:`~covseisnet.stream.NetworkStream.synchronize()` — synchronize the
      traces into the same times with interpolation.

    - :meth:`~covseisnet.stream.NetworkStream.process()` — process the traces
      with a dictionary of processing steps.

    - :meth:`~covseisnet.stream.NetworkStream.whiten()` — whiten traces in the
      spectral domain.

    - :meth:`~covseisnet.stream.NetworkStream.time_normalize()` — normalize
      the traces in the temporal domain.

    - :meth:`~covseisnet.stream.NetworkStream.assign_coordinates()` — assign
      the geographical coordinates to the traces.

    - :meth:`~covseisnet.stream.NetworkStream.download_inventory()` — get the
      inventory of each trace.

    .. tip::

        There are three main ways to create a NetworkStream object:

        #. Use the :meth:`NetworkStream.read` method to read seismic waveforms
           files into a NetworkStream object. This method is the core method
           to read seismic data into the package.

        #. Use the :func:`covseisnet.stream.read` function, which is a wrapper
           to the first method. Note that this function is directly available
           from the package root.

        #. Pass any ObsPy :class:`~obspy.core.stream.Stream` object to the
           :class:`~covseisnet.stream.NetworkStream` constructor. This is the
           case if you have a special reader for your data.

        Other standard ObsPy methods are available for instanciating a
        :class:`~covseisnet.stream.NetworkStream` object, directly documented
        in the ObsPy documentation.


    Examples
    --------

    1. Read seismic waveforms with the :meth:`NetworkStream.read` method.

    .. doctest::

        >>> import covseisnet as csn
        >>> stream = csn.NetworkStream.read()
        >>> print(type(stream))
        <class 'covseisnet.stream.NetworkStream'>
        >>> print(stream)
        NetworkStream of 3 traces from 1 station(s) (synced):
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples

    2. Read seismic waveforms with the :func:`~covseisnet.stream.read`
       function.

    .. doctest::

        >>> import covseisnet as csn
        >>> stream = csn.read()
        >>> print(type(stream))
        <class 'covseisnet.stream.NetworkStream'>

    3. Create a :class:`~covseisnet.stream.NetworkStream` object from an ObsPy
       :class:`~obspy.core.stream.Stream` object.

    .. doctest::

        >>> import obspy
        >>> import covseisnet as csn
        >>> stream = obspy.read()
        >>> network_stream = csn.NetworkStream(stream)
        >>> print(type(network_stream))
        <class 'covseisnet.stream.NetworkStream'>

    """

    def __init__(self, *args, **kwargs):
        super(NetworkStream, self).__init__(*args, **kwargs)

    def __str__(self, *args, **kwargs) -> str:
        """Print the NetworkStream object.

        This method wraps the original method of Obspy Stream with two changes:
        (1) the Stream is named NetworkStream to allow for differentiating
        both objects, and (2) it contains the flag "synced" or "not synced"
        at the first line to indicate if a synchronization is required to
        further process the traces.

        Arguments
        ---------
        *args, *kwargs: optional
            Positional and keyword arguments directly passed to the original
            function for legacy, future legacy, and interoperability.

        Returns
        -------
        str:
            The string formatted representation of the object.
        """
        # Catch orinal representation
        original_string = super(NetworkStream, self).__str__(*args, **kwargs)

        # Turn Stream to NetworkStream
        string_original = original_string.split("\n")
        first_line = string_original[0].replace("Stream", "NetworkStream")

        # Add synced flag
        synced_flag = "synced" if self.synced else "not synced"
        string_original[0] = first_line.replace(":", f" ({synced_flag}):")

        # Return
        string = "\n".join(string_original)
        return string

    def __getitem__(self, index: int) -> "Trace | NetworkStream":
        return super().__getitem__(index)

    def stats(self, index: int = 0, key: str | None = None) -> Any | Stats:
        """Stats dictionary of one of the traces in the stream.

        The purpose of this method is to extract the stats dictionary of one
        of the traces (by default the first trace) in the stream when the
        traces are synchronized. This is not to be confused with the
        :meth:`~obspy.core.stream.Trace.stats` attribute available at the
        trace level.

        Arguments
        ---------
        index: int, optional
            The index of the trace in the stream. Default to 0.
        key: str, optional
            The key of the stats dictionary. If set, the method returns the
            value of the key in the stats dictionary. If not set, the method
            returns the stats dictionary itself.

        Returns
        -------
        Any or :class:`~obspy.core.trace.Stats`
            The stats dictionary of the trace. If the key is set, the method
            returns the value of the key in the stats dictionary.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.stats()
                 network: BW
                 station: RJOB
                location:
                 channel: EHZ
               starttime: 2009-08-24T00:20:03.000000Z
                 endtime: 2009-08-24T00:20:32.990000Z
           sampling_rate: 100.0
                   delta: 0.01
                    npts: 3000
                   calib: 1.0
            back_azimuth: 100.0
             inclination: 30.0
                response: Channel Response
                From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
                Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
                4 stages:
                        Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500
                        Stage 2: CoefficientsTypeResponseStage from V to COUNTS, gain: 1.67785e+06
                        Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1
                        Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1
        """
        # Extract the stats dictionary
        stats = getattr(self[index], "stats")

        # Raise an error if the stats dictionary is not defined
        if not stats:
            raise ValueError("Stats dictionary is not defined.")

        # Return an item if key is set, otherwise the entire dictionary
        if key:
            return stats[key]
        return stats

    @classmethod
    def read(
        cls, pathname_or_url: str | BytesIO | None = None, **kwargs
    ) -> "NetworkStream":
        """Read seismic waveforms files into a
        :class:`~covseisnet.stream.NetworkStream` object.

        This function uses the :func:`obspy.core.stream.read` function to read
        the streams. A detailed list of arguments and options are available in
        the documentation. This function opens either one or multiple waveform
        files given via file name or URL using the ``pathname_or_url``
        attribute. The format of the waveform file will be automatically
        detected if not given. See the `Supported Formats` section in the
        :func:`obspy.core.stream.read` function.

        This function returns an :class:`~covseisnet.stream.NetworkStream`
        object which directly inherits from the
        :class:`obspy.core.stream.Stream` object.

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

    def times(self, *args, **kwargs) -> np.ndarray:
        """Common time vector.

        Because the :class:`~covseisnet.stream.NetworkStream` handles traces
        sampled on the same time vector, this function only returns the times
        of the first trace with the Trace method
        :meth:`~obspy.core.trace.Trace.times` if the traces are synchronized.

        Arguments
        ---------
        *args: tuple
            Arguments passed to the Trace method
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
        :meth:`obspy.core.trace.Trace.times`
        """
        # Check if the traces are synchronized
        assert self.synced, "Traces not synced, check the synchronize method."

        # Return the times of the first trace
        return self[0].times(*args, **kwargs)

    def time_extent(self) -> tuple[UTCDateTime, UTCDateTime]:
        """Get the minimal time extent of traces in a stream.

        This function returns the minimal start and end times of the traces in
        the stream. This is useful when synchronizing the traces to the same
        time vector. The start time is defined as the maximum start time of
        the traces, and the end time is defined as the minimum end time of the
        traces.

        Arguments
        ---------
        stream: :class:`~covseisnet.stream.NetworkStream` or
        :class:`~obspy.core.stream.Stream`
            The stream object.

        Returns
        -------
        tuple of :class:`~obspy.core.utcdatetime.UTCDateTime`
            The minimal start and end times of the traces.

        Example
        -------
        >>> import covseisnet as csn
        >>> stream = csn.read()
        >>> stream.time_extent()
        (UTCDateTime(2009, 8, 24, 0, 20, 3), UTCDateTime(2009, 8, 24, 0, 20, 32, 990000))
        """
        latest_starttime = max(trace.stats.starttime for trace in self)
        earliest_endtime = min(trace.stats.endtime for trace in self)
        return latest_starttime, earliest_endtime

    def cut(
        self,
        starttime: str | UTCDateTime,
        endtime: str | UTCDateTime | None = None,
        duration: float | None = None,
        **kwargs,
    ) -> None:
        """Trim traces between start and end date times.

        This function is a wrapper to the ObsPy
        :meth:`~obspy.core.stream.Stream.trim` method, but supports string
        format for the start and end times, enabling a more user-friendly
        interface. The function uses the ObsPy
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
            calculated as ``starttime + duration``. This parameter is ignored
            if the ``endtime`` parameter is set.
        **kwargs: dict, optional
            Arguments passed to the :meth:`~obspy.core.stream.Stream.trim`
            method.

        Example
        -------

        This example shows how to cut a stream between two given times. The
        stream is first read from the example data, and then cut between two
        given times.

        >>> import covseisnet as csn
        >>> stream = csn.read()
        >>> stream.cut("2009-08-24 00:20:05", "2009-08-24 00:20:12")
        >>> print(stream)
        NetworkStream of 3 traces from 1 station(s) (synced):
        BW.RJOB..EHZ | 2009-08-24T00:20:05.000000Z - 2009-08-24T00:20:12.000000Z | 100.0 Hz, 701 samples
        BW.RJOB..EHN | 2009-08-24T00:20:05.000000Z - 2009-08-24T00:20:12.000000Z | 100.0 Hz, 701 samples
        BW.RJOB..EHE | 2009-08-24T00:20:05.000000Z - 2009-08-24T00:20:12.000000Z | 100.0 Hz, 701 samples

        See Also
        --------
        :meth:`obspy.core.stream.Stream.trim`
        """
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime or starttime + duration)
        self.trim(starttime, endtime, **kwargs)

    def synchronize(
        self,
        interpolation_method: str = "linear",
        sampling_rate: float | None = None,
        **kwargs,
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
        sampling_rate: float, optional
            The sampling rate of the traces. If not set and if the traces have
            all the same sampling rate, the method uses the sampling rate of
            the first trace. If not set and if the traces have different
            sampling rates, the method raises a ValueError.
        **kwargs: dict, optional
            Additional keyword arguments passed to the
            :meth:`~obspy.core.trace.Trace.interpolate` method. Check the
            ObsPy documentation for more details on the available options.

        Raises
        ------
        ValueError
            If the traces have different sampling rates and the
            ``sampling_rate`` parameter is not set.

        See Also
        --------
        :meth:`obspy.core.trace.Trace.interpolate`
        """
        # Return if the traces are already synchronized
        if self.synced:
            return

        # Check if traces have the same sampling rate
        if not self.equal_rates:
            raise ValueError(
                "Traces have different sampling rates."
                "Use the `resample` method to resample the traces,"
                "or specify the `sampling_rate` parameter to interpolate"
                "the traces with `synchronize`."
            )

        # Find out the largest start time and the smallest end time
        time_extent = self.time_extent()
        duration = float(time_extent[-1] - time_extent[0])

        # The following information is extracted from the first trace, since
        # at this point, the traces should either have the same sampling rate
        # or the user should have specified the sampling rate.
        stats = getattr(self[0], "stats", None)

        # Get the sampling rate
        sampling_rate = getattr(stats, "sampling_rate", sampling_rate)
        if not sampling_rate:
            raise ValueError("Sampling rate is not defined.")

        # Calculate the number of samples
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

    def process(self, processing: dict[str, Any]) -> None:
        """Process the seismic traces in the stream.

        This method processes the seismic traces in the stream with a
        dictionary of processing steps. The dictionary must contain the
        processing steps as keys and the parameters as values. The method
        applies the processing steps to each trace in the stream.

        Arguments
        ---------
        processing: dict
            The dictionary of processing steps.

        Example
        -------
        >>> stream = csn.NetworkStream.read()
        >>> processing = {
        ...     "detrend": "linear",
        ...     "taper": 0.05,
        ...     "filter": {"type": "bandpass", "freqmin": 1, "freqmax": 10},
        ...     "time_normalize": {"method": "onebit"}
        }
        >>> stream.process(processing)

        """
        for method, args in processing.items():
            if isinstance(args, dict):
                getattr(self, method)(**args)
            elif isinstance(args, tuple):
                getattr(self, method)(*args)
            else:
                getattr(self, method)(args)

    def whiten(
        self,
        window_duration: float,
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
        window_duration: float
            The duration of the window for the short-time Fourier transform.
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
            constructor.

        """
        # Automatically set the sampling rate from self
        kwargs.setdefault("sampling_rate", self.sampling_rate)

        # Add window duration to kwargs
        kwargs["window_duration"] = window_duration

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

    def time_normalize(
        self,
        method: str = "onebit",
        smooth_length: int = 11,
        smooth_order: int = 1,
        epsilon: float = 1e-10,
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
        match method:

            case "onebit":
                for trace in self:
                    trace.data = signal.modulus_division(
                        trace.data, epsilon=epsilon
                    )

            case "smooth":
                for trace in self:
                    trace.data = signal.smooth_envelope_division(
                        trace.data,
                        smooth_length,
                        smooth_order,
                        epsilon,
                    )

            case _:
                raise ValueError(f"Unknown method {method}")

    def assign_coordinates(self, inventory: str | Inventory) -> None:
        """Assign the geographical coordinates to the traces.

        This method assigns the geographical coordinates of the stations to
        the traces in the stream from
        :class:`~obspy.core.inventory.inventory.Inventory` or the inventory
        file containing the inventory. The method uses the ObsPy
        :func:`~obspy.core.inventory.inventory.read_inventory` method to
        extract the coordinates of each trace in the stream if a filepath is
        provided.

        The method adds a coordinate dictionary to each trace in the stream.
        The coordinate dictionary contains the keys ``latitude``,
        ``longitude``, and ``elevation``.

        Arguments
        ---------
        inventory: str or :class:`~obspy.core.inventory.inventory.Inventory`
            The inventory file or the inventory object containing the station
            coordinates.

        Example
        -------
        >>> import covseisnet as csn
        >>> stream = csn.read()
        >>> stream.assign_coordinates("inventory.xml")

        See Also
        --------
        :func:`~obspy.core.inventory.inventory.read_inventory`
        """
        from obspy.core.inventory import read_inventory

        # Get the coordinates
        if isinstance(inventory, str):
            inventory = read_inventory(inventory)
        if not isinstance(inventory, Inventory):
            raise ValueError("Inventory is not an ObsPy Inventory object.")

        # Add entire inventory to the stream
        self.inventory = inventory

        # Loop over traces
        for trace in self:
            for network in inventory.networks:
                if trace.stats.network == network.code:
                    for station in network.stations:
                        if trace.stats.station == station.code:
                            trace.stats.coordinates = {
                                "latitude": station.latitude,
                                "longitude": station.longitude,
                                "elevation": station.elevation,
                            }

    def download_inventory(self, datacenter: str = "IRIS") -> Inventory:
        """Get the inventory of each trace in the stream.

        This method uses the ObsPy
        :meth:`~obspy.clients.fdsn.client.Client.get_stations` method to
        extract the inventory of each trace in the stream.

        Arguments
        ---------
        datacenter: str, optional
            The datacenter to use for retrieving the station coordinates. The
            default is ``"IRIS"``.

        Example
        -------
        >>> import covseisnet as csn
        >>> stream = csn.read()
        >>> stream.download_inventory(datacenter="LMU")
            Inventory created at 2024-08-20T09:10:33.403152Z
                Created by: ObsPy 1.4.1
                            https://www.obspy.org
                Sending institution: ObsPy 1.4.1,SeisComP (LMU)
                Contains:
                        Networks (3):
                                BW (3x)
                        Stations (3):
                                BW.RJOB (Jochberg, Bavaria, BW-Net) (3x)
                        Channels (0):
        """
        from obspy.clients.fdsn import Client

        # Initialize the client and the inventory
        inventory = Inventory()
        client = Client(datacenter)

        # Loop over traces and get the stations
        for trace in self:
            inventory += client.get_stations(
                network=trace.stats.network,
                station=trace.stats.station,
                location=trace.stats.location,
                channel=trace.stats.channel,
                starttime=trace.stats.starttime,
                endtime=trace.stats.endtime,
            )

        return inventory

    @property
    def synced(self) -> bool:
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
        if not self.equal_rates:
            return False

        # Assert number of samples
        if not self.equal_length:
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
    def equal_rates(self) -> bool:
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
    def equal_length(self) -> bool:
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
    def sampling_rate(self) -> float:
        """Common sampling rate in Hertz.

        This property asserts that all traces have the same sampling rate and
        returns the sampling rate of the first trace via the NetworkStream
        :meth:`stats` custom method. This is equivalent to getting
        ``stream[0].stats.sampling_rate`` directly, minus the assertion.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.sampling_rate
        100.0
        """
        # Assert sampling rate
        assert self.equal_rates, "Found different sampling rates."

        # Return the sampling rate of the first trace
        return self.stats(index=0).sampling_rate

    @property
    def npts(self) -> int:
        """Common number of samples.

        This property asserts that all traces have the same number of samples
        and returns the number of samples of the first trace via the
        NetworkStream :meth:`stats` custom method. This is equivalent to
        getting ``stream[0].stats.npts`` directly, minus the assertion.

        Example
        -------
        >>> stream = csn.read()
        >>> stream.npts
        3000
        """
        # Assert number of samples
        assert self.equal_length, "Traces have different number of samples."

        # Return the number of samples of the first trace
        return self.stats(index=0).npts

    @property
    def coordinates(self) -> list[tuple[float, float, float]]:
        """Geographical coordinates of the traces.

        This property is also available directly from looping over the traces
        and accessing the :attr:`stats.coordinates` attribute of each trace.
        """
        # Initialize the coordinates
        coordinates = []
        for trace in self:

            # Check if the coordinates are defined
            if (
                not hasattr(trace.stats, "coordinates")
                or trace.stats.coordinates is None
            ):
                raise ValueError(f"Undefined coordinates for {trace.id}.")

            # Append the coordinates
            coordinates.append(
                (
                    trace.stats.coordinates.longitude,
                    trace.stats.coordinates.latitude,
                    trace.stats.coordinates.elevation,
                )
            )
        return coordinates


def read(
    pathname_or_url: str | BytesIO | None = None, **kwargs: dict
) -> NetworkStream:
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
