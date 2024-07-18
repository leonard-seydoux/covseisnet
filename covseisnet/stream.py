"""Read and pre-process seismic data."""

import obspy


class NetworkStream(obspy.Stream):
    """Custom Stream class for seismic arrays."""

    def __init__(self, *args, **kwargs):
        """Initialize the NetworkStream object."""
        # Call the parent class constructor
        super().__init__(*args, **kwargs)

        # Infer number of stations
        self.stations = list(set([tr.stats.station for tr in self.traces]))

    def __str__(self, extended=False):
        # get longest id
        if self.traces:
            longest_id = self and max(len(tr.id) for tr in self) or 0
        else:
            longest_id = 0

        # Initialize output string
        n_traces = len(self.traces)
        n_stations = len(self.stations)
        out = f"Network Stream of {n_traces} traces from {n_stations} stations:\n"

        # Print all traces if there are less than 20
        if n_traces <= 20 or extended is True:
            out = out + "\n".join(
                [trace.__str__(longest_id) for trace in self]
            )

        # Otherwise, print only the first and last traces
        else:
            out = (
                out
                + "\n"
                + self.traces[0].__str__()
                + "\n"
                + "...\n(%i other traces)\n...\n" % (n_traces - 2)
                + self.traces[-1].__str__()
                + '\n\n[Use "print('
                + 'Stream.__str__(extended=True))" to print all Traces]'
            )
        return out


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
