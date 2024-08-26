"""
The data shown in this documentation is for demonstration purposes only. In
order to deal with seismic data download and management, this module provides
functions to download seismic data from different datacenters. 

File structure
--------------

We download the data with the ObsPy library, in particular with the Obspy
client interface for dealing with FDSN web services. For more information about
the usage of this interface, please visit the user guide about `FDSN web
service client for ObsPy
<https://docs.obspy.org/packages/obspy.clients.fdsn.html>`_. 

By default, the client downloads the data into the ``/data`` repository
located at the root of this project, as shown in the following file structure
of the project. If you would like to write the data to another location, we
recommend you to use the ``filepath_destination`` argument of the methods
presented in this module, and run them in your own script. 

::

    /
    ├── data/datasets.mseed 
    ├── covseisnet/
    ├── docs/
    ├── examples/
    ├── tests/
    ├── README.md
    ├── LICENSE
    └── pyproject.toml

Presets
-------

Note that these functions are presets to download data from specific networks
and time periods. These thwo datasets are used in the examples of the
documentation. If you modify the data download, you may need to adapt the
examples accordingly.

- :func:`~covseisnet.data.download_undervolc_data` to download data from the
  UnderVolc network between 2010-10-14T09:00:00 and 2010-10-14T16:00:00.
  During these times, we observe an elevation of the seismic activity prior to
  an eruption of the Piton de la Fournaise accompanied by a co-eruptive
  tremor. 

- :func:`~covseisnet.data.download_usarray_data` to download data from the US
  Transportable Array experiment between 2010-01-01 and 2010-03-01. In this
  case, we download only the channels LHZ from the stations R04C, O03C, M03C,
  L02A, I05A, allowing to show interesting results of ambient-noise
  cross-correlation. 

These functions all call the :func:`~covseisnet.data.download_seismic_dataset`
with specific arguments. You can also directly use these function to download
datasets that you would like to try the package on. 
"""

from os import path

import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client

from .stream import NetworkStream

DIRECTORY_PACKAGE = path.dirname(__file__)
DIRECTORY_DATA = path.join(path.dirname(DIRECTORY_PACKAGE), "data")


def download_seismic_dataset(
    starttime: UTCDateTime | str,
    endtime: UTCDateTime | str,
    network: str,
    station: str,
    channel: str = "*",
    location: str = "*",
    datacenter: str = "IRIS",
    process: dict | None = None,
    **kwargs,
) -> obspy.Stream:
    """Download seismic data from a datacenter.

    This function is a simple wwrapper to the
    :meth:`~obspy.clients.fdsn.client.Client.get_waveforms` method. It connect
    to the FDSN client using the specified ``datacenter`` (which by default is
    set to IRIS), and download seismic waveforms between the ``starttime`` and
    ``endtime``. Using the other arguments allow to specify the query to the
    datacenter. For more information, please check the Obspy documentation.

    As we aim our package to be used without explicitely calling the Obspy
    library, the dates are possible to be passed as strings.

    Arguments
    ---------
    starttime: str or :class:`~obspy.core.utcdatetime.UTCDateTime`
        The start time of the data to download.
    endtime: str or :class:`~obspy.core.utcdatetime.UTCDateTime`
        The end time of the data to download.
    network: str
        The network code to download the data from.
    station: str
        The station code to download the data from.
    channel: str, optional
        The channel code to download the data from. Default is "*".
    location: str, optional
        The location code to download the data from. Default is "*".
    datacenter: str, optional
        The datacenter to download the data from. Default is "IRIS".
    process: dict, optional
        A dictionary with the processing chain to apply to the downloaded
        data. The keys of the dictionary are the processing functions to
        apply, and the values are the arguments to pass to the processing
        function.
    **kwargs: dict
        Additional parameters to pass to the download method. These arguments
        may vary depending on the datacenter.

    Returns
    -------
    :class:`~obspy.core.stream.Stream`
        The downloaded seismic data.

    Example
    -------

        >>> import covseisnet as csn
        >>> stream = csn.data.download_seismic_dataset(
                starttime="2020-01-01 00:00",
                endtime="2020-01-01 00:01",
                channel="BHZ",
                station="TAM",
                network="G",
                datacenter="IPGP",
            )
        >>> print(stream)
        1 Trace(s) in Stream:
        G.TAM.00.BHZ | 2020-01-01T00:00:03.350000Z - \
            2020-01-01T00:01:00.000000Z | 20.0 Hz, 1134 samples
    """
    # Client
    client = Client(datacenter)

    # Turn string into UTCDateTime
    starttime = UTCDateTime(starttime)
    endtime = UTCDateTime(endtime)

    # Download data
    stream = client.get_waveforms(
        starttime=starttime,
        endtime=endtime,
        network=network,
        station=station,
        channel=channel,
        location=location,
        **kwargs,
    )

    # Transform to NetworkStream
    stream = NetworkStream(stream)

    # Raise error if no data
    if stream is None:
        raise ValueError("No data found.")

    # Preprocess
    if process is not None:
        stream.process(process)

    # Raise error if no data
    if stream is None:
        raise ValueError("No data in stream.")

    return stream


def download_undervolc_data(
    filepath_destination: str | None = None,
    starttime: str | UTCDateTime = "2010-10-14T09:00:00",
    endtime: str | UTCDateTime = "2010-10-14T16:00:00",
    network: str = "YA",
    station: str = "UV*",
    location: str = "*",
    channel: str = "HHZ",
    datacenter: str = "RESIF",
    process={
        "resample": 20,
        "merge": {"method": 1, "fill_value": 0},
        "sort": {"keys": ["station"]},
    },
    **kwargs,
) -> None:
    """Download data from the UnderVolc network.

    The argument are the same as the
    :func:`~covseisnet.data.download_seismic_dataset` with the following
    default values:

    - ``starttime``: "2010-10-14T09:00:00"
    - ``endtime``: "2010-10-14T16:00:00"
    - ``network``: "YA"
    - ``station``: "UV*"
    - ``location``: "*"
    - ``channel``: "HHZ"
    - ``datacenter``: "RESIF"
    - ``process``: resample to 20 Hz, merge the traces, and sort by station.

    And a default value for the ``filepath_destination`` argument, which is
    set to the ``/data`` repository located at the root of this project.

    """
    # Infer location
    if filepath_destination is None:
        filename = "undervolc_example.mseed"
        filepath_destination = path.join(DIRECTORY_DATA, filename)

    # Print message
    print(f"Downloading data from the {datacenter} datacenter.")

    # Download data
    stream = download_seismic_dataset(
        starttime=starttime,
        endtime=endtime,
        network=network,
        station=station,
        location=location,
        channel=channel,
        datacenter=datacenter,
        process=process,
        **kwargs,
    )
    print(stream)

    # Write stream
    stream.write(filepath_destination, format="MSEED", encoding="FLOAT64")

    # Print message
    print(f"Data saved to {filepath_destination}")


def download_usarray_data(
    filepath_destination: str | None = None,
    starttime: str | UTCDateTime = "2006-01-01",
    endtime: str | UTCDateTime = "2006-03-01",
    datacenter: str = "IRIS",
    network: str = "TA",
    station: str = "R04C,O03C,M03C,L02A,I05A",
    location: str = "*",
    channel: str = "LHZ",
    process={
        "merge": {"method": 1, "fill_value": 0},
        "resample": 0.2,
        "sort": {"keys": ["station"]},
    },
    **kwargs,
) -> None:
    """Download data from the USArray network.

    The argument are the same as the
    :func:`~covseisnet.data.download_seismic_dataset` with the following
    default values:

    - ``starttime``: "2006-01-01"
    - ``endtime``: "2006-03-01"
    - ``network``: "TA"
    - ``station``: "R04C,O03C,M03C,L02A,I05A"
    - ``location``: "*"
    - ``channel``: "LHZ"
    - ``datacenter``: "IRIS"
    - ``process``: merge the traces, resample to 0.2 Hz, and sort by station.

    And a default value for the ``filepath_destination`` argument, which is
    set to the ``/data`` repository located at the root of this project.
    """
    # Infer location
    if filepath_destination is None:
        filename = "usarray_example.mseed"
        filepath_destination = path.join(DIRECTORY_DATA, filename)

    # Print message
    print(f"Downloading data from the {datacenter} datacenter.")

    # Download data
    stream = download_seismic_dataset(
        starttime=starttime,
        endtime=endtime,
        datacenter=datacenter,
        network=network,
        station=station,
        location=location,
        channel=channel,
        process=process,
        **kwargs,
    )

    # Write stream
    stream.write(filepath_destination, format="MSEED", encoding="FLOAT64")

    # Get coordinates
    filepath_inventory = path.join(DIRECTORY_DATA, "usarray_example.xml")
    stream = NetworkStream(stream)
    inventory = stream.download_inventory(datacenter=datacenter)
    inventory.write(filepath_inventory, format="STATIONXML")

    # Print message
    print(f"Data saved to {filepath_destination}")
