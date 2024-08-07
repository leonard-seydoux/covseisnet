"""
The data shown in this documentation is for demonstration purposes only. In
order to deal with seismic data download and management, this module provides
functions to download seismic data from different datacenters. 

Most of the data management is made with the ObsPy library, in particular with
the Obspy client interface for dealing with FDSN webservices. For more
information about the usage of this interface, please visit the user guide
about `FDSN web service client for ObsPy
<https://docs.obspy.org/packages/obspy.clients.fdsn.html>`_. 

By default, the client downloads the data into the ``/data`` repository
located at the root of this project, as shown in the following file structure
of the project. If you would like to write the data to another location, we
recommend you to use the ``filepath_destination`` argument of the methods
presented in this module, and run them in your own script. 

::

    /
    ├── covseisnet/
    ├── data/
    ├── docs/
    ├── examples
    ├── tests/
    ├── README.md
    ├── LICENSE
    └── pyproject.toml

There are three presets to download data in the module:

- :func:`~covseisnet.data.download_undervolc_data` to download data from the
  UnderVolc network between 2010-10-14T09:00:00 and 2010-10-14T16:00:00.
  During these times, we observe an elevation of the seismic activity prior to
  an eruption of the Piton de la Fournaise accompanied by a co-eruptive
  tremor. 

- :func:`~covseisnet.data.download_usarray_data` to download data from the US
  Transportable Array experiment between 2010-01-01 and 2010-03-01. In this
  case, we download only the channels LHZ from the stations BGNE, J28A, sL27A,
  N23A, O28A, P33A, R27A, S32A, U29A, W31A, allowing to show interesting
  results of ambient-noise cross-correlation. 

- :func:`~covseisnet.data.download_geoscope_data` to download data from the
  GEOSCOPE network between 2020-05-01 and 2020-08-01. This data is used to
  show the results of ambient-noise cross-correlation at larger scale.

These functions all call the :func:`~covseisnet.data.download_dataset` with
specific arguments. You can also directly use these function to download
datasets that you would like to try the package on. 

"""

from os import path

import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

DIRECTORY_PACKAGE = path.dirname(__file__)
DIRECTORY_DATA = path.join(path.dirname(DIRECTORY_PACKAGE), "data")


def download_dataset(
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    datacenter: str = "IRIS",
    **kwargs,
) -> obspy.Stream | None:
    """Download seismic data from a datacenter.

    This function is a simple wwrapper to the
    :meth:`~obspy.clients.fdsn.client.Client.get_waveforms` method. It connect
    to the FDSN client using the specified ``datacenter`` (which by default is
    set to IRIS), and run the download with the
    :meth:`~obspy.clients.fdsn.client.Client.get_waveforms` method between the
    ``starttime`` and ``endtime``. Using the other arguments allow to specify
    the query to the datacenter. For more information, please check the
    Obspy documentation.

    Example
    -------

        >>> import covseisnet as csn
        >>> from obspy import UTCDateTime
        >>> stream = csn.download_dataset(
                starttime=UTCDateTime("2010-01-01 00:00"),
                endtime=UTCDateTime("2010-01-01 00:01"),
                datacenter="IRIS",
                channel="LHZ",
                station="ANMO",
                network=
            )

    Arguments
    ---------
    starttime: :class:`~obspy.UTCDateTime`
        The start time of the data to download.
    endtime: :class:`~obspy.UTCDateTime`
        The end time of the data to download.
    datacenter: str
        The datacenter to download the data from.
    **kwargs: dict
        Additional parameters to pass to the client.

    Returns
    -------
    :class:`~obspy.Stream`
        The downloaded seismic data.
    """
    # Client
    client = Client(datacenter)

    # Download data
    return client.get_waveforms(
        starttime=starttime,
        endtime=endtime,
        **kwargs,
    )


def download_undervolc_data(
    filepath_destination: str | None = None,
    starttime: UTCDateTime = UTCDateTime("2010-10-14T09:00:00"),
    endtime: UTCDateTime = UTCDateTime("2010-10-14T16:00:00"),
    datacenter: str = "RESIF",
    **kwargs,
) -> None:
    """Download data from the UnderVolc network.

    Arguments
    ---------
    starttime : :class:`~obspy.UTCDateTime`
        The start time of the data to download.
    endtime : :class:`~obspy.UTCDateTime`
        The end time of the data to download.
    """
    # Infer location
    if filepath_destination is None:
        filename = "example_undervolc.mseed"
        filepath_destination = path.join(DIRECTORY_DATA, filename)

    # Print message
    print(f"Downloading data from the {datacenter} datacenter.")

    # Set default parameters
    kwargs.setdefault("network", "YA")
    kwargs.setdefault("station", "UV*")
    kwargs.setdefault("location", "*")
    kwargs.setdefault("channel", "HHZ")

    # Download data
    stream = download_dataset(starttime=starttime, endtime=endtime, **kwargs)

    # Raise error if no data
    if stream is None:
        raise ValueError("No data found.")

    # Resample data to 20 Hz
    stream.resample(20)
    stream.merge(method=-1)
    stream.sort(keys=["station"])

    # Write stream
    stream.write(filepath_destination, format="MSEED", encoding="FLOAT64")

    # Print message
    print(f"Data saved to {filepath_destination}")


def download_usarray_data(
    filepath_destination: str | None = None,
    starttime: UTCDateTime = UTCDateTime("2010-01-01"),
    endtime: UTCDateTime = UTCDateTime("2010-03-01"),
    datacenter: str = "IRIS",
    **kwargs,
) -> None:
    """Download data from the UnderVolc network.

    Arguments
    ---------
    starttime : :class:`~obspy.UTCDateTime`
        The start time of the data to download.
    endtime : :class:`~obspy.UTCDateTime`
        The end time of the data to download.
    """
    # Infer location
    if filepath_destination is None:
        filename = "example_usarray.mseed"
        filepath_destination = path.join(DIRECTORY_DATA, filename)

    # Print message
    print(f"Downloading data from the {datacenter} datacenter.")

    # Set default parameters
    kwargs.setdefault("network", "TA")
    kwargs.setdefault(
        "station",
        "BGNE,J28A,L27A,N23A,O28A,P33A,R27A,S32A,U29A,W31A",
    )
    kwargs.setdefault("location", "*")
    kwargs.setdefault("channel", "LHZ")

    # Download data
    stream = download_dataset(starttime=starttime, endtime=endtime, **kwargs)

    # Raise error if no data
    if stream is None:
        raise ValueError("No data found.")

    # Merge
    stream.merge(method=-1)
    stream.sort(keys=["station"])

    # Write stream
    stream.write(filepath_destination, format="MSEED")

    # Print message
    print(f"Data saved to {filepath_destination}")


def download_geoscope_data(
    filepath_destination: str | None = None,
    starttime: UTCDateTime = UTCDateTime("2020-05-01"),
    endtime: UTCDateTime = UTCDateTime("2020-08-01"),
    datacenter: str = "RESIF",
    **kwargs,
) -> None:
    """Download data from the UnderVolc network.

    Arguments
    ---------
    starttime : :class:`~obspy.UTCDateTime`
        The start time of the data to download.
    endtime : :class:`~obspy.UTCDateTime`
        The end time of the data to download.
    """
    # Infer location
    if filepath_destination is None:
        filename = "example_geoscope.mseed"
        filepath_destination = path.join(DIRECTORY_DATA, filename)

    # Print message
    print(f"Downloading data from the {datacenter} datacenter.")

    # Set default parameters
    kwargs.setdefault("network", "G")
    kwargs.setdefault("station", "CLEV,CLF,CURIE,GOTF,HARD")
    kwargs.setdefault("location", "*")
    kwargs.setdefault("channel", "LHZ")

    # Download data
    stream = download_dataset(starttime=starttime, endtime=endtime, **kwargs)

    # Raise error if no data
    if stream is None:
        raise ValueError("No data found.")

    # Preprocess
    stream.merge(method=-1)
    stream.sort(keys=["station"])

    # Write stream
    stream.write(filepath_destination, format="MSEED")

    # Print message
    print(f"Data saved to {filepath_destination}")
