"""Data management.

This module provides functions to download and manage seismic data. This is
useful to download example datasets for instance.
"""

from obspy import UTCDateTime
import obspy
from obspy.clients.fdsn import Client


def download_dataset(
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    datacenter: str = "IRIS",
    **kwargs,
) -> obspy.Stream | None:
    """Download seismic data from a datacenter.

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
    filepath_destination: str = "../data/undervolc_example.mseed",
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

    # Merge
    stream.merge(method=-1)

    # Sort
    stream.sort(keys=["station"])

    # Write stream
    stream.write(filepath_destination, format="MSEED", encoding="FLOAT64")

    # Print message
    print(f"Data saved to {filepath_destination}")


def download_usarray_data(
    filepath_destination: str = "../data/usarray_example.mseed",
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

    # Sort
    stream.sort(keys=["station"])

    # Write stream
    stream.write(filepath_destination, format="MSEED")

    # Print message
    print(f"Data saved to {filepath_destination}")


def download_geoscope_data(
    filepath_destination: str = "../data/geoscope_example.mseed",
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

    # Resample data to 20 Hz
    # stream.resample(20)

    # Merge
    stream.merge(method=-1)

    # Sort
    stream.sort(keys=["station"])

    # Write stream
    stream.write(filepath_destination, format="MSEED")

    # Print message
    print(f"Data saved to {filepath_destination}")
