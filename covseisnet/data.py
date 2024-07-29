"""Data management.

This module provides functions to download and manage seismic data. This is
useful to download example datasets for instance.
"""

import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime


def download_dataset(
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    datacenter: str,
    **kwargs,
) -> obspy.Stream:
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
