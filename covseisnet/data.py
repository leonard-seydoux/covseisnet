"""Data management."""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime


def download_undervolc_data(
    filepath_destination: str = "../data/undervolc_example.mseed",
    starttime: UTCDateTime = UTCDateTime("2010-10-14T09:00:00"),
    endtime: UTCDateTime = UTCDateTime("2010-10-14T16:00:00"),
    datacenter: str = "RESIF",
    **kwargs,
) -> None:
    """Download data from the UNDerVolc network.

    Arguments
    ---------
    starttime : :class:`~obspy.UTCDateTime`
        The start time of the data to download.
    endtime : :class:`~obspy.UTCDateTime`
        The end time of the data to download.
    """
    # Print message
    print(f"Downloading data from the {datacenter} datacenter.")

    # Client
    client = Client(datacenter)

    # Set default parameters
    kwargs.setdefault("network", "YA")
    kwargs.setdefault("station", "UV*")
    kwargs.setdefault("location", "*")
    kwargs.setdefault("channel", "HHZ")

    # Download data
    stream = client.get_waveforms(
        starttime=starttime, endtime=endtime, **kwargs
    )

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
