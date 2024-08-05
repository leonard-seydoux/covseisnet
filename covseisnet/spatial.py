"""Module to deal with spatial and geographical data."""

from obspy.core.trace import Stats
from obspy.geodetics.base import locations2degrees


def pairwise_distances(
    stats: list[Stats], output_units: str = "km"
) -> list[float]:
    r"""Get the pairwise distances between the stations.

    The pairwise distances are calculated using the Euclidean distance
    between the stations. The distances are calculated using the station
    coordinates.

    Returns
    -------
    :class:`np.ndarray`
        The pairwise distances between the stations.
    """
    # Get the station coordinates
    coordinates = [stat.coordinates for stat in stats]

    # Calculate the pairwise distances of the upper triangular part
    n_stations = len(coordinates)
    pairwise_distances = []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            degrees = locations2degrees(
                coordinates[i].latitude,
                coordinates[i].longitude,
                coordinates[j].latitude,
                coordinates[j].longitude,
            )
            pairwise_distances.append(degrees)

    # Convert the pairwise distances to the output units
    match output_units:
        case "degrees":
            pairwise_distances = pairwise_distances
        case "kilometers" | "km":
            pairwise_distances = [
                distance * 111.11 for distance in pairwise_distances
            ]
        case _:
            raise ValueError(
                f"Invalid output units '{output_units}'. "
                "The output units must be 'degrees', 'kilometers', or 'miles'."
            )

    return pairwise_distances
