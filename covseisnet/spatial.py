"""Module to deal with spatial and geographical data."""

import numpy as np
from obspy.core.trace import Stats
from obspy.geodetics.base import locations2degrees


class Regular3DGrid(np.ndarray):
    r"""Base class for regular three-dimensional grid of values such as
    velocity models, travel times, and differential travel times.

    .. rubric:: Attributes

    - :attr:`lon` (:class:`numpy.ndarray`): The longitudes of the grid in
      decimal degrees.
    - :attr:`lat` (:class:`numpy.ndarray`): The latitudes of the grid in
      decimal degrees.
    - :attr:`depth` (:class:`numpy.ndarray`): The depths of the grid in
      kilometers.
    - :attr:`mesh` (:class:`list` of :class:`numpy.ndarray`): The meshgrid
      of the grid in the form ``(lon, lat, depth)``.

    .. rubric:: Methods

    - :meth:`flatten`: Flatten the grid to allow for faster calculations.
    """

    lon: np.ndarray | None
    lat: np.ndarray | None
    depth: np.ndarray | None
    mesh: list[np.ndarray] | None

    def __new__(
        cls,
        extent: tuple[float, float, float, float, float, float],
        shape: tuple[int, int, int],
    ):
        r"""
        Arguments
        ---------
        extent: tuple
            The geographical extent of the grid in the form ``(lon_min, lon_max,
            lat_min, lat_max, depth_min, depth_max)``. The longitudes and
            latitudes are in decimal degrees, and the depths in kilometers.
        shape: tuple
            The number of points in the grid in the form ``(n_lon, n_lat,
            n_depth)``.
        """
        # Extent
        obj = np.full(shape, np.nan).view(cls)

        # Grid mesh
        obj.lon = np.linspace(extent[0], extent[1], shape[0])
        obj.lat = np.linspace(extent[2], extent[3], shape[1])
        obj.depth = np.linspace(extent[4], extent[5], shape[2])
        obj.mesh = np.meshgrid(obj.lon, obj.lat, obj.depth, indexing="ij")

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)

    def __str__(self):
        ax_string = "\t{}: [{:0.2g}, {:0.2g}] with {} points\n"
        if (self.lon is None) or (self.lat is None) or (self.depth is None):
            raise ValueError("The grid axes are not defined.")
        return (
            f"{self.__class__.__name__}(\n"
            + ax_string.format(
                "lon", self.lon.min(), self.lon.max(), len(self.lon)
            )
            + ax_string.format(
                "lat", self.lat.min(), self.lat.max(), len(self.lat)
            )
            + ax_string.format(
                "depth", self.depth.min(), self.depth.max(), len(self.depth)
            )
            + f"\tmesh: {self.size} points\n"
            + f"\tmin: {self.min():.3f}\n"
            + f"\tmax: {self.max():.3f}\n"
            + ")"
        )

    @property
    def resolution(self):
        r"""Resolution of the grid in the form ``(lon_res, lat_res, depth_res)``."""
        if (self.lon is None) or (self.lat is None) or (self.depth is None):
            raise ValueError("The grid axes are not defined.")
        return (
            np.abs(self.lon[1] - self.lon[0]),
            np.abs(self.lat[1] - self.lat[0]),
            np.abs(self.depth[1] - self.depth[0]),
        )

    def flatten(self):
        r"""Flatten the grid.

        This method flattens the grid to allow for faster calculations. The
        flattened grid is a 2D array with the shape ``(n_points, 3)`` where the
        columns are the longitudes, latitudes, and depths of the grid. The
        method internally uses the :attr:`mesh` attribute to flatten the grid
        with the :attr:`numpy.ndarray.flat` attribute.
        """
        if self.mesh is None:
            raise ValueError("The grid mesh is not defined.")
        return np.array(
            [
                self.mesh[0].flat,
                self.mesh[1].flat,
                self.mesh[2].flat,
            ]
        ).T


def geographical_to_cartesian(
    lon: float, lat: float, depth: float, earth_radius: float = 6371
) -> tuple:
    r"""
    Convert geographical to Cartesian coordinates.

    Given a point of longitude :math:`\varphi`, latitude :math:`\lambda`, and
    depth :math:`d`, the Cartesian coordinates :math:`(x, y, z)` are
    calculated as follows:

    .. math::

        \begin{cases}
            x = (R - d) \cos(\lambda) \cos(\varphi) \\ y = (R - d)
            \cos(\lambda) \sin(\varphi) \\ z = (R - d) \sin(\lambda)
        \end{cases}


    where :math:`R` is the mean radius of the Earth. The depth is given in
    kilometers. And the latitude and longitude are given in radians (turned
    into radians by the function internally). Note that this transformation is
    only valid for small distances, and that no reference coordinate is
    provided. This is to calculate the relative distances between points in
    the same region.

    Arguments
    ---------
    lon, lat, depth : float
        Longitude, latitude, and depth of the point, with depth in kilometers,
        and latitude and longitude in degrees.
    earth_radius : float, optional
        Mean radius of the Earth in kilometers.

    Returns
    -------
    x, y, z : float
        Cartesian coordinates of the point.
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate Cartesian coordinates
    x = (earth_radius - depth) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (earth_radius - depth) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (earth_radius - depth) * np.sin(lat_rad)

    return x, y, z


def straight_ray_distance(
    lon1: float,
    lat1: float,
    depth1: float,
    lon2: float,
    lat2: float,
    depth2: float,
    earth_radius: float = 6371,
):
    r"""
    Straight-ray distance between two geographical coordinates.

    The straight-ray distance between two points is calculated using the
    Euclidean distance between the Cartesian coordinates of the points. The
    Cartesian coordinates are calculated using the geographical coordinates
    and the mean radius of the Earth, with the :func:`geographical_to_cartesian`
    function. The units of the longitudes and latitudes are in degrees, and the
    depths in kilometers.

    Arguments
    ---------
    lon1, lat1, depth1 : float
        Longitude, latitude, and depth of the first point.
    lon2, lat2, depth2 : float
        Longitude, latitude, and depth of the second point.
    earth_radius : float, optional
        Mean radius of the Earth in kilometers.

    Returns
    -------
    distance : float
        Straight-ray distance between the two points in kilometers.
    """
    # Convert both points to Cartesian coordinates
    x1, y1, z1 = geographical_to_cartesian(lon1, lat1, depth1, earth_radius)
    x2, y2, z2 = geographical_to_cartesian(lon2, lat2, depth2, earth_radius)

    # Calculate the distance using the 3D distance formula
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    return distance


def pairwise_great_circle_distances_from_stats(
    stats: list[Stats] | None = None,
    output_units: str = "km",
    include_diagonal: bool = True,
) -> list[float]:
    r"""Get the pairwise great-circle distances between stats coordinates.

    Calculate the pairwise great-circle distances between the coordinates of
    the different stations listed by a list of
    :class:`~obspy.core.trace.Stats` objects. The distances are calculated
    using the :func:`obspy.geodetics.base.locations2degrees`, if the stats
    contain a `coordinates` attribute. The output units can be in degrees or
    kilometers.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`, optional
        The list of stats objects containing the coordinates of the stations.
    output_units : str, optional
        The output units of the pairwise distances. The units can be 'degrees',
        'kilometers', or 'miles'.
    include_diagonal : bool, optional
        Whether to include the diagonal of the pairwise distances.

    Returns
    -------
    :class:`numpy.ndarray`
        The pairwise distances between the stations.
    """
    # Check that stats is defined
    if stats is None:
        raise ValueError("The stats are not defined.")

    # Get the station coordinates
    coordinates = [stat.coordinates for stat in stats]

    # Calculate the pairwise distances of the upper triangular part
    distance_to_diagonal = 0 if include_diagonal else 1
    n_stations = len(coordinates)
    pairwise_distances = []
    for i in range(n_stations):
        for j in range(i + distance_to_diagonal, n_stations):
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
