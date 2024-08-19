"""Module to deal with spatial and geographical data."""

import numpy as np
from obspy.core.trace import Stats
from obspy.geodetics.base import locations2degrees

from .correlation import CrossCorrelationMatrix


def convert_to_cartesian(lon, lat, depth, earth_radius=6371):
    """
    Convert geographical coordinates (latitude, longitude, depth) to Cartesian coordinates (x, y, z).

    Parameters:
    lat (float): Latitude in degrees.
    lon (float): Longitude in degrees.
    depth (float): Depth in kilometers (positive inward).
    R (float): Mean radius of the Earth in kilometers.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate Cartesian coordinates
    x = (earth_radius - depth) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (earth_radius - depth) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (earth_radius - depth) * np.sin(lat_rad)

    return x, y, z


def direct_distance(lon1, lat1, depth1, lon2, lat2, depth2, R=6371):
    """
    Calculate the direct distance between two points on Earth given latitude, longitude, and depth.

    Parameters:
    lat1, lon1, depth1 (float): Latitude, longitude, and depth of the first point.
    lat2, lon2, depth2 (float): Latitude, longitude, and depth of the second point.
    R (float): Mean radius of the Earth in kilometers.

    Returns:
    float: The direct distance between the two points in kilometers.
    """
    # Convert both points to Cartesian coordinates
    x1, y1, z1 = convert_to_cartesian(lon1, lat1, depth1, R)
    x2, y2, z2 = convert_to_cartesian(lon2, lat2, depth2, R)

    # Calculate the distance using the 3D distance formula
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    return distance


def pairwise_distances(
    stats: list[Stats],
    output_units: str = "km",
    include_diagonal: bool = True,
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


class Regular3DGrid(np.ndarray):
    """Class to create a regular grid."""

    lon: np.ndarray | None
    lat: np.ndarray | None
    depth: np.ndarray | None
    mesh: list[np.ndarray] | None

    def __new__(
        cls,
        extent: tuple[float, float, float, float, float, float],
        shape: tuple[int, int, int],
    ):
        """Create a regular grid.

        Arguments
        ---------
        extent: tuple
            The extent of the grid in the form (xmin, xmax, ymin, ymax, zmin,
            zmax).
        shape: tuple
            The number of points in the grid in the form (nx, ny, nz).
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
                "lon", self.lon.min(), self.lon.max(), self.shape[0]
            )
            + ax_string.format(
                "lat", self.lat.min(), self.lat.max(), self.shape[1]
            )
            + ax_string.format(
                "depth", self.depth.min(), self.depth.max(), self.shape[2]
            )
            + f"\tmesh: {self.size} points\n"
            + f"\tmin: {self.min():.3f}\n"
            + f"\tmax: {self.max():.3f}\n"
            + ")"
        )

    def __repr__(self):
        return str(self)

    def flatten(self):
        """Flatten the grid."""
        if self.mesh is None:
            raise ValueError("The grid mesh is not defined.")
        return np.array(
            [
                self.mesh[0].flat,
                self.mesh[1].flat,
                self.mesh[2].flat,
            ]
        ).T


class VelocityModel(Regular3DGrid):
    """Base class to create a velocity model."""

    def __new__(cls, **kwargs):
        return super().__new__(cls, **kwargs)


class ConstantVelocityModel(VelocityModel):
    """Class to create a constant velocity model."""

    constant_velocity: float

    def __new__(cls, velocity: float, **kwargs):
        obj = super().__new__(cls, **kwargs)
        obj[...] = velocity
        obj.constant_velocity = velocity
        return obj


class TravelTime3DGrid(Regular3DGrid):
    """Class to create a 3D travel time grid."""

    receiver_position: list | tuple | np.ndarray | None
    velocity_model: VelocityModel | ConstantVelocityModel | None

    def __new__(
        cls,
        receiver_position: list | tuple | np.ndarray,
        velocity_model: VelocityModel | ConstantVelocityModel,
    ):
        """Create a 3D travel time grid.

        Arguments
        ---------
        grid: :class:`~covseisnet.spatial.Regular3DGrid`
            The regular grid on which the travel times are calculated.
        velocity_model: :class:`~covseisnet.velocity_model.VelocityModel`
            The velocity model used to calculate the travel times.
        """
        # Create the object
        obj = velocity_model.copy().view(cls)
        obj[...] = np.nan

        # Attributions
        obj.receiver_position = receiver_position
        obj.velocity_model = velocity_model

        # Calculate the travel times
        if isinstance(velocity_model, ConstantVelocityModel):
            obj[...] = obj._calculate_travel_times_constant_velocity(
                velocity_model.constant_velocity
            )

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.receiver_position = getattr(obj, "receiver_position", None)
        self.velocity_model = getattr(obj, "velocity_model", None)

    def _calculate_travel_times_constant_velocity(self, velocity):
        """Calculate the travel times for a constant velocity model."""
        # Calculate the travel times
        travel_times = np.empty(self.shape)
        if self.receiver_position is None:
            raise ValueError("The receiver position is not defined.")
        for i, position in enumerate(self.flatten()):
            distance = direct_distance(
                *self.receiver_position,
                *position,
            )
            travel_times.flat[i] = distance / velocity

        return travel_times.reshape(self.shape, order="F")


# class BackProjection(Regular3DGrid):
#     """Class to perform back-projection on cross-correlation functions."""

#     def __new__(cls, travel_times: list[TravelTime3DGrid]):
#         obj = super().__new__(cls, travel_times.shape)

# def __init__(self, travel_times: list[TravelTime3DGrid]):
#     """Create a Beam object by providing a traveltime object and specifying
#     the number of time windows.

#     Parameters
#     ----------
#     travel_times : TravelTime3DGrid
#         The travel times object.
#     """
#     # Attributions
#     self.grid = travel_times[0].grid
#     self.likelihood = np.zeros(self.grid.shape)

# def calculate_likelihood(
#     self, cross_correlation: CrossCorrelationMatrix, lags: np.ndarray
# ):
#     """Calculate the likelihood of the back-projection.

#     Parameters
#     ----------
#     cross_correlation : np.ndarray
#         The cross-correlation function.
#     """
#     # Calculate the likelihood
#     for i, source_position in enumerate(self.grid.flatten()):

#         # Calculate the travel time
#         travel_time = self.travel_times.values.flat[i]

#         # Calculate the likelihood
#         self.likelihood.flat[i] = np.sum(cross_correlation * travel_time)
