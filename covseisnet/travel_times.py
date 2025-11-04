"""Module to deal with velocity models."""

import numpy as np
import warnings
from obspy.core.trace import Stats

from .spatial import straight_ray_distance, Regular3DGrid
from .velocity import VelocityModel


class TravelTimes(Regular3DGrid):
    r"""Three-dimensional travel time grid.

    This object is a three-dimensional grid of travel times between sources
    and a receiver. The sources are defined on the grid coordinates of the
    velocity model. The receiver is defined by its geographical coordinates,
    provided by the users. Depeding on the velocity model, the travel times
    are calculated using appropriate methods:

    - If the velocity model is a
      :class:`~covseisnet.velocity.VelocityModel`, the travel times
      are calculated using the straight ray distance between the sources and
      the receiver, with the :func:`~covseisnet.spatial.straight_ray_distance`.
    """

    stats: Stats | None
    receiver_coordinates: tuple[float, float, float] | None
    velocity_model: VelocityModel | None

    def __new__(
        cls,
        velocity_model: VelocityModel,
        stats: Stats | None = None,
        receiver_coordinates: tuple[float, float, float] | None = None,
    ):
        r"""
        Arguments
        ---------
        velocity_model: :class:`~covseisnet.velocity_model.VelocityModel`
            The velocity model used to calculate the travel times.
        stats: :class:`~obspy.core.trace.Stats`, optional
            The :class:`~obspy.core.trace.Stats` object of the receiver. If
            provided, the receiver coordinates are extracted from the stats.
        receiver_coordinates: tuple, optional
            The geographical coordinates of the receiver in the form
            ``(longitude, latitude, depth)``. The longitudes and latitudes are
            in decimal degrees, and the depths in kilometers. If the ``stats``
            is provided, the receiver coordinates are extracted from the stats
            and this argument is ignored.
        """
        # Create the object
        obj = velocity_model.copy().view(cls)
        obj[...] = np.nan

        # Check if stats is defined
        if stats is None and receiver_coordinates is None:
            raise ValueError(
                "One of stats or receiver coordinates must be defined."
            )
        if stats is not None and receiver_coordinates is not None:
            raise ValueError(
                "Only one of stats or receiver coordinates must be defined."
            )

        # Attributions
        if stats is not None:
            receiver_coordinates = (
                stats.coordinates["longitude"],
                stats.coordinates["latitude"],
                -1e-3 * stats.coordinates["elevation"],
            )
        if receiver_coordinates is None:
            raise ValueError("The receiver position is not defined.")
        obj.receiver_coordinates = receiver_coordinates
        obj.velocity_model = velocity_model

        # Calculate the travel times
        if isinstance(velocity_model, VelocityModel):
            obj[...] = obj.compute_travel_times(obj.receiver_coordinates)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", np.array([np.nan]))
        self.lat = getattr(obj, "lat", np.array([np.nan]))
        self.depth = getattr(obj, "depth", np.array([np.nan]))
        self.mesh = getattr(obj, "mesh", [np.array([np.nan])])
        self.velocity_model = getattr(obj, "velocity_model", None)
        self.stats = getattr(obj, "stats", None)
        self.receiver_coordinates = getattr(obj, "receiver_coordinates", None)

    def __sub__(self, other):
        if not isinstance(other, TravelTimes):
            raise ValueError("The object must be a TravelTimes object.")
        if not np.allclose(self.velocity_model, other.velocity_model):
            raise ValueError("The velocity model must be the same.")
        return DifferentialTravelTimes(self, other)

    def compute_travel_times(
        self, receiver_coordinates: tuple[float, float, float]
    ):
        r"""Calculate the travel times within a constant velocity model.

        Calculates the travel times of waves that travel at a constant velocity on
        the straight line between the sources and a receiver.

        The sources are defined on the grid coordinates of the velocity model. The
        receiver is defined by its geographical coordinates, provided by the
        users. The method internally uses the
        :func:`~covseisnet.spatial.straight_ray_distance` function to calculate the
        straight ray distance between the sources and the receiver.
        """
        if self.velocity_model is None:
            raise ValueError("The velocity model is not defined.")

        # Check if receiver coordinates are defined
        if receiver_coordinates is None and self.receiver_coordinates is None:
            raise ValueError("The receiver position is not defined.")

        # Initialize the travel times
        travel_times = np.full(self.velocity_model.shape, np.nan)

        # Split cases
        if self.velocity_model.is_constant():
            for i, position in enumerate(self.velocity_model.flatten()):
                distance = straight_ray_distance(
                    *receiver_coordinates, *position
                )
                travel_times.flat[i] = (
                    distance / self.velocity_model.constant_velocity
                )
        else:
            import pykonal

            # Origin of the grid in spherical coordinates
            origin = [
                self.velocity_model.lat.max(),
                self.velocity_model.lon.min(),
                self.velocity_model.depth.max(),
            ]
            pykonal_origin = pykonal.transformations.geo2sph(origin)  # type: ignore
            rho_min, theta_min, lbd_min = pykonal_origin

            # Extract resolutions
            lon_res, lat_res, depth_res = self.velocity_model.resolution
            lon_res = np.deg2rad(lon_res)
            lat_res = np.deg2rad(lat_res)

            # Initialize the Eikonal solver
            solver = pykonal.solver.PointSourceSolver(coord_sys="spherical")

            # Define the computational grid
            solver.velocity.min_coords = pykonal_origin
            solver.velocity.node_intervals = depth_res, lat_res, lon_res
            num_lon, num_lat, num_dep = self.velocity_model.shape
            solver.velocity.npts = num_dep, num_lat, num_lon
            solver.velocity.values = np.flip(
                np.flip(np.swapaxes(self.velocity_model, 0, 2), axis=1), axis=0
            )

            # Convert the geographical coordinates of the receiver to spherical coordinates
            rec_lon, rec_lat, rec_dep = receiver_coordinates
            rho_rec, theta_rec, lbd_rec = pykonal.transformations.geo2sph(  # type: ignore
                np.array([rec_lat, rec_lon, rec_dep])
            )
            solver.src_loc = np.array([rho_rec, theta_rec, lbd_rec])

            # Check if source is within grid:
            if (
                (solver.src_loc[0] < rho_min)
                or (solver.src_loc[0] > rho_min + num_dep * depth_res)
                or (solver.src_loc[1] < theta_min)
                or (solver.src_loc[1] > theta_min + num_lat * lat_res)
                or (solver.src_loc[2] < lbd_min)
                or (solver.src_loc[2] > lbd_min + num_lon * lon_res)
            ):
                warnings.warn(
                    "Receiver is outside the grid! Pykonal will return Infs."
                )

            solver.solve()

            travel_times[...] = np.flip(
                np.flip(np.swapaxes(solver.tt.values, 0, 2), axis=2), axis=1
            )

        return travel_times


class DifferentialTravelTimes(TravelTimes):
    r"""Three-dimensional differential travel time grid.

    This object is a three-dimensional grid of differential travel times
    between two travel time grids. The sources are defined on the grid
    coordinates of the velocity model. The two receivers are defined in each
    of the travel time grids. The differential travel times are calculated by
    subtracting the travel times of the second grid from the first grid. These
    differential travel times are useful to perform back-projection on the
    cross-correlation functions. Note that the differential travel times do
    not depend on the way the travel times are calculated, as long as the
    sources are defined on the grid coordinates of the velocity model.
    """

    receiver_coordinates: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    )

    def __new__(cls, travel_times_1: TravelTimes, travel_times_2: TravelTimes):
        r"""
        Arguments
        ---------
        travel_times_1: :class:`~covseisnet.travel_times.TravelTimes`
            The first travel time grid.
        travel_times_2: :class:`~covseisnet.travel_times.TravelTimes`
            The second travel time grid.
        """
        # Create the object
        obj = travel_times_1.copy().view(cls)

        # Calculate the differential travel times
        obj[...] = travel_times_1.__array__() - travel_times_2.__array__()

        # Attributions
        if travel_times_1.receiver_coordinates is None:
            raise ValueError("The receiver position is not defined.")
        if travel_times_2.receiver_coordinates is None:
            raise ValueError("The receiver position is not defined.")
        obj.receiver_coordinates = (
            travel_times_1.receiver_coordinates,
            travel_times_2.receiver_coordinates,
        )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", np.array([np.nan]))
        self.lat = getattr(obj, "lat", np.array([np.nan]))
        self.depth = getattr(obj, "depth", np.array([np.nan]))
        self.mesh = getattr(obj, "mesh", [np.array([np.nan])])
        self.stats = getattr(obj, "stats", None)
        self.velocity_model = getattr(obj, "velocity_model", None)
        self.receiver_coordinates = getattr(obj, "receiver_coordinates", None)


def calculate_travel_times(
    velocity: VelocityModel,
    receiver_coordinates: tuple[float, float, float] | None = None,
    stats: Stats | None = None,
):
    r"""Calculate the travel times within a constant velocity model.

    Calculates the travel times of waves that travel at a constant velocity on
    the straight line between the sources and a receiver.

    The sources are defined on the grid coordinates of the velocity model. The
    receiver is defined by its geographical coordinates, provided by the
    users. The method internally uses the
    :func:`~covseisnet.spatial.straight_ray_distance` function to calculate the
    straight ray distance between the sources and the receiver.
    """
    # Create the instance
    return TravelTimes(
        velocity, stats=stats, receiver_coordinates=receiver_coordinates
    )
