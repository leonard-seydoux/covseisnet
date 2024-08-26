"""Module to deal with velocity models."""

import numpy as np
from obspy.core.trace import Stats

from .spatial import straight_ray_distance, Regular3DGrid
from .velocity import ConstantVelocityModel, VelocityModel


class TravelTimes(Regular3DGrid):
    r"""Three-dimensional travel time grid.

    This object is a three-dimensional grid of travel times between sources
    and a receiver. The sources are defined on the grid coordinates of the
    velocity model. The receiver is defined by its geographical coordinates,
    provided by the users. Depeding on the velocity model, the travel times
    are calculated using appropriate methods:

    - If the velocity model is a
      :class:`~covseisnet.velocity.ConstantVelocityModel`, the travel times
      are calculated using the straight ray distance between the sources and
      the receiver, with the :func:`~covseisnet.spatial.straight_ray_distance`.
    """

    stats: Stats | None
    receiver_coordinates: tuple[float, float, float] | None
    velocity_model: VelocityModel | ConstantVelocityModel | None

    def __new__(
        cls,
        velocity_model: VelocityModel | ConstantVelocityModel,
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
        if isinstance(velocity_model, ConstantVelocityModel):
            obj[...] = travel_times_constant_velocity(
                velocity_model, obj.receiver_coordinates
            )

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)
        self.velocity_model = getattr(obj, "velocity_model", None)
        self.stats = getattr(obj, "stats", None)
        self.receiver_coordinates = getattr(obj, "receiver_coordinates", None)


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
        obj[...] = travel_times_1 - travel_times_2

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
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)
        self.stats = getattr(obj, "stats", None)
        self.velocity_model = getattr(obj, "velocity_model", None)
        self.receiver_coordinates = getattr(obj, "receiver_coordinates", None)


def travel_times_constant_velocity(
    velocity: ConstantVelocityModel,
    receiver_coordinates: tuple[float, float, float],
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
    # Calculate the travel times
    travel_times = np.full(velocity.shape, np.nan)
    if receiver_coordinates is None:
        raise ValueError("The receiver position is not defined.")
    for i, position in enumerate(velocity.flatten()):
        distance = straight_ray_distance(*receiver_coordinates, *position)
        travel_times.flat[i] = distance / velocity.constant_velocity
    return travel_times.reshape(travel_times.shape, order="F")
