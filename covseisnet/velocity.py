"""Module to deal with velocity models."""

from .spatial import Regular3DGrid


class VelocityModel(Regular3DGrid):
    r"""Base class to create a velocity model."""

    def __new__(cls, **kwargs):
        return super().__new__(cls, **kwargs)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)


class ConstantVelocityModel(VelocityModel):
    r"""Class to create a constant velocity model."""

    constant_velocity: float | None

    def __new__(cls, velocity: float, **kwargs):
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
        velocity: float
            The constant velocity of the model in kilometers per second.
        """
        obj = super().__new__(cls, **kwargs)
        obj[...] = velocity
        obj.constant_velocity = velocity
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)
        self.constant_velocity = getattr(obj, "constant_velocity", None)
