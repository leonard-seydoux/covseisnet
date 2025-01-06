"""Module to deal with velocity models."""

import numpy as np
import os

from .spatial import Regular3DGrid


class VelocityModel(Regular3DGrid):
    r"""Base class to create a velocity model."""

    velocity: np.ndarray | float | None

    def __new__(cls, velocity, **kwargs):
        obj = super().__new__(cls, **kwargs)
        obj[...] = velocity
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)

    def resample(self, shape: tuple[int, int, int]):
        r"""Resample the velocity model to a new shape.

        Arguments
        ---------
        shape : tuple
            The new shape of the grid in the form ``(n_lon, n_lat, n_depth)``.

        Returns
        -------
        velocity_model : VelocityModel
            The velocity model with the new resolution.
        """
        lon = np.arange(self.extent[0], self.extent[1], shape[0])
        lat = np.arange(self.extent[2], self.extent[3], shape[1])
        depth = np.arange(self.extent[4], self.extent[5], shape[2])
        return self.cast_to_new_grid(lon, lat, depth)

    def cast_to_new_grid(self, new_longitudes, new_latitudes, new_depths):
        r"""Interpolate velocity model onto new grid."""
        from scipy.interpolate import RegularGridInterpolator

        # new Regular3DGrid instance
        extent = (
            new_longitudes.min(),
            new_longitudes.max(),
            new_latitudes.min(),
            new_latitudes.max(),
            new_depths.min(),
            new_depths.max(),
        )
        shape = (len(new_longitudes), len(new_latitudes), len(new_depths))
        new_grid = Regular3DGrid(extent=extent, shape=shape)
        # initialize interpolators: points outside the original
        # grid are given the values of the nearest neighbors
        interpolator_inside = RegularGridInterpolator(
            (self.lon, self.lat, self.depth),
            self,  # .velocity,
            method="linear",
        )
        interpolator_outside = RegularGridInterpolator(
            (self.lon, self.lat, self.depth),
            self,  # .velocity,
            method="nearest",
            fill_value=None,
            bounds_error=False,
        )
        # bounds of the new grid
        lon_min = self.lon.min()
        lon_max = self.lon.max()
        lat_min = self.lat.min()
        lat_max = self.lat.max()
        dep_min = self.depth.min()
        dep_max = self.depth.max()
        # new grid
        lon_g, lat_g, dep_g = new_grid.mesh
        # find points inside and outside the original grid
        inside = (
            (lon_g >= lon_min)
            & (lon_g <= lon_max)
            & (lat_g >= lat_min)
            & (lat_g <= lat_max)
            & (dep_g >= dep_min)
            & (dep_g <= dep_max)
        )
        outside = ~inside
        # new field
        new_field = np.zeros(
            new_grid.shape, dtype=self.dtype  # self.velocity.dtype
        )
        new_field[inside] = interpolator_inside(
            (lon_g[inside], lat_g[inside], dep_g[inside])
        )
        new_field[outside] = interpolator_outside(
            (lon_g[outside], lat_g[outside], dep_g[outside])
        )
        #
        extent = (
            new_longitudes.min(),
            new_longitudes.max(),
            new_latitudes.min(),
            new_latitudes.max(),
            new_depths.min(),
            new_depths.max(),
        )
        new_velocity_model = VelocityModel(
            extent=extent, shape=new_grid.shape, velocity=new_field
        )
        return new_velocity_model


# class VelocityModel3D(VelocityModel):
#     r"""Class to create a 3D velocity model."""

#     velocity: np.ndarray | None

#     def __new__(cls, velocity3d: np.ndarray, **kwargs):
#         r"""
#         Arguments
#         ---------
#         extent: tuple
#             The geographical extent of the grid in the form ``(lon_min, lon_max,
#             lat_min, lat_max, depth_min, depth_max)``. The longitudes and
#             latitudes are in decimal degrees, and the depths in kilometers.
#         shape: tuple
#             The number of points in the grid in the form ``(n_lon, n_lat,
#             n_depth)``.
#         velocity3d: np.ndarray
#             The 3d velocity of the model in kilometers per second.
#         """
#         obj = super().__new__(cls, **kwargs)
#         obj[...] = velocity3d
#         obj.velocity = velocity3d
#         return obj

#     def __array_finalize__(self, obj):
#         if obj is None:
#             return
#         self.lon = getattr(obj, "lon", None)
#         self.lat = getattr(obj, "lat", None)
#         self.depth = getattr(obj, "depth", None)
#         self.mesh = getattr(obj, "mesh", None)
#         self.velocity = getattr(obj, "velocity", None)


# class VelocityModel0D(VelocityModel):
#     r"""Class to create a constant velocity model."""

#     velocity: float | None

#     def __new__(cls, velocity0d: float, **kwargs):
#         r"""
#         Arguments
#         ---------
#         extent: tuple
#             The geographical extent of the grid in the form ``(lon_min, lon_max,
#             lat_min, lat_max, depth_min, depth_max)``. The longitudes and
#             latitudes are in decimal degrees, and the depths in kilometers.
#         shape: tuple
#             The number of points in the grid in the form ``(n_lon, n_lat,
#             n_depth)``.
#         velocity0d: float
#             The constant velocity of the model in kilometers per second.
#         """
#         obj = super().__new__(cls, **kwargs)
#         obj[...] = velocity0d
#         obj.velocity = velocity0d
#         return obj

#     def __array_finalize__(self, obj):
#         if obj is None:
#             return
#         self.lon = getattr(obj, "lon", None)
#         self.lat = getattr(obj, "lat", None)
#         self.depth = getattr(obj, "depth", None)
#         self.mesh = getattr(obj, "mesh", None)
#         self.velocity = getattr(obj, "velocity", None)


def model_from_grid(longitude, latitude, depth, velocity):
    r"""Create a velocity model from a grid of coordinates and velocities.

    Arguments
    ---------
    longitude : np.ndarray
        The longitudes of the grid in decimal degrees.
    latitude : np.ndarray
        The latitudes of the grid in decimal degrees.
    depth : np.ndarray
        The depths of the grid in kilometers.
    velocity : np.ndarray
        The 3d velocity of the model in kilometers per second.

    Returns
    -------
    velocity_model_3d : VelocityModel3D
        Instance of `VelocityModel3D` built with the grid of coordinates and
        velocities.
    """
    # Get extent
    extent = (
        longitude.min(),
        longitude.max(),
        latitude.min(),
        latitude.max(),
        depth.min(),
        depth.max(),
    )

    # Get shape
    shape = velocity.shape

    # Instantiate
    return VelocityModel(extent=extent, shape=shape, velocity=velocity)
