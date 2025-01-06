"""Module to deal with velocity models."""

import numpy as np

from scipy.interpolate import RegularGridInterpolator

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
        self.lon = getattr(obj, "lon", np.array([np.nan]))
        self.lat = getattr(obj, "lat", np.array([np.nan]))
        self.depth = getattr(obj, "depth", np.array([np.nan]))
        self.mesh = getattr(obj, "mesh", [np.array([np.nan])])

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
        lon = np.linspace(self.extent[0], self.extent[1], shape[0])
        lat = np.linspace(self.extent[2], self.extent[3], shape[1])
        depth = np.linspace(self.extent[4], self.extent[5], shape[2])
        return self.interpolate(lon, lat, depth)

    def interpolate(
        self,
        lon,
        lat,
        depth,
        interpolation_method="linear",
        extrapolation_method="nearest",
    ):
        r"""Interpolate velocity model onto new grid.

        Arguments
        ---------
        lon : np.ndarray
            The longitudes of the new grid in decimal degrees.
        lat : np.ndarray
            The latitudes of the new grid in decimal degrees.
        depth : np.ndarray
            The depths of the new grid in kilometers.
        interpolation_method : str, optional
            The interpolation method used by `scipy.interpolate.RegularGridInterpolator`.
            Default is 'linear'.
        """

        # Instanciate new grid
        new_extent = (
            lon.min(),
            lon.max(),
            lat.min(),
            lat.max(),
            depth.min(),
            depth.max(),
        )
        new_shape = (len(lon), len(lat), len(depth))
        new_grid = Regular3DGrid(extent=new_extent, shape=new_shape)

        # Initialize interpolators. Points outside the original grid are given
        # the values of the nearest neighbors.
        interpolator = RegularGridInterpolator(
            (self.lon, self.lat, self.depth),
            self,
            method=interpolation_method,
        )
        extrapolator = RegularGridInterpolator(
            (self.lon, self.lat, self.depth),
            self,
            method=extrapolation_method,
            fill_value=None,  # type: ignore
            bounds_error=False,
        )

        # Find points inside and outside the original grid
        lon, lat, depth = new_grid.mesh
        inside = (
            (lon >= self.lon.min())  # extent[0])
            & (lon <= self.extent[1])
            & (lat >= self.extent[2])
            & (lat <= self.extent[3])
            & (depth >= self.extent[4])
            & (depth <= self.extent[5])
        )

        # New field
        velocity = np.zeros(new_grid.shape, dtype=self.dtype)

        # Interpolate and extrapolate
        velocity[inside] = interpolator(
            (lon[inside], lat[inside], depth[inside])
        )
        velocity[~inside] = extrapolator(
            (lon[~inside], lat[~inside], depth[~inside])
        )

        return VelocityModel(
            extent=new_extent, shape=new_grid.shape, velocity=velocity
        )


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
