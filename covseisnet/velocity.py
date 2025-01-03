"""Module to deal with velocity models."""

import numpy as np
import os

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


class VelocityModel3D(VelocityModel):
    r"""Class to create a 3D velocity model."""

    velocity: np.ndarray | None

    def __new__(cls, velocity3d: np.ndarray, **kwargs):
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
        velocity3d: np.ndarray
            The 3d velocity of the model in kilometers per second.
        """
        obj = super().__new__(cls, **kwargs)
        obj[...] = velocity3d
        obj.velocity = velocity3d
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)
        self.velocity = getattr(obj, "velocity", None)

    @classmethod
    def read_from_hdf5(
        cls, data_field: str, path: str, axis_labels: tuple[str, str, str]
    ):
        r"""Instantiate from an hdf5 file.

        The hdf5 file must have the following data sets:
        - longitude
        - latitude
        - depth
        - `data_field`
        with `data_field` being the name of the target field, e.g., 'Vs'.

        Arguments
        ---------
        data_field : str
            Name of the target geolocated field in the hdf5 file.
        path : str
            Path to the hdf5 file.
        axis_labels : tuple[str, str, str]
            3-tuple of any combination of {'longitude', 'latitude', 'depth'}.
            These axis labels indicate the structure of the 3d grid in the
            hdf5 file.

        Returns
        -------
        velocity_model_3d : VelocityModel3D
            Instance of `VelocityModel3D` built with the hdf5 file at `path`.
        """
        import h5py as h5

        OUTPUT_AXIS_LABELS = ("longitude", "latitude", "depth")
        if not os.path.isfile(path):
            print(f"Could not find {path}.")
            return
        with h5.File(path, mode="r") as fin:
            if data_field not in fin:
                print(f"Could not find {data_field} in {path}.")
                return
            # !!!!!!!! here we assume that the coordinates along
            #          each axis are sorted in increasing order !!!!!!!!
            longitude = np.unique(fin["longitude"][()])
            latitude = np.unique(fin["latitude"][()])
            depth = np.unique(fin["depth"][()])
            velocity = fin[data_field][()]
        original_axis_positions = (
            axis_labels.index(OUTPUT_AXIS_LABELS[0]),
            axis_labels.index(OUTPUT_AXIS_LABELS[1]),
            axis_labels.index(OUTPUT_AXIS_LABELS[2]),
        )
        output_axis_positions = (0, 1, 2)
        # prepare arguments to instantiate `VelocityModel3D`
        velocity = np.moveaxis(velocity, original_axis_positions, output_axis_positions)
        num_lon = len(longitude)
        num_lat = len(latitude)
        num_dep = len(depth)
        shape = (num_lon, num_lat, num_dep)
        extent = (
            longitude.min(),
            longitude.max(),
            latitude.min(),
            latitude.max(),
            depth.min(),
            depth.max(),
        )
        return cls(extent=extent, shape=shape, velocity3d=velocity)


class VelocityModel0D(VelocityModel):
    r"""Class to create a constant velocity model."""

    velocity: float | None
      
    def __new__(cls, velocity0d: float, **kwargs):
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
        velocity0d: float
            The constant velocity of the model in kilometers per second.
        """
        obj = super().__new__(cls, **kwargs)
        obj[...] = velocity0d
        obj.velocity = velocity0d
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", None)
        self.lat = getattr(obj, "lat", None)
        self.depth = getattr(obj, "depth", None)
        self.mesh = getattr(obj, "mesh", None)
        self.velocity = getattr(obj, "velocity", None)
