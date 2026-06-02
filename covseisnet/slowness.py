"""Slowness grids and plane-wave time delays for beamforming.

This module defines the :class:`SlownessGrid` — a two-dimensional grid
parameterised by apparent slowness magnitude and azimuth (or equivalently by
the Cartesian slowness components :math:`u_x`, :math:`u_y`) — and the
:class:`PlaneWaveDelays` subclass that pre-computes the time delay of a
plane wave at each station for every grid point.

Contrary to :class:`~covseisnet.spatial.GeographicalGrid`, which lives in
three-dimensional geographical space (longitude, latitude, depth), a
:class:`SlownessGrid` lives in two-dimensional horizontal slowness space.
It is the natural input for plane-wave beamforming, whereas
:class:`~covseisnet.travel_times.TravelTimes` is the natural input for
matched-field processing.

Classes
-------
- :class:`~covseisnet.slowness.SlownessGrid` — regular 2-D grid in
  slowness–azimuth space.
- :class:`~covseisnet.slowness.PlaneWaveDelays` — pre-computed inter-station
  time delays for every slowness grid point.
"""

import numpy as np
from obspy.core.trace import Stats

from .spatial import station_local_coordinates


class SlownessGrid(np.ndarray):
    r"""Regular two-dimensional grid in apparent-slowness space.

    The grid covers the apparent slowness magnitude
    :math:`s \in [s_{\min}, s_{\max}]` and the propagation azimuth
    :math:`\theta \in [0°, 360°)` measured clockwise from north.  At each
    grid point :math:`(s, \theta)` we can store any scalar quantity (e.g.
    beamformed power).

    Internally the grid is stored as a 2-D NumPy array of shape
    ``(n_slowness, n_azimuth)``.  The corresponding axis values are
    available as :attr:`slowness` and :attr:`azimuth`.

    The Cartesian slowness components are

    .. math::

        u_x = s \sin\theta, \qquad u_y = s \cos\theta

    and are available via the :attr:`ux` and :attr:`uy` properties.

    Arguments
    ---------
    slowness_max : float
        Maximum apparent slowness in s/km.
    n_slowness : int
        Number of slowness samples (including zero).
    n_azimuth : int
        Number of azimuth samples over :math:`[0°, 360°)`.
    fill_value : float, optional
        Initial fill value for the grid array.  Default is :obj:`numpy.nan`.

    Example
    -------

    .. plot::

        import covseisnet as csn

        grid = csn.slowness.SlownessGrid(
            slowness_max=0.5, n_slowness=30, n_azimuth=72
        )
        print(grid)
    """

    slowness: np.ndarray
    azimuth: np.ndarray

    def __new__(
        cls,
        slowness_max: float,
        n_slowness: int,
        n_azimuth: int,
        fill_value: float = np.nan,
    ) -> "SlownessGrid":
        r"""
        Arguments
        ---------
        slowness_max : float
            Maximum apparent slowness in s/km.
        n_slowness : int
            Number of slowness samples (from 0 to ``slowness_max``).
        n_azimuth : int
            Number of azimuth samples uniformly distributed over
            :math:`[0°, 360°)`.
        fill_value : float, optional
            Value used to initialise the grid.  Default is :obj:`numpy.nan`.
        """
        obj = np.full((n_slowness, n_azimuth), fill_value).view(cls)
        obj.slowness = np.linspace(0, slowness_max, n_slowness)
        obj.azimuth = np.linspace(0, 360, n_azimuth, endpoint=False)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.slowness = getattr(obj, "slowness", np.array([np.nan]))
        self.azimuth = getattr(obj, "azimuth", np.array([np.nan]))

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"\tslowness: [{self.slowness.min():.4f}, "
            f"{self.slowness.max():.4f}] s/km with {len(self.slowness)} points\n"
            f"\tazimuth:  [{self.azimuth.min():.1f}, "
            f"{self.azimuth.max():.1f}]° with {len(self.azimuth)} points\n"
            f"\tmesh: {self.size:,} points\n"
            ")"
        )

    @property
    def ux(self) -> np.ndarray:
        r"""East Cartesian slowness :math:`u_x = s \sin\theta` (s/km)."""
        s, az = np.meshgrid(self.slowness, self.azimuth, indexing="ij")
        return s * np.sin(np.radians(az))

    @property
    def uy(self) -> np.ndarray:
        r"""North Cartesian slowness :math:`u_y = s \cos\theta` (s/km)."""
        s, az = np.meshgrid(self.slowness, self.azimuth, indexing="ij")
        return s * np.cos(np.radians(az))

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Meshgrid ``(slowness, azimuth)`` of shape ``(n_slowness, n_azimuth)``."""
        return np.meshgrid(self.slowness, self.azimuth, indexing="ij")

    def maximum_coordinates(self) -> tuple[float, float]:
        r"""Slowness and azimuth at the grid maximum.

        Returns
        -------
        slowness, azimuth : float
            Apparent slowness in s/km and azimuth in degrees at the point of
            maximum value.
        """
        i, j = np.unravel_index(np.nanargmax(self), self.shape)
        return float(self.slowness[i]), float(self.azimuth[j])


class PlaneWaveDelays(SlownessGrid):
    r"""Time delays of a plane wave at each station over the slowness grid.

    For a plane wave with Cartesian slowness components
    :math:`(u_x, u_y) = s(\sin\theta, \cos\theta)` the delay at station
    :math:`i` located at flat-Earth coordinates :math:`(x_i, y_i)` is

    .. math::

        \tau_i(u_x, u_y) = u_x\, x_i + u_y\, y_i

    This object stores the full delay tensor of shape
    ``(n_stations, n_slowness, n_azimuth)`` as the :attr:`delays` attribute.
    The underlying :class:`SlownessGrid` data array is unused (filled with
    NaN) and serves only as a container for the grid metadata.

    Station coordinates are computed with
    :func:`~covseisnet.spatial.station_local_coordinates`.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.  Each must have a
        ``coordinates`` attribute with ``longitude`` and ``latitude`` keys.
    slowness_max : float
        Maximum apparent slowness in s/km.
    n_slowness : int
        Number of slowness samples (from 0 to ``slowness_max``).
    n_azimuth : int
        Number of azimuth samples over :math:`[0°, 360°)`.

    Attributes
    ----------
    delays : :class:`numpy.ndarray`
        Delay tensor of shape ``(n_stations, n_slowness, n_azimuth)`` in
        seconds.
    stats : list of :class:`~obspy.core.trace.Stats`
        The input stats list.

    Example
    -------

    Compute the delays for a 5-station array and print the shape:

    >>> import covseisnet as csn
    >>> stream = csn.read("docs/source/data/undervolc.mseed")
    >>> stream.assign_coordinates("docs/source/data/undervolc.xml")
    >>> stats = [tr.stats for tr in stream]
    >>> from covseisnet.slowness import PlaneWaveDelays
    >>> delays = PlaneWaveDelays(stats, slowness_max=0.5, n_slowness=50, n_azimuth=72)
    >>> print(delays.delays.shape)
    (15, 50, 72)
    """

    delays: np.ndarray
    stats: list[Stats]

    def __new__(
        cls,
        stats: list[Stats],
        slowness_max: float,
        n_slowness: int,
        n_azimuth: int,
    ) -> "PlaneWaveDelays":
        r"""
        Arguments
        ---------
        stats : list of :class:`~obspy.core.trace.Stats`
            Stats objects of the network stations.
        slowness_max : float
            Maximum apparent slowness in s/km.
        n_slowness : int
            Number of slowness samples.
        n_azimuth : int
            Number of azimuth samples.
        """
        obj = super().__new__(
            cls,
            slowness_max=slowness_max,
            n_slowness=n_slowness,
            n_azimuth=n_azimuth,
        )
        obj.stats = stats
        x, y = station_local_coordinates(stats)
        obj.delays = (
            obj.ux[np.newaxis, :, :] * x[:, np.newaxis, np.newaxis]
            + obj.uy[np.newaxis, :, :] * y[:, np.newaxis, np.newaxis]
        )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        self.delays = getattr(obj, "delays", None)
        self.stats = getattr(obj, "stats", None)
