"""Matched-field processing on a geographical grid.

This module provides the :class:`MatchedFieldProcessing` class, which
evaluates the Bartlett matched-field processor (MFP) power over a
:class:`~covseisnet.spatial.GeographicalGrid` using pre-computed travel times
and a :class:`~covseisnet.covariance.CovarianceMatrix` as input.

The Bartlett MFP is the three-dimensional generalisation of beamforming: the
steering vector is now the replica field of a spherical wave rather than a
plane wave, so the processor can resolve source depth as well as horizontal
position.

The MVDR matched-field processor (Capon) and MUSIC are planned for future
releases.

Classes
-------
- :class:`~covseisnet.matchedfield.MatchedFieldProcessing` — Bartlett MFP
  power on a geographical grid.
"""

import numpy as np

from .covariance import CovarianceMatrix
from .spatial import GeographicalGrid
from .travel_times import TravelTimes


class MatchedFieldProcessing(GeographicalGrid):
    r"""Bartlett matched-field processing power on a geographical grid.

    Given a :class:`~covseisnet.covariance.CovarianceMatrix`
    :math:`\mathbf{C} \in \mathbb{C}^{N \times N}` and a replica vector

    .. math::

        \mathbf{w}(\mathbf{r}, f) = \exp\!\bigl(-2\pi i f \boldsymbol{t}(
        \mathbf{r})\bigr)

    where :math:`\boldsymbol{t}(\mathbf{r})` is the vector of travel times
    from grid point :math:`\mathbf{r}` to every receiver, the Bartlett MFP
    power is

    .. math::

        P_{\mathrm{Bartlett}}(\mathbf{r}, f)
        = \mathbf{w}^\dagger \mathbf{C} \mathbf{w} \;/\; N^2

    where :math:`N` is the number of stations.

    The result is stored as a 3-D array of shape
    ``(n_lon, n_lat, n_depth)`` that inherits from
    :class:`~covseisnet.spatial.GeographicalGrid`.

    The MVDR processor minimises output power subject to a distortionless
    constraint on the replica:

    .. math::

        P_{\mathrm{MVDR}}(\mathbf{r}, f)
        = \frac{1}{\mathbf{w}^\dagger \mathbf{C}^{-1} \mathbf{w}}

    .. note::

        **Planned extensions**

        - *MUSIC pseudo-spectrum*: projects replica vectors onto the noise
          subspace of the covariance matrix; offers higher resolution than
          Bartlett for well-separated sources.

    Arguments
    ---------
    travel_times : dict[str, :class:`~covseisnet.travel_times.TravelTimes`]
        Dictionary mapping station identifiers to their travel-time grids.
        All grids must share the same geographical extent and shape.

    Attributes
    ----------
    travel_times : dict[str, :class:`~covseisnet.travel_times.TravelTimes`]
        The input travel-time grids.

    Example
    -------

    Compute the MFP power for a synthetic spherical wave:

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt

        import covseisnet as csn
        from covseisnet.matchedfield import MatchedFieldProcessing
        from covseisnet.synthetic import spherical_wave_covariance

        stream = csn.read("docs/source/data/undervolc.mseed")
        stream.assign_coordinates("docs/source/data/undervolc.xml")
        stats = [tr.stats for tr in stream]

        extent  = (14.8, 15.4, 37.6, 38.2, 0.0, 5.0)
        shape   = (20, 20, 5)
        velocity_model = csn.velocity.VelocityModel(
            extent=extent, shape=shape, velocity=3.0
        )
        source = (15.1, 37.9, 2.0)

        travel_times = {
            str(s): csn.calculate_travel_times(velocity_model, stats=s)
            for s in stats
        }

        frequency = 2.0
        slowness  = 1.0 / 3.0
        cov = spherical_wave_covariance(stats, frequency, slowness, source)

        mfp = MatchedFieldProcessing(travel_times)
        mfp.compute_bartlett(cov, frequency)

        fig, ax = plt.subplots()
        lons, lats, _ = mfp.mesh
        ax.pcolormesh(lons[:, :, 0], lats[:, :, 0], mfp[:, :, 0], cmap="inferno")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Bartlett MFP power (surface slice)")
        plt.tight_layout()
    """

    travel_times: dict[str, TravelTimes]

    def __new__(
        cls,
        travel_times: dict[str, TravelTimes],
    ) -> "MatchedFieldProcessing":
        r"""
        Arguments
        ---------
        travel_times : dict[str, :class:`~covseisnet.travel_times.TravelTimes`]
            Dictionary of travel-time grids, one per receiver.  All grids must
            share the same geographical extent and shape.
        """
        reference = next(iter(travel_times.values()))
        obj = np.full(reference.shape, np.nan).view(cls)
        obj.lon = reference.lon.copy()
        obj.lat = reference.lat.copy()
        obj.depth = reference.depth.copy()
        obj.mesh = reference.mesh
        obj.travel_times = travel_times
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lon = getattr(obj, "lon", np.array([np.nan]))
        self.lat = getattr(obj, "lat", np.array([np.nan]))
        self.depth = getattr(obj, "depth", np.array([np.nan]))
        self.mesh = getattr(obj, "mesh", [np.array([np.nan])])
        self.travel_times = getattr(obj, "travel_times", None)

    def compute_bartlett(
        self,
        covariance: CovarianceMatrix,
        frequency: float,
    ) -> None:
        r"""Compute the Bartlett MFP power in-place.

        Evaluates

        .. math::

            P(\mathbf{r}) = \mathbf{w}^\dagger(\mathbf{r}, f)\,
                            \mathbf{C}\,
                            \mathbf{w}(\mathbf{r}, f) \;/\; N^2

        for every grid point :math:`\mathbf{r}` and stores the result in the
        array.

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            Spectral covariance matrix at one time–frequency bin, of shape
            ``(n_stations, n_stations)``.
        frequency : float
            Frequency in Hz at which to evaluate the steering vectors.

        Raises
        ------
        ValueError
            If ``covariance`` does not have exactly 2 dimensions.
        """
        if covariance.ndim != 2:
            raise ValueError(
                f"covariance must be 2-D, got shape {covariance.shape}."
            )
        keys = list(self.travel_times)
        n_stations = len(keys)
        tt = np.stack([self.travel_times[k].__array__() for k in keys], axis=0)

        steering = np.exp(-2j * np.pi * frequency * tt)
        n_grid = self.size
        steering_flat = steering.reshape(n_stations, n_grid)

        power = np.real(
            np.einsum(
                "si,st,ti->i", steering_flat.conj(), covariance, steering_flat
            )
        )
        self[...] = power.reshape(self.shape) / n_stations**2

    def compute_mvdr(
        self,
        covariance: CovarianceMatrix,
        frequency: float,
        regularization: float = 1e-6,
    ) -> None:
        r"""Compute the MVDR (Capon) MFP power in-place.

        Evaluates

        .. math::

            P_{\mathrm{MVDR}}(\mathbf{r}) =
            \frac{1}{\mathbf{w}^\dagger(\mathbf{r}, f)\,
                     \mathbf{C}^{-1}\,
                     \mathbf{w}(\mathbf{r}, f)}

        where :math:`\mathbf{C}^{-1}` is the regularised inverse of the
        covariance matrix.  The regularisation prevents numerical instability
        when the matrix is nearly singular:

        .. math::

            \tilde{\mathbf{C}} = \mathbf{C} + \delta \operatorname{tr}(
            \mathbf{C}) \mathbf{I}

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            Spectral covariance matrix at one time–frequency bin, of shape
            ``(n_stations, n_stations)``.
        frequency : float
            Frequency in Hz at which to evaluate the steering vectors.
        regularization : float, optional
            Diagonal loading factor as a fraction of the trace.  Default
            is ``1e-6``.

        Raises
        ------
        ValueError
            If ``covariance`` does not have exactly 2 dimensions.
        """
        if covariance.ndim != 2:
            raise ValueError(
                f"covariance must be 2-D, got shape {covariance.shape}."
            )
        keys = list(self.travel_times)
        n_stations = len(keys)
        tt = np.stack([self.travel_times[k].__array__() for k in keys], axis=0)

        cov_reg = np.array(covariance, dtype=complex)
        cov_reg += regularization * np.trace(cov_reg) * np.eye(n_stations)
        cov_inv = np.linalg.inv(cov_reg)

        steering = np.exp(-2j * np.pi * frequency * tt)
        n_grid = self.size
        steering_flat = steering.reshape(n_stations, n_grid)

        denominator = np.real(
            np.einsum(
                "si,st,ti->i", steering_flat.conj(), cov_inv, steering_flat
            )
        )
        self[...] = (1.0 / np.maximum(denominator, 1e-30)).reshape(self.shape)

    def maximum_coordinates(self) -> tuple[float, float, float]:
        r"""Return the coordinates of the grid maximum.

        Returns
        -------
        lon, lat, depth : float
            Longitude (degrees), latitude (degrees) and depth (km) of the
            grid point with maximum power.
        """
        i, j, k = np.unravel_index(np.nanargmax(self), self.shape)
        return float(self.lon[i]), float(self.lat[j]), float(self.depth[k])
