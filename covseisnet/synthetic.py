"""Synthetic covariance matrices for seismic arrays.

This module provides functions to generate theoretical and estimated synthetic
covariance matrices from analytical models. These are useful for validating
array processing methods, exploring the relationship between wavefield physics
and covariance matrix structure, and for producing reference covariances to
compare against observed data.

The station geometry is described by a list of
:class:`~obspy.core.trace.Stats` objects with assigned geographical
coordinates. All spatial calculations delegate to
:func:`~covseisnet.spatial.station_local_coordinates` and
:func:`~covseisnet.spatial.pairwise_distance_matrix_from_stats` from the
:mod:`covseisnet.spatial` module.

Functions
---------
- :func:`~covseisnet.synthetic.plane_wave_field` — monochromatic plane-wave
  field at each station.
- :func:`~covseisnet.synthetic.plane_wave_covariance` — rank-1 covariance
  matrix of a monochromatic plane wave.
- :func:`~covseisnet.synthetic.surface_noise_covariance` — theoretical
  isotropic 2D (surface-wave) noise covariance via the Bessel function
  :math:`J_0`.
- :func:`~covseisnet.synthetic.volume_noise_covariance` — theoretical
  isotropic 3D (body-wave) noise covariance via the sinc function.
- :func:`~covseisnet.synthetic.spherical_wave_field` — monochromatic
  spherical wave field from a point source.
- :func:`~covseisnet.synthetic.spherical_wave_covariance` — rank-1 covariance
  matrix of a monochromatic spherical wave.
- :func:`~covseisnet.synthetic.random_noise_covariance` — covariance matrix
  estimated from independent Gaussian noise snapshots.
"""

import numpy as np
from scipy.special import j0
from obspy.core.trace import Stats

from .covariance import CovarianceMatrix
from .spatial import (
    straight_ray_distance,
    station_local_coordinates,
    pairwise_distance_matrix_from_stats,
)


def plane_wave_field(
    stats: list[Stats],
    frequency: float,
    slowness: float,
    azimuth: float,
) -> np.ndarray:
    r"""Monochromatic plane-wave field at each station.

    Computes the complex wavefield :math:`\psi_i` of a monochromatic plane
    wave propagating in a homogeneous medium with apparent slowness
    :math:`s_0` and frequency :math:`f`, recorded at station :math:`i`
    located at local east-north coordinates :math:`(x_i, y_i)`:

    .. math::

        \psi_i = \exp\!\left(-\imath\, k \bigl(\sin\theta\, x_i
                 + \cos\theta\, y_i\bigr)\right)

    where :math:`k = 2\pi f s_0` is the apparent wavenumber (rad/km), and
    :math:`\theta` is the propagation azimuth measured clockwise from north.
    The local coordinates are computed with a flat-Earth projection (see
    :func:`~covseisnet.spatial.station_local_coordinates`).

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.  Each must have a
        ``coordinates`` attribute with ``longitude`` and ``latitude`` keys.
    frequency : float
        Wave frequency in hertz.
    slowness : float
        Apparent slowness in seconds per kilometre (s/km).
    azimuth : float
        Propagation direction measured clockwise from north in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Complex array of shape ``(n_stations,)`` with the wavefield
        amplitude and phase at each station.

    See also
    --------
    :func:`~covseisnet.synthetic.plane_wave_covariance`

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt

        import covseisnet as csn
        from covseisnet.synthetic import plane_wave_field

        # Build a circular array of 12 stations
        n = 12
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        lons = 0.5 * np.cos(angles) / 111.195 + 2.0
        lats = 0.5 * np.sin(angles) / 111.195 + 48.0
        stats = [
            csn.NetworkStream.read()[0].stats.__class__()
            for _ in range(n)
        ]
        for i, s in enumerate(stats):
            s.coordinates = {"longitude": lons[i], "latitude": lats[i],
                             "elevation": 0.0}

        wavefield = plane_wave_field(stats, frequency=1.0, slowness=0.3,
                                     azimuth=45.0)
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        sc = ax.scatter(angles, np.abs(wavefield), c=np.angle(wavefield),
                        cmap="hsv", vmin=-np.pi, vmax=np.pi, s=100)
        plt.colorbar(sc, label="Phase (rad)")
        ax.set_title("Plane-wave phase across the array")

    """
    x, y = station_local_coordinates(stats)
    wavenumber = 2 * np.pi * frequency * slowness
    azimuth_rad = np.radians(azimuth)
    phase = wavenumber * (np.sin(azimuth_rad) * x + np.cos(azimuth_rad) * y)
    return np.exp(-1j * phase)


def plane_wave_covariance(
    stats: list[Stats],
    frequency: float,
    slowness: float,
    azimuth: float,
) -> CovarianceMatrix:
    r"""Rank-1 covariance matrix of a monochromatic plane wave.

    Computes the outer product of the plane-wave field
    :math:`\boldsymbol{\psi}` (see
    :func:`~covseisnet.synthetic.plane_wave_field`):

    .. math::

        C_{ij} = \psi_i\, \psi_j^*

    The resulting matrix is Hermitian, positive semi-definite, and has rank 1,
    meaning its eigenvalue spectrum consists of a single non-zero eigenvalue
    equal to :math:`N` (the number of stations) and :math:`N-1` zero
    eigenvalues.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.
    frequency : float
        Wave frequency in hertz.
    slowness : float
        Apparent slowness in seconds per kilometre (s/km).
    azimuth : float
        Propagation direction measured clockwise from north in degrees.

    Returns
    -------
    :class:`~covseisnet.covariance.CovarianceMatrix`
        Hermitian covariance matrix of shape ``(n_stations, n_stations)``.

    See also
    --------
    :func:`~covseisnet.synthetic.plane_wave_field`
    :func:`~covseisnet.synthetic.spherical_wave_covariance`
    """
    wavefield = plane_wave_field(stats, frequency, slowness, azimuth)
    covariance = wavefield[:, None] * wavefield[None, :].conj()
    return CovarianceMatrix(covariance)


def surface_noise_covariance(
    stats: list[Stats],
    frequency: float,
    slowness: float,
) -> CovarianceMatrix:
    r"""Theoretical covariance matrix for isotropic 2D (surface-wave) noise.

    Under the assumption of isotropic surface-wave noise distributed uniformly
    around the array, the cross-spectral density between stations :math:`i`
    and :math:`j` separated by a distance :math:`d_{ij}` is given by the
    zero-order Bessel function of the first kind :cite:p:`lobkis_on_2001`:

    .. math::

        C_{ij}(f) = J_0\!\left(k\, d_{ij}\right)

    where :math:`k = 2\pi f s_0` is the wavenumber. The diagonal entries
    equal 1 (since :math:`J_0(0) = 1`). The matrix is real, symmetric, and
    Toeplitz (for a linear array).  This is the theoretical limit reached by
    averaging sufficiently many independent plane-wave snapshots distributed
    uniformly in azimuth.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.  Each must have a
        ``coordinates`` attribute with ``longitude``, ``latitude``, and
        ``elevation`` keys.
    frequency : float
        Wave frequency in hertz.
    slowness : float
        Apparent slowness in seconds per kilometre (s/km).

    Returns
    -------
    :class:`~covseisnet.covariance.CovarianceMatrix`
        Real-valued (promoted to complex) Hermitian covariance matrix of
        shape ``(n_stations, n_stations)``.

    See also
    --------
    :func:`~covseisnet.synthetic.volume_noise_covariance`
    """
    distances = pairwise_distance_matrix_from_stats(stats)
    wavenumber = 2 * np.pi * frequency * slowness
    covariance = j0(wavenumber * distances).astype(complex)
    return CovarianceMatrix(covariance)


def volume_noise_covariance(
    stats: list[Stats],
    frequency: float,
    slowness: float,
) -> CovarianceMatrix:
    r"""Theoretical covariance matrix for isotropic 3D (body-wave) noise.

    Under the assumption of uniformly distributed isotropic noise sources
    in three dimensions (body-wave diffuse field), the cross-spectral density
    between two receivers separated by distance :math:`d_{ij}` is
    :cite:p:`weaver_diffuse_2001`:

    .. math::

        C_{ij}(f) = \frac{\sin(k\, d_{ij})}{k\, d_{ij}}

    where :math:`k = 2\pi f s_0` is the wavenumber.  This is the unnormalised
    sinc function.  The diagonal entries equal 1 in the limit
    :math:`d_{ij} \to 0`.

    Note
    ----
    NumPy's :func:`numpy.sinc` uses the *normalised* convention
    :math:`\operatorname{sinc}(x) = \sin(\pi x)/(\pi x)`.  The physical sinc
    :math:`\sin(kd)/(kd)` is therefore evaluated as
    ``numpy.sinc(k * d / numpy.pi)``.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.  Each must have a
        ``coordinates`` attribute with ``longitude``, ``latitude``, and
        ``elevation`` keys.
    frequency : float
        Wave frequency in hertz.
    slowness : float
        Apparent slowness in seconds per kilometre (s/km).

    Returns
    -------
    :class:`~covseisnet.covariance.CovarianceMatrix`
        Real-valued (promoted to complex) Hermitian covariance matrix of
        shape ``(n_stations, n_stations)``.

    See also
    --------
    :func:`~covseisnet.synthetic.surface_noise_covariance`
    """
    distances = pairwise_distance_matrix_from_stats(stats)
    wavenumber = 2 * np.pi * frequency * slowness
    covariance = np.sinc(wavenumber * distances / np.pi).astype(complex)
    return CovarianceMatrix(covariance)


def spherical_wave_field(
    stats: list[Stats],
    frequency: float,
    slowness: float,
    source: tuple[float, float, float],
) -> np.ndarray:
    r"""Monochromatic spherical wave field from a point source.

    Computes the wavefield :math:`\psi_i` of an outgoing monochromatic
    spherical wave emitted by a point source at ``source`` and recorded at
    each station :math:`i`:

    .. math::

        \psi_i = \frac{1}{r_i + \varepsilon}
                 \exp\!\left(-\imath\, k\, r_i\right)

    where :math:`r_i` is the straight-ray distance (km) between the source
    and station :math:`i`, :math:`k = 2\pi f s_0` the wavenumber, and
    :math:`\varepsilon = 10^{-6}` km a regularisation constant that avoids
    division by zero for a station co-located with the source.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.  Each must have a
        ``coordinates`` attribute with ``longitude``, ``latitude``, and
        ``elevation`` keys.
    frequency : float
        Wave frequency in hertz.
    slowness : float
        Apparent slowness in seconds per kilometre (s/km).
    source : tuple of float
        Source position as ``(longitude, latitude, depth)`` where longitude
        and latitude are in decimal degrees and depth is in kilometres
        (positive downward below sea level).

    Returns
    -------
    :class:`numpy.ndarray`
        Complex array of shape ``(n_stations,)`` with the wavefield
        amplitude and phase at each station.

    See also
    --------
    :func:`~covseisnet.synthetic.spherical_wave_covariance`
    :func:`~covseisnet.synthetic.plane_wave_field`
    """
    wavenumber = 2 * np.pi * frequency * slowness
    src_lon, src_lat, src_depth = source
    radii = np.array(
        [
            straight_ray_distance(
                src_lon,
                src_lat,
                src_depth,
                s.coordinates["longitude"],
                s.coordinates["latitude"],
                -1e-3 * s.coordinates.get("elevation", 0.0),
            )
            for s in stats
        ]
    )
    return np.exp(-1j * wavenumber * radii) / (radii + 1e-6)


def spherical_wave_covariance(
    stats: list[Stats],
    frequency: float,
    slowness: float,
    source: tuple[float, float, float],
) -> CovarianceMatrix:
    r"""Rank-1 covariance matrix of a monochromatic spherical wave.

    Computes the outer product of the spherical-wave field
    :math:`\boldsymbol{\psi}` (see
    :func:`~covseisnet.synthetic.spherical_wave_field`):

    .. math::

        C_{ij} = \psi_i\, \psi_j^*
               = \frac{\exp\!\bigl(-\imath\, k\,(r_i - r_j)\bigr)}
                      {(r_i + \varepsilon)(r_j + \varepsilon)}

    Like the plane-wave covariance, this matrix is Hermitian, positive
    semi-definite, and has rank 1.  Unlike the plane-wave case, the amplitude
    of each off-diagonal entry decreases with the distance from the source,
    reflecting geometrical spreading.  This model is useful for testing
    source-localisation algorithms.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.
    frequency : float
        Wave frequency in hertz.
    slowness : float
        Apparent slowness in seconds per kilometre (s/km).
    source : tuple of float
        Source position as ``(longitude, latitude, depth)`` where longitude
        and latitude are in decimal degrees and depth is in kilometres
        (positive downward below sea level).

    Returns
    -------
    :class:`~covseisnet.covariance.CovarianceMatrix`
        Hermitian covariance matrix of shape ``(n_stations, n_stations)``.

    See also
    --------
    :func:`~covseisnet.synthetic.spherical_wave_field`
    :func:`~covseisnet.synthetic.plane_wave_covariance`
    """
    wavefield = spherical_wave_field(stats, frequency, slowness, source)
    covariance = wavefield[:, None] * wavefield[None, :].conj()
    return CovarianceMatrix(covariance)


def random_noise_covariance(
    stats: list[Stats],
    n_snapshots: int = 100,
    seed: int | None = None,
) -> CovarianceMatrix:
    r"""Covariance matrix of spatially uncorrelated Gaussian noise.

    Estimates the covariance matrix of white (spatially uncorrelated) noise
    by averaging :math:`M` outer products of independent random snapshots:

    .. math::

        \hat{\mathbf{C}} = \frac{1}{M}\sum_{m=1}^{M}
        \boldsymbol{n}_m\, \boldsymbol{n}_m^{\dagger}

    where each snapshot :math:`\boldsymbol{n}_m \in \mathbb{C}^N` is drawn
    from a standard complex normal distribution (independent standard normals
    for real and imaginary parts).  In expectation,
    :math:`\mathbb{E}[\hat{\mathbf{C}}] = \mathbf{I}_N`, and the result
    converges to the identity matrix as :math:`M \to \infty`.  This model
    corresponds to a fully incoherent field with maximum coherence spectral
    width :math:`\sigma = 1`.

    Arguments
    ---------
    stats : list of :class:`~obspy.core.trace.Stats`
        Stats objects of the network stations.  Only the number of stations
        :math:`N = \texttt{len(stats)}` is used.
    n_snapshots : int, optional
        Number of independent snapshots :math:`M` to average.  Larger values
        yield a result closer to :math:`\mathbf{I}_N`.  Defaults to 100.
    seed : int or None, optional
        Seed for the random number generator.  Pass an integer for
        reproducible results.  Defaults to ``None`` (non-deterministic).

    Returns
    -------
    :class:`~covseisnet.covariance.CovarianceMatrix`
        Complex Hermitian covariance matrix of shape
        ``(n_stations, n_stations)``.
    """
    rng = np.random.default_rng(seed)
    n_stations = len(stats)
    covariance = np.zeros((n_stations, n_stations), dtype=complex)
    for _ in range(n_snapshots):
        snapshot = rng.standard_normal(n_stations) + 1j * rng.standard_normal(
            n_stations
        )
        covariance += snapshot[:, None] * snapshot[None, :].conj()
    covariance /= n_snapshots
    return CovarianceMatrix(covariance)
