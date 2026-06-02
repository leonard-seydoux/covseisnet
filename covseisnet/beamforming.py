"""Frequency-domain plane-wave beamforming.

This module provides the :class:`PlaneWaveBeamforming` class, which evaluates
the Bartlett (conventional) beamformer power over a
:class:`~covseisnet.slowness.SlownessGrid` using a
:class:`~covseisnet.covariance.CovarianceMatrix` as input.

The Bartlett beamformer represents the classical steered-response-power
estimator.  The minimum-variance distortionless-response (MVDR/Capon)
beamformer and the MUSIC pseudo-spectrum are planned for future releases.

Classes
-------
- :class:`~covseisnet.beamforming.PlaneWaveBeamforming` — Bartlett
  plane-wave beamformer output on a slowness grid.
"""

import numpy as np

from .covariance import CovarianceMatrix
from .slowness import PlaneWaveDelays, SlownessGrid


class PlaneWaveBeamforming(SlownessGrid):
    r"""Bartlett plane-wave beamforming power on a slowness grid.

    Given a :class:`~covseisnet.covariance.CovarianceMatrix`
    :math:`\mathbf{C} \in \mathbb{C}^{N \times N}` and a steering vector

    .. math::

        \mathbf{a}(f, u_x, u_y) = \exp\!\bigl(-2\pi i f \boldsymbol{\tau}(u_x,
        u_y)\bigr)

    where :math:`\boldsymbol{\tau}` is the vector of inter-station delays from
    a :class:`~covseisnet.slowness.PlaneWaveDelays` object, the Bartlett
    beamformed power is

    .. math::

        P_{\mathrm{Bartlett}}(f, u_x, u_y)
        = \mathbf{a}^\dagger \mathbf{C} \mathbf{a}
        \;\bigg/\; N^2

    where :math:`N` is the number of stations and the denominator normalises
    the output to the interval :math:`[0, 1]` for a unit-power source.

    The result is stored as a 2-D array of shape ``(n_slowness, n_azimuth)``
    that inherits from :class:`~covseisnet.slowness.SlownessGrid`.

    .. note::

        **Planned extensions**

        - *Temporal beamforming*: applying the same steering in the time
          domain, which can be efficient for broadband, transient sources.
        - *MVDR / Capon beamformer*: minimises output power subject to a
          distortionless constraint; suppresses correlated noise but requires
          matrix inversion.
        - *MUSIC pseudo-spectrum*: projects steering vectors onto the noise
          subspace of the covariance matrix; provides higher resolution than
          Bartlett but requires knowledge of the signal subspace rank.

    Arguments
    ---------
    delays : :class:`~covseisnet.slowness.PlaneWaveDelays`
        Pre-computed inter-station plane-wave delays.

    Example
    -------

    Compute the beamforming power for a synthetic plane wave:

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt

        import covseisnet as csn
        from covseisnet.slowness import PlaneWaveDelays
        from covseisnet.beamforming import PlaneWaveBeamforming
        from covseisnet.synthetic import plane_wave_covariance

        # Build a small circular array of 8 stations
        n = 8
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        lons = 0.4 * np.cos(angles) / 111.195 + 2.0
        lats = 0.4 * np.sin(angles) / 111.195 + 48.0
        stats = [csn.NetworkStream.read()[0].stats.__class__() for _ in range(n)]
        for i, s in enumerate(stats):
            s.coordinates = {
                "longitude": lons[i], "latitude": lats[i], "elevation": 0.0
            }

        # Synthetic plane-wave covariance
        frequency = 1.0
        slowness  = 0.30
        azimuth   = 45.0
        cov = plane_wave_covariance(stats, frequency, slowness, azimuth)

        # Beamforming
        delays = PlaneWaveDelays(stats, slowness_max=0.6, n_slowness=60, n_azimuth=120)
        bf = PlaneWaveBeamforming(delays)
        bf.compute_bartlett(cov, frequency)

        # Plot
        S, A = delays.mesh
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.pcolormesh(np.radians(A), S, bf, cmap="inferno")
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.set_title("Bartlett beamforming power")
        plt.tight_layout()
    """

    def __new__(
        cls,
        delays: PlaneWaveDelays,
    ) -> "PlaneWaveBeamforming":
        r"""
        Arguments
        ---------
        delays : :class:`~covseisnet.slowness.PlaneWaveDelays`
            Pre-computed inter-station plane-wave delays.
        """
        obj = np.full(delays.shape, np.nan).view(cls)
        obj.slowness = delays.slowness.copy()
        obj.azimuth = delays.azimuth.copy()
        obj._delays = delays
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        self._delays = getattr(obj, "_delays", None)

    def compute_bartlett(
        self,
        covariance: CovarianceMatrix,
        frequency: float,
    ) -> None:
        r"""Compute the Bartlett beamforming power in-place.

        Evaluates

        .. math::

            P(u_x, u_y) = \mathbf{a}^\dagger(f,u_x,u_y)\,
                          \mathbf{C}\,
                          \mathbf{a}(f,u_x,u_y) \;/\; N^2

        for every grid point and stores the result in the array.

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            The spectral covariance matrix at one time–frequency bin, of
            shape ``(n_stations, n_stations)``.
        frequency : float
            The frequency in Hz at which to evaluate the steering vectors.

        Raises
        ------
        ValueError
            If ``covariance`` does not have exactly 2 dimensions.
        """
        if covariance.ndim != 2:
            raise ValueError(
                f"covariance must be 2-D, got shape {covariance.shape}."
            )
        n_stations = covariance.shape[0]
        delays = self._delays.delays

        steering = np.exp(-2j * np.pi * frequency * delays)
        power = np.einsum(
            "sjk,st,tjk->jk", steering.conj(), covariance, steering
        )

        self[...] = np.real(power) / n_stations**2
