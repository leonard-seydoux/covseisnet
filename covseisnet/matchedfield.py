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
from .spatial import GeographicalGrid, straight_ray_distance
from .travel_times import TravelTimes
from .velocity import VelocityModel


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

    def compute_bartlett_delay_taper(
        self,
        covariance: CovarianceMatrix,
        frequency: float,
        sigma_tau: float = 1.0,
        taper: str = "gaussian",
    ) -> None:
        r"""Bartlett MFP with a delay-domain taper on the covariance.

        Multiplies each cross-spectral entry by a smooth window evaluated at
        the differential travel time between the corresponding station pair:

        .. math::

            \tilde{C}_{jk} = C_{jk} \cdot
            w\!\left(\frac{t_j(\mathbf{r}_0) - t_k(\mathbf{r}_0)}{\sigma_\tau}
            \right)

        where :math:`t_j` are the travel-time grids averaged over all grid
        points (a proxy for the typical inter-station delay at the centre of
        the search volume) and :math:`w` is either a Gaussian or a raised-
        cosine (Hann) taper.

        The taper is the frequency-domain dual of time-domain smoothing of
        the cross-correlation: it rolls off contributions from station pairs
        whose differential travel time exceeds :math:`\sigma_\tau`, which
        are precisely the pairs that generate grating lobes when the
        wavelength is shorter than the grid spacing.  The result is
        **deterministic** — no random sampling, no frequency loop.

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            Spectral covariance matrix, shape ``(n_stations, n_stations)``.
        frequency : float
            Frequency in Hz.
        sigma_tau : float, optional
            Taper width in seconds.  A good starting value is the one-way
            travel-time across one grid cell: ``grid_spacing_km / velocity``.
            Larger values preserve more long-delay coherence (less
            smoothing); smaller values suppress lobes more aggressively at
            the cost of spatial resolution.  Default 1.0 s.
        taper : {'gaussian', 'hann'}, optional
            Window function.  ``'gaussian'`` applies
            :math:`\exp(-\tau_{jk}^2 / 2\sigma_\tau^2)`;
            ``'hann'`` applies a raised cosine that reaches zero at
            :math:`|\tau_{jk}| = \sigma_\tau`.  Default ``'gaussian'``.

        Raises
        ------
        ValueError
            If ``covariance`` does not have exactly 2 dimensions, or if
            ``taper`` is not recognised.
        """
        if covariance.ndim != 2:
            raise ValueError(
                f"covariance must be 2-D, got shape {covariance.shape}."
            )
        if taper not in ("gaussian", "hann"):
            raise ValueError(
                f"taper must be 'gaussian' or 'hann', got {taper!r}."
            )

        keys = list(self.travel_times)
        n_stations = len(keys)
        tt = np.stack([self.travel_times[k].__array__() for k in keys], axis=0)

        # Representative inter-station delays: mean travel time per station
        # (averaged over grid), shape (n_stations,)
        tt_mean = tt.reshape(n_stations, -1).mean(axis=1)
        # Differential travel-time matrix between all station pairs, (n, n)
        dtau = tt_mean[:, None] - tt_mean[None, :]

        if taper == "gaussian":
            w = np.exp(-(dtau**2) / (2.0 * sigma_tau**2))
        else:  # hann
            x = np.clip(np.abs(dtau) / sigma_tau, 0.0, 1.0)
            w = 0.5 * (1.0 + np.cos(np.pi * x))

        cov_tapered = np.array(covariance, dtype=complex) * w

        steering = np.exp(-2j * np.pi * frequency * tt)
        steering_flat = steering.reshape(n_stations, self.size)

        power = np.real(
            np.einsum(
                "si,st,ti->i", steering_flat.conj(), cov_tapered, steering_flat
            )
        )
        self[...] = power.reshape(self.shape) / n_stations**2

    def compute_bartlett_phase_stacking(
        self,
        covariance: CovarianceMatrix,
        frequency: float,
        n_perturbations: int = 200,
        dt_max: float = 2.0,
        seed: "int | None" = None,
    ) -> None:
        r"""Max-pooled Bartlett MFP with random per-station phase perturbations.

        Overcomes the grid-spacing resolution limit by dithering the
        covariance phase instead of refining the grid.  For each of the
        ``n_perturbations`` trials an independent random delay
        :math:`\delta t_j \sim \mathcal{U}(0, dt_{max})` is drawn for every
        station :math:`j`.  The delay is absorbed as a phase rotation on the
        covariance:

        .. math::

            \tilde{C}_{jk} = C_{jk}\,
            e^{\,2\pi i f (\delta t_j - \delta t_k)}

        which is algebraically identical to evaluating the Bartlett map with
        travel times :math:`t_j(\mathbf{r}) + \delta t_j`.  The element-wise
        maximum over all trials is stored in-place:

        .. math::

            P_{\text{stack}}(\mathbf{r}) =
            \max_{n}\, P^{(n)}_{\text{Bartlett}}(\mathbf{r})

        Because each random shift moves the effective search point by a
        sub-cell offset in delay space, the pooled map recovers a
        near-continuous scan without recomputing travel-time grids.

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            Spectral covariance matrix, shape ``(n_stations, n_stations)``.
        frequency : float
            Frequency in Hz.
        n_perturbations : int, optional
            Number of random phase-shift trials.  Default 200.
        dt_max : float, optional
            Upper bound of the uniform delay distribution in seconds.
            Should be at least as large as the travel-time difference across
            one grid cell; defaults to 2.0 s.
        seed : int, optional
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If ``covariance`` does not have exactly 2 dimensions.
        """
        if covariance.ndim != 2:
            raise ValueError(
                f"covariance must be 2-D, got shape {covariance.shape}."
            )
        rng = np.random.default_rng(seed)
        keys = list(self.travel_times)
        n_stations = len(keys)
        tt = np.stack([self.travel_times[k].__array__() for k in keys], axis=0)

        steering = np.exp(-2j * np.pi * frequency * tt)
        n_grid = self.size
        steering_flat = steering.reshape(n_stations, n_grid)

        cov = np.array(covariance, dtype=complex)
        pooled = np.full(n_grid, -np.inf)

        for _ in range(n_perturbations):
            # Independent per-station delays; absorbed into the covariance phase.
            dt = rng.uniform(0.0, dt_max, size=n_stations)
            phi = np.exp(2j * np.pi * frequency * dt)  # (n_stations,)
            cov_pert = cov * np.outer(phi, phi.conj())
            power = np.real(
                np.einsum(
                    "si,st,ti->i",
                    steering_flat.conj(),
                    cov_pert,
                    steering_flat,
                )
            )
            np.maximum(pooled, power, out=pooled)

        self[...] = pooled.reshape(self.shape) / n_stations**2

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

    def locate_refine(
        self,
        covariance: CovarianceMatrix,
        frequency: float,
        center: "tuple[float, float, float] | None" = None,
        n_iter: int = 4,
        zoom_factor: float = 5.0,
        shape: tuple = (20, 20, 10),
    ) -> "MatchedFieldProcessing":
        r"""Iterative grid refinement for sub-wavelength localization.

        At each iteration a new grid is built centred on the current best
        estimate, with a spacing ``zoom_factor`` times finer than the
        previous level.  Starting from the coarse Bartlett maximum the
        search region shrinks geometrically, converging to sub-wavelength
        precision after a handful of steps.

        This is the cleanest deterministic solution to the grid-aliasing
        problem: rather than fighting grating lobes on a fixed grid, the
        grid itself adapts so that the main lobe always spans multiple
        nodes.

        Only supported for constant-velocity models.

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            Spectral covariance matrix, shape ``(n_stations, n_stations)``.
        frequency : float
            Frequency in Hz.
        center : tuple of three floats, optional
            ``(lon, lat, depth)`` starting point.  Defaults to the maximum
            of the current (coarse) grid.
        n_iter : int, optional
            Number of zoom levels.  Default 4 (zoom-factor 5 → 5^4 = 625×
            finer than the initial spacing).
        zoom_factor : float, optional
            Linear refinement factor per iteration.  Default 5.
        shape : tuple of three ints, optional
            Grid shape ``(n_lon, n_lat, n_depth)`` at every zoom level.
            Default ``(20, 20, 10)``.

        Returns
        -------
        mfp_fine : :class:`MatchedFieldProcessing`
            The finest-level MFP map (can be plotted with
            :func:`~covseisnet.plot.grid3d`).
        """
        ref_tt = next(iter(self.travel_times.values()))
        if not ref_tt.velocity_model.is_constant():
            raise ValueError(
                "locate_refine requires a constant-velocity model."
            )
        velocity = float(ref_tt.velocity_model.constant_velocity)

        # Initial half-widths: one full coarse grid cell in each dimension
        half_lon = abs(float(self.lon[1] - self.lon[0]))
        half_lat = abs(float(self.lat[1] - self.lat[0]))
        half_dep = (
            abs(float(self.depth[1] - self.depth[0]))
            if len(self.depth) > 1
            else 1.0
        )

        current = list(
            center if center is not None else self.maximum_coordinates()
        )
        mfp_fine = self

        for _ in range(n_iter):
            half_lon /= zoom_factor
            half_lat /= zoom_factor
            half_dep /= zoom_factor
            lon0, lat0, dep0 = current
            extent_fine = (
                lon0 - half_lon,
                lon0 + half_lon,
                lat0 - half_lat,
                lat0 + half_lat,
                max(0.0, dep0 - half_dep),
                dep0 + half_dep,
            )
            vm_fine = VelocityModel(
                extent=extent_fine, shape=shape, velocity=velocity
            )
            tt_fine = {
                key: TravelTimes(
                    vm_fine,
                    receiver_coordinates=tt.receiver_coordinates,
                )
                for key, tt in self.travel_times.items()
            }
            mfp_fine = MatchedFieldProcessing(tt_fine)
            mfp_fine.compute_bartlett(covariance, frequency)
            current = list(mfp_fine.maximum_coordinates())

        return mfp_fine

    # ------------------------------------------------------------------
    # Gridless localization helpers
    # ------------------------------------------------------------------

    def _bartlett_power_at_point(
        self,
        lon: float,
        lat: float,
        depth: float,
        covariance: np.ndarray,
        frequency: float,
    ) -> float:
        r"""Bartlett power at an arbitrary geographic point.

        Computes straight-ray travel times from ``(lon, lat, depth)`` to
        every receiver, forms the replica vector, and returns the Bartlett
        power.  Only supported for constant-velocity travel-time models.

        Arguments
        ---------
        lon, lat, depth : float
            Source longitude (degrees), latitude (degrees), depth (km).
        covariance : array-like, shape (n_stations, n_stations)
            Spectral covariance matrix.
        frequency : float
            Frequency in Hz.

        Returns
        -------
        power : float
            Bartlett MFP power normalised by :math:`N^2`.
        """
        keys = list(self.travel_times)
        n = len(keys)
        times = np.empty(n)
        for i, key in enumerate(keys):
            tt = self.travel_times[key]
            vm = tt.velocity_model
            if not vm.is_constant():
                raise ValueError(
                    "Gridless localization requires a constant-velocity model."
                )
            dist = straight_ray_distance(
                lon, lat, depth, *tt.receiver_coordinates
            )
            times[i] = dist / float(vm.constant_velocity)
        w = np.exp(-2j * np.pi * frequency * times)
        return float(np.real(w.conj() @ covariance @ w)) / n**2

    def locate_optimize(
        self,
        covariance: "CovarianceMatrix",
        frequency: float,
        bounds: "tuple | None" = None,
        seed: "int | None" = None,
    ) -> tuple[float, float, float]:
        r"""Gridless localization via differential evolution.

        Maximizes the Bartlett MFP power

        .. math::

            P_{\mathrm{Bartlett}}(\mathbf{r}) =
            \mathbf{w}^\dagger \mathbf{C} \mathbf{w} \;/\; N^2

        over a continuous search domain using
        :func:`scipy.optimize.differential_evolution`.  This avoids the
        grid-spacing limitation of :meth:`compute_bartlett` and is useful
        when the wavelength is shorter than the grid cell size.

        Only supported for constant-velocity travel-time models.

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            Spectral covariance matrix at one time–frequency bin, shape
            ``(n_stations, n_stations)``.
        frequency : float
            Frequency in Hz.
        bounds : list of three (min, max) pairs, optional
            Search bounds ``[(lon_min, lon_max), (lat_min, lat_max),
            (depth_min, depth_max)]``.  Defaults to the extent of the grid.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        lon, lat, depth : float
            Coordinates of the maximum-power location.
        """
        from scipy.optimize import differential_evolution

        if bounds is None:
            bounds = [
                (float(self.lon.min()), float(self.lon.max())),
                (float(self.lat.min()), float(self.lat.max())),
                (float(self.depth.min()), float(self.depth.max())),
            ]

        def neg_power(x):
            return -self._bartlett_power_at_point(
                x[0], x[1], x[2], covariance, frequency
            )

        result = differential_evolution(
            neg_power, bounds, seed=seed, tol=1e-9, maxiter=1000
        )
        return float(result.x[0]), float(result.x[1]), float(result.x[2])

    def locate_mcmc(
        self,
        covariance: "CovarianceMatrix",
        frequency: float,
        bounds: "tuple | None" = None,
        n_samples: int = 5000,
        n_burn: int = 1000,
        step_size: "tuple | None" = None,
        beta: float = 1.0,
        seed: "int | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Gridless localization via Metropolis-Hastings MCMC.

        Samples the posterior

        .. math::

            p(\mathbf{r}) \propto
            P_{\mathrm{Bartlett}}(\mathbf{r})^{\,\beta}

        using a Gaussian random-walk Metropolis-Hastings algorithm.  Because
        the sampler operates in continuous space, it can resolve source
        locations at sub-grid-cell precision regardless of the dominant
        wavelength.

        The Bartlett power serves as an unnormalised likelihood: when
        :math:`\beta = 1` the posterior is proportional to the power;
        larger :math:`\beta` concentrates samples closer to the maximum.

        Only supported for constant-velocity travel-time models.

        Arguments
        ---------
        covariance : :class:`~covseisnet.covariance.CovarianceMatrix`
            Spectral covariance matrix at one time–frequency bin, shape
            ``(n_stations, n_stations)``.
        frequency : float
            Frequency in Hz.
        bounds : list of three (min, max) pairs, optional
            Hard bounds ``[(lon_min, lon_max), (lat_min, lat_max),
            (depth_min, depth_max)]``.  Defaults to the extent of the grid.
        n_samples : int, optional
            Number of posterior samples to return (after burn-in).
            Default 5000.
        n_burn : int, optional
            Number of burn-in steps to discard.  Default 1000.
        step_size : tuple of three floats, optional
            Standard deviations of the isotropic Gaussian proposal in
            ``(lon, lat, depth)``.  Defaults to one tenth of each bound
            range.
        beta : float, optional
            Inverse temperature.  Default 1.0.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, 3)
            Posterior samples as ``[lon, lat, depth]`` rows.
        log_prob : np.ndarray, shape (n_samples,)
            Log-probability (up to a constant) at each sample.
        """
        rng = np.random.default_rng(seed)

        if bounds is None:
            bounds = [
                (float(self.lon.min()), float(self.lon.max())),
                (float(self.lat.min()), float(self.lat.max())),
                (float(self.depth.min()), float(self.depth.max())),
            ]

        (lon_min, lon_max), (lat_min, lat_max), (dep_min, dep_max) = bounds

        if step_size is None:
            step_size = (
                0.1 * (lon_max - lon_min),
                0.1 * (lat_max - lat_min),
                max(0.1 * (dep_max - dep_min), 0.1),
            )
        sig = np.array(step_size, dtype=float)

        def log_prob(lon, lat, depth):
            if not (
                lon_min <= lon <= lon_max
                and lat_min <= lat <= lat_max
                and dep_min <= depth <= dep_max
            ):
                return -np.inf
            p = self._bartlett_power_at_point(
                lon, lat, depth, covariance, frequency
            )
            return beta * np.log(p) if p > 0 else -np.inf

        # Initialise chain at the grid maximum (warm start)
        i, j, k = np.unravel_index(np.nanargmax(self), self.shape)
        current = np.array(
            [float(self.lon[i]), float(self.lat[j]), float(self.depth[k])]
        )
        current_lp = log_prob(*current)

        samples = np.empty((n_samples, 3))
        log_probs = np.empty(n_samples)

        sample_idx = 0
        for step in range(n_burn + n_samples):
            proposal = current + rng.normal(0.0, sig)
            proposal_lp = log_prob(*proposal)
            if np.log(rng.uniform()) < proposal_lp - current_lp:
                current = proposal
                current_lp = proposal_lp
            if step >= n_burn:
                samples[sample_idx] = current
                log_probs[sample_idx] = current_lp
                sample_idx += 1

        return samples, log_probs
