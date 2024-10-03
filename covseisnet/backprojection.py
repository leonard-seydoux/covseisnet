import numpy as np

from .spatial import Regular3DGrid
from .travel_times import DifferentialTravelTimes
from .correlation import CrossCorrelationMatrix


class DifferentialBackProjection(Regular3DGrid):
    r"""Differential travel-times backprojection from cross-correlation
     functions.

    Using differential travel times, this object calculates the likelihood of
    the back-projection for a set of cross-correlation functions.
    """

    moveouts: dict[str, DifferentialTravelTimes]
    pairs: list[str]

    def __new__(
        cls, differential_travel_times: dict[str, DifferentialTravelTimes]
    ):
        r"""
        Arguments
        ---------
        differential_travel_times: dict
            The differential travel times between several pairs of receivers.
            Each key of the dictionary is a pair of receivers, and the value is
            a :class:`~covseisnet.travel_times.DifferentialTravelTimes` object.
        """
        # Create the object
        pairs = list(differential_travel_times.keys())
        obj = differential_travel_times[pairs[0]].copy().view(cls)
        obj[...] = 0
        obj.moveouts = differential_travel_times
        obj.pairs = pairs
        return obj

    def calculate_likelihood(
        self,
        cross_correlation: CrossCorrelationMatrix,
        normalize: bool = True,
    ):
        r"""Calculate the likelihood of the back-projection.

        This method calculates the likelihood of the back-projection for a set
        of cross-correlation functions. The likelihood is calculated by
        summing the cross-correlation functions for various differential
        travel times. The likelihood is then normalized by the sum of the
        likelihood.

        The likelihood :math:`\mathcal{L}(\varphi, \lambda, z)` is calculated
        as:

        .. math::

            \mathcal{L}(\varphi, \lambda, z) = \sum_{i = 1}^N C_i(\tau -
            \delta \tau_{i}(\varphi, \lambda, z))

        where :math:`C_{i}` is the cross-correlation function for the pair of
        receivers :math:`i`, :math:`\tau` is the cross-correlation lag, and
        :math:`\delta \tau_{i}(\varphi, \lambda, z)` is the differential travel
        time for the pair of receivers :math:`i` at the grid point
        :math:`(\varphi, \lambda, z)`. Once calculated, the likelihood is
        normalized by the sum of the likelihood:

        .. math::

            \mathcal{L}(\varphi, \lambda, z) = \frac{\mathcal{L}(\varphi,
            \lambda, z)}{\int \mathcal{L}(\varphi, \lambda, z) d\varphi d\lambda
            dz}

        Arguments
        ----------
        cross_correlation :
        :class:`~covseisnet.correlation.CrossCorrelationMatrix`
            The cross-correlation functions.
        """
        # Calculate the likelihood
        zero_lag = cross_correlation.shape[-1] // 2
        sampling_rate = cross_correlation.sampling_rate
        for i_pair, pair in enumerate(self.pairs):
            lags = np.round(self.moveouts[pair] * sampling_rate).astype(int)
            for i in range(self.size):
                lag = lags.flat[i]
                try:
                    self.flat[i] += cross_correlation[i_pair, zero_lag + lag]
                except IndexError:
                    continue

        # Normalize the likelihood
        self /= self.sum()

        # Renormalize the likelihood if necessary
        if normalize:
            self /= self.max()

    def calculate_likelihood_bp(
        self,
        cross_correlation: CrossCorrelationMatrix,
        normalize: bool = True,
    ):
        r"""Calculate the likelihood of the back-projection.

        This method calculates the likelihood of the back-projection for a set
        of cross-correlation functions. The likelihood is calculated by
        summing the cross-correlation functions for various differential
        travel times. The likelihood is then normalized by the sum of the
        likelihood.

        The likelihood :math:`\mathcal{L}(\varphi, \lambda, z)` is calculated
        as:

        .. math::

            \mathcal{L}(\varphi, \lambda, z) = \sum_{i = 1}^N C_i(\tau -
            \delta \tau_{i}(\varphi, \lambda, z))

        where :math:`C_{i}` is the cross-correlation function for the pair of
        receivers :math:`i`, :math:`\tau` is the cross-correlation lag, and
        :math:`\delta \tau_{i}(\varphi, \lambda, z)` is the differential travel
        time for the pair of receivers :math:`i` at the grid point
        :math:`(\varphi, \lambda, z)`. Once calculated, the likelihood is
        normalized by the sum of the likelihood:

        .. math::

            \mathcal{L}(\varphi, \lambda, z) = \frac{\mathcal{L}(\varphi,
            \lambda, z)}{\int \mathcal{L}(\varphi, \lambda, z) d\varphi d\lambda
            dz}

        Arguments
        ----------
        cross_correlation :
        :class:`~covseisnet.correlation.CrossCorrelationMatrix`
            The cross-correlation functions.
        """
        try:
            import beampower as bp
        except ImportError:
            print("beampower is not installed")
            return
        moveouts_arr_sec = np.array([self.moveouts[p] for p in self.pairs])
        moveouts_arr_samp = np.int32(
            np.round(moveouts_arr_sec * cross_correlation.sampling_rate)
        )
        # print(moveouts_arr_samp.shape)
        # print(moveouts_arr_samp.shape)
        moveouts_arr_samp = moveouts_arr_samp.reshape(
            moveouts_arr_samp.shape[0], -1
        ).T[:, :, np.newaxis]
        print(moveouts_arr_samp.shape)
        weights_phase = np.ones(
            (cross_correlation.shape[0], 1, 1), dtype=np.float32
        )
        # weights_phase[:, 1, 0] = (
        #     0.0  # horizontal component traces do not contribute to P-wave beam
        # )
        # weights_phase[:, 1, 0] = (
        #     1.0  # vertical component traces contribute to P-wave beam
        # )
        # weights_phase[:, :2, 1] = (
        #     1.0  # horizontal component traces contribute to S-wave beam
        # )
        # weights_phase[:, 2, 1] = (
        #     0.0  # vertical component traces do not contribute to S-wave beam
        # )
        weights_sources = np.ones(
            moveouts_arr_samp.shape[:-1], dtype=np.float32
        )
        # cross_correlation = cross_correlation[:, np.newaxis, :]

        # Calculate the likelihood with beampower
        self.flat = bp.beampower.beamform(
            cross_correlation[:, np.newaxis, :],
            moveouts_arr_samp,
            weights_phase,
            weights_sources,
            mode="differential",
            reduce=None,
        ).flat

        # Normalize the likelihood
        self /= self.sum()

        # Renormalize the likelihood if necessary
        if normalize:
            self /= self.max()

    def maximum_coordinates(self):
        r"""Return the coordinates of the maximum likelihood."""
        i, j, k = np.unravel_index(np.argmax(self), self.shape)
        return self.lon[i], self.lat[j], self.depth[k]
