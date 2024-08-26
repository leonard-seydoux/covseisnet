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
        half_size = cross_correlation.shape[-1] // 2
        for i_pair, pair in enumerate(self.pairs):
            moveouts = self.moveouts[pair]
            for i in range(self.size):
                moveout = moveouts.flat[i]
                idx = int(cross_correlation.sampling_rate * moveout)
                try:
                    self.flat[i] += cross_correlation[i_pair, half_size + idx]
                except IndexError:
                    continue

        # Normalize the likelihood
        self /= self.sum()

        # Renormalize the likelihood if necessary
        if normalize:
            self /= self.max()
