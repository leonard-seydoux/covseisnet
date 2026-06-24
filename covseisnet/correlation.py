"""Cross-correlation matrix in time domain."""

import numpy as np

from obspy.core.trace import Stats
from scipy.signal import windows, correlation_lags
from scipy.ndimage import gaussian_filter1d

from .covariance import CovarianceMatrix
from .signal import bandpass_filter, hilbert_envelope
from .signal import ShortTimeFourierTransform


class CrossCorrelationMatrix(np.ndarray):
    r"""Cross-correlation matrix in the time domain.

    The correlation matrix is stored with pairwise correlations along the
    first axis. Pair metadata are stored explicitly in the ``pairs`` attribute,
    because the station-pair axis is a compressed representation of the upper
    triangular part of the original covariance matrix.
    """

    def __new__(
        cls,
        input_array: np.ndarray,
        stats: list[Stats] | list[dict] | None = None,
        stft: ShortTimeFourierTransform | None = None,
        pairs: list[tuple[str, str]] | list[str] | None = None,
    ) -> "CrossCorrelationMatrix":
        r"""Create a cross-correlation matrix.

        Parameters
        ----------
        input_array: :class:`numpy.ndarray`
            Correlation array. The first axis is expected to represent station
            pairs whenever pair metadata are provided.
        stats: list, optional
            Station metadata inherited from the covariance matrix.
        stft: :class:`~covseisnet.signal.ShortTimeFourierTransform`, optional
            STFT parameters inherited from the covariance matrix.
        pairs: list, optional
            Pair metadata associated with the first axis. Each pair should be a
            tuple ``(station_i, station_j)``. For metadata-free objects, strings
            such as ``"pair 0"`` are also accepted.
        """
        input_array = np.asarray(input_array)

        obj = input_array.view(cls)
        obj.stats = stats
        obj.stft = stft
        obj.pairs = pairs
        obj._validate_pair_axis()
        return obj

    def __array_finalize__(self, obj):
        r"""Finalize the array after NumPy view casting or slicing."""
        if obj is None:
            return
        self.stats = getattr(obj, "stats", None)
        self.stft = getattr(obj, "stft", None)
        self.pairs = getattr(obj, "pairs", None)

    def __reduce__(self):
        r"""Reduce the object for pickling."""
        pickled_state = super().__reduce__()
        return (
            pickled_state[0],
            pickled_state[1],
            (pickled_state[2], self.__dict__),
        )

    def __setstate__(self, state):
        r"""Restore the object after unpickling."""
        ndarray_state, attributes = state
        super().__setstate__(ndarray_state)
        self.__dict__.update(attributes)

    @staticmethod
    def _get_station_name(stats: Stats | dict) -> str:
        r"""Return the station name from an ObsPy Stats object or a dict."""
        if isinstance(stats, dict):
            return stats["station"]
        return stats.station

    def _validate_pair_axis(self):
        r"""Validate that pair metadata match the first correlation axis."""
        if self.pairs is None:
            return

        if self.ndim == 0:
            raise ValueError("Pair metadata cannot be attached to a scalar.")

        n_pairs = self.shape[0]
        if len(self.pairs) != n_pairs:
            raise ValueError(
                f"The correlation matrix has {n_pairs} pairs along its first "
                f"axis, but {len(self.pairs)} pair metadata entries were "
                "provided."
            )

        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError(
                "Pair metadata must be unique to allow pair-based selection, "
                f"got {self.pairs}."
            )

    @property
    def stations(self) -> list[str] | None:
        r"""Return station names when station metadata are available."""
        if self.stats is None:
            return None
        return [self._get_station_name(stats) for stats in self.stats]

    @property
    def pair_indices(self) -> dict[tuple[str, str] | str, int]:
        r"""Dictionary mapping pair metadata to first-axis indices."""
        if self.pairs is None:
            raise ValueError("Pair metadata are not available.")
        return {pair: i for i, pair in enumerate(self.pairs)}

    def _normalize_pair(self, pair: tuple[str, str] | str):
        r"""Return a pair key present in the object, allowing reversed order."""
        indices = self.pair_indices

        if pair in indices:
            return pair

        if isinstance(pair, tuple) and len(pair) == 2:
            reversed_pair = (pair[1], pair[0])
            if reversed_pair in indices:
                return reversed_pair

        raise KeyError(
            f"Unknown pair {pair!r}. Available pairs are: {self.pairs}."
        )

    def pair(self, station_i: str, station_j: str) -> np.ndarray:
        r"""Return one pairwise cross-correlation.

        The returned object is a plain :class:`numpy.ndarray`, because the pair
        axis has been consumed and the result is no longer a
        :class:`CrossCorrelationMatrix`.
        """
        pair = self._normalize_pair((station_i, station_j))
        return np.asarray(
            np.ndarray.__getitem__(self, self.pair_indices[pair])
        )

    def select_pairs(
        self, pairs: list[tuple[str, str] | str]
    ) -> "CrossCorrelationMatrix":
        r"""Return a correlation matrix restricted to selected pairs.

        The pair metadata are updated consistently with the first axis.
        """
        normalized_pairs = [self._normalize_pair(pair) for pair in pairs]
        indices = [self.pair_indices[pair] for pair in normalized_pairs]

        return CrossCorrelationMatrix(
            np.asarray(self)[indices],
            stats=self.stats,
            stft=self.stft,
            pairs=normalized_pairs,
        )

    @property
    def sampling_rate(self) -> float:
        r"""Return the sampling rate in Hz."""
        if self.stats is None:
            raise ValueError("Stats are needed to get the sampling rate.")
        return self.stats[0].sampling_rate

    def envelope(self, **kwargs) -> "CrossCorrelationMatrix":
        r"""Hilbert envelope of the correlation matrix."""
        kwargs.setdefault("axis", -1)
        return CrossCorrelationMatrix(
            hilbert_envelope(self, **kwargs),
            stats=self.stats,
            stft=self.stft,
            pairs=self.pairs,
        )

    def taper(self, max_percentage: float = 0.1) -> "CrossCorrelationMatrix":
        r"""Taper the correlation matrix along the lag axis."""
        return CrossCorrelationMatrix(
            self * windows.tukey(self.shape[-1], max_percentage),
            stats=self.stats,
            stft=self.stft,
            pairs=self.pairs,
        )

    def smooth(self, sigma: float = 1, **kwargs) -> "CrossCorrelationMatrix":
        r"""Use a Gaussian kernel to smooth the correlation matrix."""
        return CrossCorrelationMatrix(
            gaussian_filter1d(self, sigma=sigma, **kwargs),
            stats=self.stats,
            stft=self.stft,
            pairs=self.pairs,
        )

    def flat(self):
        r"""Flatten all dimensions except the lag axis."""
        return self.reshape(-1, self.shape[-1])

    def bandpass(
        self, frequency_band: tuple | list, filter_order: int = 4
    ) -> "CrossCorrelationMatrix":
        r"""Bandpass filter the correlation functions."""
        correlation_flat = self.flat()
        correlation_filtered = bandpass_filter(
            correlation_flat,
            self.sampling_rate,
            frequency_band,
            filter_order,
        )
        correlation_filtered = correlation_filtered.reshape(self.shape)

        return CrossCorrelationMatrix(
            correlation_filtered,
            stats=self.stats,
            stft=self.stft,
            pairs=self.pairs,
        )


def calculate_cross_correlation_matrix(
    covariance_matrix: CovarianceMatrix,
    include_autocorrelation: bool = True,
) -> tuple[np.ndarray, list, CrossCorrelationMatrix]:
    r"""Extract correlation in time domain from the given covariance matrix.

    The returned :class:`CrossCorrelationMatrix` stores the pair metadata in its
    ``pairs`` attribute, so pairwise correlations can be retrieved with
    :meth:`CrossCorrelationMatrix.pair` or selected with
    :meth:`CrossCorrelationMatrix.select_pairs`.
    """
    # Two-sided covariance matrix.
    covariance_matrix_twosided = covariance_matrix.twosided()

    # Extract upper triangular station pairs.
    distance_to_diagonal = 0 if include_autocorrelation else 1
    covariance_triu = covariance_matrix_twosided.triu(k=distance_to_diagonal)

    # Extract pair names from the covariance station metadata.
    if covariance_matrix.stats is not None:
        stations = covariance_matrix.stations
        pairs = []
        for i in range(len(stations)):
            for j in range(i + distance_to_diagonal, len(stations)):
                pairs.append((stations[i], stations[j]))
    else:
        pairs = [f"pair {i}" for i in range(covariance_triu.shape[-1])]

    # Change axes position to have the pair axis first.
    covariance_triu = covariance_triu.transpose(2, 0, 1)

    # Get transform parameters.
    if covariance_matrix.stft is None:
        raise ValueError(
            "STFT parameters are needed to calculate correlation."
        )
    n_samples = len(covariance_matrix.stft.win)
    sampling_rate = covariance_matrix.stft.fs

    # Inverse Fourier transform.
    correlation = np.fft.fftshift(np.fft.ifft(covariance_triu), axes=-1).real

    # Calculate lags.
    lags = correlation_lags(n_samples, n_samples, mode="same") / sampling_rate

    # Turn into CrossCorrelationMatrix.
    correlation = CrossCorrelationMatrix(
        correlation,
        stats=covariance_matrix.stats,
        stft=covariance_matrix.stft,
        pairs=pairs,
    )

    return lags, pairs, correlation
