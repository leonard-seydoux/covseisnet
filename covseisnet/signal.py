"""
The package provides tools for spectral analysis of seismic data. The main
class is the :class:`~ShortTimeFourierTransform` class, which extends the
:class:`~scipy.signal.ShortTimeFFT` class to provide a more user-friendly
interface for the Short-Time Fourier Transform with streams. 

Other functions are provided to normalize seismic traces in the spectral
domain, such as the :func:`~modulus_division` and :func:`~smooth_modulus_division` functions. Also, we provide the :func:`~smooth_envelope_division` function to normalize seismic traces by a smooth version of its envelope.
"""

from typing import Literal

import matplotlib.dates as mdates
import numpy as np
from obspy.core.stream import Stream
from obspy.core.trace import Trace, Stats
from scipy import signal


class ShortTimeFourierTransform(signal.ShortTimeFFT):

    def __init__(
        self,
        window_duration: float = 2.0,
        window_step: None | float = None,
        window_function: str = "hann",
        sampling_rate: float = 1.0,
        **kwargs,
    ) -> None:
        """Short-Time Fourier Transform instance.

        This class extends the :class:`scipy.signal.ShortTimeFFT` class to
        provide a more user-friendly interface for the Short-Time Fourier
        Transform with ObsPy and Covseisnet Streams.

        Arguments
        ---------
        window_duration: float, optional
            The duration of the window in seconds. Default is 2.0 seconds.
        window_step: float, optional
            The step between windows in seconds. Default is half the window
            duration.
        window_function: str, optional
            The window function to use. Default is the Hann window.
        sampling_rate: float, optional
            The sampling rate of the data. Default is 1.0 Hz. In most methods
            the sampling rate is set automatically from the data.
        **kwargs: dict, optional
            Additional keyword arguments are passed to the
            :meth:`scipy.signal.ShortTimeFFT.stft` method.

        Notes
        -----

        By default, the :class:`scipy.signal.ShortTimeFFT` class pads the data
        with zeros on both sides to avoid edge effects, and enable the
        analysis of edge samples (see the `tutorial on the Short-Time Fourier
        Transform <https://docs.scipy.org/doc/scipy/tutorial/signal.html#tutorial-stft>`_
        in the SciPy documentation). Because the main purpose of this class
        is to analyse the spatial coherence of seismic data, this default
        behaviour is disabled. Indeed, the edges appear wrongly coherent
        because of the zero-padding. This is why the representation of the
        Short-Time Fourier Transform indicates the number of samples and
        windows that are out-of-bounds. These samples are discarded from the
        analysis. Note that this is not a real issue since the covariance
        matrix analysis is performed on sliding windows.

        Examples
        --------
        >>> from covseisnet.signal import ShortTimeFourierTransform
        >>> stft = ShortTimeFourierTransform(
        ...     window_duration=2.0,
        ...     window_step=1.0,
        ...     window_function="hann",
        ...     sampling_rate=100.0,
        ... )
        >>> print(stft)
        Short-Time Fourier Transform instance
            Sampling rate: 100.0 Hz
            Frequency range: 0.50 to 50.00 Hz
            Frequency resolution: 0.5 Hz
            Frequency bins: 101
            Window length: 200 samples
            Window step: 100 samples
            Out-of-bounds: 101 sample(s), 1 window(s)

        See also
        --------
        :class:`scipy.signal.ShortTimeFFT`
        """
        # Define apodization window
        window_size = int(window_duration * sampling_rate)
        window_array = signal.windows.get_window(window_function, window_size)

        # Define step between windows
        window_step = window_step or window_duration / 2
        window_step_size = int(window_step * sampling_rate)

        # Initialize the Short-Time Fourier Transform class
        kwargs.setdefault("fs", sampling_rate)
        super().__init__(window_array, window_step_size, **kwargs)

    def __str__(self):
        k0, p0 = self.lower_border_end
        out = "Short-Time Fourier Transform instance\n"
        out += f"\tSampling rate: {self.fs} Hz\n"
        out += f"\tFrequency range: {self.f[0] :.2f} to {self.f[-1] :.2f} Hz\n"
        out += f"\tFrequency resolution: {self.delta_f} Hz\n"
        out += f"\tFrequency bins: {self.f_pts}\n"
        out += f"\tWindow length: {len(self.win)} samples\n"
        out += f"\tWindow step: {self.hop} samples\n"
        out += f"\tOut-of-bounds: {k0} sample(s), {p0} window(s)"
        return out

    @property
    def sampling_rate(self) -> float:
        """The sampling rate of the Short-Time Fourier Transform.

        This property returns the sampling rate of the Short-Time Fourier
        Transform in Hertz.

        Returns
        -------
        float
            The sampling rate of the Short-Time Fourier Transform in Hertz.
        """
        return self.fs

    @property
    def frequencies(self) -> np.ndarray:
        """The frequencies of the Short-Time Fourier Transform.

        This property returns the frequencies of the Short-Time Fourier
        Transform in Hertz. The frequencies are calculated from the sampling
        rate and the number of frequency bins.

        Returns
        -------
        numpy.ndarray
            The frequencies of the Short-Time Fourier Transform in Hertz.
        """
        return self.f

    def times(self, stats: Stats, **kwargs) -> np.ndarray:
        """The times of the window centers in seconds.

        This method returns the times of the window centers in seconds. The
        times are calculated from the start time of the trace and the window
        step.

        Arguments
        ---------
        stats : :class:`~obspy.core.trace.Stats`
            The stats object of the trace.

        Returns
        -------
        numpy.ndarray
            The times of the window centers in matplotlib datenum format.
        """
        times = self.t(stats.npts, **kwargs).copy()
        return times / 86400 + mdates.date2num(stats.starttime.datetime)

    def valid_window_bounds(self, stats: Stats, **kwargs) -> dict:
        """The bounds of the Short-Time Fourier Transform.

        This method calculate the valid window bounds of the Short-Time
        Fourier Transform in seconds. The bounds are calculated from the
        start time of the trace, the window duration, and the window step.

        Arguments
        ---------
        stats : :class:`~obspy.core.trace.Stats`
            The stats object of the trace.

        Returns
        -------
        tuple
            The lower and upper bounds of the Short-Time Fourier Transform in
            seconds.
        """
        _, p0 = self.lower_border_end
        _, p1 = self.upper_border_begin(stats.npts)
        return {"p0": p0, "p1": p1}

    def transform(
        self,
        trace: Trace,
        detrend: Literal["linear", "constant"] = "linear",
        **kwargs,
    ) -> tuple:
        """Short-time Fourier Transform of a trace.

        This method calculates the Short-Time Fourier Transform of a trace
        using the :meth:`scipy.signal.ShortTimeFFT.stft_detrend` method. The
        method returns the times of the window centers in matplotlib datenum
        format, the frequencies of the spectrogram, and the short-time spectra
        of the trace.

        Prior to the Short-Time Fourier Transform, the method removes the
        linear trend from the trace to avoid edge effects. The method also
        discards the out-of-bounds samples and windows that are padded with
        zeros by the :class:`scipy.signal.ShortTimeFFT`.

        Arguments
        ---------
        trace : :class:`~obspy.core.trace.Trace`
            The trace to transform.
        detrend : str, optional
            The detrending method. Default is "linear".
        **kwargs: dict, optional
            Additional keyword arguments are passed to the
            :meth:`scipy.signal.ShortTimeFFT.stft_detrend` method. The same
            arguments are passed to the :meth:`scipy.signal.ShortTimeFFT.t`
            method to get the times of the window centers.

        Returns
        -------
        times : numpy.ndarray
            The times of the window centers in matplotlib datenum format.
        frequencies : numpy.ndarray
            The frequencies of the spectrogram.
        short_time_spectra : numpy.ndarray
            The short-time spectra of the trace with shape ``(n_frequencies,
            n_times)``.

        Examples
        --------
        >>> import covseisnet as csn
        >>> stft = csn.ShortTimeFourierTransform(
        ...     window_duration=2.0,
        ...     window_step=1.0,
        ...     window_function="hann",
        ...     sampling_rate=100.0,
        ... )
        >>> stream = csn.read()
        >>> trace = stream[0]
        >>> times, frequencies, short_time_spectra = stft.transform(trace)
        >>> print(times.shape, frequencies.shape, short_time_spectra.shape)
        (28,) (101,) (101, 28)
        """
        # Extract the lower and upper border
        kwargs.update(self.valid_window_bounds(trace.stats))

        # Calculate the Short-Time Fourier Transform
        short_time_spectra = self.stft_detrend(trace.data, detrend, **kwargs)

        # Get frequencies
        frequencies = self.frequencies

        # Get times in matplotlib datenum format
        times = self.times(trace.stats, **kwargs)

        return times, frequencies, short_time_spectra

    def map_transform(self, stream: Stream, **kwargs) -> tuple:
        """Transform a stream into the spectral domain.

        This method transforms a stream into the spectral domain by applying
        the Short-Time Fourier Transform to each trace of the stream. The
        method returns the times of the spectrogram in matplotlib datenum
        format, the frequencies of the spectrogram, and the spectrogram of the
        stream.

        This method is basically a wrapper around the :meth:`transform` method
        that applies the Short-Time Fourier Transform to each trace of the
        stream.

        Note that the stream must be ready for processing, i.e., it must pass
        the quality control from the property :attr:`~covseisnet.stream.Stream.is_ready_to_process`.

        Arguments
        ---------
        stream : :class:`~obspy.core.stream.Stream`
            The stream to transform.

        Returns
        -------
        times : numpy.ndarray
            The times of the spectrogram in matplotlib datenum format.
        frequencies : numpy.ndarray
            The frequencies of the spectrogram.
        short_time_spectra : numpy.ndarray
            The spectrogram of the stream with shape ``(n_trace, n_frequencies,
            n_times)``.
        **kwargs: dict, optional
            Additional keyword arguments are passed to the :meth:`transform`
            method.

        Examples
        --------
        >>> import covseisnet as csn
        >>> stft = csn.ShortTimeFourierTransform(
        ...     window_duration=2.0,
        ...     window_step=1.0,
        ...     window_function="hann",
        ...     sampling_rate=100.0,
        ... )
        >>> stream = csn.read()
        >>> times, frequencies, short_time_spectra = stft.map_transform(stream)
        >>> print(times.shape, frequencies.shape, short_time_spectra.shape)
        (28,) (101,) (3, 101, 28)
        """
        # Calculate the Short-Time Fourier Transform
        short_time_spectra = []
        for trace in stream:
            times, frequencies, spectra = self.transform(trace, **kwargs)
            short_time_spectra.append(spectra)

        return times, frequencies, np.array(short_time_spectra)


def modulus_division(
    x: int | float | complex | np.ndarray, epsilon=1e-10
) -> np.ndarray:
    r"""Division of a number by the absolute value or its modulus.

    Given a complex number (or array) :math:`x = a e^{i\phi}`,
    where :math:`a` is the modulus and :math:`\phi` the phase, the function
    returns the unit-modulus complex number such as

    .. math::

              \mathbb{C} \ni \tilde{x} = \frac{x}{|x| + \epsilon} \approx e^{i\phi}

    This method normalizes the input complex number by dividing it by its
    modulus plus a small epsilon value to prevent division by zero,
    effectively scaling the modulus to 1 while preserving the phase.

    Note that this function also work with real-valued arrays, in which case
    the modulus is the absolute value of the real number. It is useful to
    normalize seismic traces in the temporal domain. It writes

    .. math::

        \mathbb{R} \ni \tilde{x} = \frac{x}{|x| + \epsilon} \approx \text{sign}(x)

    Arguments
    ---------
    x: numpy.ndarray
        The complex-valued data to extract the unit complex number from.
    epsilon: float, optional
        A small value added to the modulus to avoid division by zero. Default
        is 1e-10.

    Returns
    -------
    numpy.ndarray
        The unit complex number with the same phase as the input data, or the
        sign of the real number if the input is real-valued.
    """
    return x / (np.abs(x) + epsilon)


def smooth_modulus_division(
    x: np.ndarray,
    smooth: None | int = None,
    order: int = 1,
    epsilon: float = 1e-10,
) -> np.ndarray:
    r"""Modulus division of a complex number with smoothing.

    Given a complex array :math:`x[n] = a[n] e^{i\phi[n]}`, where :math:`a[n]`
    is the modulus and :math:`\phi[n]` the phase, the function returns the
    normalized complex array :math:`\tilde{x}[n]` such as

    .. math::

        \tilde{x}[n] = \frac{x[n]}{\mathcal S a[n] + \epsilon}

    where :math:`\mathcal S a[n]` is a smoothed version of the modulus array
    :math:`a[n]`. The smoothing function is performed with the Savitzky-Golay
    filter, and the order and length of the filter are set by the ``smooth``
    and ``order`` parameters.

    Arguments
    ---------
    x: :class:`numpy.ndarray`
        The spectra to detrend. Must be of shape ``(n_frequencies, n_times)``.
    smooth: int
        Smoothing window size in points.
    order: int
        Smoothing order. Check the scipy function
        :func:`~scipy.signal.savgol_filter` function for more details.

    Keyword arguments
    -----------------
    epsilon: float, optional
        A regularizer for avoiding zero division.

    Returns
    -------
    The spectrum divided by the smooth modulus spectrum.
    """
    smooth_modulus = signal.savgol_filter(np.abs(x), smooth, order, axis=0)
    return x / (smooth_modulus + epsilon)


def smooth_envelope_division(
    x: np.ndarray, smooth: int, order: int = 1, epsilon: float = 1e-10
) -> np.ndarray:
    r"""Normalize seismic traces by a smooth version of its envelope.

    This function normalizes seismic traces by a smooth version of its
    envelope. The envelope is calculated with the Hilbert transform, and then
    smoothed with the Savitzky-Golay filter. The order and length of the
    filter are set by the ``smooth`` and ``order`` parameters.

    Considering the seismic trace :math:`x(t)`, the normalized trace
    :math:`\hat x(t)` is obtained with

    .. math::

        \hat x(t) = \frac{x(t)}{\mathcal{A}x(t) + \epsilon}

    where :math:`A` is an smoothing operator applied to Hilbert envelope of
    the trace :math:`x(t)`.

    Arguments
    ---------
    x: numpy.ndarray
        The seismic trace to normalize.
    smooth: int
        The length of the Savitzky-Golay filter for smoothing the envelope.
    order: int
        The order of the Savitzky-Golay filter for smoothing the envelope.
    epsilon: float, optional
        Regularization parameter in division, set to ``1e-10`` by default.

    Returns
    -------
    numpy.ndarray
        The normalized seismic trace.
    """
    envelope = np.abs(np.array(signal.hilbert(x)))
    smooth_envelope = signal.savgol_filter(envelope, smooth, order)
    return x / (smooth_envelope + epsilon)


def entropy(
    x: np.ndarray, epsilon: float = 1e-10, axis: int = -1
) -> np.ndarray:
    r"""Entropy calculation.

    Entropy calculated from a given distribution of values. The entropy is
    defined as

    .. math::

        h = - \sum_{n=0}^N n x_n \log(x_n + \epsilon)

    where :math:`x_n` is the distribution of values. This function assumes the
    distribution is normalized by its sum.

    Arguments
    ---------
    x: :class:`numpy.ndarray`
        The distribution of values.
    epsilon: float, optional
        The regularization parameter for the logarithm.
    **kwargs: dict, optional
        Additional keyword arguments passed to the :func:`numpy.sum` function.
        Typically, the axis along which the sum is performed.
    """
    return np.sum(-x * np.log(x + epsilon), axis=axis)


def diversity(x: np.ndarray, epsilon: float = 1e-10, **kwargs) -> np.ndarray:
    r"""Shanon diversity index calculation.

    Shanon diversity index calculated from a given distribution of values. The
    diversity index is defined as

    .. math::

        D = \exp(h + \epsilon)

    where :math:`h` is the entropy of the distribution of values. This function
    assumes the distribution is normalized by its sum.

    Arguments
    ---------
    x: :class:`numpy.ndarray`
        The distribution of values.
    epsilon: float, optional
        The regularization parameter for the logarithm.
    **kwargs: dict, optional
        Additional keyword arguments passed to the :func:`numpy.sum` function.
        Typically, the axis along which the sum is performed.
    """
    return np.exp(entropy(x, epsilon, **kwargs))


def width(x: np.ndarray, **kwargs) -> np.ndarray:
    r"""Width calculation.

    Width calculated from a given distribution of values. The width is defined
    as

    .. math::

        \sigma = \sum_{n=0}^N n x_n

    where :math:`x_n` is the distribution of values. This function assumes the
    distribution is normalized by its sum.

    Arguments
    ---------
    x: :class:`numpy.ndarray`
        The distribution of values.
    **kwargs: dict, optional
        Additional keyword arguments passed to the :func:`numpy.sum` function.
        Typically, the axis along which the sum is performed.
    """
    kwargs.setdefault("axis", -1)
    indices = np.arange(x.shape[kwargs["axis"]])
    return np.multiply(x, indices).sum(**kwargs)


def bandpass_filter(x, sampling_rate, frequency_band, filter_order=4):
    """Bandpass filter the signal.

    Apply a Butterworth bandpass filter to the signal. Uses
    :func:`~scipy.signal.butter` and :func:`~scipy.signal.filtfilt` to
    avoid phase shift.

    Parameters
    ----------
    x: :class:`~numpy.ndarray`
        The signal to filter.
    sampling_rate: float
        The sampling rate in Hz.
    frequency_band: tuple
        The frequency band to filter in Hz.
    filter_order: int, optional
        The order of the Butterworth filter.

    Returns
    -------
    :class:`~numpy.ndarray`
        The filtered signal.

    """
    # Turn frequencies into normalized frequencies
    nyquist = 0.5 * sampling_rate
    normalized_frequency_band = [f / nyquist for f in frequency_band]

    # Extract filter
    b, a = signal.butter(
        filter_order,
        normalized_frequency_band,
        btype="band",
    )

    # Apply filter
    return signal.filtfilt(b, a, x, axis=-1)


def hilbert_envelope(x: np.ndarray, **kwargs) -> np.ndarray:
    r"""Hilbert envelope calculation.

    Calculate the envelope of a signal using the Hilbert transform. The
    envelope is defined as

    .. math::

        \text{envelope}(x) = \sqrt{x^2 + \text{hilbert}(x)^2}

    Arguments
    ---------
    x: :class:`numpy.ndarray`
        The signal to calculate the envelope from.

    Returns
    -------
    :class:`numpy.ndarray`
        The envelope of the signal.
    """
    return np.abs(np.array(signal.hilbert(x, **kwargs)))
