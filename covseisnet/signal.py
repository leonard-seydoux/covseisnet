"""Spectral tools."""

import matplotlib.dates as mdates
import numpy as np
import obspy
from scipy import signal


class ShortTimeFourierTransform(signal.ShortTimeFFT):

    def __init__(
        self,
        window_duration_sec: float = 2.0,
        window_step_sec: None | float = None,
        window_function: str = "hann",
        sampling_rate: None | float = 1.0,
        **kwargs,
    ) -> None:
        """Short-Time Fourier Transform instance.

        This class extends the :class:`scipy.signal.ShortTimeFFT` class to
        provide a more user-friendly interface for the Short-Time Fourier
        Transform with ObsPy and Covseisnet Streams.
        """
        # Define apodization window
        window_size = int(window_duration_sec * sampling_rate)
        window_array = signal.windows.get_window(window_function, window_size)

        # Define step between windows
        if window_step_sec is not None:
            window_step = int(window_step_sec * sampling_rate)
        else:
            window_step = window_size // 2

        # Set sampling rate
        kwargs.setdefault("fs", sampling_rate)

        # Initialize the Short-Time Fourier Transform
        super().__init__(window_array, window_step, **kwargs)

    def __str__(self):
        k0, p0 = self.lower_border_end
        out = "Short-Time Fourier Transform instance\n"
        out += f"\tSampling rate: {self.fs} Hz\n"
        out += f"\tFrequency range: {self.f[1] :.2f} to {self.f[-1] :.2f} Hz\n"
        out += f"\tFrequency resolution: {self.delta_f} Hz\n"
        out += f"\tFrequency bins: {self.f_pts}\n"
        out += f"\tWindow length: {len(self.win)} samples\n"
        out += f"\tWindow step: {self.hop} samples\n"
        out += f"\tOut-of-bounds: {k0} sample(s), {p0} window(s)"
        return out

    def transform(self, trace: obspy.core.trace.Trace) -> tuple:
        """Short-time Fourier Transform of a trace.

        Arguments
        ---------
        trace : :class:`~obspy.core.trace.Trace`
            The trace to transform.
        **kwargs: dict, optional
            Additional keyword arguments are passed to the
            :meth:`scipy.signal.ShortTimeFFT.stft` method.

        Returns
        -------
        times : list
            The times of the window centers in matplotlib datenum format.
        frequencies : numpy.ndarray
            The frequencies of the spectrogram.
        spectrogram : numpy.ndarray
            The spectrogram of the trace.
        """
        # Get trace data
        data = trace.data
        npts = trace.stats.npts

        # Extract the lower and upper border
        _, p0 = self.lower_border_end
        _, p1 = self.upper_border_begin(npts)

        # Calculate the Short-Time Fourier Transform
        short_time_spectra = self.stft_detrend(data, "linear", p0=p0, p1=p1)

        # Get frequencies
        frequencies = self.f

        # Get times
        starttime_matplotlib = mdates.date2num(trace.stats.starttime.datetime)
        times = self.t(npts, p0=p0, p1=p1).copy()
        times /= 86400
        times += starttime_matplotlib

        return times, frequencies, short_time_spectra

    def map_transform(
        self,
        stream: obspy.core.stream.Stream,
    ) -> tuple:
        """Transform a stream into the spectral domain.

        Arguments
        ---------
        stream : :class:`~obspy.core.stream.Stream`
            The stream to transform.

        Returns
        -------
        times : list
            The times of the spectrogram in class:`~obspy.UTCDateTime` format.
        frequencies : numpy.ndarray
            The frequencies of the spectrogram.
        spectrogram : numpy.ndarray
            The spectrogram of the stream.
        """
        # Calculate the Short-Time Fourier Transform
        short_time_spectra = []
        for trace in stream:
            times, frequencies, short_time_spectrum = self.transform(trace)
            short_time_spectra.append(short_time_spectrum)

        return times, frequencies, np.array(short_time_spectra)


def modulus_division(x: np.ndarray, epsilon=1e-10) -> np.ndarray:
    r"""Modulus division of a complex number.

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

        \tilde{x}[n] = \frac{x[n]}{s(a)[n] + \epsilon}

    where :math:`s(a)[n]` is a smoothed version of the modulus array
    :math:`a[n]`. The smoothing function is performed with the Savitzky-Golay
    filter, and the order and length of the filter are set by the ``smooth``
    and ``order`` parameters.].

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
    x: np.ndarray, smooth: int, order: int, epsilon: float = 1e-10
) -> np.ndarray:
    r"""Normalize seismic traces by a smooth version of its envelope.

    This function normalizes seismic traces by a smooth version of its
    envelope. The envelope is calculated with the Hilbert transform, and then
    smoothed with the Savitzky-Golay filter. The order and length of the
    filter are set by the ``smooth`` and ``order`` parameters.

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
    envelope = np.abs(signal.hilbert(x))
    smooth_envelope = signal.savgol_filter(envelope, smooth, order)
    return x / (smooth_envelope + epsilon)
