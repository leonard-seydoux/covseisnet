"""
The package provides tools for spectral analysis of seismic data. The main
class is the :class:`~ShortTimeFourierTransform` class, which extends the
:class:`~scipy.signal.ShortTimeFFT` class to provide a more user-friendly
interface for the Short-Time Fourier Transform with streams. 

Other functions are provided to normalize seismic traces in the spectral
domain, such as the :func:`~modulus_division` and :func:`~smooth_modulus_division` functions. Also, we provide the :func:`~smooth_envelope_division` function to normalize seismic traces by a smooth version of its envelope.
"""

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

        Arguments
        ---------
        window_duration_sec: float, optional
            The duration of the window in seconds. Default is 2.0 seconds.
        window_step_sec: float, optional
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
        ...     window_duration_sec=2.0,
        ...     window_step_sec=1.0,
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

    def transform(
        self, trace: obspy.core.trace.Trace, detrend="linear", **kwargs
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
        ...     window_duration_sec=2.0,
        ...     window_step_sec=1.0,
        ...     window_function="hann",
        ...     sampling_rate=100.0,
        ... )
        >>> stream = csn.read()
        >>> trace = stream[0]
        >>> times, frequencies, short_time_spectra = stft.transform(trace)
        >>> print(times.shape, frequencies.shape, short_time_spectra.shape)
        (28,) (101,) (101, 28)
        """
        # Get trace data
        data = trace.data
        npts = trace.stats.npts

        # Extract the lower and upper border
        _, p0 = self.lower_border_end
        _, p1 = self.upper_border_begin(npts)
        kwargs.setdefault("p0", p0)
        kwargs.setdefault("p1", p1)

        # Calculate the Short-Time Fourier Transform
        short_time_spectra = self.stft_detrend(data, detrend, **kwargs)

        # Get frequencies
        frequencies = self.f

        # Get times in matplotlib datenum format
        starttime_matplotlib = mdates.date2num(trace.stats.starttime.datetime)
        times = self.t(npts, **kwargs).copy()
        times /= 86400
        times += starttime_matplotlib

        return times, frequencies, short_time_spectra

    def map_transform(
        self,
        stream: obspy.core.stream.Stream,
        **kwargs: dict,
    ) -> tuple:
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
        ...     window_duration_sec=2.0,
        ...     window_step_sec=1.0,
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


def modulus_division(x: np.ndarray, epsilon=1e-10) -> np.ndarray:
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
    x: np.ndarray, smooth: int, order: int, epsilon: float = 1e-10
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
    envelope = np.abs(signal.hilbert(x))
    smooth_envelope = signal.savgol_filter(envelope, smooth, order)
    return x / (smooth_envelope + epsilon)
