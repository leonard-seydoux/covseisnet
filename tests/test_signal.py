"""Test of the ArrayStream class."""

import numpy as np

from covseisnet import read
from covseisnet import ShortTimeFourierTransform
from covseisnet import signal

# Seed the random number generator for reproducibility.
np.random.seed(42)


def test_stft_instance():
    """Check default ShortTimeFourierTransform sampling rate."""
    stft = ShortTimeFourierTransform()
    assert stft.sampling_rate == stft.fs == 1.0


def test_stft_times():
    """Check default ShortTimeFourierTransform times."""
    stream = read()
    stft = ShortTimeFourierTransform(sampling_rate=stream.sampling_rate)
    assert all(stft.times(stream.traces[0].stats) != 0)


def test_map_transform_shape():
    """Check the map_transform method."""
    stream = read()
    stft = ShortTimeFourierTransform(sampling_rate=stream.sampling_rate)
    times, frequencies, spectra = stft.map_transform(stream)
    assert spectra.shape == (
        len(stream),
        len(frequencies),
        len(times),
    )


def test_transform_shape():
    """Check the transform method."""
    stream = read()
    stft = ShortTimeFourierTransform(sampling_rate=stream.sampling_rate)
    times, frequencies, spectrum = stft.transform(stream.traces[0])
    assert spectrum.shape == (len(frequencies), len(times))


def test_modulus_division():
    # Scalar tests
    assert np.allclose(signal.modulus_division(2), 1)
    assert np.allclose(signal.modulus_division(0 + 4j), 1j)
    assert np.allclose(signal.modulus_division(2 + 2j), np.exp(1j * np.pi / 4))
    # Array tests
    x = np.random.randn(5)
    assert np.allclose(signal.modulus_division(x), np.sign(x))
    x = np.random.randn(5) + 1j * np.random.randn(5)
    assert np.allclose(signal.modulus_division(x), x / np.abs(x))


def test_smooth_modulus_division():
    x = np.random.randn(100)
    y = signal.smooth_modulus_division(x, smooth=11)
    assert len(y) == len(x)


def test_smooth_envelope_division():
    x = np.random.randn(100)
    y = signal.smooth_envelope_division(x, smooth=11)
    assert len(y) == len(x)
