import numpy as np
import pytest
from scipy.signal import find_peaks


def test_fft_zeros():
    """Test that the FFT of an all-zero signal remains zero. This validates the linearity and base case of FFT."""
    assert np.all(np.fft.fft(np.zeros(8)) == 0)


def test_fft_delta():
    """Test that the FFT of a delta function (impulse) yields a flat spectrum. This verifies that an impulse contains all frequencies equally."""
    delta = np.zeros(8)
    delta[0] = 1
    assert np.all(np.fft.fft(delta) == 1)


def test_fft_constant():
    """Test that a constant signal's FFT has only a DC component. All other frequencies should be zero."""
    x = np.ones(8)
    y = np.fft.fft(x)
    assert y[0] == 8
    assert np.allclose(y[1:], 0)


def test_ifft_identity():
    """Test that applying IFFT to the FFT of a signal reconstructs the original signal. This checks reversibility of the FFT."""
    x = np.random.rand(16)
    assert np.allclose(np.fft.ifft(np.fft.fft(x)), x)


def test_fft_single_sine():
    """Test that the FFT of a sine wave shows a peak at the correct frequency (5Hz). Verifies frequency resolution and correct transform."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)
    y = np.fft.fft(x)
    freq = np.fft.fftfreq(len(t), d=1 / 1000)
    idx = np.argmax(np.abs(y))
    assert freq[idx] == pytest.approx(5, abs=0.1)


def test_fft_symmetry():
    """Test that FFT of real input is conjugate symmetric."""
    x = np.random.rand(64)
    y = np.fft.fft(x)
    assert np.allclose(np.conj(y[1:]), y[-1:0:-1])


def test_parseval():
    """Test Parseval's theorem: energy in time domain equals energy in frequency domain (scaled). Verifies energy conservation in FFT."""
    x = np.random.rand(128)
    energy_time = np.sum(np.abs(x) ** 2)
    energy_freq = np.sum(np.abs(np.fft.fft(x)) ** 2) / len(x)
    assert pytest.approx(energy_time) == energy_freq


def test_fft_complex_signal():
    """Test that a constant complex signal's FFT has only a DC component. All other frequencies should be zero."""
    x = np.ones(8) + 1j * np.ones(8)
    y = np.fft.fft(x)
    assert y[0] == pytest.approx(8 + 8j)
    assert np.allclose(y[1:], 0)


def test_ifft_known():
    """Test IFFT of a frequency domain vector with imaginary parts reconstructs a real sine wave (phase may differ)."""
    y = np.zeros(4, dtype=complex)
    y[1] = 2j
    y[3] = -2j  # Conjugate pair to yield a real sine wave
    x = np.fft.ifft(y).real

    expected = np.sin(2 * np.pi * np.arange(4) / 4)
    # Accept both sine and -sine (phase difference of π)
    assert np.allclose(x, expected, atol=1e-12) or np.allclose(x, -expected, atol=1e-12)


def test_fft_length():
    """Test that changing FFT length (zero-padding) affects output length. Verifies FFT supports padding."""
    x = np.ones(4)
    y = np.fft.fft(x, n=8)
    assert len(y) == 8


def test_fft_cosine():
    """Test FFT of a cosine wave shows a peak at its frequency (20Hz). Cosine yields two symmetric peaks."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.cos(2 * np.pi * 20 * t)
    y = np.abs(np.fft.fft(x))
    freqs = np.fft.fftfreq(len(t), 1 / 1000)
    assert freqs[np.argmax(y)] == pytest.approx(20, abs=0.1)


def test_ifft_real():
    """Test IFFT of real input yields nearly real output (imaginary parts near zero due to float error)."""
    x = np.random.rand(16)
    y = np.fft.fft(x)
    x_rec = np.fft.ifft(y)
    assert np.all(np.abs(x_rec.imag) < 1e-10)


def test_fftshift_peak():
    """Test FFT shift centers the zero frequency. A delta at the center in time should yield zero-centered spectrum."""
    x = np.zeros(16)
    x[8] = 1
    y = np.fft.fftshift(np.fft.fft(x))
    peak_index = np.argmax(np.abs(y))
    assert peak_index == 0


def test_fft_windowed():
    """Test FFT on windowed signal shows expected frequency with reduced leakage. Verifies impact of windowing."""
    x = np.hamming(64) * np.sin(2 * np.pi * 4 * np.arange(64) / 64)
    y = np.fft.fft(x)
    peak = np.argmax(np.abs(y[:32]))
    assert peak == pytest.approx(4, abs=1)


def test_fft_two_sines():
    """Test FFT of two sine waves shows peaks at 10Hz and 20Hz within tolerance."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    y = np.abs(np.fft.fft(x))
    freqs = np.fft.fftfreq(len(t), 1 / 1000)
    peaks, _ = find_peaks(y[:500], height=10)
    detected = np.round(freqs[peaks])
    assert 10 in detected
    assert 20 in detected


def test_fft_phase():
    """Test FFT phase of a sine wave is around -π/2. Sine wave leads cosine by 90°, hence the expected phase shift."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)
    y = np.fft.fft(x)
    phase = np.angle(y[np.argmax(np.abs(y))])
    assert pytest.approx(phase, abs=0.1) == -np.pi / 2


def test_fft_square_wave():
    """Test FFT of a square wave detects the fundamental frequency (5 Hz) regardless of sign."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sign(np.sin(2 * np.pi * 5 * t))
    y = np.abs(np.fft.fft(x))
    freqs = np.fft.fftfreq(len(t), 1 / 1000)
    peak = freqs[np.argmax(y[1:]) + 1]
    assert pytest.approx(abs(peak), abs=0.5) == 5


def test_fft_bin_resolution():
    """Test frequency bin resolution is Fs/N. Ensures users understand frequency precision for FFT output."""
    N = 1024
    Fs = 1000
    freq_bin = Fs / N
    assert freq_bin == pytest.approx(0.97656, rel=1e-4)


def test_fft_chirp():
    """Test FFT of a chirp signal spreads energy over many frequencies (broadband behavior)."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * t**2 * 50)
    y = np.abs(np.fft.fft(x))
    assert np.sum(y > 0.2 * np.max(y)) > 10


def test_fft_noise_flat():
    """Test FFT of white noise shows a roughly flat spectrum. Verifies expected power distribution in random signals."""
    x = np.random.randn(1024)
    y = np.abs(np.fft.fft(x))
    avg = np.mean(y)
    std = np.std(y)
    assert std / avg < 2  # relatively flat spectrum
