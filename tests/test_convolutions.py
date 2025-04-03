import numpy as np
from scipy.linalg import circulant


def test_conv_vs_fft_multiplication():
    """Test that linear convolution equals pointwise multiplication in frequency when inputs are zero-padded."""
    x = np.random.rand(64)
    h = np.random.rand(64)
    N = len(x) + len(h) - 1  # total length for linear convolution
    y_time = np.convolve(x, h)
    y_freq = np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(h, N)).real
    assert np.allclose(y_time, y_freq, atol=1e-10)


def test_convolution_matrix_diagonalization():
    """
    Test that a circulant convolution matrix is diagonalized by the FFT.
    FFT diagonalizes the convolution operator: C = F⁻¹ D F, and C @ x = IFFT(FFT(h) * FFT(x))
    """
    N = 8
    h = np.random.rand(N)
    x = np.random.rand(N)

    # Build circulant matrix from h
    C = circulant(h)

    # Convolution via matrix multiplication
    y_time = C @ x

    # Diagonalization via FFT: C = F⁻¹ D F
    F = np.fft.fft(np.eye(N))  # Fourier matrix
    F_inv = np.fft.ifft(np.eye(N))  # Inverse Fourier matrix
    D = np.diag(np.fft.fft(h))  # Diagonal matrix of FFT(h)

    y_freq = F_inv @ (D @ (F @ x))  # y = F⁻¹ D F x

    assert np.allclose(y_time, y_freq.real, atol=1e-12)
