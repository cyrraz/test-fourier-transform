import numpy as np
import torch
import torch.nn.functional as F
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


def test_conv_fft():
    """Test equivalence of cross-correlation via `conv1d` and FFT, ensuring results match within tolerance."""
    # Generate random input and kernel tensors with shapes (1,1,10) and (1,1,5) respectively.
    x = torch.randn(1, 1, 10, dtype=torch.float64)
    h = torch.randn(1, 1, 5, dtype=torch.float64)

    # Get signal and kernel lengths, and compute full cross-correlation length.
    n, k = x.shape[2], h.shape[2]
    N = n + k - 1

    # Compute cross-correlation in the time domain using conv1d with appropriate padding.
    y_time = F.conv1d(F.pad(x, (k - 1, k - 1)), h).squeeze()

    # Compute FFT-based cross-correlation:
    # Flatten tensors, compute FFTs with zero-padding, multiply with conjugate, then IFFT.
    x_fft = torch.fft.fft(x.view(-1), n=N)
    h_fft = torch.fft.fft(h.view(-1), n=N)
    y_freq = torch.fft.ifft(x_fft * torch.conj(h_fft)).real
    # Align the FFT result with the conv1d output.
    y_freq = torch.roll(y_freq, shifts=(k - 1))

    # Assert that both methods produce nearly identical results.
    assert torch.allclose(y_time, y_freq, atol=1e-10)


def test_conv_fft_2d():
    """
    Test equivalence of 2D cross-correlation using time-domain and FFT-based methods.
    """
    # Create random 2D input and kernel tensors with shapes (1,1,10,10) and (1,1,5,5)
    x = torch.randn(1, 1, 10, 10, dtype=torch.float64)
    h = torch.randn(1, 1, 5, 5, dtype=torch.float64)

    # Determine spatial dimensions and calculate the full cross-correlation size.
    n1, n2 = x.shape[2], x.shape[3]  # Input height and width.
    k1, k2 = h.shape[2], h.shape[3]  # Kernel height and width.
    N1, N2 = n1 + k1 - 1, n2 + k2 - 1  # Full cross-correlation dimensions.

    # Compute time-domain cross-correlation using conv2d with symmetric padding.
    # F.pad pads in the order (pad_left, pad_right, pad_top, pad_bottom).
    y_time = F.conv2d(F.pad(x, (k2 - 1, k2 - 1, k1 - 1, k1 - 1)), h).squeeze()

    # Compute FFT-based cross-correlation:
    # Use 2D FFT with zero-padding to the full cross-correlation size.
    x_fft = torch.fft.fft2(x[0, 0], s=(N1, N2))
    h_fft = torch.fft.fft2(h[0, 0], s=(N1, N2))
    y_freq = torch.fft.ifft2(x_fft * torch.conj(h_fft)).real
    # Align FFT result with conv2d output by applying a circular shift.
    y_freq = torch.roll(y_freq, shifts=(k1 - 1, k2 - 1), dims=(0, 1))

    # Assert that both methods yield nearly identical results.
    assert torch.allclose(y_time, y_freq, atol=1e-10)


def test_conv_fft_3d():
    """Test equivalence of 3D cross-correlation using time-domain conv3d and FFT-based methods."""
    # Generate random input and kernel tensors:
    # - x: 3D input tensor with shape (batch=1, channel=1, depth=10, height=10, width=10)
    # - h: 3D kernel tensor with shape (batch=1, channel=1, depth=5, height=5, width=5)
    x = torch.randn(1, 1, 10, 10, 10, dtype=torch.float64)
    h = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64)

    # Extract spatial dimensions for the input tensor (x):
    # n_d: depth, n_h: height, n_w: width.
    n_d, n_h, n_w = x.shape[2], x.shape[3], x.shape[4]

    # Extract spatial dimensions for the kernel tensor (h):
    # k_d: depth, k_h: height, k_w: width.
    k_d, k_h, k_w = h.shape[2], h.shape[3], h.shape[4]

    # Calculate the full cross-correlation output dimensions for each spatial axis:
    # The full output dimension is given by: input_size + kernel_size - 1.
    N_d = n_d + k_d - 1  # Full output depth.
    N_h = n_h + k_h - 1  # Full output height.
    N_w = n_w + k_w - 1  # Full output width.

    # Compute time-domain cross-correlation using conv3d:
    # Apply symmetric padding to the input tensor so that the output size matches the full cross-correlation size.
    # F.pad for 3D tensors expects the padding order: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
    # Each spatial dimension is padded by (kernel_size - 1) on both sides.
    x_padded = F.pad(x, (k_w - 1, k_w - 1, k_h - 1, k_h - 1, k_d - 1, k_d - 1))
    # Perform 3D convolution:
    # In this context, conv3d computes cross-correlation (no kernel flipping) by sliding the kernel over the padded input.
    y_time = F.conv3d(
        x_padded, h
    ).squeeze()  # Remove singleton dimensions from the result.

    # Compute FFT-based cross-correlation:
    # First, compute the 3D Fast Fourier Transform (FFT) for both the input and kernel.
    # The FFT transforms the data from the spatial domain to the frequency domain.
    # The parameter s=(N_d, N_h, N_w) zero-pads the input to the full cross-correlation size.
    x_fft = torch.fft.fftn(x[0, 0], s=(N_d, N_h, N_w))
    h_fft = torch.fft.fftn(h[0, 0], s=(N_d, N_h, N_w))
    # Multiply the FFT of the input with the complex conjugate of the FFT of the kernel:
    # Multiplication in the frequency domain corresponds to cross-correlation in the spatial domain.
    y_freq = torch.fft.ifftn(
        x_fft * torch.conj(h_fft)
    ).real  # Inverse FFT transforms the result back to the spatial domain; take the real part.

    # Align the FFT result with the time-domain result by applying a circular shift:
    # The shift of (k_d - 1, k_h - 1, k_w - 1) along the respective dimensions compensates for the padding offset.
    y_freq = torch.roll(y_freq, shifts=(k_d - 1, k_h - 1, k_w - 1), dims=(0, 1, 2))

    # Verify that both methods yield nearly identical results within an absolute tolerance of 1e-10:
    # torch.allclose checks element-wise closeness between the two tensors.
    assert torch.allclose(y_time, y_freq, atol=1e-10)


def conv4d(input, weight):
    """
    Perform 4D convolution by stacking 3D convolution operations along the fourth spatial dimension.

    Parameters:
    - input: a tensor of shape (N, C_in, D1, D2, D3, D4) where:
        N    : batch size.
        C_in : number of input channels.
        D1, D2, D3, D4: spatial dimensions of the input.
    - weight: a tensor of shape (C_out, C_in, k1, k2, k3, k4) where:
        C_out: number of output channels.
        C_in : number of input channels (must match the input).
        k1, k2, k3, k4: kernel sizes along the four spatial dimensions.

    Returns:
    - output: a tensor of shape (N, C_out, O1, O2, O3, O4) where:
        O1 = D1 - k1 + 1, O2 = D2 - k2 + 1, O3 = D3 - k3 + 1, O4 = D4 - k4 + 1.

    The function computes cross-correlation (i.e. convolution without kernel flipping)
    by iterating over the fourth spatial dimension of the input and kernel. For each offset
    along the fourth dimension, a 3D convolution is performed over the first three spatial dimensions,
    and the results are summed to yield the final 4D convolution output.
    """
    # Retrieve dimensions from the input tensor.
    N, C_in, D1, D2, D3, D4 = input.shape
    # Retrieve dimensions from the kernel tensor.
    C_out, C_in_weight, k1, k2, k3, k4 = weight.shape
    # Ensure that the number of channels in the input matches that expected by the kernel.
    assert C_in == C_in_weight, "Mismatch between input channels and kernel channels."

    # Calculate the output spatial dimensions for a valid convolution (no extra padding here):
    # Each output dimension is computed as: input_dimension - kernel_dimension + 1.
    O1 = D1 - k1 + 1
    O2 = D2 - k2 + 1
    O3 = D3 - k3 + 1
    O4 = D4 - k4 + 1

    # Initialize the output tensor with zeros. It will store the results of the 4D convolution.
    output = torch.zeros(
        (N, C_out, O1, O2, O3, O4), dtype=input.dtype, device=input.device
    )

    # Loop over each output index along the fourth spatial dimension.
    # The 4D convolution is computed by summing over contributions from each slice of the kernel along this dimension.
    for i in range(O4):
        # Initialize an accumulator to sum the results from each kernel slice (over the fourth dimension).
        slice_sum = None
        # Loop over each offset in the kernel's fourth dimension.
        for r in range(k4):
            # Select the slice from the input corresponding to the current output index plus the kernel offset.
            # This effectively fixes the 4th dimension, resulting in a 5D tensor of shape:
            # (N, C_in, D1, D2, D3) which can be processed by a 3D convolution.
            x_slice = input[..., i + r]

            # Extract the corresponding 3D kernel slice from the weight tensor.
            # The resulting kernel slice has shape (C_out, C_in, k1, k2, k3) and is used for the 3D convolution.
            w_slice = weight[..., r]

            # Perform a 3D convolution on the selected slice of the input with the corresponding kernel slice.
            # F.conv3d computes the convolution over the three spatial dimensions (D1, D2, D3).
            conv_result = F.conv3d(x_slice, w_slice)
            # The shape of conv_result is (N, C_out, O1, O2, O3).

            # Accumulate the convolution result from this kernel slice.
            slice_sum = conv_result if slice_sum is None else slice_sum + conv_result

        # After processing all kernel slices for the current output index, assign the accumulated result
        # to the corresponding slice of the output tensor along the fourth spatial dimension.
        output[..., i] = slice_sum

    # Return the complete 4D convolution output.
    return output


def test_conv_fft_4d():
    """
    Test the equivalence of 4D cross-correlation computed via a time-domain method (using conv4d)
    and an FFT-based method. Random input and kernel tensors are generated, and both methods are compared
    to ensure that they yield nearly identical results within a specified numerical tolerance.
    """
    # Generate a random 4D input tensor with shape:
    # (batch_size=1, channels=1, dim1=10, dim2=10, dim3=10, dim4=10)
    x = torch.randn(1, 1, 10, 10, 10, 10, dtype=torch.float64)

    # Generate a random 4D kernel tensor with shape:
    # (batch_size=1, channels=1, kernel_dim1=5, kernel_dim2=5, kernel_dim3=5, kernel_dim4=5)
    h = torch.randn(1, 1, 5, 5, 5, 5, dtype=torch.float64)

    # Extract the spatial dimensions of the input tensor.
    n1, n2, n3, n4 = x.shape[2], x.shape[3], x.shape[4], x.shape[5]

    # Extract the kernel sizes from the kernel tensor.
    k1, k2, k3, k4 = h.shape[2], h.shape[3], h.shape[4], h.shape[5]

    # Compute the full cross-correlation dimensions for each spatial axis.
    # For full cross-correlation, the output size is: input_size + kernel_size - 1.
    N1, N2, N3, N4 = n1 + k1 - 1, n2 + k2 - 1, n3 + k3 - 1, n4 + k4 - 1

    # Pad the input tensor symmetrically along each spatial dimension so that the output of the convolution
    # corresponds to the full cross-correlation result. F.pad requires padding in the order:
    # (pad_last_dim_left, pad_last_dim_right, pad_second_last_left, pad_second_last_right, ...).
    x_padded = F.pad(
        x,
        (
            k4 - 1,
            k4 - 1,  # Padding for the 4th spatial dimension.
            k3 - 1,
            k3 - 1,  # Padding for the 3rd spatial dimension.
            k2 - 1,
            k2 - 1,  # Padding for the 2nd spatial dimension.
            k1 - 1,
            k1 - 1,
        ),
    )  # Padding for the 1st spatial dimension.

    # Compute the time-domain cross-correlation using the defined conv4d function.
    # The conv4d function applies 4D convolution by stacking 3D convolutions along the fourth spatial dimension.
    # The squeeze() function removes any singleton dimensions to facilitate comparison.
    y_time = conv4d(x_padded, h).squeeze()

    # Compute the FFT-based cross-correlation:
    # 1. Compute the 4D Fast Fourier Transform (FFT) of the original input and kernel.
    #    The 's' parameter zero-pads the tensor to the full cross-correlation dimensions.
    x_fft = torch.fft.fftn(x[0, 0], s=(N1, N2, N3, N4))
    h_fft = torch.fft.fftn(h[0, 0], s=(N1, N2, N3, N4))

    # 2. Multiply the FFT of the input with the complex conjugate of the FFT of the kernel.
    #    In the frequency domain, multiplication corresponds to cross-correlation in the spatial domain.
    y_freq = torch.fft.ifftn(
        x_fft * torch.conj(h_fft)
    ).real  # Perform the inverse FFT and take the real part.

    # Align the FFT-based result with the time-domain result by applying a circular shift.
    # The shift compensates for the offset introduced by the symmetric padding, shifting each spatial axis by (kernel_size - 1).
    y_freq = torch.roll(
        y_freq, shifts=(k1 - 1, k2 - 1, k3 - 1, k4 - 1), dims=(0, 1, 2, 3)
    )

    # Verify that the results from the time-domain and FFT-based methods are nearly identical.
    # torch.allclose checks that all elements of the two tensors are equal within an absolute tolerance of 1e-10.
    assert torch.allclose(y_time, y_freq, atol=1e-10)
