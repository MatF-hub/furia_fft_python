import cmath
import math
import numpy as np

def _dft_1d(x: np.ndarray, inverse: bool) -> np.ndarray:
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            if not inverse:
                X[k] += x[n] * cmath.exp(-2j * math.pi * k * n / N)
            else:
                X[k] += x[n] * cmath.exp(2j * math.pi * k * n / N)

    return X

def _dft_nd(x: np.ndarray, inverse: bool) -> np.ndarray:
    result = x.astype(complex)

    for axis in range(result.ndim):
        result = np.apply_along_axis(
            lambda v: _dft_1d(v, inverse),
            axis,
            result
        )

    return result

def dft(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(x).__name__}")

    if x.ndim == 1:
        return _dft_1d(x, inverse=False)
    elif x.ndim >= 2:
        return _dft_nd(x, inverse=False)
    else:
        raise ValueError(f"Input must be at least 1D, got {x.ndim}D")

def idft(X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(X).__name__}")

    if X.ndim == 1:
        return _dft_1d(X, inverse=True) / len(X)
    elif X.ndim >= 2:
        return _dft_nd(X, inverse=True) / (X.shape[0] * X.shape[1])
    else:
        raise ValueError(f"Input must be at least 1D, got {X.ndim}D")

# radix-2 Cooley–Tukey recursive FFT
def _radix2_recursive_fft_1d(x: np.ndarray) -> np.ndarray:
    N = len(x)

    if N & (N - 1):
        raise ValueError("Sequence length must be power of 2")

    if N == 1:
        return x

    if N == 2:
        return np.array([x[0] + x[1], x[0] - x[1]], dtype=complex)

    WN = cmath.exp(-2j * math.pi / N)
    W = [WN ** k for k in range(N // 2)]

    even_indices = np.array([x[2*r] for r in range(N // 2)])
    odd_indices = np.array([x[2*r+1] for r in range(N // 2)])

    A = _radix2_recursive_fft_1d(even_indices)
    B = _radix2_recursive_fft_1d(odd_indices)

    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        X[k] = A[k] + W[k] * B[k]
        X[k + N // 2] = A[k] - W[k] * B[k]

    return X

def bit_reverse_ordering(x: np.ndarray):
    N = len(x)

    bit_reversed_index = 0

    for index in range(1, N):

        # Start at highest bit
        bit = N // 2

        # Find the bit that needs to change
        while bit_reversed_index & bit:
            bit_reversed_index ^= bit
            bit //= 2

        bit_reversed_index ^= bit

        if index < bit_reversed_index:
            x[index], x[bit_reversed_index] = x[bit_reversed_index], x[index]

# radix-2 Cooley–Tukey iterative FFT
def _radix2_fft_1d(x: np.ndarray) -> np.ndarray:
    N = len(x)

    if N == 1:
        return x

    if N == 2:
        return np.array([x[0] + x[1], x[0] - x[1]], dtype=complex)
    
    x = np.asarray(x, dtype=complex).copy()

    # Bit-reversal ordering 
    bit_reverse_ordering(x)

    # FFT butterfly stages, we will work backward.
    size = 2 #first butterfly stage has size 2. notice W(size=2)=1.

    while size <= N:
        half = size // 2

        # Twiddle factor for this stage
        W = np.exp(-2j * np.pi / size)

        # Process each group of `size` elements
        for start in range(0, N, size):

            w = 1.0

            # Perform butterflies
            for k in range(half):
                even = x[start + k]
                odd = x[start + k + half] * w

                x[start + k] = even + odd
                x[start + k + half] = even - odd

                w *= W

        size *= 2

    return x

def _fft_1d(x: np.ndarray) -> np.ndarray:
    N = len(x)

    if N & (N - 1):
        return Bluestein_fft_1d(x)
    else:
        return _radix2_fft_1d(x)

def _fft_nd(x: np.ndarray) -> np.ndarray:
    result = x.astype(complex)

    for axis in range(result.ndim):
        result = np.apply_along_axis(_fft_1d, axis, result)

    return result

def fft(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(x).__name__}")

    if x.ndim == 1:
        return _fft_1d(x)
    elif x.ndim >= 2:
        return _fft_nd(x)
    else:
        raise ValueError(f"Input must be at least 1D, got {x.ndim}D")

# Use conjugate trick to compute ifft using fft.
def _ifft_1d(X: np.ndarray) -> np.ndarray:
    N = len(X)
    X_conj = np.conj(X)
    result = _fft_1d(X_conj)
    return np.conj(result) / N

def _ifft_nd(X: np.ndarray) -> np.ndarray:
    result = X.astype(complex)

    for axis in range(result.ndim):
        result = np.apply_along_axis(_ifft_1d, axis, result)

    return result

def ifft(X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(X).__name__}")

    if X.ndim == 1:
        return _ifft_1d(X)
    elif X.ndim >= 2:
        return _ifft_nd(X)
    else:
        raise ValueError(f"Input must be at least 1D, got {X.ndim}D")

def circular_convolution(x: np.ndarray, y: np.ndarray):
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")    
    X = fft(x)
    Y = fft(y)
    return ifft(X*Y)

def Bluestein_fft_1d(x: np.ndarray):
    N = len(x)

    # Use binary representation length to find next power of 2.
    M = 2**((N-1).bit_length())
    # For bluestein we need next power of 2, s.t:  M >= 2*N-1 && M <= 4*N-3
    while (M < 2*N-1):
        M *= 2

    # define padded signals using chirps.
    x_padded = np.zeros(M, dtype=complex)
    y_padded = np.zeros(M, dtype=complex)

    # Twiddle factor
    W = cmath.exp(1j*2*math.pi/N)

    chirp_1 = np.array([W ** ( -0.5 * n ** 2) for n in range(N)], dtype=complex)
    x_padded[:N]=x*chirp_1
    chirp_2 = np.array([W ** (0.5 * n**2) for n in range(-N + 1, N)], dtype=complex)
    y_padded[:N] = chirp_2[N-1:]
    y_padded[M-N+1:M] = chirp_2[:N-1]

    return chirp_1 * circular_convolution(x_padded, y_padded)[:N]

if __name__ == "__main__":
    print("=" * 60)
    print("1D FFT (Power-of-2)")
    print("=" * 60)
    N = 1024
    x = np.array([np.sin(2 * np.pi * i / N) + 0.5 * np.cos(4 * np.pi * i / N) for i in range(N)], dtype=complex)

    X_dft = dft(x)
    X_fft = fft(x)
    X_numpy = np.fft.fft(x)

    max_err = max(np.max(np.abs(X_dft - X_fft)), np.max(np.abs(X_fft - X_numpy)))
    all_close = np.allclose(X_dft, X_fft) and np.allclose(X_fft, X_numpy)
    print(f"DFT, FFT, NumPy all close: {all_close} | max error: {max_err:.2e}")

    x_idft = idft(X_dft)
    x_ifft = ifft(X_fft)
    x_numpy_ifft = np.fft.ifft(X_numpy)

    print(f"Reconstruction errors (x - inverse(forward(x))):")
    print(f"  IDFT(DFT(x)): {np.max(np.abs(x - x_idft)):.2e}")
    print(f"  IFFT(FFT(x)): {np.max(np.abs(x - x_ifft)):.2e}")
    print(f"  NumPy:        {np.max(np.abs(x - x_numpy_ifft)):.2e}")

    print("\n" + "=" * 60)
    print("2D FFT")
    print("=" * 60)
    np.random.seed(42)
    x_2d = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)

    X_dft_2d = dft(x_2d)
    X_fft_2d = fft(x_2d)
    X_numpy_2d = np.fft.fft2(x_2d)

    max_err_2d = max(np.max(np.abs(X_dft_2d - X_fft_2d)), np.max(np.abs(X_fft_2d - X_numpy_2d)))
    all_close_2d = np.allclose(X_dft_2d, X_fft_2d) and np.allclose(X_fft_2d, X_numpy_2d)
    print(f"DFT, FFT, NumPy all close: {all_close_2d} | max error: {max_err_2d:.2e}")

    x_idft_2d = idft(X_dft_2d)
    x_ifft_2d = ifft(X_fft_2d)
    x_numpy_ifft_2d = np.fft.ifft2(X_numpy_2d)

    print(f"Reconstruction errors (x - inverse(forward(x))):")
    print(f"  IDFT(DFT(x)): {np.max(np.abs(x_2d - x_idft_2d)):.2e}")
    print(f"  IFFT(FFT(x)): {np.max(np.abs(x_2d - x_ifft_2d)):.2e}")
    print(f"  NumPy:        {np.max(np.abs(x_2d - x_numpy_ifft_2d)):.2e}")

    print("\n" + "=" * 60)
    print("BLUESTEIN FFT (Non-Power-of-2)")
    print("=" * 60)
    non_pow2_sizes = [100, 333, 500, 1000]

    for size in non_pow2_sizes:
        x_non_pow2 = np.random.randn(size) + 1j * np.random.randn(size)

        X_dft_non_pow2 = dft(x_non_pow2)
        X_fft_non_pow2 = fft(x_non_pow2)
        X_numpy_non_pow2 = np.fft.fft(x_non_pow2)

        all_close = np.allclose(X_dft_non_pow2, X_fft_non_pow2) and np.allclose(X_fft_non_pow2, X_numpy_non_pow2)

        x_idft_non_pow2 = idft(X_dft_non_pow2)
        x_ifft_non_pow2 = ifft(X_fft_non_pow2)
        x_numpy_ifft_non_pow2 = np.fft.ifft(X_numpy_non_pow2)

        print(f"N={size:4d}: {all_close}")
        print(f"  Reconstruction: IDFT {np.max(np.abs(x_non_pow2 - x_idft_non_pow2)):.2e}, IFFT {np.max(np.abs(x_non_pow2 - x_ifft_non_pow2)):.2e}, NumPy {np.max(np.abs(x_non_pow2 - x_numpy_ifft_non_pow2)):.2e}")
