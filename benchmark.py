import core
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    # Power-of-2 benchmarks
    print("=" * 90)
    print("POWER-OF-2 SEQUENCES")
    print("=" * 90)
    sizes_pow2 = [2**k for k in range(4, 11)]
    num_repeats = 10

    dft_times = []
    fft_recursive_times = []
    fft_iterative_times = []
    numpy_times = []

    print("Size\t\tDFT (ms)\tRec FFT (ms)\tIter FFT (ms)\tNumPy (ms) [ref]")
    print("-" * 90)

    for size in sizes_pow2:
        x = np.random.randn(size) + 1j * np.random.randn(size)

        dft_total = 0
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            core.dft(x)
            dft_total += time.perf_counter() - t0
        dft_avg = (dft_total / num_repeats) * 1000

        fft_recursive_total = 0
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            core._radix2_recursive_fft_1d(x)
            fft_recursive_total += time.perf_counter() - t0
        fft_recursive_avg = (fft_recursive_total / num_repeats) * 1000

        fft_iterative_total = 0
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            core._radix2_fft_1d(x)
            fft_iterative_total += time.perf_counter() - t0
        fft_iterative_avg = (fft_iterative_total / num_repeats) * 1000

        numpy_total = 0
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            np.fft.fft(x)
            numpy_total += time.perf_counter() - t0
        numpy_avg = (numpy_total / num_repeats) * 1000

        dft_times.append(dft_avg)
        fft_recursive_times.append(fft_recursive_avg)
        fft_iterative_times.append(fft_iterative_avg)
        numpy_times.append(numpy_avg)

        rec_vs_iter = fft_recursive_avg / fft_iterative_avg if fft_iterative_avg > 0 else 0
        print(f"{size}\t\t{dft_avg:.4f}\t\t{fft_recursive_avg:.4f}\t\t{fft_iterative_avg:.4f}\t\t{numpy_avg:.4f}")

    # Non-power-of-2 benchmarks
    print("\n" + "=" * 90)
    print("NON-POWER-OF-2 SEQUENCES (BLUESTEIN)")
    print("=" * 90)
    sizes_non_pow2 = [100, 333, 500, 1000, 2000, 5000]

    dft_times_non_pow2 = []
    fft_bluestein_times = []
    numpy_times_non_pow2 = []

    print("Size\t\tDFT (ms)\t\tBluestein (ms)\tNumPy (ms) [ref]")
    print("-" * 90)

    for size in sizes_non_pow2:
        x = np.random.randn(size) + 1j * np.random.randn(size)

        dft_total = 0
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            core.dft(x)
            dft_total += time.perf_counter() - t0
        dft_avg = (dft_total / num_repeats) * 1000

        fft_bluestein_total = 0
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            core.fft(x)
            fft_bluestein_total += time.perf_counter() - t0
        fft_bluestein_avg = (fft_bluestein_total / num_repeats) * 1000

        numpy_total = 0
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            np.fft.fft(x)
            numpy_total += time.perf_counter() - t0
        numpy_avg = (numpy_total / num_repeats) * 1000

        dft_times_non_pow2.append(dft_avg)
        fft_bluestein_times.append(fft_bluestein_avg)
        numpy_times_non_pow2.append(numpy_avg)

        print(f"{size}\t\t{dft_avg:.4f}\t\t{fft_bluestein_avg:.4f}\t\t\t{numpy_avg:.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Power-of-2 plot
    ax1.loglog(sizes_pow2, dft_times, 'o-', linewidth=2.5, markersize=8, label='DFT O(N²)', color='red')
    ax1.loglog(sizes_pow2, fft_recursive_times, 's-', linewidth=2.5, markersize=8, label='FFT Recursive', color='blue')
    ax1.loglog(sizes_pow2, fft_iterative_times, '^-', linewidth=2.5, markersize=8, label='FFT Iterative', color='darkgreen')
    ax1.loglog(sizes_pow2, numpy_times, '--', linewidth=1.5, alpha=0.5, label='NumPy (C-backend ref)', color='gray')

    ax1.set_xlabel("Sequence Length (N)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Execution Time (ms)", fontsize=11, fontweight='bold')
    ax1.set_title("Power-of-2: Recursive vs Iterative FFT", fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=10, loc='upper left')

    # Non-power-of-2 plot
    ax2.loglog(sizes_non_pow2, dft_times_non_pow2, 'o-', linewidth=2.5, markersize=8, label='DFT O(N²)', color='red')
    ax2.loglog(sizes_non_pow2, fft_bluestein_times, 's-', linewidth=2.5, markersize=8, label='FFT Bluestein', color='blue')
    ax2.loglog(sizes_non_pow2, numpy_times_non_pow2, '--', linewidth=1.5, alpha=0.5, label='NumPy (C-backend ref)', color='gray')

    ax2.set_xlabel("Sequence Length (N)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Execution Time (ms)", fontsize=11, fontweight='bold')
    ax2.set_title("Non-Power-of-2: DFT vs Bluestein FFT", fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig('fft_benchmark.png', dpi=150, bbox_inches='tight')
    print("Plot saved to fft_benchmark.png")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\nPower-of-2 at N={sizes_pow2[-1]}:")
    print(f"  DFT vs Recursive FFT: {dft_times[-1]/fft_recursive_times[-1]:.0f}x speedup")
    print(f"  DFT vs Iterative FFT: {dft_times[-1]/fft_iterative_times[-1]:.0f}x speedup")
    print(f"  Recursive vs Iterative: {fft_recursive_times[-1]/fft_iterative_times[-1]:.2f}x ({'Iterative' if fft_iterative_times[-1] < fft_recursive_times[-1] else 'Recursive'} faster)")
    print(f"  (NumPy C-backend is {numpy_times[-1]/fft_iterative_times[-1]:.2e}x faster for reference)")
    print(f"\nNon-power-of-2 at N={sizes_non_pow2[-1]}:")
    print(f"  DFT vs Bluestein: {dft_times_non_pow2[-1]/fft_bluestein_times[-1]:.0f}x speedup")
    print(f"  (NumPy C-backend is {numpy_times_non_pow2[-1]/fft_bluestein_times[-1]:.2e}x faster for reference)")
