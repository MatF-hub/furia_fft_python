import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import core

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = sorted(glob.glob(os.path.join(base_dir, "images", "image_*.png")))
    highpass_cutoff = 5

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("L").resize((128, 128))
        img_array = np.array(img, dtype=complex)

        X = core.fft(img_array)
        magnitude = np.log1p(np.abs(X))
        threshold = 5e-2 * np.abs(X).max()
        
        # Image compressed
        p = 0.30  # Keep 30% of coefficients
        k = int(p * X.size)

        # Find the k largest magnitudes
        indices = np.argpartition(magnitude.ravel(), -k)[-k:]

        # Create a sparse FFT array
        X_compressed = np.zeros_like(X)
        X_compressed.ravel()[indices] = X.ravel()[indices]
        magnitude_compressed = np.log1p(np.abs(X_compressed))
        ratio = np.count_nonzero(np.abs(X_compressed))/np.count_nonzero(np.abs(X))

        # reconstruct from compressed
        img_reconstructed = core.ifft(X_compressed).real
        error = np.max(np.abs(img_array - img_reconstructed))

        axes[row, 0].imshow(np.real(img_array), cmap="gray")
        axes[row, 0].set_title(f"Original Image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(magnitude, cmap="hot")
        axes[row, 1].set_title(f"FFT Magnitude:\n(N° coeff.: {np.count_nonzero(np.abs(X))})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(magnitude_compressed, cmap="hot")
        axes[row, 2].set_title(f"FFT Compressed Magnitude:\n(N° coeff.: {np.count_nonzero(np.abs(X_compressed))})")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(np.real(img_reconstructed), cmap="gray")
        axes[row, 3].set_title(f"Reconstructed (err: {error:.2e})")
        axes[row, 3].axis("off")

        print(f"{img_path}: error = {error:.2e}")

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_compression.png"), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'image_filtering.png')}")
    plt.show()
