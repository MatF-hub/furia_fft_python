import core
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

if __name__ == "__main__":
    image_paths = sorted(glob.glob("images/image_*.png"))
    highpass_cutoff = 5

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("L").resize((128, 128))
        img_array = np.array(img, dtype=complex)

        X = core.fft(img_array)
        magnitude = np.log1p(np.abs(X))
        img_reconstructed = core.ifft(X)
        error = np.max(np.abs(img_array - img_reconstructed))

        # Define high pass filter for corner detection
        h, w = X.shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        H = 1.0 - np.exp(-(dist**2) / (2 * highpass_cutoff**2))

        X_shifted = np.fft.fftshift(X)   #Use numpy fftshift just for lazyness
        X_filtered = np.fft.ifftshift(X_shifted * H) #Use numpy fftshift just for lazyness
        img_edges = core.ifft(X_filtered)

        axes[row, 0].imshow(np.abs(img_array), cmap="gray")
        axes[row, 0].set_title(f"Original: {img_path.split('/')[-1]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(magnitude, cmap="hot")
        axes[row, 1].set_title("FFT Magnitude")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(np.abs(img_reconstructed), cmap="gray")
        axes[row, 2].set_title(f"Reconstructed (err: {error:.2e})")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(np.abs(img_edges), cmap="gray")
        axes[row, 3].set_title(f"Edge Detection\n(C={highpass_cutoff})")
        axes[row, 3].axis("off")

        print(f"{img_path}: error = {error:.2e}")

    os.makedirs('output', exist_ok=True)
    plt.tight_layout()
    plt.savefig('output/image_filtering.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to output/image_filtering.png")
    plt.show()
