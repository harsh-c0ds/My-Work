import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.restoration import richardson_lucy

# -----------------------------
# Paths (Windows)
# -----------------------------
figdir = r"D:\results\figures"

# -----------------------------
# Load data
# -----------------------------
img = io.imread(f"{figdir}\\noisy.png", as_gray=True)
psf = io.imread(f"{figdir}\\psf.png", as_gray=True)

psf = psf / psf.sum()

# -----------------------------
# Richardson–Lucy deconvolution
# -----------------------------
recon = richardson_lucy(img, psf, iterations=30)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Noisy')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(recon, cmap='gray')
plt.title('Reconstruction')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(psf, cmap='hot')
plt.title('PSF')
plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# Save result
# -----------------------------
io.imsave(f"{figdir}\\reconstructed.png", recon)
