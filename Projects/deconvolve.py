import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.restoration import richardson_lucy
import os

# -------------------------------------------------
# Paths
# -------------------------------------------------
figdir = r"D:\Projects\results\figures"

# -------------------------------------------------
# Load LINEAR PSF from MATLAB
# -------------------------------------------------
PSF = loadmat(os.path.join(figdir, 'psf_linear.mat'))['PSF']
PSF = PSF.astype(np.float64)

# Safety checks
PSF = np.nan_to_num(PSF)
PSF[PSF < 0] = 0
PSF /= PSF.sum()

# -------------------------------------------------
# Physical regularization: PSF smoothing
# -------------------------------------------------
# Models detector pixel size / bandwidth limit
PSF = gaussian_filter(PSF, sigma=0.6)
PSF /= PSF.sum()

# -------------------------------------------------
# Load blurred + noisy image
# -------------------------------------------------
img = io.imread(os.path.join(figdir, 'noisy.png'), as_gray=True)
img = img.astype(np.float64)
img = np.nan_to_num(img)
img /= img.max()

# -------------------------------------------------
# Richardson–Lucy (EARLY STOPPING = REGULARIZATION)
# -------------------------------------------------
recon = richardson_lucy(
    img,
    PSF,
    num_iter=12,     # <<< IMPORTANT: do NOT increase blindly
    clip=True
)

recon = np.clip(recon, 0, 1)

# -------------------------------------------------
# Save result (PNG-safe)
# -------------------------------------------------
out_path = os.path.join(figdir, 'reconstructed.png')
io.imsave(out_path, (recon * 255).astype(np.uint8))

print(f"Reconstruction saved to: {out_path}")

# -------------------------------------------------
# Visualization
# -------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Noisy')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(recon, cmap='gray')
plt.title('Reconstructed (regularized)')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(PSF, cmap='hot')
plt.title('PSF')
plt.axis('off')

plt.tight_layout()
plt.show()
