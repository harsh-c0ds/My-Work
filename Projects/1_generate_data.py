import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.restoration import richardson_lucy
import os

from utils_forward import load_psf, forward_model, normalize

# -------------------------------------------------
# Paths
# -------------------------------------------------
figdir = r"D:\Projects\results\figures"
os.makedirs(figdir, exist_ok=True)

psf_path = os.path.join(figdir, 'psf_linear.mat')

# -------------------------------------------------
# Load PSF
# -------------------------------------------------
PSF = load_psf(psf_path)

# -------------------------------------------------
# Ground truth (synthetic object)
# -------------------------------------------------
N = PSF.shape[0]
gt = np.zeros((N, N))

gt[N//2-20:N//2+20, N//2-5:N//2+5] = 1.0
gt[N//2-5:N//2+5, N//2-20:N//2+20] = 1.0

io.imsave(
    os.path.join(figdir, 'ground_truth.png'),
    (gt * 255).astype(np.uint8)
)

# -------------------------------------------------
# Forward model
# -------------------------------------------------
noisy = forward_model(gt, PSF, noise_std=0.01)

io.imsave(
    os.path.join(figdir, 'noisy.png'),
    (noisy * 255).astype(np.uint8)
)

# -------------------------------------------------
# Richardson–Lucy deconvolution (baseline)
# -------------------------------------------------
recon = richardson_lucy(
    noisy,
    PSF,
    num_iter=12,
    clip=True
)

recon = normalize(recon)

io.imsave(
    os.path.join(figdir, 'reconstructed.png'),
    (recon * 255).astype(np.uint8)
)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(gt, cmap='gray')
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(noisy, cmap='gray')
plt.title('Blurred + Noise')
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(recon, cmap='gray')
plt.title('RL Reconstruction')
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(PSF, cmap='hot')
plt.title('PSF')
plt.axis('off')

plt.tight_layout()
plt.show()

print("All outputs saved to:", figdir)
