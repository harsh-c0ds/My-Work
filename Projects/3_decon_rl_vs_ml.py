import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from skimage import io
from skimage.restoration import richardson_lucy, denoise_tv_chambolle
from skimage.metrics import mean_squared_error, structural_similarity
import os

from utils_forward import load_psf, normalize

# --------------------------------------------------
# Paths
# --------------------------------------------------
figdir = r"D:\Projects\results\figures"
os.makedirs(figdir, exist_ok=True)

psf_path = os.path.join(figdir, 'psf_linear.mat')

# --------------------------------------------------
# Load ground truth
# --------------------------------------------------
gt = io.imread(os.path.join(figdir, 'ground_truth.png'), as_gray=True)
gt = normalize(gt)

# --------------------------------------------------
# Load PSF (shared utility)
# --------------------------------------------------
PSF = load_psf(psf_path)
PSF_flip = PSF[::-1, ::-1]

# --------------------------------------------------
# Load noisy image
# --------------------------------------------------
y = io.imread(os.path.join(figdir, 'noisy.png'), as_gray=True)
y = normalize(y)

# --------------------------------------------------
# RL reconstruction
# --------------------------------------------------
rl = richardson_lucy(
    y,
    PSF,
    num_iter=12,
    clip=True
)
rl = normalize(rl)

# --------------------------------------------------
# ML + TV reconstruction
# --------------------------------------------------
x = y.copy()
alpha = 0.15
lam = 0.02
niter = 50

for _ in range(niter):
    Hx = fftconvolve(x, PSF, mode='same') + 1e-8
    grad = fftconvolve(1 - y / Hx, PSF_flip, mode='same')

    x -= alpha * grad
    x = np.clip(x, 0, None)

    x = denoise_tv_chambolle(
        x,
        weight=lam,
        channel_axis=None
    )

x = normalize(x)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
rmse_rl = np.sqrt(mean_squared_error(gt, rl))
rmse_ml = np.sqrt(mean_squared_error(gt, x))

ssim_rl = structural_similarity(gt, rl, data_range=1.0)
ssim_ml = structural_similarity(gt, x, data_range=1.0)

print("=== Quantitative Comparison ===")
print(f"RL     → RMSE = {rmse_rl:.4f}, SSIM = {ssim_rl:.4f}")
print(f"ML+TV  → RMSE = {rmse_ml:.4f}, SSIM = {ssim_ml:.4f}")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(gt, cmap='gray')
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(rl, cmap='gray')
plt.title(f'RL\nRMSE={rmse_rl:.3f}, SSIM={ssim_rl:.3f}')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(x, cmap='gray')
plt.title(f'ML+TV\nRMSE={rmse_ml:.3f}, SSIM={ssim_ml:.3f}')
plt.axis('off')

plt.tight_layout()
plt.show()
