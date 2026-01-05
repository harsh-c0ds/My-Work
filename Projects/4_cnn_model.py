import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io

from utils_forward import (
    load_psf,
    forward_model,
    normalize
)

from skimage.restoration import richardson_lucy, denoise_tv_chambolle
from scipy.signal import fftconvolve

# ==================================================
# CONFIG
# ==================================================
FIGDIR = r"D:\Projects\results\figures"
USE_TV = True
USE_RL = True

os.makedirs(FIGDIR, exist_ok=True)

# ==================================================
# Ground truth
# ==================================================
def generate_gt(N):
    gt = np.zeros((N, N))
    gt[N//2-20:N//2+20, N//2-3:N//2+3] = 1
    gt[N//2-3:N//2+3, N//2-20:N//2+20] = 1
    return gt

# ==================================================
# RL Deconvolution
# ==================================================
def rl_decon(y, PSF, niter=12):
    rl = richardson_lucy(y, PSF, num_iter=niter, clip=True)
    return normalize(rl)

# ==================================================
# MAP-TV
# ==================================================
def map_tv(y, PSF, niter=50, alpha=0.15, lam=0.02):
    PSF_flip = PSF[::-1, ::-1]
    x = y.copy()

    for _ in range(niter):
        Hx = fftconvolve(x, PSF, mode='same') + 1e-8
        grad = fftconvolve(1 - y / Hx, PSF_flip, mode='same')
        x -= alpha * grad
        x = np.clip(x, 0, None)
        x = denoise_tv_chambolle(x, weight=lam, channel_axis=None)

    return normalize(x)

# ==================================================
# MAIN
# ==================================================
def main():

    # Load PSF (shared utility)
    PSF = load_psf(os.path.join(FIGDIR, 'psf_linear.mat'))
    N = PSF.shape[0]

    # Ground truth
    gt = generate_gt(N)
    io.imsave(os.path.join(FIGDIR, 'ground_truth.png'),
              (gt * 255).astype(np.uint8))

    # Forward model
    y = forward_model(gt, PSF, noise_std=0.01)
    io.imsave(os.path.join(FIGDIR, 'noisy.png'),
              (y * 255).astype(np.uint8))

    # Reconstructions
    rl = tv = None

    if USE_RL:
        rl = rl_decon(y, PSF)
        io.imsave(os.path.join(FIGDIR, 'reconstructed_RL.png'),
                  (rl * 255).astype(np.uint8))

    if USE_TV:
        tv = map_tv(y, PSF)
        io.imsave(os.path.join(FIGDIR, 'reconstructed_MAP_TV.png'),
                  (tv * 255).astype(np.uint8))

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    plt.figure(figsize=(14,4))
    plt.subplot(1,4,1); plt.imshow(gt, cmap='gray'); plt.title('GT'); plt.axis('off')
    plt.subplot(1,4,2); plt.imshow(y, cmap='gray'); plt.title('Noisy'); plt.axis('off')
    plt.subplot(1,4,3); plt.imshow(rl, cmap='gray'); plt.title('RL'); plt.axis('off')
    plt.subplot(1,4,4); plt.imshow(tv, cmap='gray'); plt.title('MAP-TV'); plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
