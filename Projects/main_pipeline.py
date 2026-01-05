import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.restoration import richardson_lucy, denoise_tv_chambolle
from skimage.metrics import mean_squared_error, structural_similarity
from scipy.signal import fftconvolve

# Import your shared utility
from utils_forward import load_psf, forward_model, normalize

# ==================================================
# CONFIGURATION & PATHS
# ==================================================
FIGDIR = r"D:\Projects\results\figures"
os.makedirs(FIGDIR, exist_ok=True)
PSF_PATH = os.path.join(FIGDIR, 'psf_linear.mat')

# Hyperparameters
RL_ITER = 12
MAP_TV_ITER = 50
MAP_TV_ALPHA = 0.15
MAP_TV_LAMBDA = 0.02

# ==================================================
# CORE FUNCTIONS
# ==================================================

def generate_ground_truth(N):
    """Generates the synthetic cross object."""
    gt = np.zeros((N, N))
    gt[N//2-20:N//2+20, N//2-5:N//2+5] = 1.0
    gt[N//2-5:N//2+5, N//2-20:N//2+20] = 1.0
    return gt

def run_map_tv(y, PSF, niter=50, alpha=0.15, lam=0.02):
    """Executes Maximum Likelihood with Total Variation regularization."""
    PSF_flip = PSF[::-1, ::-1]
    x = y.copy()
    for k in range(niter):
        Hx = fftconvolve(x, PSF, mode='same') + 1e-8
        grad = fftconvolve(1 - y / Hx, PSF_flip, mode='same')
        x -= alpha * grad
        x = np.clip(x, 0, None)
        x = denoise_tv_chambolle(x, weight=lam, channel_axis=None)
    return normalize(x)

# ==================================================
# MAIN PIPELINE
# ==================================================

def main():
    # 1. Load PSF
    print("Loading PSF...")
    PSF = load_psf(PSF_PATH)
    N = PSF.shape[0]

    # 2. Generate Ground Truth & Save
    print("Generating Ground Truth...")
    gt = generate_ground_truth(N)
    io.imsave(os.path.join(FIGDIR, 'ground_truth.png'), (gt * 255).astype(np.uint8))

    # 3. Apply Forward Model (Blur + Noise)
    print("Generating Noisy Observation...")
    y = forward_model(gt, PSF, noise_std=0.01)
    io.imsave(os.path.join(FIGDIR, 'noisy.png'), (y * 255).astype(np.uint8))

    # 4. Richardson-Lucy Deconvolution
    print(f"Running RL Deconvolution ({RL_ITER} iterations)...")
    rl_recon = richardson_lucy(y, PSF, num_iter=RL_ITER, clip=True)
    rl_recon = normalize(rl_recon)
    io.imsave(os.path.join(FIGDIR, 'reconstructed_RL.png'), (rl_recon * 255).astype(np.uint8))

    # 5. MAP-TV Deconvolution
    print(f"Running MAP-TV Deconvolution ({MAP_TV_ITER} iterations)...")
    tv_recon = run_map_tv(y, PSF, niter=MAP_TV_ITER, alpha=MAP_TV_ALPHA, lam=MAP_TV_LAMBDA)
    io.imsave(os.path.join(FIGDIR, 'reconstructed_MAP_TV.png'), (tv_recon * 255).astype(np.uint8))

    # 6. Quantitative Evaluation
    rmse_rl = np.sqrt(mean_squared_error(gt, rl_recon))
    ssim_rl = structural_similarity(gt, rl_recon, data_range=1.0)
    
    rmse_tv = np.sqrt(mean_squared_error(gt, tv_recon))
    ssim_tv = structural_similarity(gt, tv_recon, data_range=1.0)

    print("\n=== RESULTS SUMMARY ===")
    print(f"RL    -> RMSE: {rmse_rl:.4f}, SSIM: {ssim_rl:.4f}")
    print(f"MAP-TV -> RMSE: {rmse_tv:.4f}, SSIM: {ssim_tv:.4f}")

    # 7. Final Visualization
    plt.figure(figsize=(16, 4))
    titles = ['Ground Truth', 'Blurred/Noisy', f'RL (SSIM: {ssim_rl:.2f})', f'MAP-TV (SSIM: {ssim_tv:.2f})']
    imgs = [gt, y, rl_recon, tv_recon]
    
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 4, i+1)
        plt.imshow(img, cmap='gray' if i < 4 else 'hot')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'final_comparison.png'))
    plt.show()

if __name__ == "__main__":
    main()