import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from skimage import io
from skimage.restoration import denoise_tv_chambolle
import os

from utils_forward import load_psf, normalize

# --------------------------------------------------
# Paths
# --------------------------------------------------
figdir = r"D:\Projects\results\figures"
os.makedirs(figdir, exist_ok=True)

psf_path = os.path.join(figdir, 'psf_linear.mat')

# --------------------------------------------------
# Load PSF (shared utility)
# --------------------------------------------------
PSF = load_psf(psf_path)
PSF_flip = PSF[::-1, ::-1]   # adjoint PSF

# --------------------------------------------------
# Load observation
# --------------------------------------------------
y = io.imread(os.path.join(figdir, 'noisy.png'), as_gray=True)
y = normalize(y)

# --------------------------------------------------
# Initialization
# --------------------------------------------------
x = y.copy()
alpha = 0.15        # stable step size
lam = 0.02          # TV strength
niter = 50

# --------------------------------------------------
# Optimization loop (Poisson MAP + TV)
# --------------------------------------------------
for k in range(niter):

    Hx = fftconvolve(x, PSF, mode='same') + 1e-8

    # Poisson negative log-likelihood gradient
    grad = fftconvolve(1 - y / Hx, PSF_flip, mode='same')

    # Gradient descent step
    x -= alpha * grad
    x = np.clip(x, 0, None)

    # TV proximal step
    x = denoise_tv_chambolle(
        x,
        weight=lam,
        channel_axis=None
    )

    if k % 10 == 0:
        residual = np.mean(np.abs(Hx - y))
        print(f"Iter {k:02d} | Residual: {residual:.4e}")

# --------------------------------------------------
# Normalize & save
# --------------------------------------------------
x = normalize(x)

io.imsave(
    os.path.join(figdir, 'reconstructed_ML_TV.png'),
    (x * 255).astype(np.uint8)
)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(y, cmap='gray')
plt.title('Blurred + Noise')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(x, cmap='gray')
plt.title('MAP-TV Reconstruction')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(PSF, cmap='hot')
plt.title('PSF')
plt.axis('off')

plt.tight_layout()
plt.show()
