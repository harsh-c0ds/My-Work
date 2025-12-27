import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import fftconvolve
from skimage import io
from skimage.restoration import denoise_tv_chambolle
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
figdir = r"D:\Projects\results\figures"

# --------------------------------------------------
# Load PSF
# --------------------------------------------------
PSF = loadmat(os.path.join(figdir,'psf_linear.mat'))['PSF']
PSF = PSF.astype(np.float64)
PSF /= PSF.sum()

PSF_flip = PSF[::-1, ::-1]   # adjoint PSF

# --------------------------------------------------
# Load image
# --------------------------------------------------
y = io.imread(os.path.join(figdir,'noisy.png'), as_gray=True)
y = y.astype(np.float64)
y /= y.max()

# --------------------------------------------------
# Initialization
# --------------------------------------------------
x = y.copy()        # initial guess
alpha = 0.8         # step size
lam = 0.02          # TV weight (IMPORTANT)
niter = 50

# --------------------------------------------------
# ML-TV optimization loop
# --------------------------------------------------
for k in range(niter):

    # Forward model
    Hx = fftconvolve(x, PSF, mode='same') + 1e-8

    # Poisson log-likelihood gradient
    grad = fftconvolve(1 - y / Hx, PSF_flip, mode='same')

    # Gradient descent step
    x = x - alpha * grad

    # Positivity constraint
    x = np.clip(x, 0, None)

    # TV proximal step
    x = denoise_tv_chambolle(
        x,
        weight=lam,
        channel_axis=None
    )

    if k % 10 == 0:
        print(f"Iteration {k}/{niter}")

# --------------------------------------------------
# Normalize & save
# --------------------------------------------------
x /= x.max()
io.imsave(
    os.path.join(figdir,'reconstructed_ML_TV.png'),
    (x*255).astype(np.uint8)
)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(y, cmap='gray')
plt.title('Noisy')

plt.subplot(1,3,2)
plt.imshow(x, cmap='gray')
plt.title('ML + TV Reconstruction')

plt.subplot(1,3,3)
plt.imshow(PSF, cmap='hot')
plt.title('PSF')

plt.axis('off')
plt.tight_layout()
plt.show()
