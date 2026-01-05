import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter


def load_psf(psf_mat_path, smooth_sigma=0.6):
    """
    Load and sanitize LINEAR PSF from MATLAB.
    """
    PSF = loadmat(psf_mat_path)['PSF']
    PSF = PSF.astype(np.float64)

    PSF = np.nan_to_num(PSF)
    PSF[PSF < 0] = 0

    if PSF.sum() == 0:
        raise ValueError("PSF sum is zero — check MATLAB PSF export")

    PSF /= PSF.sum()

    # Detector / bandwidth regularization
    if smooth_sigma is not None:
        PSF = gaussian_filter(PSF, sigma=smooth_sigma)
        PSF /= PSF.sum()

    return PSF


def forward_model(x, PSF, noise_std=0.01):
    """
    Forward optical model: blur + Gaussian noise.
    """
    blurred = fftconvolve(x, PSF, mode='same')

    noise = noise_std * np.random.randn(*blurred.shape)
    y = blurred + noise

    return np.clip(y, 0, 1)


def normalize(img):
    """
    Normalize image to [0,1]
    """
    img = np.nan_to_num(img)
    m = img.max()
    return img / m if m > 0 else img
