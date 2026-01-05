import numpy as np
import os
from skimage import io
from scipy.io import loadmat
from scipy.signal import fftconvolve
from skimage.restoration import richardson_lucy

# --------------------------------------------------
# Paths
# --------------------------------------------------
base = r"D:\Projects"
figdir = os.path.join(base,'results','figures')
dsdir = os.path.join(base,'dataset')

for split in ['train','val']:
    for sub in ['input','target']:
        os.makedirs(os.path.join(dsdir,split,sub), exist_ok=True)

# --------------------------------------------------
# Load PSF
# --------------------------------------------------
PSF = loadmat(os.path.join(figdir,'psf_linear.mat'))['PSF']
PSF = PSF.astype(np.float64)
PSF /= PSF.sum()

# --------------------------------------------------
# Generate samples
# --------------------------------------------------
def make_object(N):
    x = np.zeros((N,N))
    cx, cy = np.random.randint(50,200,size=2)
    x[cx-10:cx+10, cy-2:cy+2] = 1
    x[cx-2:cx+2, cy-10:cy+10] = 1
    return x

np.random.seed(0)

for i in range(120):

    gt = make_object(256)
    blur = fftconvolve(gt, PSF, mode='same')
    noisy = np.clip(blur + 0.01*np.random.randn(*blur.shape),0,1)

    rl = richardson_lucy(noisy, PSF, num_iter=12, clip=True)

    split = 'train' if i < 100 else 'val'

    io.imsave(
        os.path.join(dsdir,split,'input',f'{i:03d}.png'),
        (rl*255).astype(np.uint8)
    )
    io.imsave(
        os.path.join(dsdir,split,'target',f'{i:03d}.png'),
        (gt*255).astype(np.uint8)
    )

print("Dataset generation complete.")
