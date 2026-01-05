clear; close all; clc;

lambda = 550e-9;
NA = 0.1;
N = 256;
dx = 5e-6;

fx = (-N/2:N/2-1) / (N*dx);
[Fx, Fy] = meshgrid(fx, fx);
rho = sqrt(Fx.^2 + Fy.^2);

fc = NA / lambda;
pupil = double(rho <= fc);

PSF = abs(fftshift(fft2(pupil))).^2;
PSF = PSF / sum(PSF(:));

figure;
imagesc(log10(PSF + eps));
axis image;
colormap hot;
colorbar;
title('Log-scaled PSF');

mkdir('../results');
mkdir('../results/figures');
imwrite(mat2gray(log10(PSF + eps)), '../results/figures/psf.png');
