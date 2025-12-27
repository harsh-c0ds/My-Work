clear; close all; clc;

%% Parameters
lambda = 550e-9;      % wavelength (m)
NA = 0.1;
N = 256;              % grid size
dx = 5e-6;            % pupil sampling (m)

outdir = 'D:\Projects\results\figures';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% Frequency grid
fx = (-N/2:N/2-1)/(N*dx);
[Fx, Fy] = meshgrid(fx, fx);
rho = sqrt(Fx.^2 + Fy.^2);

fc = NA/lambda;
pupil = double(rho <= fc);

%% Aberrations (Zernike-like, simple)
W_defocus = 2*pi*0.3*(rho/fc).^2;
W_astig   = 2*pi*0.2*(Fx.^2 - Fy.^2)/fc^2;
W = (W_defocus + W_astig) .* pupil;

%% Complex pupil
P = pupil .* exp(1i*W);

%% PSF (LINEAR!)
PSF = abs(fftshift(fft2(P))).^2;
PSF = PSF / sum(PSF(:));   % normalize energy

%% Save linear PSF for Python
save(fullfile(outdir,'psf_linear.mat'),'PSF');

%% Save log-PSF for visualization only
PSF_log = log10(PSF + eps);
imwrite(mat2gray(PSF_log), fullfile(outdir,'psf_log.png'));

%% Display
figure;
subplot(1,2,1)
imagesc(PSF), axis image
title('Linear PSF')

subplot(1,2,2)
imagesc(PSF_log), axis image
title('Log-scaled PSF')
colormap hot
