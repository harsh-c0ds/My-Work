clear; close all; clc;

%% Parameters
lambda = 550e-9;      % wavelength (m)
NA = 0.1;
N = 256;              % grid size
dx = 5e-6;            % pupil sampling (m)

outdir = 'D:\Projects\results\figures';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% --------------------------------------------------
%% Ground truth object (LINEAR)
%% --------------------------------------------------
gt = phantom('Modified Shepp-Logan', N);
gt = gt / max(gt(:));     % normalize to [0,1]

% Save ground truth
imwrite(gt, fullfile(outdir,'ground_truth.png'));
save(fullfile(outdir,'ground_truth.mat'),'gt');

%% --------------------------------------------------
%% Frequency grid (pupil plane)
%% --------------------------------------------------
fx = (-N/2:N/2-1)/(N*dx);
[Fx, Fy] = meshgrid(fx, fx);
rho = sqrt(Fx.^2 + Fy.^2);

fc = NA/lambda;
pupil = double(rho <= fc);

%% --------------------------------------------------
%% Aberrations (simple Zernike-like)
%% --------------------------------------------------
W_defocus = 2*pi*0.3*(rho/fc).^2;
W_astig   = 2*pi*0.2*(Fx.^2 - Fy.^2)/fc^2;
W = (W_defocus + W_astig) .* pupil;

%% --------------------------------------------------
%% Complex pupil
%% --------------------------------------------------
P = pupil .* exp(1i*W);

%% --------------------------------------------------
%% PSF (LINEAR — this is what Python MUST use)
%% --------------------------------------------------
PSF = abs(fftshift(fft2(P))).^2;
PSF = PSF / sum(PSF(:));   % normalize energy

% Save linear PSF for Python
save(fullfile(outdir,'psf_linear.mat'),'PSF');

%% --------------------------------------------------
%% Log-scaled PSF (VISUALIZATION ONLY)
%% --------------------------------------------------
PSF_log = log10(PSF + eps);
imwrite(mat2gray(PSF_log), fullfile(outdir,'psf_log.png'));

%% --------------------------------------------------
%% Display
%% --------------------------------------------------
figure;
subplot(1,3,1)
imagesc(gt), axis image off
title('Ground Truth')

subplot(1,3,2)
imagesc(PSF), axis image off
title('Linear PSF')

subplot(1,3,3)
imagesc(PSF_log), axis image off
title('Log-scaled PSF')
colormap hot
