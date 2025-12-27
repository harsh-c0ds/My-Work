clear; close all; clc;

outdir = 'D:\Projects\results\figures';
load(fullfile(outdir,'psf_linear.mat'));

N = size(PSF,1);

%% Object (point + structure)
obj = zeros(N);
obj(N/2,N/2) = 1;        % point source
obj = obj + 0.1*phantom(N);

obj = obj / max(obj(:));

%% Convolution
blurred = real(ifft2(fft2(obj).*fft2(ifftshift(PSF))));

%% Poisson noise
blurred = blurred / max(blurred(:));
noisy = imnoise(blurred,'poisson');

%% Save
imwrite(noisy, fullfile(outdir,'noisy.png'));

%% Display
figure;
subplot(1,3,1), imshow(obj,[]), title('Object')
subplot(1,3,2), imshow(blurred,[]), title('Blurred')
subplot(1,3,3), imshow(noisy,[]), title('Noisy')
