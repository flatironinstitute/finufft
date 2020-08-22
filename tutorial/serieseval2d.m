% Demo evaluating a 2D Fourier series at arbitrary points in quasi-optimal
% time via FINUFFT, in MATLAB. Barnett 6/3/20
clear; close all;

% we work in [0,2pi)^2. Set up a 2D Fourier series
kmax = 500;                   % bandlimit per dim
k = -kmax:kmax-1;             % freq indices in each dim
N1 = 2*kmax; N2 = N1;         % # modes in each dim
[k1 k2] = ndgrid(k,k);        % grid of freq indices
rng(0);
fk =  randn(N1,N2)+1i*randn(N1,N2);  % iid random complex mode data
% let's scale the amplitudes vs (k1,k2) to give a Gaussian random field with
% isotropic (periodized) Matern kernel (ie, covariance is Yukawa for alpha=1)...
k0 = 30;                     % freq scale parameter
alpha = 3.7;                 % power; alpha>2 to converge in L^2
fk = fk .* ((k1.^2+k2.^2)/k0^2 + 1).^(-alpha/2);     % sqrt of spectral density

M = 1e6; x = 2*pi*rand(1,M); y = 2*pi*rand(1,M);     % random target points
tol = 1e-9;
tic; c = finufft2d2(x,y,+1,tol,fk); toc   % evaluate Fourier series at (x,y)'s
% Elapsed time is 0.130059 seconds.

j = 1;                        % do math check on 1st target...
c1 = sum(sum(fk.*exp(1i*(k1*x(j)+k2*y(j)))));
abs(c1-c(j)) / norm(c,inf)

figure(1); clf;
jplot = 1:1e5;          % indices to plot
scatter(x(jplot),y(jplot),1.0,real(c(jplot)),'filled'); axis tight equal
xlabel('x'); ylabel('y'); colorbar; title('Re f(x,y)');
set(gcf,'paperposition',[0 0 8 7]); print -dpng ../docs/pics/fser2d.png
