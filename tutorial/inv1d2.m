% demo of inversion of the 1D type 2 NUFFT, ie fitting a Fourier series to
% function values at scattered points.
% Barnett 12/6/23
clear; close all; addpath utils; addpath ../matlab

N=3e5;   % num unknown coeffs (ie twice the max frequency)
ks = -floor(N/2) + (0:N-1);      % row vec of the frequency indices
M=2*N;           % number of scattered points on the periodic domain
wellcond=true;       % use to choose example
toep=true;           % use to choose matvec method
rng(0);                     % fix seed for reproducibility
if wellcond, x = 2*pi*((0:M-1)' + 2*rand(M,1))/M;   % jittered pts (will be well conditioned)
else, x = 2*pi*rand(M,1);             % iid random (will be ill conditioned)
end

% choose known (complex) coeffs, corresponding to above freqs indices
ftrue = (randn(N,1) + 1i*randn(N,1))/sqrt(N);

tol = 1e-12;
y = finufft1d2(x,+1,tol,ftrue);       % data = eval this Fourier series
%y = y + 1e-6*(randn(M,1) + 1i*randn(M,1));   % add noise (6-digit meas acc)

if N*M<1e7          % expensive dense direct solve, to check what it's doing
  A = exp(1i*x(:)*ks);            % outer prod
  fdirect = A\y;    % ouch!
  fprintf('rel l2 coeff err of dense solve: %.3g\n', norm(fdirect-ftrue)/norm(ftrue))
end

rhs = finufft1d1(x,y,-1,tol,N);      % compute A^* y

if ~toep   % iterative solve of normal eqns, each iteration a pair of NUFFTs
  tic
  [f,flag,relres,iter] = pcg(@(f) applyAHA(f,x,1e-6), rhs, 1e-6, N);
  fprintf('CG-NUFFT relres %.3g done in %d iters, %.3g s\n', relres,iter,toc)
else        % iterative solve of normal eqns, each iteration a padded FFT
  v = finufft1d1(x, ones(size(x)), -1, tol, 2*N-1);  % Toep vec, inds -(N-1):(N+1)
  vhat = fft([v;0]);
  tic
  [f,flag,relres,iter] = pcg(@(f) applyToep(f,vhat), rhs, 1e-6, N);
  fprintf('CG-Toep relres %.3g done in %d iters, %.3g s\n', relres,iter,toc)
end
  
yrecon = finufft1d2(x,+1,tol,f);
fprintf('\trel l2 resid of Af=y: %.3g\n', norm(yrecon-y)/norm(y))
fprintf('\trel l2 coeff err: %.3g\n', norm(f-ftrue)/norm(ftrue))
ng = 10*N; xg = 2*pi*(0:ng)/ng;       % fine plot grid
ytrueg = finufft1d2(xg,+1,1e-12,ftrue);  % eval true series on plot grid
yg = finufft1d2(xg,+1,1e-12,f);      % eval the recon series on plot grid
fprintf('\tabs max err: %.3g\n', norm(yg-ytrueg,inf))

figure();
subplot(2,1,1); plot(xg,real(ytrueg),'r-'); hold on;
plot(xg,real(yg),'b-');
plot(x,real(y),'k.','markersize',5); legend('true f(x)', 'recon series f(x)', 'data points (x_j,y_j)')
%axis([0 200/N -3*sqrt(N) 3*sqrt(N)]);   % show a few wiggles
dx = 2e-3;
if wellcond, x0=0; else, x0 = 0.920; end            % view domain start
axis([x0 x0+dx -3 3]);   % show a few wiggles
subplot(2,1,2); semilogy(xg,abs(ytrueg-yg),'b-'); hold on;
plot(x,abs(yrecon-y),'k.','markersize',5); legend('error in f(x)', 'error at data points')
axis tight; v=axis; axis([x0 x0+dx 1e-7 1])
set(gcf,'paperposition',[0 0 8 6]);
if wellcond, print -dpng ../docs/pics/inv1d2err_wellcond.png
else, print -dpng ../docs/pics/inv1d2err.png
end

% [other methods (CG adj nor eqns, PCG, not as good in my tests)...?]
