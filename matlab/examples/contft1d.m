% Demo of approximating continuous 1D Fourier transforms using the NUFFT,
% in MATLAB. Barnett 8/19/20
clear; close all; addpath utils

n = 50;
f = @(x) cos(n*acos(x));    % function is the T_{50} Chebychev poly on [-1,1]

% we want to compute hat{f}(k), its FT, at a bunch of k values (targets).
% Let's say they are generic at first
M=1e6;
kmax = 1e3;
k = kmax * (2*rand(1,M)-1);    % freq targets arbitrary, out to some high freq

tol = 1e-10;

N = 1000;
[xj,wj] = lgwt(N,-1,1);   % quadrature scheme for smooth funcs on [-1,1]
fj = f(xj);               % func values
fhat = finufft1d3(xj, wj.*fj, +1, tol, k);

N = 2000;
[xj,wj] = lgwt(N,-1,1);   % quadrature scheme for smooth funcs on [-1,1]
fj = f(xj);               % func values
fhat2 = finufft1d3(xj, wj.*fj, +1, tol, k);

norm(fhat-fhat2)/norm(fhat)

  