% Demo of approximating continuous 1D Fourier transforms using the NUFFT and
% a quadrature rule, in MATLAB. Barnett 8/20/20
clear; close all; addpath utils

a=0; b=1;                    % interval
f = @(x) cos(50*x.^2);      % smooth function defined on (a,b)

figure(1); xx=linspace(a-.3,b+.3,3e3); plot(xx,f(xx).*(xx>a & xx<b),'-');
hold on; [xj,wj] = lgwt(200,a,b); plot(xj,f(xj),'.','markersize',10);
axis tight; v=axis; v(3:4)=1.2*v(3:4); axis(v);
xlabel('x'); ylabel('f(x)');
title('function whose FT is sought, and Gauss-Legendre nodes'); drawnow
set(gcf,'paperposition',[0 0 10 2]); print -dpng ../../docs/pics/contft1d.png

M = 1e6;                   % # targets we want to compute the FT hat{f}(k) at
kmax = 500;

% ---------- nonuniform targets (type 3) ------------------------------------

k = kmax * (2*rand(1,M)-1);    % freq targets arbitrary, out to some high freq

N = 200;
[xj,wj] = lgwt(N,a,b);   % N-node quadrature scheme for smooth funcs on (a,b)

tol = 1e-10;
tic;
fhat = finufft1d3(xj, f(xj).*wj, +1, tol, k);
toc

figure(2); plot(k, [real(fhat),imag(fhat)], '.');
v=axis; v(1)=0; axis(v); xlabel('$k$','interpreter','latex');
h=legend('Re $\hat{f}(k)$', 'Im $\hat{f}(k)$'); set(h,'interpreter','latex');
set(gcf,'paperposition',[0 0 10 2]); print -dpng ../../docs/pics/contft1dans.png

Ns = 100:10:220;         % convergence study in N
for i=1:numel(Ns), N=Ns(i);
  [xj,wj] = lgwt(N,a,b);   % N-node quadrature scheme for smooth funcs on (a,b)
  fhats{i} = finufft1d3(xj, f(xj).*wj, +1, tol, k);
end
f0 = norm(fhats{end},inf);  % compute rel sup norm of fhat vs highest-N case...
for i=1:numel(Ns)-1, errsup(i) = norm(fhats{i}-fhats{end},inf)/f0; end
figure(2); semilogy(Ns(1:end-1),errsup,'+-'); title('N-convergence of FT');
xlabel('N'); ylabel('sup error in $\hat{f}$','interpreter','latex');
axis tight;
set(gcf,'paperposition',[0 0 5 3]); print -dpng ../../docs/pics/contft1dN.png


% ---------- uniform targets (type 1) ------------------------------------  k = 

k = kmax * (-M:(M-1))/M;   % particular uniform grid, spacing 2*kmax/M

N = 200;
[xj,wj] = lgwt(N,a,b);   % N-node quadrature scheme for smooth funcs on (a,b)
cj = f(xj)*wj;

fhat3 = finufft1d3(xj, cj, +1, tol, k);   % old type-3, to check

sc = 2*pi/(2*kmax/M);    % x rescale factor
tic;
fhat = finufft1d1(sc*xj, cj, +1, tol);
toc
norm(fhat-fhat3,inf)

% if your grid has different offset, you will need other simple affine
% transforms
