% Demo of approximating 1D Fourier transforms of functions using the NUFFT and
% a quadrature rule, in MATLAB. Barnett 8/20/20
clear; close all; addpath utils

a=0; b=1;                    % interval
f = @(x) cos(50*x.^2);      % smooth function defined on (a,b)

figure(1); xx=linspace(a-.3,b+.3,3e3); plot(xx,f(xx).*(xx>a & xx<b),'-');
hold on; [xj,wj] = lgwt(200,a,b); plot(xj,f(xj),'.','markersize',10);
axis tight; v=axis; v(3:4)=1.2*v(3:4); axis(v);
xlabel('x'); ylabel('f(x)');
title('function whose FT is sought, and Gauss-Legendre nodes'); drawnow
set(gcf,'paperposition',[0 0 10 2]); print -dpng ../docs/pics/contft1d.png

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
set(gcf,'paperposition',[0 0 10 2]); print -dpng ../docs/pics/contft1dans.png

Ns = 100:10:220;         % convergence study in N
for i=1:numel(Ns), N=Ns(i);
  [xj,wj] = lgwt(N,a,b);   % N-node quadrature scheme for smooth funcs on (a,b)
  fhats{i} = finufft1d3(xj, f(xj).*wj, +1, tol, k);
end
f0 = norm(fhats{end},inf);  % compute rel sup norm of fhat vs highest-N case...
for i=1:numel(Ns)-1, errsup(i) = norm(fhats{i}-fhats{end},inf)/f0; end
clear fhats
figure(3); semilogy(Ns(1:end-1),errsup,'+-'); title('N-convergence of FT');
xlabel('N'); ylabel('sup error in $\hat{f}$','interpreter','latex');
axis tight;
set(gcf,'paperposition',[0 0 5 3]); print -dpng ../docs/pics/contft1dN.png


% ---------- uniform targets (type 1) ------------------------------------

dk = 2*kmax/M;              % spacing of target k grid
k = dk * (-M/2:(M/2-1));    % a particular uniform M-grid of this spacing
N = 200;
[xj,wj] = lgwt(N,a,b);      % N-node quadrature scheme for smooth funcs on (a,b)
cj = f(xj).*wj;             % strengths (will be same for type 1)
tic;                        % faster new way: dk stretch to 1, equiv squeeze xj
fhat = finufft1d1(dk*xj, cj, +1, tol, M);   % type 1, requesting M modes
toc
fhat3 = finufft1d3(xj, cj, +1, tol, k);   % old type-3, to check
norm(fhat-fhat3,inf)

% ----------- singular function with uniform targets ----------------------

f = @(x) cos(50*x.^2)./sqrt(x);    % singular function defined on (0,1), zero elsewhere
Ns = 180:20:240;                   % N values to check convergence
for i=1:numel(Ns), N=Ns(i);
  [xj,wj] = lgwt(N,a,b);           % N-node scheme for smooth funcs on (0,1)
  wj = 2*xj.*wj; xj = xj.*xj;      % convert to rule for -1/2 power singularity
  fhats{i} = finufft1d1(dk*xj, f(xj).*wj, +1, tol, M);    % type 1 as above
end
f0 = norm(fhats{end},inf);   % compute rel sup norm of fhat vs highest-N case
for i=1:numel(Ns)-1, errsup(i) = norm(fhats{i}-fhats{end},inf)/f0; end
disp([Ns(1:3); errsup(1:3)]')
fhat = fhats{end}; figure(4); plot(k, [real(fhat),imag(fhat)], '.');
axis([0 500 -.05 .7]); xlabel('$k$','interpreter','latex');
h=legend('Re $\hat{f}(k)$', 'Im $\hat{f}(k)$'); set(h,'interpreter','latex');
set(gcf,'paperposition',[0 0 9 3]); print -dpng ../docs/pics/contft1dsing.png

% if your grid has a different offset, you will need to shift to be like the
% above (if M even), then post-apply a phase factor
