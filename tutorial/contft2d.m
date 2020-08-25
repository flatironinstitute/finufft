% Demo of approximating 2D Fourier transform of a polar characteristic function
% using the NUFFT and a planar quadrature rule, in MATLAB. Barnett 8/21/20
clear; close all; addpath utils

g = @(t) 1 + 0.3*cos(3*t);                             % boundary shape
% build the quadrature...
n = 300;                                               % # theta nodes
t = 2*pi*(1:n)/n; wt = (2*pi/n);                       % theta nodes, const weights
bx = cos(t).*g(t); by = sin(t).*g(t);                  % boundary points
m = 70;                                               % # r nodes
[xr,wr] = lgwt(m,0,1);                                 % rule for (0,1)
xj = nan(n*m,1); yj = xj; wj = xj;
for i=1:n                                              % loop over angles
  r = g(t(i)); jj = (1:m) + (i-1)*m;                   % this radius; index list
  xj(jj) = cos(t(i))*r*xr; yj(jj) = sin(t(i))*r*xr;    % line of nodes
  wj(jj) = wt*r^2*xr.*wr;            % theta weight times rule for r.dr on (0,r)
end

figure(1); clf; plot([bx bx(1)],[by by(1)],'-'); hold on; plot(xj,yj,'.'); axis equal;
xlabel('x_1'); ylabel('x_2'); title('Domain \Omega and nodes for it (case n=50, m=20)');
%set(gcf,'paperposition',[0 0 6 5]); print -dpng ../docs/pics/contft2dnodes.png

% choose freq target grid and evaluate the FT on it...
kmax = 100;
M1 = 1e3;                      % frequency target grid size in each dimension
dk = 2*kmax/M1;
k1 = dk * (-M1/2:(M1/2-1));    % same 1D freq grid as before
tol = 1e-9;
tic
fhat = finufft2d1(dk*xj, dk*yj, wj, +1, tol, M1, M1);   % M1^2 output nodes
toc

figure(2); clf; imagesc(k1,k1,log10(abs(fhat))'); axis xy equal tight;
%set(gca,'ydir','normal');
caxis([-4 .4]);
colorbar; xlabel('k_1'); ylabel('k_2'); title('2D Fourier transform $\log_{10} |\hat{f}({\bf k})|$, where ${\bf k}:=(k_1,k_2)$','interpreter','latex');
set(gcf,'paperposition',[0 0 6 6]); print -dpng ../docs/pics/contft2dans.png

% convergence study... (repeats most of the above): n=70, m=280 good for 1e-10
ms = 50:10:120;
clear fhats errsup
for i=1:numel(ms), m = ms(i); n=300;
  t = 2*pi*(1:n)/n; wt = (2*pi/n);                       % theta nodes, const weight
  bx = cos(t).*g(t); by = sin(t).*g(t);                  % boundary points
  [xr,wr] = lgwt(m,0,1);                                 % rule for (0,1)
  xj = nan(n*m,1); yj = xj; wj = xj;
  for l=1:n                                              % loop over angles
    r = g(t(l)); jj = (1:m) + (l-1)*m;                   % this radius; index list
    xj(jj) = cos(t(l))*r*xr; yj(jj) = sin(t(l))*r*xr;    % line of nodes
    wj(jj) = wt*r^2*xr.*wr;            % theta weight times rule for r.dr on (0,r)
  end
  fhat = finufft2d1(dk*xj, dk*yj, wj, +1, tol, M1, M1);   % rescale x, y as in 1D
  fhats{i} = fhat(:);        % as col vec
end
f0 = norm(fhats{end},inf);
for i=1:numel(ms)-1, errsup(i) = norm(fhats{i}-fhats{end},inf)/f0; end
disp([ms(1:end-1); errsup]')
% we see m=90 (n=270) has converged
