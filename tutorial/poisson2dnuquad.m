% Demo solving Poisson eqn on nonuniform grid in [0,2pi)^2 periodic box.
% The NU grid has a quadrature rule associated with it, meaning that no
% inverse-NUFFT is needed. We use simple tensor-prod grid to illustrate.
% Barnett 2/14/20
clear; close all;

% source function in [0,2pi)^2 (eps-periodic, zero-mean to be consistent)
w0 = 0.1;  % width of bumps
src = @(x,y) exp(-0.5*((x-1).^2+(y-2).^2)/w0^2)-exp(-0.5*((x-3).^2+(y-5).^2)/w0^2);

% A) Solve -\Delta u = f, on regular grid via FFT, to warm up..................
disp('FFT on reg grid...');
ns = 40:20:120;   % convergence study of grid points per side
ns = 2*ceil(ns/2);      % insure even
for i=1:numel(ns), n = ns(i);
  x = 2*pi*(0:n-1)/n;   % grid
  [xx yy] = ndgrid(x,x);   % ordering: x fast, y slow
  f = src(xx,yy);         % eval source on grid
  fhat = ifft2(f);        % Fourier series coeffs by Euler-F projection
  k = [0:n/2-1 -n/2:-1];   % Fourier mode grid
  [kx ky] = ndgrid(k,k);
  kfilter = 1./(kx.^2+ky.^2);    % inverse -Laplacian in Fourier space
  kfilter(1,1) = 0;   % kill the average, zero mode (even if inconsistent)
  kfilter(n/2+1,:) = 0; kfilter(:,n/2+1) = 0; % kill n/2 modes since non-symm
  u = fft2(kfilter.*fhat);   % do it
  u = real(u);
  fprintf('n=%d:\t\tu(0,0) = %.15e\n',n,u(1,1))   % check conv at a point
end
% we observe spectral convergence to 14 digits
%fhat(37,15)               % check a mode
figure(1); subplot(1,2,1); imagesc(log10(abs(fhat))); axis equal xy tight; colorbar; title('FFT: log_{10} abs fhat');
figure(2); subplot(1,2,1); imagesc(x,x,f'); colorbar('southoutside');  % show it
axis xy equal tight; title('source term f'); xlabel('x'); ylabel('y');
subplot(1,2,2); imagesc(x,x,u'); colorbar('southoutside');
axis xy equal tight; title('FFT solution u'); xlabel('x'); ylabel('y'); drawnow
set(gcf,'paperposition',[0 0 10 5]); print -dpng ../docs/pics/pois_fft.png

% B) solve on general nonuniform tensor prod grid............................
tol = 1e-12;       % precision
fprintf('NUFFT on tensor-prod NU (known-quadrature) grid... tol=%g\n',tol);
% smooth maps from [0,2pi) -> [0,2pi) that generate 1d quadrature rules
mapx = @(t) t + 0.5*sin(t); mapxp = @(t) 1 + 0.5*cos(t);       % its exact deriv
mapy = @(t) t + 0.4*sin(2*t); mapyp = @(t) 1 + 0.8*cos(2*t);
% (note I chose map(0)=0 so that the origin remains "on grid" for conv test)

ns = 80:40:240; ns = 2*ceil(ns/2);  % convergence study of grid points per side
for i=1:numel(ns), n = ns(i);
  t = 2*pi*(0:n-1)/n;         % unif grid
  xm = mapx(t); ym = mapy(t);  % actual 1d grids
  xw = mapxp(t); yw = mapyp(t);  % 1d quadrature weight jacobian factors
  ww = xw'*yw / n^2;      % 2d quadr weights, including 1/(2pi)^2 in E-F integr
  [xx yy] = ndgrid(xm,ym);       % 2d NU pts
  f = src(xx,yy);
  if i==1, figure(3); mesh(xm,ym,f'); view(2); axis equal; axis([0 2*pi 0 2*pi]); title('f on mesh'); end
  Nk = 0.5*n; Nk = 2*ceil(Nk/2);  % modes to trust due to quadr err (dep on map)
  o.modeord = 1;      % fft output mode ordering
  fhat = finufft2d1(xx(:),yy(:),f(:).*ww(:),1,tol,Nk,Nk,o);  % do E-F
  % note: since tensor-prod case, could do stack of 1d1 NUFFTs faster :)
  k = [0:Nk/2-1 -Nk/2:-1];   % Fourier mode grid
  [kx ky] = ndgrid(k,k);
  kfilter = 1./(kx.^2+ky.^2);  % inverse -Laplacian in Fourier space (as above)
  kfilter(1,1) = 0; kfilter(Nk/2+1,:) = 0; kfilter(:,Nk/2+1) = 0;
  u = finufft2d2(xx(:),yy(:),-1,tol,kfilter.*fhat,o);   % eval filtered F series @ NU
  u = reshape(real(u),[n n]);
  fprintf('n=%d:\tNk=%d\tu(0,0) = %.15e\n',n,Nk,u(1,1))  % conv at same pt
end
%fhat(37,15)            % check a mode
figure(1); subplot(1,2,2); imagesc(log10(abs(fhat))); axis equal xy tight; colorbar; title('NUFFT: log_{10} abs fhat');
figure(4); subplot(1,2,1); mesh(xm,ym,f'); view(2); colorbar('southoutside');
axis equal tight; title('source term f'); xlabel('x'); ylabel('y');
subplot(1,2,2); mesh(xm,ym,u'); view(2); colorbar('southoutside');
axis equal tight; title('NUFFT solution u'); xlabel('x'); ylabel('y');

% C) solve on general nonuniform (still quadrature) grid.......................
tol = 1e-12;       % precision
fprintf('NUFFT on general NU (known-quadrature) grid... tol=%g\n',tol);
% smooth bijec map from [0,2pi)^2 -> [0,2pi)^2, generates quadrature rule
map = @(t,s) [t + 0.5*sin(t) + 0.2*sin(2*s); s + 0.3*sin(2*s) + 0.3*sin(s-t)];
mapJ = @(t,s) [1 + 0.5*cos(t), 0.4*cos(2*s); ...
               -0.3*cos(s-t),  1+0.6*cos(2*s)+0.3*cos(s-t)]; % 2x2 Jacobian
% (note I chose map(0,0)=(0,0) so the origin remains "on grid" for conv test)

ns = 80:40:240; ns = 2*ceil(ns/2);  % convergence study of grid points per side
for i=1:numel(ns), n = ns(i);
  t = 2*pi*(0:n-1)/n;           % 1d unif grid
  [tt ss] = ndgrid(t,t);
  xxx = map(tt(:)',ss(:)');
  xx = reshape(xxx(1,:),[n n]); yy = reshape(xxx(2,:),[n n]);  % 2d NU pts
  J = mapJ(tt(:)',ss(:)');
  detJ = J(1,1:n^2).*J(2,n^2+1:end) - J(2,1:n^2).*J(1,n^2+1:end);
  ww = detJ / n^2;      % 2d quadr weights, including 1/(2pi)^2 in E-F integr
  f = src(xx,yy);
  if i==1, figure(3); mesh(xx,yy,f); view(2); axis equal; axis([0 2*pi 0 2*pi]); title('f on mesh'); end
  Nk = 0.5*n; Nk = 2*ceil(Nk/2);  % modes to trust due to quadr err (dep on map)
  o.modeord = 1;      % fft output mode ordering
  fhat = finufft2d1(xx(:),yy(:),f(:).*ww(:),1,tol,Nk,Nk,o);  % step 1: do E-F
  k = [0:Nk/2-1 -Nk/2:-1];   % Fourier mode grid
  [kx ky] = ndgrid(k,k);
  kfilter = 1./(kx.^2+ky.^2);  % inverse -Laplacian in Fourier space (as above)
  kfilter(1,1) = 0; kfilter(Nk/2+1,:) = 0; kfilter(:,Nk/2+1) = 0;
  u = finufft2d2(xx(:),yy(:),-1,tol,kfilter.*fhat,o);   % eval filtered F series @ NU
  u = reshape(real(u),[n n]);
  fprintf('n=%d:\tNk=%d\tu(0,0) = %.15e\n',n,Nk,u(1,1))  % conv at same pt
end
%fhat(37,15)            % check a mode
figure(1); subplot(1,2,2); imagesc(log10(abs(fhat))); axis equal xy tight; colorbar; title('NUFFT: log_{10} abs fhat');
figure(4); subplot(1,2,1); mesh(xx,yy,f); view(2); colorbar('southoutside');
axis equal tight; title('source term f'); xlabel('x'); ylabel('y');
subplot(1,2,2); mesh(xx,yy,u); view(2); colorbar('southoutside');
axis equal tight; title('NUFFT solution u'); xlabel('x'); ylabel('y');

% Note: if you really wanted to have an adaptive grid, using Fourier modes
% is a waste, since you need as many modes as nodes in a uniform FFT solver;
% you may as well use an FFT solver. For a fully adaptive fast Poisson solver
% use a "box-code", ie, periodic FMM applied to a quad-tree quadrature scheme.
