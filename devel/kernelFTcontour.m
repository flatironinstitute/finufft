% study complex plane for estimating FT of exp sqrt kernel

clear; %close all
be = 30;   % beta param
th = pi/2; %3*pi/4;
sqrtr = @(z) exp(-0.5i*th)*sqrt(exp(1i*th)*z); % sqrt w/ branch cut at angle pi - th.
sqrtrm = @(z) exp(0.5i*th)*sqrt(exp(-1i*th)*z); % sqrt w/ branch cut at angle pi + th.
%f = @(x) exp(be*sqrt(1-x.^2));
f = @(x) exp(be*sqrtr(1-x).*sqrtrm(1+x));  % factor 1-x^2 to move branches
%f = @(x) exp(be*1i*sqrt(x.^2-1));  % ack
%f = @(x) exp(be*1i*sqrt(x-1).*sqrt(x+1));  % confused

figure; % 2d C-plane plot
g=-1:0.02:3;
%g=-5:0.01:10;
[xx yy] = meshgrid(g);
zz = xx+1i*yy;

for rho=1.2; %1:0.01:1.5   % -------- rho = scaled freq,  loop for anim
  k = rho*be;          % freq to compute FT at
  F = @(z) f(z).*exp(1i*k*z);    % new definition (sign)
  Fz = F(zz);
  hold off; imagesc(g,g,real(Fz)); axis xy equal tight; colorbar;
  z0 = 1i*rho/sqrt(1-rho^2); % saddle pt
  siz = abs(F(z0)); if ~isfinite(siz), siz=1e2; end
  caxis(siz*2*[-1 1]);
  hold on; plot([-1 1],[0 0],'k.'); plot(real(z0),imag(z0),'wx');
  axis([min(g) max(g) min(g) max(g)]);
  title(sprintf('\\rho=%g\n',rho));
  drawnow;
end           % ----------

% contours of real part of upstairs in exp(-beta.p(x))...  (beta is asymp)
figure; contourf(g,g,real(log(Fz)),(-10:10)+log(siz));
axis equal xy tight; colorbar;
hold on; plot([-1 1],[0 0],'w.','markersize',10); plot(real(z0),imag(z0),'wx');
title(sprintf('\\rho=%g\n',rho));

% add elliptic coords:
[uu vv] = meshgrid(0:0.1:2,0:0.1:2*pi); zze = cosh(uu+1i*vv);
h=mesh(real(zze),imag(zze),0*zze);
set(h,'alphadata',0*zze,'alphadatamapping','direct','facealpha','flat');

if 0 % attempt at contour for rho<1: (crap)
t0=rho/sqrt(1-rho^2);
a=(1-t0^2)/2; R = (1+t0^2)/2; n = 1e3; t = (0.5:n-0.5)/n*2*pi; % contour
z2 = a+R*exp(1i*t);
z = 1i*sqrt(-z2);   % branch cut at angle 0
plot(real(z),imag(z),'-'); hold on; zends=[z(1) z(end)];
plot(real(zends),imag(zends),'.','markersize',10); axis equal
end

if 0 % 1d real axis
X = 4; x=-X:0.003:X;
figure; subplot(2,1,1); plot(x,[real(F(x));imag(F(x))],'-');
ylabel('F(x)= f(x)e^{ikx}'); axis([-X X -5 5]);
subplot(2,1,2); semilogy(x,abs(F(x)),'-'); ylabel('|f(x)|');
end


if 0
figure; subplot(1,3,1); imagesc(g,g,log10(abs(Fz))); colorbar;
axis xy tight equal;
hold on; plot(1i*rho/sqrt(1-rho^2),'w.');
%caxis([-1 5]);
subplot(1,3,2); contourf(g,g,real(Fz)); colorbar; axis xy tight equal;
subplot(1,3,3); contourf(g,g,imag(Fz)); colorbar; axis xy tight equal;
end
