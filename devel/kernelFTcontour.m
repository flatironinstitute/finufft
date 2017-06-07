% study complex plane for estimating FT of exp sqrt kernel.
% adapted 5/15/17 for aliasing sum m>0.  (Go down for single-rho contours)

% show aliasing sum m>0 formula in C plane, at value of |k|<N/2 and x
clear
rat = 2.0; w=14; % sigma upsampling, width in grid pts
gam = 0.95;     % safety factor
be = gam*pi*w*(1-1/(2*rat));   % beta shape param
th = pi/2; %3*pi/4;        % angle to rotate sqrt branch cuts down by
sqrtr = @(z) exp(-0.5i*th)*sqrt(exp(1i*th)*z); % sqrt w/ branch cut at angle pi - th.
sqrtrm = @(z) exp(0.5i*th)*sqrt(exp(-1i*th)*z); % sqrt w/ branch cut at angle pi + th.
phi = @(x) exp(be*sqrtr(1-x).*sqrtrm(1+x));  % factor 1-x^2 to move branches, without e^-beta prefactor
%phi = @(x) exp(be*sqrt(1-x.^2));  % without e^-beta prefactor
xi = -pi*w/(2*rat);  % xi = k.alpha = scaled freq being tested
if 0
gx=.7:.001:1.3; gy = -.1:.001:.5; % zoom
%g=-.2:0.01:3; gx=g; gy=g;   % full
[xx yy] = meshgrid(gx,gy); zz = xx+1i*yy;
figure;
for xoal=0:.002:2/w      % loop over target x (grid frac part times w/2)....
  Fz = exp(1i*xi*(zz+xoal))./(exp(-1i*pi*w*(zz+xoal))-1).*phi(zz);
  Fz = Fz.*exp(1i*pi*w*(zz+xoal)); % exclude m=1 from semi-inf sum
  imagesc(gx,gy,real(Fz)); hold on;
  plot([-1 1],[0 0],'k.','markersize',10); hold off;  
  axis xy equal; axis([min(gx) max(gx) min(gy) max(gy)]);
  caxis(10*[-1 1]); colorbar;
  title(sprintf('w=%d, \\beta=%g, xi=%g, x/\\alpha=%g\n',w,be,xi,xoal)); drawnow
end                    %.....
return
end

if 0 %  plot integrand vs t, on up (-1) contour...
figure; t=logspace(-6,1,1e3);
for chi=0%:.001:2/w
  F = exp(be*sqrt(2i*t+t.^2) - xi*t) ./ (exp(pi*w*(-1i*chi+t)) - 1);
  loglog(t,abs([real(F);imag(F)]),'-'); xlabel('t'); ylabel('integrand')
  title(sprintf('w=%d, \\beta=%g, xi=%g, \\chi=%g\n',w,be,xi,chi)); drawnow
end
axis tight
return
end

if 0 %  plot single-xi freq integrand vs t, on up (-1) contour...
figure; t = linspace(0,2,1e3); %t=logspace(-3,0.5,1e3);
xi = 2*pi*w*(1-1/(2*rat));   % beyond cutoff of beta
F = exp(be*sqrt(2i*t+t.^2) - xi*t); sum(F)*(t(2)-t(1))
plot(t,[real(F);imag(F)],'-'); xlabel('t'); ylabel('integrand')
title(sprintf('w=%d, \\beta=%g, xi=%g\n',w,be,xi)); axis tight;
%return
end

if 1 %  plot single-xi freq integrand vs t, on J1 across contour...
figure; t = linspace(0,.1,1e3);
xi = 30*pi*w; %pi*w*(1-1/(2*rat));   % beyond cutoff of beta
rho = xi/be;
a = rho^2/(rho^2-1); de = a-1; z=a+1i*t; % delta gap, z contour J2 up
z = 1+t;  % J1 across
F = exp(be*sqrt(1-z.^2) + 1i*xi*z); sum(F)*(t(2)-t(1))
plot(t,[real(F);imag(F)],'-'); hold on; z0 = rho/sqrt(rho^2-1);plot(z0-1,0,'r+');
xlabel('t'); ylabel('integrand')
title(sprintf('w=%d, \\beta=%g, xi=%g\n',w,be,xi)); axis tight;
return
end

% check bnds on quad solve for Re sqrt(1-z^2) :  5/16/17 p.4
%figure; t=0:.001:1; d=.1;b=d-t.^2;plot(t,-b+sqrt(b.^2+4*(1+d)*t.^2),'-');
%hold on;plot(t,2*sqrt(1+d)*t,'r-');  % UB in t^2<d
%hold on;plot(t,2*(t.^2-d)+2*sqrt(1+d)*t,'m-');   % UB in t^2>d




%%%%%%%%%%%%%%%%%%%%%%%%% single-rho value expts ........................

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
g=-1:0.01:3;
%g=-5:0.01:10;
[xx yy] = meshgrid(g);
zz = xx+1i*yy;

for rho=5; %1:0.01:1.5   % -------- rho = scaled freq,  loop for anim
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

