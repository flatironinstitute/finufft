% make saddle point figs. Barnett 5/2/17

clear
% aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
be = 30;   % beta freq param
dx = 0.02; gx=-2.4:dx:2.7; gy=-1.2:dx:3; % plotting
[xx yy] = meshgrid(gx,gy);
zz = xx+1i*yy;

rho = 0.8;  % case (a)
p = @(z) sqrt(1-z.^2) + 1i*rho*z;       % Olver's -p func upstairs in exp
F = @(z) exp(be*p(z));                  % integrand
figure; imagesc(gx,gy,real(F(zz)));
z0 = 1i*rho/sqrt(1-rho^2);  % saddle pt
siz = abs(F(z0)); caxis(siz*2*[-1 1]);
axis xy tight equal; v=axis; colorbar('southoutside')
xlabel('Re z'); ylabel('Im z');
hold on; plot([-1 1],[0 0],'k.','markersize',10);
contour(gx,gy,real(p(zz)),real(p(z0))*[1 1],'--k'); % level curve re p @ saddle
plot(real(z0),imag(z0),'w*','markersize',10);
t = [1:dx:max(gx)]; h=0.02; y=h*sin(1.2*t/h); plot(t,y,'k-'); plot(-t,y,'k-');
axis(v);

title(sprintf('Case (a): \\rho=%g\n',rho));
set(gcf,'paperposition',[0 0 4 5]);
print -depsc2 saddlea.eps

% do the inset...
figure; %axes('position',[0.6 0.7 0.2 0.2]);
u0=atanh(rho);
u=(0:100)/100*u0; v=asin(sqrt(1-rho^2)./(cosh(u)-rho*sinh(u)));
patch(0.01+[u 0], [v pi/2],0.8*[1 1 1],'linestyle','none');
hold on; plot(u,v,'k--','linewidth',2);
text(0.15*u0,0.8*pi/2,'bad');
plot(u0,pi/2,'*','markersize',10); plot(0,0,'k.','markersize',10);
v = (0:100)/100*pi/2; u = atanh(rho)*sin(v); % example contour in ellip coords
plot(u,v,'k-');
axis([0 u0 0 pi/2]);
set(gca,'xtick',[0 u0],'xticklabel',{'0','u_0'}); xlabel('u');
set(gca,'ytick',[0 pi/2],'yticklabel',{'0','\pi/2'}); ylabel('v');
set(gcf,'paperposition',[0 0 1.5 1.5]);
print -depsc2 saddleauv.eps
% now add this to saddlea in xfig -> saddlea_lab

% bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
rho = 1.2;  % case (b)
th = pi/2;     % how much to rotate branches of sqrt!
sqrtr = @(z) exp(-0.5i*th)*sqrt(exp(1i*th)*z); % sqrt w/ branch cut at angle pi - th.
sqrtrm = @(z) exp(0.5i*th)*sqrt(exp(-1i*th)*z); % sqrt w/ branch cut at angle pi + th.
p = @(z) sqrtr(1-z).*sqrtrm(1+z) + 1i*rho*z;
F = @(z) exp(be*p(z));
figure; imagesc(gx,gy,real(F(zz)));
z0 = [-1 1]*rho/sqrt(rho^2-1);  % saddle pts
caxis([-1 1]);
axis xy tight equal; v=axis; colorbar('southoutside')
xlabel('Re z'); ylabel('Im z');
hold on; plot([-1 1],[0 0],'k.','markersize',10);
plot([min(gx) max(gx); -1 1],zeros(2),'k--');
n=1e3; t=(0:n)/n*2*pi; plot(z0(1)*cos(t),z0(1)/rho*sin(t),'k--');
plot(real(z0),imag(z0),'w*','markersize',10);
t = [min(gy):dx:0]; y=1+h*sin(1.2*t/h); plot(y,t,'k-'); plot(-y,t,'k-');
axis(v);

title(sprintf('Case (b): \\rho=%g\n',rho));
set(gcf,'paperposition',[0 0 4 5]);
print -depsc2 saddleb.eps
% now add contour in xfig -> saddleb_lab

% cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc, needs ft.m
beta = 30;
phi = @(z) exp(beta*(sqrt(1-z.^2)-1)) .* (abs(z)<=1);   % ES kernel
k=0:1e-2:2*beta;
L=1.0; phihat = ft(phi,L,k);

% saddle approx:
r = k/beta;  % rho
% (a), r<1 form...
sad = exp(beta*(sqrt(1-r.^2)-1)) .* (1-r.^2).^(-.75) * sqrt(2*pi/beta);
jtail = r>1; 
rt = r(jtail);
% (b), r>1 form...
sad(jtail) = sqrt(2*pi/beta)*exp(-beta)*2*sin(-pi/4+beta*sqrt(rt.^2-1)).*(rt.^2-1).^(-.75);    % use 2 saddle pts for k>beta

figure; semilogy(r,abs(phihat),'-'); axis([0 max(r) 1e-16 1e0]);
hold on; semilogy(r,abs(sad),'-'); semilogy(r,abs(sad-phihat),'--');
plot([1 1], [1e-12 1],'r--');
%legend('true (quadrature)','asymptotic (Thm. 4)','diff');
legend('true','Thm. 4','difference');
xlabel('$\rho$','interpreter','latex');
ylabel('$|\hat\phi(\rho\beta)|$','interpreter','latex');
text(0,1e2,'(c)');
set(gcf,'paperposition',[0 0 3 3]);
print -depsc2 saddleft.eps
