% plots of ES kernel for paper. Barnett 4/11/17
% needs: vline, ft, gauss.

clear
R = 2.0;           % upsampling ratio
beta = 4;
phi = @(z) exp(beta*(sqrt(1-z.^2)-1)) .* (abs(z)<=1);   % ES kernel
KB = @(z) besseli(0,beta*sqrt(1-z.^2))/besseli(0,beta) .* (abs(z)<=1);  % KB
wg = 0.46;  % gauss width parameter
TG = @(z) exp(-wg*beta*z.^2) .* (abs(z)<=1);  % trunc Gauss
figure; set(gcf,'position',[500 500 1000 300]);
subplot(1,3,1); z = -1.1:1e-3:1.1;
h(3)=plot(z,TG(z),'m:'); hold on; plot([-1 1], TG([-1 1]), 'm.','markersize',10);
h(2)=plot(z,KB(z),'g--'); hold on; plot([-1 1], KB([-1 1]), 'g.','markersize',10);
h(1)=plot(z,phi(z),'-'); plot([-1 1], phi([-1 1]), 'b.','markersize',10);
axis tight
xlabel('$z$','interpreter','latex'); ylabel('$\phi(z)$','interpreter','latex')
text(-1,0.9,'(a)');
legend(h,'ES','K-B','Gauss','location','south');

subplot(1,3,2);
%ns = 13; beta = 2.3*ns;
beta = 30;
phi = @(z) exp(beta*(sqrt(1-z.^2)-1)) .* (abs(z)<=1);   % ES kernel
KB = @(z) besseli(0,beta*sqrt(1-z.^2))/besseli(0,beta) .* (abs(z)<=1);  % KB
TG = @(z) exp(-wg*beta*z.^2) .* (abs(z)<=1);  % trunc Gauss
z = 0:1e-3:1.1;
h(3)=semilogy(z,TG(z),'m:'); hold on; plot(1, TG(1), 'm.','markersize',10);
h(2)=semilogy(z,KB(z),'g--'); plot(1, KB(1), 'g.','markersize',10);
h(1)=semilogy(z,phi(z),'-'); plot(1, phi(1), 'b.','markersize',10);
xlabel('$z$','interpreter','latex'); ylabel('$\phi(z)$','interpreter','latex')
axis([0 max(z) 1e-14 1e0]);
text(0.05,3e-2,'(b)');
%legend(h,'ES','K-B','Gauss','location','southwest');

subplot(1,3,3);
k=0:1e-2:1.5*beta;
L=1.0; phihat = ft(phi,L,k); KBhat = ft(KB,L,k);    % use quadrature for FT
TGhat = ft(TG,L,k);
semilogy(k,abs(TGhat),'m:'); hold on;
semilogy(k,abs(KBhat),'g--'); semilogy(k,abs(phihat),'-');
%s = sqrt(k.^2-beta^2); KBhatex = (2/besseli(0,beta))*sin(s)./s;
%semilogy(k,abs(KBhatex),'k:'); % check Logan's formula, yup
xlabel('$\xi$','interpreter','latex');
ylabel('$|\hat\phi(\xi)|$','interpreter','latex')
vline(beta);
text(beta+1,1e-3,'$\xi=\beta$','interpreter','latex','color',[1 0 0]);
axis([0 max(k) 1e-14 1e0]);
text(3,1e-2,'(c)');

set(gcf,'paperposition',[0 0 9.5 2.5]);
print -depsc2 kernel.eps
