% K-B pair for talk. Barnett 11/7/17
% needs: vline, ft, gauss.

clear
R = 2.0;           % upsampling ratio
beta = 5;  %ns = 13; beta = 2.3*ns;
%KB = @(z) besseli(0,beta*sqrt(1-z.^2))/besseli(0,beta) .* (abs(z)<=1);  % KB
KB = @(z) besseli(0,beta*sqrt(1-z.^2)) .* (abs(z)<=1);  % KB

figure; set(gcf,'position',[500 500 1000 300]);

subplot(1,2,1); z = -1.1:1e-3:1.1;
h(2)=plot(z,KB(z),'-'); hold on; plot([-1 1], KB([-1 1]), '.','markersize',10);
axis tight
v = axis;
xlabel('$z$','interpreter','latex'); ylabel('$\phi_{KB}(z)$','interpreter','latex')
text(-1,v(4)*0.9,sprintf('$\\beta=%g$',beta),'interpreter','latex','color',[0 0 0])

subplot(1,2,2);
wid = 3.0*beta;
k=-wid:1e-2:wid;  % symm
L=1.0; KBhat = ft(KB,L,k);    % use quadrature for FT
plot(k,real(KBhat),'-');
xlabel('$\xi$','interpreter','latex');
ylabel('$\hat\phi_{KB}(\xi)$','interpreter','latex')
vline(beta*[-1 1]); hline(0,'k-');
axis tight
v = axis;
text(beta+1,v(4)*0.7,'$\xi=\beta$','interpreter','latex','color',[1 0 0]);

set(gcf,'paperposition',[0 0 9.5 2.5]);
print -depsc2 kbpair.eps
