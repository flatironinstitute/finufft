% plot of FT for aliasing fig. Barnett 4/20/17

clear
R = 2.0;           % upsampling ratio
beta = 30;
phi = @(z) exp(beta*(sqrt(1-z.^2)-1)) .* (abs(z)<=1);   % ES kernel
k=0:1e-2:1.5*beta;
L=1.0; phihat = ft(phi,L,k); 
semilogy(k,abs(phihat),'k-','linewidth',2);
axis([0 max(k) 1e-14 1e0]);
box off
set(gca,'xtick',[],'ytick',[]);
set(gcf,'paperposition',[0 0 7 5]);
print -depsc2 alias.eps
