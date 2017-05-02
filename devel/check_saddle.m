% check the saddle-pt approx to FT of ES kernel 4/25/17

clear
R = 2.0;           % upsampling ratio
beta = 15; % 30;
phi = @(z) exp(beta*(sqrt(1-z.^2)-1)) .* (abs(z)<=1);   % ES kernel
k=0:1e-2:100*beta;
%k=0:1e-2:100*beta;  % check way out! (slow since gauss quadr pts)
L=1.0; phihat = ft(phi,L,k);
figure; semilogy(k,abs(phihat),'-'); axis([0 max(k) 1e-16 1e0]);

% saddle approx:
r = k/beta;  % rho
sad = exp(beta*(sqrt(1-r.^2)-1)) .* (1-r.^2).^(-.75) * sqrt(2*pi/beta);
jtail = r>1; sad(jtail) = 2*real(sad(jtail));  % use 2 saddle pts for k>beta
rt = r(jtail);
sad(jtail) = sqrt(2*pi/beta)*exp(-beta)*2*sin(-pi/4+beta*sqrt(rt.^2-1)).*(rt.^2-1).^(-.75);

hold on; semilogy(k,abs(sad),'-'); semilogy(k,abs(sad-phihat),'--');
legend('true','saddle','difference');
set(gca,'xscale','log'); plot(k,0.1*exp(-beta)*[(r/10).^-1;(r/10).^(-3/2)],'k--'); legend('true','saddle','difference','k^{-1}','k^{-3/2}'); % check decay, doesn't match
%semilogy(k,abs(phihat-2*exp(-beta)*sin(k)./k),'m-'); % appears to be k^-2
%legend('true','saddle','difference','w/o top hat');


figure; plot(k,abs(sad./phihat),'-'); title('ratio'); axis([0 beta 1 1.2]);


% z0=1i*r./sqrt(1-r.^2); figure; plot(r, [real(z0);imag(z0)],'-'); title('z_0');
