% Demo evaluating a 1D Fourier series at arbitrary points in quasi-optimal
% time via FINUFFT, in MATLAB.
% Barnett 6/3/20
clear; close all;

L = 10;   % periodic interval [0,L)

kmax = 500;         % bandlimit
k = -kmax:kmax-1;   % freq indices (negative up through positive mode ordering)
N = 2*kmax;         % # modes

% make some convenient Fourier coefficients...
rng(0);
fk = randn(N,1)+1i*randn(N,1);   % iid random complex data, column vec
k0 = 100;            % freq scale of Gaussian random field w/ Gaussian covar
fk = fk .* exp(-(k/k0).^2).';    % scale the amplitudes, kills high freqs

M = 1e6; x = L*rand(1,M);        % make random target points in [0,L)
tol = 1e-12;
x_scaled = x * (2*pi/L);         % don't forget to scale to 2pi-periodic!
tic; c = finufft1d2(x_scaled,+1,tol,fk); toc   % evaluate Fourier series at x's
% takes 0.026 sec

% exhaustive math check by naive evaluation (12 seconds)
tic; cn = 0*c; for m=k, cn = cn + fk(m+N/2+1)*exp(1i*m*x_scaled.'); end, toc
norm(c-cn,inf)

% naive evaluation with reversed loop order (29 seconds)
%tic; cn = 0*c; for j=1:M, cn(j) = exp(1i*k*x_scaled(j)) * fk; end, toc  % dot
%norm(c-cn,inf)

figure(1); clf;
Mp = 1e4;               % how many pts to plot
jplot = 1:Mp;           % indices to plot
plot(x(jplot),real(c(jplot)),'b.'); axis tight; xlabel('x'); ylabel('Re f(x)');
set(gcf,'paperposition',[0 0 10 2.5]); print -dpng ../docs/pics/fser1d.png

% Extra stuff beyond the webpage:
% check with evaluation of same series on Mp uniform points via FFT
fk_pad = [zeros((Mp-N)/2,1); fk; zeros((Mp-N)/2,1)];  % column pad with zeros
fi = Mp * ifft(fftshift(fk_pad));       % evaluate
yi = L*(0:Mp-1)/Mp;           % spatial grid corresp to the FFT eval
hold on; plot(yi,real(fi),'r.'); legend('NU pts','unif pts');   % eye norm is ok

% math check: send the unif pts into NUFFT and compare answers
ci = finufft1d2(yi * (2*pi/L),+1,tol,fk); 
fprintf('max error on %d unif test pts; %.3g\n',Mp,norm(fi-ci,inf))
