% Demo evaluating a 1D Fourier series at arbitrary points in quasi-optimal
% time via FINUFFT, in MATLAB.
% Barnett 6/3/20
clear; close all;

L = 10;   % periodic interval [0,L)

kmax = 300;         % bandlimit
k = -kmax:kmax-1;   % freq indices (negative up through positive mode ordering)
N = 2*kmax;         % # modes

% make some convenient Fourier coefficients...
rng(0);
fk = randn(N,1)+1i*randn(N,1);   % iid random complex data, column vec
k0 = 50;            % freq scale of Gaussian random field w/ Gaussian covar
fk = fk .* exp(-(k/k0).^2).';    % scale the amplitudes, kills high freqs

M = 1e5; x = L*rand(1,M);        % make random target points in [0,L)
tol = 1e-12;
x_scaled = x * (2*pi/L);         % don't forget to scale to 2pi-periodic!
tic; c = finufft1d2(x_scaled,+1,tol,fk); toc   % evaluate Fourier series at x

figure(1); clf;
Mp = 3e3;               % how many pts to plot
jplot = 1:Mp;           % indices to plot
plot(x(jplot),c(jplot),'b.'); axis tight; xlabel('x'); ylabel('f(x)');

% check with evaluation of same series on Mp uniform points via FFT
fk_pad = [zeros((Mp-N)/2,1); fk; zeros((Mp-N)/2,1)];  % column pad with zeros
fi = Mp * ifft(fftshift(fk_pad));       % evaluate
yi = L*(0:Mp-1)/Mp;           % spatial grid corresp to the FFT eval
hold on; plot(yi,fi,'r.'); legend('NU pts','unif pts');   % eye norm is ok

% math check: send the unif pts into NUFFT and compare answers
ci = finufft1d2(yi * (2*pi/L),+1,tol,fk); 
fprintf('max error on %d unif test pts; %.3g\n',Mp,norm(fi-ci,inf))
