% demo making indep samples from a 1D Gaussian random field, defined by its
% power spectral density \hat{C}(k), using Fourier methods without any BCs.
% Case of NU target pts, hence needs FINUFFT. (Note FFT is ok for unif targs.)
% Barnett 5/24/19
clear

% choose a GRF spectral density function \hat{C}(k)...
form = 'y';   % 'g' is for checking the variance; 'y' is what you care about.
if form=='g'        % Gaussian form (can get superalg convergence wrt K)
  K0 = 2e3;         % freq width, makes smooth below scale 1/K0
  Chat = @(k) (1/sqrt(2*pi)/K0)*exp(-(k/K0).^2/2);
elseif form=='y'    % desired Yukawa form (merely algebraic conv wrt K)
  alpha = 1.0; beta = 100; gamma = 1;    % sqrt(gamma/beta)=0.1 spatial scale
  Chat = @(k) (gamma*k.^2 + beta).^-alpha;
end

% user params...
L = 1;   % we want to evaluate on domain length D=[0,L].
M = 1e4;           % # targs
x = rand(1,M)*L;   % user can choose any target pts
tol = 1e-6;        % desired NUFFT precision
K = 1e4;           % max freq to go out to in Fourier transform for GRF

% alg param setup...
eta = 1.5;         % k-quad scheme convergence param, >=1; maybe =1 is enough?
dk_nyq = pi/L;     % Nyquist k-grid spacing for the width of target domain
dk = dk_nyq/eta;   % k-grid
P = 2*pi/dk        % actual period the NUFFT thinks there is
N = 2*K/dk;        % how many modes, if # k quadr pts.
N = 2*ceil(N/2)    % make even integer
k = (-N/2:N/2-1)*dk;  % the k grid

% check the integral of spectral density (non-stochastic)... 1 for form='g'
I = dk*sum(Chat(k))   % plain trap rule
%figure; loglog(k,Chat(k),'+-'); xlabel('k'); title('check k-quadr scheme');

nr = 3e2;   % # reps, ie indep samples from GRF drawn.. 1e3 takes 3 sec
fprintf('sampling %d times... ',nr);
gvar = zeros(M,1);    % accumulate the ptwise variance
figure;
for r=1:nr
  fk = ((randn(1,N)+1i*randn(1,N))/sqrt(2)) .* sqrt(dk*Chat(k));   % spec ampl
  g = finufft1d2(x/P,+1,tol,fk);                % do it, outputting g at targs
  if r<=5, h=plot(x,real(g),'.'); hold on; end  % show a few samples
  gvar = gvar + abs(g).^2;
end
gvar = gvar/nr;
fprintf('done. mean ptwise var g = %.3f\n',mean(gvar))

xlabel('x'); h2=plot(x,gvar, 'ko'); legend([h h2],'samples g(x)','est var g(x)')
v=axis; v(1:2)=[0 L]; axis(v);
title(sprintf('1D GRF type %s: a few samples, and ptwise variance',form));
