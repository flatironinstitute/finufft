clear
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-3;   % requested accuracy
o.debug = 1;      % choose 1 for timing breakdown text output
o.nthreads = 0;   % omit, or use 0, to use default num threads.
M       = 1e9;    % # of NU pts - crashes out at 1e9.
N       = 1e6;    % # of modes (approx total, used in all dims)

j = ceil(0.93*M);                               % target pt index to test
k = ceil(0.24*M);                               % freq targ pt index to test

tic;
fprintf('1D type 1: using %d modes...\n',N)
x = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);
[f ier] = finufft1d1(x,c,isign,eps,N,o);
fprintf('done in %.3g s, ier=%d\n',toc,ier)
if ~ier
  nt = ceil(0.37*N);                              % pick a mode index
  fe = (1/M)*sum(c.*exp(1i*isign*nt*x));          % exact
  of1 = floor(N/2)+1;                             % mode index offset
  fprintf('rel err in F[%d] is %.3g\n',nt,abs((fe-f(nt+of1))/fe))
end
