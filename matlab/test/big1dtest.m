% Explore large problems: timing, RAM usage, matlab interface.
% Barnett 3/28/17
clear
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-3;   % requested accuracy
o.debug = 1;      % choose 1 for timing breakdown text output
o.spread_sort=0;
M       = 2.2e9;    % # of NU pts - when >=2e31, answer is wrong, zero ***
N       = 1e6;    % # of modes (approx total, used in all dims)

j = ceil(0.93*M);                               % target pt index to test

if 0
fprintf('generating x & c data (single-threaded and slow)...\n')
x = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);
fprintf('1D type 1: using %d modes...\n',N)
tic;
f = finufft1d1(x,c,isign,eps,N,o);
fprintf('done in %.3g s\n',toc)
if ~ier
  nt = ceil(0.37*N);                              % pick a mode index
  fe = sum(c.*exp(1i*isign*nt*x));                % exact
  of1 = floor(N/2)+1;                             % mode index offset
  fprintf('rel err in F[%d] is %.3g\n',nt,abs((fe-f(nt+of1))/fe))
end
end

if 1
fprintf('generating x data (single-threaded and slow)...\n')
x = pi*(2*rand(M,1)-1);
f = randn(N,1)+1i*randn(N,1);
fprintf('1D type 2: using %d modes...\n',N)
tic
c = finufft1d2(x,isign,eps,f,o);        % Out of memory iff >=2^31
fprintf('done in %.3g s\n',toc)
ms=numel(f); mm = (ceil(-ms/2):floor((ms-1)/2))';  % mode index list
ce = sum(f.*exp(1i*isign*mm*x(j)));             % crucial f, mm same shape
fprintf('1D type-2: rel err in c[%d] is %.3g\n',j,abs((ce-c(j))/ce))
end

% conclusion: we get zeros output if >=2^31. Fix this issue w/ mex interface.
