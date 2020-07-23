% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1.
% Single-precision case.
% Lu 5/11/2020. Barnett added timing, tweaked.
clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential
tol     = 1e-5;   % requested accuracy (cannot ask for much more in single prec)
M       = 2e5;
N       = 1e5;    % # of modes (approx total, used in all dims)
type = 1;
n_modes = N;      % n_dims inferred from length of this
ntrans = 3;

x = pi*(2*rand(1,M,'single')-1);                % choose NU points
c = randn(1,M*ntrans,'single')+1i*randn(1,M*ntrans,'single');     % strengths

% set options then plan the transform...
opts.debug=2;
opts.floatprec = 'single';   % tells it to make a single-precision plan
disp('starting...'), tic
plan = finufft_plan(type,n_modes,isign,ntrans,tol,opts);

plan.setpts(x);                                 % send in NU pts

f = plan.execute(c);                               % do the transform
disp('done.'); toc

% if you do not want to do more transforms of this size, clean up...
delete(plan);

% check the error of one output...
nt = ceil(0.37*N);                              % pick a mode index
t = ceil(0.7*ntrans);                           % pick a transform in stack
fe = sum(c(M*(t-1)+(1:M)).*exp(1i*isign*nt*x));           % exact
of1 = floor(N/2) + 1 + N*(t-1);                           % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-f(nt+of1))/norm(f,Inf))
