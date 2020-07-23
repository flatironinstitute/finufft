% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1.
% Lu 5/11/2020. Barnett added timing, tweaked.
clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential
tol     = 1e-9;   % requested accuracy
M       = 1e6;
N       = 1e6;    % # of modes (approx total, used in all dims)
type = 1;
n_modes = N;      % n_dims inferred from length of this
ntrans = 2;

x = pi*(2*rand(1,M)-1);                         % choose NU points
c = randn(1,M*ntrans)+1i*randn(1,M*ntrans);     % choose stack of strengths

disp('starting...'), tic     % just time FINUFFT not the data creation
opts.debug=2;    % set options then plan the transform...
plan = finufft_plan(type,n_modes,isign,ntrans,tol,opts);

plan.setpts(x);                                 % send in NU pts

f = plan.execute(c);                               % do the transform
disp('done.'); toc

% if you do not want to do more transforms of this size, clean up...
delete(plan);

% check the error of one output...
nt = ceil(0.37*N);                              % pick a mode index
t = ceil(0.7*ntrans);                           % pick a transform in stack
fe = sum(c(M*(t-1)+(1:M)).*exp(1i*isign*nt*x));        % exact
of1 = floor(N/2) + 1 + N*(t-1);                        % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-f(nt+of1))/norm(f,Inf))
