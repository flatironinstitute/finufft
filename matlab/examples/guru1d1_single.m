% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1.
% Single-precision case.
% Lu 5/11/2020. Barnett added timing, tweaked.
clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-5;   % requested accuracy (cannot ask for much more in single prec)
M       = 2e5;
N       = 1e5;    % # of modes (approx total, used in all dims)
type = 1;
n_modes = N;      % n_dims inferred from length of this
n_transf = 1;

disp('starting...'), tic
% set options then plan the transform...
opts.debug=2;
opts.spread_debug=0;
opts.floatprec = 'single';   % tells it to make a single-precision plan
plan = finufft_plan(type,n_modes,isign,n_transf,eps,opts);

x = pi*(2*rand(1,M,'single')-1);                % choose NU points
plan.setpts(x,[],[]);                           % send them in

c = randn(1,M,'single')+1i*randn(1,M,'single'); % choose strengths
f = plan.exec(c);                               % do the transform

disp('done.'); toc
% if you do not want to do more transforms of this size, clean up...
delete(plan);

% check the error of one output...
nt = ceil(0.37*N);                              % pick a mode index
fe = sum(c.*exp(1i*isign*nt*x));                % exact
of1 = floor(N/2)+1;                             % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-f(nt+of1))/norm(f,Inf))
