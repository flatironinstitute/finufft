% matlab guru interface 1D type 1, single-precision case.
% Barnett 7/5/2020.
clear

%parameters
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-5;   % requested accuracy
M       = 2e5;
N       = 1e5;    % # of modes (approx total, used in all dims)
type=1;
n_modes = N;  % n_dims inferred from length of this
n_transf=1;

%init pts, strength
x = pi*(2*rand(1,M,'single')-1);
c = randn(1,M,'single')+1i*randn(1,M,'single');

disp('starting...'), tic
%opts
opts.debug=2;
opts.spread_debug=0;
opts.floatprec = 'single';   % tells it to make a single-precision plan

%plan
plan = finufft_plan(type,n_modes,isign,n_transf,eps,opts);

%set pts
plan.finufft_setpts(x,[],[]);

%exec
f = plan.finufft_exec(c);
disp('done.'); toc

%error
nt = ceil(0.37*N);                              % pick a mode index
fe = sum(c.*exp(1i*isign*nt*x));                % exact
of1 = floor(N/2)+1;                             % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-f(nt+of1))/norm(fe,Inf))
