% matlab guru interface 1D type 1.
% Lu 5/11/2020. Barnett added timing, tweaked interface.
clear

%parameters
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-6;   % requested accuracy
M       = 2e6;
N       = 1e6;    % # of modes (approx total, used in all dims)
type=1;
n_modes = N;  % n_dims inferred from length of this
n_transf=1;

%init pts, strength
x = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);

disp('starting...'), tic
%opts
opts.debug=2;
opts.spread_debug=0;

%plan
plan = finufft_plan(type,n_modes,isign,n_transf,eps,opts);

%set pts
plan.setpts(x,[],[]);

%exec
f = plan.exec(c);
delete(plan);
disp('done.'); toc

%error
nt = ceil(0.37*N);                              % pick a mode index
fe = sum(c.*exp(1i*isign*nt*x));                % exact
of1 = floor(N/2)+1;                             % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-f(nt+of1))/norm(f,Inf))
