% matlab guru interface 1d1.
% Lu 5/11/2020
clear

%parameters
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-6;   % requested accuracy
M       = 2.2e3;
N       = 1e3;    % # of modes (approx total, used in all dims)
type=1;
n_dims=1;
n_modes = [N;1;1];
n_transf=1;
blksize=1;

%init pts, strength
x = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);

%opts
opts = nufft_opts();
opts.set_debug(1);

%plan
plan = nufft_plan();

%make plan
plan.nufft_makeplan(type,n_dims,n_modes,isign,n_transf,eps,blksize,opts);

%set pts
plan.nufft_setpts(M,x,[],[],N,[],[],[]);

%excute
[f,ier] = plan.nufft_excute(c); 

%error
nt = ceil(0.37*N);                              % pick a mode index
fe = sum(c.*exp(1i*isign*nt*x));                % exact
of1 = floor(N/2)+1;                             % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs((fe-f(nt+of1))/fe))
