% MATLAB FINUFFT GPU interface demo for 1D type 1 transform.
clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential
tol     = 1e-8;   % requested accuracy
M       = 1e7;
N       = 1e7;    % # of modes (approx total, used in all dims)
type = 1;
n_modes = [N];      % n_dims inferred from length of this
ntrans = 1;

x = pi*(2*rand(1,M)-1);                         % choose NU points
c = randn(1,M*ntrans)+1i*randn(1,M*ntrans);     % choose stack of strengths
xg = gpuArray(x); cg = gpuArray(c);             % move to GPU

opts.debug=0;    % set options then plan the transform...
opts.floatprec = 'double';   % tells it to make a single-precision plan

disp('starting...'), tic     % just time cuFINUFFT, not the data creation

opts.gpu_method=2;
plan = cufinufft_plan(type,n_modes,isign,ntrans,tol,opts);   % make plan

plan.setpts(xg);                                 % send in NU pts

fg = plan.execute(cg);                           % do transform (to fg on GPU)

tgpu = toc;
fprintf('done in %.3g s: throughput (excl H<->D) is %.3g NUpt/s\n',tgpu,M/tgpu)

% check the error of only one output...
nt = ceil(0.37*N);                              % pick a mode index
t = ceil(0.7*ntrans);                           % pick a transform in stack
fe = sum(c(M*(t-1)+(1:M)).*exp(1i*isign*nt*x));        % exact (done on CPU)
of1 = floor(N/2) + 1 + N*(t-1);                        % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-fg(nt+of1))/norm(fg,Inf))

% if you do not want to do more transforms of this size, clean up...
delete(plan);

