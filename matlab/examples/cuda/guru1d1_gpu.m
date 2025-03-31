% MATLAB double-precision FINUFFT GPU demo for 1D type 1 transform.
clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential
tol     = 1e-8;   % requested accuracy
M       = 1e7;
N       = 1e7;    % # of modes
type = 1;
n_modes = [N];    % n_dims inferred from length of this
ntrans = 1;       % number of transforms (>1: demo many-vector interface)

x = pi*(2*rand(M,1)-1);                         % choose NU points
c = randn(M,ntrans)+1i*randn(M,ntrans);         % choose stack of strengths
xg = gpuArray(x); cg = gpuArray(c);             % move to GPU

opts.debug=1;    % set options then plan the transform...
opts.floatprec = 'double';   % tells it to make a double-precision plan

disp('starting...'), tic     % just time cuFINUFFT, not the data creation

opts.gpu_method=2;
plan = cufinufft_plan(type,n_modes,isign,ntrans,tol,opts);   % make plan

plan.setpts(xg);                                 % send in NU pts

fg = plan.execute(cg);                           % do transform (to fg on GPU)

tgpu = toc;
fprintf('done in %.3g s: throughput (excl H<->D) is %.3g NUpt/s\n',...
        tgpu, M*ntrans/tgpu)

% check the error of only one output...
t = ceil(0.7*ntrans);                           % pick a transform in stack
if ntrans>1, ct = cg(:,t); ft = fg(:,t); else, ct = cg; ft = fg; end
nt = ceil(0.37*N);                              % pick a mode index
fe = sum(ct.*exp(1i*isign*nt*x));               % exact (done on CPU)
of1 = floor(N/2)+1;                             % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-ft(nt+of1))/norm(ft,Inf))

% if you do not want to do more transforms of this size, clean up...
delete(plan);
