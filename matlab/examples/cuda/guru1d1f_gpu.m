% MATLAB single-precision FINUFFT GPU demo for 1D type 1 transform.
clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential
tol     = 1e-3;   % requested accuracy
M       = 1e8;    % # pts
N       = 1e6;    % # of modes; note N*eps('single') limits accuracy
type = 1;
n_modes = [N];    % n_dims inferred from length of this
ntrans = 1;       % number of transforms (>1: demo many-vector interface)

xg = pi*(2*gpuArray.rand(M,1,'single')-1);             % float32 NU pts on GPU
% stack of float32 strengths on GPU...
cg = gpuArray.randn(M,ntrans,'single')+1i*gpuArray.randn(M,ntrans,'single');

opts.debug=1;    % set options then plan the transform...
opts.floatprec = 'single';   % tells it to make a single-precision plan
opts.gpu_method=2;

dev = gpuDevice();           % needed for timing
disp('starting...'), tic     % just time cuFINUFFT, not the data creation

plan = cufinufft_plan(type,n_modes,isign,ntrans,tol,opts);   % make plan

plan.setpts(xg);                                 % send in NU pts

fg = plan.execute(cg);                           % do transform (to fg on GPU)

wait(dev); tgpu = toc;                           % since GPU async
fprintf('done in %.3g s: throughput (excl H<->D) is %.3g NUpt/s\n',...
        tgpu, M*ntrans/tgpu)

% if you do not want to do more transforms of this size, clean up...
delete(plan);

% check the error of only one output also on GPU...
t = ceil(0.7*ntrans);                           % pick a transform in stack
if ntrans>1, ct = cg(:,t); ft = fg(:,t); else, ct = cg; ft = fg; end
nt = ceil(0.37*N);                              % pick a mode index
fe = sum(ct.*exp(1i*isign*nt*xg));              % exact
of1 = floor(N/2)+1;                             % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-ft(nt+of1))/norm(ft,Inf))
