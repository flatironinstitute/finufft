clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential
tol     = 1e-8;   % requested accuracy
M       = 1e6;
N       = 1e6;    % # of modes (approx total, used in all dims)
type = 1;
n_modes = [N];      % n_dims inferred from length of this
ntrans = 1;

%x = pi*(2*rand(1,M,'single')-1);                % choose NU points
%c = randn(1,M*ntrans,'single')+1i*randn(1,M*ntrans,'single');     % strengths
x = pi*(2*rand(1,M)-1);                         % choose NU points
c = randn(1,M*ntrans)+1i*randn(1,M*ntrans);     % choose stack of strengths

xg = gpuArray(x); cg = gpuArray(c);                % move to GPU

opts.debug=1;    % set options then plan the transform...
opts.floatprec = 'double';   % tells it to make a single-precision plan
%opts.floatprec = 'single';   % tells it to make a single-precision plan

disp('starting...'), tic     % just time FINUFFT not the data creation

plan = finufft_plan(type,n_modes,isign,ntrans,tol,opts);

opts.gpu_method=2;
plang = cufinufft_plan(type,n_modes,isign,ntrans,tol,opts);

plan.setpts(x);                                 % send in NU pts

plang.setpts(xg);                                 % send in NU pts

f = plan.execute(c);                               % do the transform

fg = plang.execute(cg);                               % do the transform

disp('done.'); toc

% relative diff CPU vs GPU
norm(c-cg)/norm(c)
norm(x-xg)/norm(x)
norm(f-fg)/norm(f)
%f(1:10)
%fg(1:10)

% check the error of one output...
nt = ceil(0.37*N);                              % pick a mode index
t = ceil(0.7*ntrans);                           % pick a transform in stack
fe = sum(c(M*(t-1)+(1:M)).*exp(1i*isign*nt*x));        % exact
of1 = floor(N/2) + 1 + N*(t-1);                        % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-fg(nt+of1))/norm(fg,Inf))

% if you do not want to do more transforms of this size, clean up...
delete(plan);
delete(plang);
