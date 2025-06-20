% MATLAB double-precision gpuArray 2D type 1 nonuniform FFT demo via CUFINUFFT
% Libin Lu & Alex Barnett, 3/30/25.
clear

% set required parameters...
isign   = +1;           % sign of imaginary unit in exponential
tol     = 1e-9;         % requested accuracy
M       = 1e8;          % # nonuniform points
N1 = 1e4; N2 = 5e3;     % # of modes in each dim (x,y)
type = 1;
n_modes = [N1;N2];      % n_dims inferred from length of this array
ntrans = 1;             % if >1, demo the many-vector interface

% make random NU points (xg,yg) and random complex strengths cg on GPU...
xg = (2*pi)*gpuArray.rand(M,1);
yg = (2*pi)*gpuArray.rand(M,1);
cg = gpuArray.randn(M,ntrans)+1i*gpuArray.randn(M,ntrans);

opts.debug = 1;              % set options then plan the transform...
opts.floatprec = 'double';   % tells it to make a double-precision plan
opts.gpu_method = 2;         % "SM" method

dev = gpuDevice();           % needed for timing
disp('starting...'), tic     % time CUFINUFFT, not the data creation
plang = cufinufft_plan(type,n_modes,isign,ntrans,tol,opts);

plang.setpts(xg,yg);                                 % send in NU pts

fg = plang.execute(cg);                              % do the transform

wait(dev); tgpu = toc;                               % since GPU async
fprintf('done in %.3g s: throughput (excl. H<->D) is %.3g NUpt/s\n',...
        tgpu, ntrans*M/tgpu)

% if you do not want to do more transforms of this size, clean up...
delete(plang);

% check the error of only one output, also using GPU...
t = ceil(0.7*ntrans);                             % pick a transform in stack
if ntrans>1, ct = cg(:,t); ft = fg(:,:,t); else, ct = cg; ft = fg; end
nt1 = ceil(0.37*N1); nt2 = ceil(-0.41*N2);        % pick a mode index
fe = sum(ct.*exp(1i*isign*(nt1*xg+nt2*yg)));      % exact ans to working prec
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1;         % mode index offsets
fprintf('2D type-1: rel err in F[%d,%d] is %.3g\n', nt1, nt2,...
        abs(fe-ft(nt1+of1,nt2+of2))/norm(ft(:),Inf) )  % careful: not mat norm
