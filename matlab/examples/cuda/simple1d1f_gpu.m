% MATLAB single-precision FINUFFT GPU demo for 1D type 1 transform.
clear

M = 1e8;
x = 2*pi*gpuArray.rand(M,1,'single');   % random pts in [0,2pi]^2
y = 2*pi*gpuArray.rand(M,1,'single');
% iid random complex data...
c = gpuArray.randn(M,1,'single')+1i*gpuArray.randn(M,1,'single');

N1 = 10000; N2 = 5000;                   % desired Fourier mode array sizes
tol = 1e-3;

dev = gpuDevice();                       % crucial for valid timing
tic
f = cufinufft2d1(x,y,c,+1,tol,N1,N2);    % do it (all opts default)
%opts.gpu_method=2; f = cufinufft2d1(x,y,c,+1,tol,N1,N2,opts); % do it with opts
wait(dev)                                % crucial for valid timing
tgpu = toc;
fprintf('done in %.3g s: throughput (excl H<->D) is %.3g NUpt/s\n',tgpu,M/tgpu)

% check the error of only one output, also on GPU...
nt = ceil(0.47*N);                       % pick a mode index in -N/2,..,N/2-1
fe = sum(c.*exp(1i*isign*nt*x));         % exact
of = floor(N/2)+1;                       % mode index offset
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-f(nt+of))/norm(f,Inf))
