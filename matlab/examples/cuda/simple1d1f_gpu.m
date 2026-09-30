% MATLAB single-precision FINUFFT GPU demo for 1D type 1 transform.
clear

% docs-start: simple1d1f-gpu
M = 1e8;
x = 2*pi*gpuArray.rand(M,1,'single');   % random pts in [0,2pi]^2
y = 2*pi*gpuArray.rand(M,1,'single');
% iid random complex data...
c = gpuArray.randn(M,1,'single')+1i*gpuArray.randn(M,1,'single');

N1 = 10000; N2 = 5000;                   % desired Fourier mode array sizes
tol = 1e-3;
% docs-end: simple1d1f-gpu

% docs-start: simple1d1f-gpu-timed
dev = gpuDevice();                       % crucial for valid timing
tic
f = cufinufft2d1(x,y,c,+1,tol,N1,N2);    % do it (all opts default)
%opts.gpu_method=2; f = cufinufft2d1(x,y,c,+1,tol,N1,N2,opts); % do it with opts
wait(dev)                                % crucial for valid timing
tgpu = toc;
% docs-end: simple1d1f-gpu-timed
fprintf('done in %.3g s: throughput (excl H<->D) is %.3g NUpt/s\n',tgpu,M/tgpu)

% check the error of only one output, also on GPU...
nt1 = ceil(0.47*N1); nt2 = ceil(0.47*N2);       % pick mode indices in -Ni/2,..,Ni/2-1
fe = sum(c.*exp(1i*(nt1*x + nt2*y)));           % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1;       % mode index offsets
fprintf('rel err in F[%d,%d] is %.3g\n',nt1,nt2,abs(fe-f(nt1+of1,nt2+of2))/norm(f(:),Inf))
