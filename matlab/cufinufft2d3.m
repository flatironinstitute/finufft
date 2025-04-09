% CUFINUFFT2D3   GPU 2D complex nonuniform FFT, type 3 (nonuniform to nonuniform).
%
% f = cufinufft2d3(x,y,c,isign,eps,s,t)
% f = cufinufft2d3(x,y,c,isign,eps,s,t,opts)
%
% This computes on the GPU, to relative precision eps, via a fast algorithm:
%
%              nj
%     f[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j])),  for k = 1, ..., nk
%              j=1
%   Inputs:
%     x,y   coordinates of nonuniform sources in R^2, each a length-nj vector.
%     c     length-nj complex vector of source strengths. If numel(c)>nj,
%           expects a stack of vectors (eg, a nj*ntrans matrix) each of which is
%           transformed with the same source and target locations.
%     isign if >=0, uses + sign in exponential, otherwise - sign.
%     eps   relative precision requested (generally between 1e-15 and 1e-1)
%     s,t   frequency coordinates of nonuniform targets in R^2,
%           each a length-nk vector.
%     opts   optional struct with optional fields controlling the following:
%     opts.debug:   0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT).
%     opts.gpu_method:  0 (auto, default), 1 (GM or GM-sort), 2 (SM).
%     opts.gpu_sort:  0 (do not sort NU pts), 1 (sort when GM method, default).
%     opts.gpu_kerevalmeth:  0 (slow reference). 1 (Horner ppoly, default).
%     opts.gpu_maxsubprobsize:  max # NU pts per subprob (gpu_method=2 only).
%     opts.gpu_binsize{x,y,z}:  various binsizes in GM-sort/SM (for experts).
%     opts.gpu_maxbatchsize:   0 (auto, default), or many-vector batch size.
%     opts.gpu_device_id:  sets the GPU device ID (experts only).
%   Outputs:
%     f     length-nk complex vector of values at targets, or, if ntrans>1,
%           a matrix of size (nk,ntrans)
%
% Notes:
%  * For CUFINUFFT all array I/O is in the form of gpuArrays (on-device).
%  * The precision of gpuArray input x controls whether the double or
%    single precision GPU library is called; all array inputs must match in
%    location (ie, be gpuArrays), and in precision.
%  * The vectorized (many vector) interface, ie ntrans>1, can be faster
%    than repeated calls with the same nonuniform points. Note that here the
%    I/O data ordering is stacked not interleaved. See ../docs/matlab_gpu.rst
%  * For more details about the opts fields, see ../docs/c_gpu.rst
%  * See ERRHANDLER, VALID_* and CUFINUFFT_PLAN for possible warning/error IDs.
%  * Full documentation is online at http://finufft.readthedocs.io
%
% See also CUFINUFFT_PLAN.
function f = cufinufft2d3(x,y,c,isign,eps,s,t,o)

valid_setpts(true,3,2,x,y,[],s,t,[]);
o.floatprec=underlyingType(x);                      % should be 'double' or 'single'
n_transf = valid_ntr(x,c);
p = cufinufft_plan(3,2,isign,n_transf,eps,o);
p.setpts(x,y,[],s,t,[]);
f = p.execute(c);
