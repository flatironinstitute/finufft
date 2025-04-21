% CUFINUFFT1D2   GPU 1D complex nonuniform FFT, type 2 (uniform to nonuniform).
%
% c = cufinufft1d2(x,isign,eps,f)
% c = cufinufft1d2(x,isign,eps,f,opts)
%
% This computes on the GPU, to relative precision eps, via a fast algorithm:
%
%    c[j] = SUM   f[k1] exp(+/-i k1 x[j])      for j = 1,...,nj
%            k1
%     where sum is over -ms/2 <= k1 <= (ms-1)/2.
%
%  Inputs:
%     x     length-nj vector of real-valued locations of nonuniform sources
%     f     complex Fourier coefficients. If a vector, length sets ms
%           (with mode ordering given by opts.modeord). If a matrix, each
%           of ntrans columns is transformed with the same nonuniform targets.
%     isign if >=0, uses + sign in exponential, otherwise - sign.
%     eps   relative precision requested (generally between 1e-15 and 1e-1)
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
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.gpu_spreadinterponly: 0 (do NUFFT, default), 1 (only spread/interp)
%  Outputs:
%     c     complex column vector of nj answers at targets, or,
%           if ntrans>1, matrix of size (nj,ntrans).
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
function c = cufinufft1d2(x,isign,eps,f,o)

valid_setpts(true,2,1,x);
o.floatprec=underlyingType(x);             % should be 'double' or 'single'
[ms,n_transf]=size(f);                     % if f a col vec, n_transf=1, but...
if ms==1, ms=n_transf; n_transf=1; end     % allow a single row vec as valid f
p = cufinufft_plan(2,ms,isign,n_transf,eps,o);
p.setpts(x);
c = p.execute(f);
