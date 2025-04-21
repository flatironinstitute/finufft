% CUFINUFFT3D2   GPU 3D complex nonuniform FFT, type 2 (uniform to nonuniform).
%
% c = cufinufft3d2(x,y,z,isign,eps,f)
% c = cufinufft3d2(x,y,z,isign,eps,f,opts)
%
% This computes on the GPU, to relative precision eps, via a fast algorithm:
%
%    c[j] =   SUM   f[k1,k2,k3] exp(+/-i (k1 x[j] + k2 y[j] + k3 z[j]))
%           k1,k2,k3
%                            for j = 1,..,nj
%     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,
%                       -mu/2 <= k3 <= (mu-1)/2.
%
%  Inputs:
%     x,y,z real-valued coordinates of nonuniform targets,
%           each a vector of length nj
%     f     complex Fourier coefficient array, whose size sets (ms,mt,mu).
%           (Mode ordering given by opts.modeord, in each dimension.)
%           If a 4D array, 4th dimension sets ntrans, and each of ntrans
%           3D arrays is transformed with the same nonuniform targets.
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
function c = cufinufft3d2(x,y,z,isign,eps,f,o)

valid_setpts(true,2,3,x,y,z);
o.floatprec=underlyingType(x);             % should be 'double' or 'single'
[ms,mt,mu,n_transf] = size(f);             % if f 3D array, n_transf=1
p = cufinufft_plan(2,[ms;mt;mu],isign,n_transf,eps,o);
p.setpts(x,y,z);
c = p.execute(f);
