% FINUFFT3D1   3D complex nonuniform FFT of type 1 (nonuniform to uniform).
%
% f = finufft3d1(x,y,z,c,isign,eps,ms,mt,mu)
% f = finufft3d1(x,y,z,c,isign,eps,ms,mt,mu,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
%
%                       nj
%     f[k1,k2,k3] =    SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
%                      j=1
%
%     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
%         -mu/2 <= k3 <= (mu-1)/2.
%
%   Inputs:
%     x,y,z real-valued coordinates of nonuniform sources,
%           each a length-nj vector
%     c     length-nj complex vector of source strengths. If numel(c)>nj,
%           expects a stack of vectors (eg, a nj*ntrans matrix) each of which is
%           transformed with the same source locations.
%     isign if >=0, uses + sign in exponential, otherwise - sign.
%     eps   relative precision requested (generally between 1e-15 and 1e-1)
%     ms,mt,mu  number of Fourier modes requested in x,y and z; each may be
%           even or odd.
%           In either case the mode range is integers lying in [-m/2, (m-1)/2]
%     opts   optional struct with optional fields controlling the following:
%     opts.debug:   0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.spread_debug: spreader: 0 (no text, default), 1 (some), or 2 (lots)
%     opts.spread_sort:  0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.spread_kerevalmeth:  0: exp(sqrt()), 1: Horner ppval (faster)
%     opts.spread_kerpad: (iff kerevalmeth=0)  0: don't pad to mult of 4, 1: do
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT)
%     opts.spread_thread:   for ntrans>1 only. 0:auto, 1:seq multi, 2:par, etc
%     opts.maxbatchsize:  for ntrans>1 only. max blocking size, or 0 for auto.
%     opts.nthreads:   number of threads, or 0: use all available (default)
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: [DEPRECATED] has no effect
%   Outputs:
%     f     size (ms,mt,mu) complex array of Fourier coefficients
%           (ordering given by opts.modeord in each dimension; ms fastest, mu
%           slowest), or, if ntrans>1, a 4D array of size (ms,mt,mu,ntrans).
%
% Notes:
%  * The vectorized (many vector) interface, ie ntrans>1, can be much faster
%    than repeated calls with the same nonuniform points. Note that here the I/O
%    data ordering is stacked rather than interleaved. See ../docs/matlab.rst
%  * The class of input x (double vs single) controls whether the double or
%    single precision library are called; precisions of all data should match.
%  * For more details about the opts fields, see ../docs/opts.rst
%  * See ERRHANDLER, VALID_* and FINUFFT_PLAN for possible warning/error IDs.
%  * Full documentation is online at http://finufft.readthedocs.io

function f = finufft3d1(x,y,z,c,isign,eps,ms,mt,mu,o)

if nargin<10, o.dummy=1; end
valid_setpts(1,3,x,y,z);
o.floatprec=class(x);                      % should be 'double' or 'single'
n_transf = valid_ntr(x,c);
p = finufft_plan(1,[ms;mt;mu],isign,n_transf,eps,o);
p.setpts(x,y,z);
f = p.execute(c);
