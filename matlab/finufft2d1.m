% FINUFFT2D1   2D complex nonuniform FFT of type 1 (nonuniform to uniform).
%
% f = finufft2d1(x,y,c,isign,eps,ms,mt)
% f = finufft2d1(x,y,c,isign,eps,ms,mt,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
%
%                   nj
%     f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
%                  j=1
%
%     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.
%
%   Inputs:
%     x,y    locations of NU sources on the square [-3pi,3pi]^2, each length nj
%     c      size-nj complex array of source strengths
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)
%     ms,mt  number of Fourier modes requested in x & y; each may be even or odd
%            in either case the mode range is integers lying in [-m/2, (m-1)/2]
%     opts   optional struct with optional fields controlling the following:
%     opts.debug:   0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.spread_debug: spreader, (no text) 1 (some) or 2 (lots)
%     opts.spread_sort:  0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.spread_kerevalmeth:  0: exp(sqrt()), 1: Horner ppval (faster)
%     opts.spread_kerpad: (iff kerevalmeth=0)  0: don't pad to mult of 4, 1: do
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT)
%     opts.spread_thread:   for ntrans>1 only. 0:auto, 1:seq multi, 2:par, etc
%     opts.maxbatchsize:  for ntrans>1 only. max blocking size, or 0 for auto.
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default)
%   Outputs:
%     f     size (ms*mt) double complex array of Fourier transform values
%           (ordering given by opts.modeord in each dimension, ms fast, mt slow)
%
% Notes:
%  * All available threads are used; control how many with maxNumCompThreads.
%  * The above documents the simple (single-transform) interface. To transform
%    ntrans vectors together with the same nonuniform points, add a final
%    dimension of size ntrans>1 to the f and c arrays. See ../docs/matlab.rst
%  * For more details about the opts fields, see ../docs/opts.rst
%  * See ERRHANDLER for list of possible warning/error IDs.
%  * Full documentation is given in ../finufft-manual.pdf and online at
%    http://finufft.readthedocs.io

function f = finufft2d1(x,y,c,isign,eps,ms,mt,o)

if nargin<8, o.dummy=1; end
nj=numel(x);
n_transf = round(numel(c)/numel(x));
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if n_transf*nj~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(1,[ms;mt],isign,n_transf,eps,o);
p.finufft_setpts(x,y,[],[],[],[]);
f = p.finufft_exec(c);
