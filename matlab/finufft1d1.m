function [f ier] = finufft1d1(x,c,isign,eps,ms,o)
% FINUFFT1D1   1D complex nonuniform FFT of type 1 (nonuniform to uniform).
%
% [f ier] = finufft1d1(x,c,isign,eps,ms)
% [f ier] = finufft1d1(x,c,isign,eps,ms,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
%
%               nj
%     f(k1) =  SUM c[j] exp(+/-i k1 x(j))  for -ms/2 <= k1 <= (ms-1)/2
%              j=1
%   Inputs:
%     x     location of sources on interval [-3pi,3pi], length nj
%     c     size-nj complex array of source strengths
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps     precision requested (>1e-16)
%     ms     number of Fourier modes computed, may be even or odd;
%            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
%     opts   optional struct with optional fields controlling the following:
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default)
%   Outputs:
%     f     size-ms double complex array of Fourier transform values
%     ier   0 if success, else:
%           1 : eps too small (transform still performed at closest eps)
%           2 : size of arrays to malloc exceed MAX_NF
%           3 : spreader: fine grid too small compared to spread (kernel) width
%           4 : spreader: if chkbnds=1, nonuniform pt out of range [-3pi,3pi]^d
%           5 : spreader: array allocation error
%           6 : spreader: illegal direction (should be 1 or 2)
%           7 : upsampfac too small (should be >1.0)
%           8 : upsampfac not a value with known Horner poly eval rule
%           9 : ntrans invalid in "many" (vectorized) or guru interface
%          10 : transform type invalid (guru)
%          11 : general allocation failure
%          12 : dimension invalid (guru)
%
% Notes:
%  * All available threads are used; control how many with maxNumCompThreads.
%  * The above documents the simple (single-transform) interface. To transform
%    ntrans vectors together with the same nonuniform points, add a final
%    dimension of size ntrans>1 to the f and c arrays. See ../docs/matlab.rst
%  * For more details about the opts fields, see ../docs/opts.rst
%  * Full documentation is given in ../finufft-manual.pdf and online at
%    http://finufft.readthedocs.io


if nargin<6, o.dummy=1; end            % make a dummy options struct
n_transf = round(numel(c)/numel(x));   % back out how many transf
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(1,ms,isign,n_transf,eps,o);
p.finufft_setpts(x,[],[],[],[],[]);
[f,ier] = p.finufft_exec(c);
