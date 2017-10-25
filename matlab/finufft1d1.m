function [f ier] = finufft1d1(x,c,isign,eps,ms,o)
% FINUFFT1D1
%
% [f ier] = finufft1d1(x,c,isign,eps,ms)
% [f ier] = finufft1d1(x,c,isign,eps,ms,opts)
%
% Type-1 1D complex nonuniform FFT.
%
%               nj
%     f(k1) =  SUM c[j] exp(+/-i k1 x(j))  for -ms/2 <= k1 <= (ms-1)/2
%              j=1                            
%   Inputs:
%     x     location of sources on interval [-pi,pi], length nj
%     c     size-nj complex array of source strengths
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps     precision requested (>1e-16)
%     ms     number of Fourier modes computed, may be even or odd;
%            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
%     opts.debug: 0 (silent), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts in spreader), 1 (sort, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
%     opts.modeord: 0 (CMCL increasing mode ordering), 1 (FFT mode ordering)
%   Outputs:
%     f     size-ms double complex array of Fourier transform values
%            (increasing mode ordering)
%     ier - 0 if success, else:
%                     1 : eps too small
%		      2 : size of arrays to malloc exceed MAX_NF
%                     other codes: as returned by cnufftspread

if nargin<6, o=[]; end
debug=0; if isfield(o,'debug'), debug = o.debug; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
spread_sort=1; if isfield(o,'spread_sort'), spread_sort=o.spread_sort; end
fftw=0; if isfield(o,'fftw'), fftw=o.fftw; end
modeord=0; if isfield(o,'modeord'), modeord=o.modeord; end
nj=numel(x);
if numel(c)~=nj, error('c must have the same number of elements as x'); end

mex_id_ = 'o int = finufft1d1m(i double, i double[], i dcomplex[], i int, i double, i double, o dcomplex[x], i int, i int, i int, i int, i int)';
[ier, f] = finufft(mex_id_, nj, x, c, isign, eps, ms, debug, nthreads, spread_sort, fftw, modeord, ms);

% ---------------------------------------------------------------------------
