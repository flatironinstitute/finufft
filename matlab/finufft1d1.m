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
%     x     location of sources on interval [-3pi,3pi], length nj
%     c     size-nj complex array of source strengths
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps     precision requested (>1e-16)
%     ms     number of Fourier modes computed, may be even or odd;
%            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE, default), 1 (use FFTW_MEASURE)
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default)
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%   Outputs:
%     f     size-ms double complex array of Fourier transform values
%     ier - 0 if success, else:
%           1 : eps too small
%           2 : size of arrays to malloc exceed MAX_NF
%           other codes: as returned by spreadinterp


% Alex prototyping how the simple & "auto-detect many" interface could look.
% 5/17/20.
% Have not tested ntransf>1 yet.

if nargin<6, o.dummy=1; end
n_transf = round(numel(c)/numel(x));   % back out how many transf
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(1,ms,isign,n_transf,eps,o);
p.finufft_setpts(x,[],[],[],[],[]);
[f,ier] = p.finufft_exec(c);
