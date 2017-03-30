function [c ier] = finufft1d2(x,isign,eps,f,o)
% FINUFFT1D2
%
% [c ier] = finufft1d2(x,isign,eps,f)
% [c ier] = finufft1d2(x,isign,eps,f,opts)
%
% Type-2 1D complex nonuniform FFT.
%
%    c[j] = SUM   f[k1] exp(+/-i k1 x[j])      for j = 1,...,nj
%            k1 
%     where sum is over -ms/2 <= k1 <= (ms-1)/2.
%
%  Inputs:
%     x     location of NU targets on interval [-pi,pi], length nj
%     f     complex Fourier transform values (increasing mode ordering)
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)
%     opts.debug: 0 (silent), 1 (timing breakdown), 2 (debug info).
%     opts.maxnalloc - largest number of array elements for internal alloc
%                      (0 has no effect)
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts in spreader), 1 (sort, default)
%  Outputs:
%     c     complex double array of nj answers at targets
%     ier - 0 if success, else:
%                     1 : eps too small
%	       	      2 : size of arrays to malloc exceed opts.maxnalloc
%                     other codes: as returned by cnufftspread

if nargin<5, o=[]; end
debug=0; if isfield(o,'debug'), debug = o.debug; end
maxnalloc=0; if isfield(o,'maxnalloc'), maxnalloc = o.maxnalloc; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
spread_sort=1; if isfield(o,'spread_sort'), spread_sort=o.spread_sort; end
nj=numel(x);
ms=numel(f);
% c = complex(zeros(nj,1));   % todo: change all output to inout & prealloc...

mex_id_ = 'o int = finufft1d2m(i double, i double[], o dcomplex[x], i int, i double, i double, i dcomplex[], i int, i double, i int, i int)';
[ier, c] = finufft(mex_id_, nj, x, isign, eps, ms, f, debug, maxnalloc, nthreads, spread_sort, nj);

% ---------------------------------------------------------------------------
