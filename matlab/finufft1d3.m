function [f ier] = finufft1d3(x,c,isign,eps,s,o)
% FINUFFT1D3
%
% [f ier] = finufft1d3(x,c,isign,eps,s)
% [f ier] = finufft1d3(x,c,isign,eps,s,opts)
%
%              nj
%     f[k]  =  SUM   c[j] exp(+-i s[k] x[j]),      for k = 1, ..., nk
%              j=1
%   Inputs:
%     x      location of NU sources in R (real line).
%     c      size-nj double complex array of source strengths
%     s      frequency locations of NU targets in R.
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)
%     opts.debug: 0 (silent), 1 (timing breakdown), 2 (debug info).
%     opts.maxnalloc - largest number of array elements for internal alloc
%                      (0 has no effect)
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts in spreader), 1 (sort, default)
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s
%     returned value - 0 if success, else:
%                      1 : eps too small
%		       2 : size of arrays to malloc exceed opts.maxnalloc

if nargin<6, o=[]; end
debug=0; if isfield(o,'debug'), debug = o.debug; end
maxnalloc=0; if isfield(o,'maxnalloc'), maxnalloc = o.maxnalloc; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
spread_sort=1; if isfield(o,'spread_sort'), spread_sort=o.spread_sort; end
nj=numel(x);
nk=numel(s);
if numel(c)~=nj, error('c must have the same number of elements as x'); end

mex_id_ = 'o int = finufft1d3m(i double, i double[], i dcomplex[x], i int, i double, i double, i double[], o dcomplex[x], i int, i double, i int, i int)';
[ier, f] = finufft(mex_id_, nj, x, c, isign, eps, nk, s, debug, maxnalloc, nthreads, spread_sort, nj, nk);

% ------------------------------------------------------------------------
