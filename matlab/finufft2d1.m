function [f ier] = finufft2d1(x,y,c,isign,eps,ms,mt,o)
% FINUFFT2D1
%
% [f ier] = finufft2d1(x,y,c,isign,eps,ms,mt)
% [f ier] = finufft2d1(x,y,c,isign,eps,ms,mt,opts)
%
% Type-1 2D complex nonuniform FFT.
%
%                 1   nj
%     f[k1,k2] =  --  SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
%                 nj  j=1
% 
%     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2, and nj>0.
%                        if nj=0, f identically zero.
%   Inputs:
%     x,y   locations of NU sources on the square [-pi,pi]^2, each length nj
%     c     size-nj complex array of source strengths
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps     precision requested (>1e-16)
%     ms,mt  number of Fourier modes requested in x & y; each may be even or odd
%           in either case the mode range is integers lying in [-m/2, (m-1)/2]
%     opts.debug: 0 (silent), 1 (timing breakdown), 2 (debug info).
%     opts.maxnalloc - largest number of array elements for internal alloc
%                      (0 has no effect)
%     opts.nthreads sets requested number of threads (else automatic)
%   Outputs:
%     f     size (ms*mt) double complex array of Fourier transform values
%            (increasing mode ordering)
%     ier - 0 if success, else:
%                     1 : eps too small
%		      2 : size of arrays to malloc exceed opts.maxnalloc
%                     other codes: as returned by cnufftspread

if nargin<8, o=[]; end
debug=0; if isfield(o,'debug'), debug = o.debug; end
maxnalloc=0; if isfield(o,'maxnalloc'), maxnalloc = o.maxnalloc; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
nj=numel(x);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(c)~=nj, error('c must have the same number of elements as x'); end

mex_id_ = 'o int = finufft2d1m(i double, i double[], i double[], i dcomplex[], i int, i double, i double, i double, o dcomplex[xx], i int, i double, i int)';
[ier, f] = finufft(mex_id_, nj, x, y, c, isign, eps, ms, mt, debug, maxnalloc, nthreads, ms, mt);

% ---------------------------------------------------------------------------
