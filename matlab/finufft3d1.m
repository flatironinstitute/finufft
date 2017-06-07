function [f ier] = finufft3d1(x,y,z,c,isign,eps,ms,mt,mu,o)
% FINUFFT3D1
%
% [f ier] = finufft3d1(x,y,z,c,isign,eps,ms,mt,mu)
% [f ier] = finufft3d1(x,y,z,c,isign,eps,ms,mt,mu,opts)
%
% Type-1 3D complex nonuniform FFT.
%
%                       nj
%     f[k1,k2,k3] =    SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
%                      j=1
% 
%     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
%         -mu/2 <= k3 <= (mu-1)/2.
%
%   Inputs:
%     x,y,z locations of NU sources on [-pi,pi]^3, each length nj
%     c     size-nj complex array of source strengths
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps     precision requested (>1e-16)
%     ms,mt,mu number of Fourier modes requested in x,y and z; each may be
%           even or odd.
%           In either case the mode range is integers lying in [-m/2, (m-1)/2]
%     opts.debug: 0 (silent), 1 (timing breakdown), 2 (debug info).
%     opts.maxnalloc - largest number of array elements for internal alloc
%                      (0 has no effect)
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts in spreader), 1 (sort, default)
%   Outputs:
%     f     size (ms*mt*mu) double complex array of Fourier transform values
%            (increasing mode ordering in each dimension, ms fastest, mu
%             slowest).
%     ier - 0 if success, else:
%                     1 : eps too small
%		      2 : size of arrays to malloc exceed opts.maxnalloc
%                     other codes: as returned by cnufftspread

if nargin<10, o=[]; end
debug=0; if isfield(o,'debug'), debug = o.debug; end
maxnalloc=0; if isfield(o,'maxnalloc'), maxnalloc = o.maxnalloc; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
spread_sort=1; if isfield(o,'spread_sort'), spread_sort=o.spread_sort; end
nj=numel(x);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
if numel(c)~=nj, error('c must have the same number of elements as x'); end
mtot = ms*mt*mu;                        % since mwrap can't output 3D arrays :(

mex_id_ = 'o int = finufft3d1m(i double, i double[], i double[], i double[], i dcomplex[], i int, i double, i double, i double, i double, o dcomplex[x], i int, i double, i int, i int)';
[ier, f] = finufft(mex_id_, nj, x, y, z, c, isign, eps, ms, mt, mu, debug, maxnalloc, nthreads, spread_sort, mtot);

f = reshape(f,[ms mt mu]);              % make a 3D array


% ---------------------------------------------------------------------------
