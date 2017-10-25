function [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u,o)
% FINUFFT3D3
%
% [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u)
% [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u,opts)
%
%              nj
%     f[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j] + u[k] z[j])),
%              j=1
%                              for k = 1, ..., nk
%   Inputs:
%     x,y,z  location of NU sources in R^3, each length nj.
%     c      size-nj double complex array of source strengths
%     s,t,u   frequency locations of NU targets in R^3.
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)
%     opts.debug: 0 (silent), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts in spreader), 1 (sort, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s,t,u
%     returned value - 0 if success, else:
%                      1 : eps too small
%		       2 : size of arrays to malloc exceed MAX_NF

if nargin<10, o=[]; end
debug=0; if isfield(o,'debug'), debug = o.debug; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
spread_sort=1; if isfield(o,'spread_sort'), spread_sort=o.spread_sort; end
fftw=0; if isfield(o,'fftw'), fftw=o.fftw; end
nj=numel(x);
nk=numel(s);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
if numel(c)~=nj, error('c must have the same number of elements as x'); end
if numel(t)~=nk, error('t must have the same number of elements as s'); end
if numel(u)~=nk, error('u must have the same number of elements as s'); end

mex_id_ = 'o int = finufft3d3m(i double, i double[], i double[], i double[], i dcomplex[x], i int, i double, i double, i double[], i double[], i double[], o dcomplex[x], i int, i int, i int, i int)';
[ier, f] = finufft(mex_id_, nj, x, y, z, c, isign, eps, nk, s, t, u, debug, nthreads, spread_sort, fftw, nj, nk);

% ------------------------------------------------------------------------
