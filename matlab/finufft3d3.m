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
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE, default), 1 (use FFTW_MEASURE)
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s,t,u
%     returned value - 0 if success, else:
%                      1 : eps too small
%                      2 : size of arrays to malloc exceed MAX_NF

if nargin<10, o=[]; end
opts = finufft_opts(o);
nj=numel(x);
nk=numel(s);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
if numel(c)~=nj, error('c must have the same number of elements as x'); end
if numel(t)~=nk, error('t must have the same number of elements as s'); end
if numel(u)~=nk, error('u must have the same number of elements as s'); end

mex_id_ = 'o int = finufft3d3m(i double, i double[], i double[], i double[], i dcomplex[x], i int, i double, i double, i double[], i double[], i double[], o dcomplex[x], i double[])';
[ier, f] = finufft(mex_id_, nj, x, y, z, c, isign, eps, nk, s, t, u, opts, nj, nk);
