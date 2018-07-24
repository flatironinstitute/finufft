function [f ier] = finufft2d3(x,y,c,isign,eps,s,t,o)
% FINUFFT2D3
%
% [f ier] = finufft2d3(x,y,c,isign,eps,s,t)
% [f ier] = finufft2d3(x,y,c,isign,eps,s,t,opts)
%
%              nj
%     f[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j])),  for k = 1, ..., nk
%              j=1
%   Inputs:
%     x,y    location of NU sources in R^2, each length nj.
%     c      size-nj double complex array of source strengths
%     s,t    frequency locations of NU targets in R^2.
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE, default), 1 (use FFTW_MEASURE)
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s,t
%     returned value - 0 if success, else:
%                      1 : eps too small
%                      2 : size of arrays to malloc exceed MAX_NF

if nargin<8, o=[]; end
opts = finufft_opts(o);
nj=numel(x);
nk=numel(s);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(c)~=nj, error('c must have the same number of elements as x'); end
if numel(t)~=nk, error('t must have the same number of elements as s'); end

mex_id_ = 'o int = finufft2d3m(i double, i double[], i double[], i dcomplex[x], i int, i double, i double, i double[], i double[], o dcomplex[x], i double[])';
[ier, f] = finufft(mex_id_, nj, x, y, c, isign, eps, nk, s, t, opts, nj, nk);

% ------------------------------------------------------------------------
