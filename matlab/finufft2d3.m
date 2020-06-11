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
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s,t
%     returned value - 0 if success, else:
%                      1 : eps too small
%                      2 : size of arrays to malloc exceed MAX_NF
%
% All available threads are used; control how many with maxNumCompThreads

if nargin<8, o.dummy=1; end
n_transf = round(numel(c)/numel(x));
nj=numel(x);
nk=numel(s);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(t)~=nk, error('t must have the same number of elements as s'); end
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(3,2,isign,n_transf,eps,o);
p.finufft_setpts(x,y,[],s,t,[]);
[f,ier] = p.finufft_exec(c);
