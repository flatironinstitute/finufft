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
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE, default), 1 (use FFTW_MEASURE)
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s
%     returned value - 0 if success, else:
%                      1 : eps too small
%                      2 : size of arrays to malloc exceed MAX_NF

if nargin<6, o=[]; end
opts = finufft_opts(o);
nj=numel(x);
nk=numel(s);
if numel(c)~=nj, error('c must have the same number of elements as x'); end

mex_id_ = 'o int = finufft1d3m(i double, i double[], i dcomplex[x], i int, i double, i double, i double[], o dcomplex[x], i double[])';
[ier, f] = finufft(mex_id_, nj, x, c, isign, eps, nk, s, opts, nj, nk);

% ------------------------------------------------------------------------
