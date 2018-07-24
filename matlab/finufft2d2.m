function [c ier] = finufft2d2(x,y,isign,eps,f,o)
% FINUFFT2D2
%
% [c ier] = finufft2d2(x,y,isign,eps,f)
% [c ier] = finufft2d2(x,y,isign,eps,f,opts)
%
% Type-2 2D complex nonuniform FFT.
%
%    c[j] =  SUM   f[k1,k2] exp(+/-i (k1 x[j] + k2 y[j]))  for j = 1,..,nj
%           k1,k2
%     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,
%
%  Inputs:
%     x,y   location of NU targets on the square [-3pi,3pi]^2, each length nj
%     f     size (ms,mt) complex Fourier transform value matrix
%           (mode ordering given by opts.modeord in each dimension)
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE, default), 1 (use FFTW_MEASURE)
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default).
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%  Outputs:
%     c     complex double array of nj answers at the targets.
%     ier - 0 if success, else:
%           1 : eps too small
%           2 : size of arrays to malloc exceed MAX_NF
%           other codes: as returned by cnufftspread

if nargin<6, o=[]; end
opts = finufft_opts(o);
nj=numel(x);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
[ms,mt]=size(f);

mex_id_ = 'o int = finufft2d2m(i double, i double[], i double[], o dcomplex[x], i int, i double, i double, i double, i dcomplex[], i double[])';
[ier, c] = finufft(mex_id_, nj, x, y, isign, eps, ms, mt, f, opts, nj);

% ---------------------------------------------------------------------------
