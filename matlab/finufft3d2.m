function [c ier] = finufft3d2(x,y,z,isign,eps,f,o)
% FINUFFT3D2
%
% [c ier] = finufft3d2(x,y,z,isign,eps,f)
% [c ier] = finufft3d2(x,y,z,isign,eps,f,opts)
%
% Type-2 3D complex nonuniform FFT.
%
%    c[j] =   SUM   f[k1,k2,k3] exp(+/-i (k1 x[j] + k2 y[j] + k3 z[j]))
%           k1,k2,k3       
%                            for j = 1,..,nj
%     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,
%                       -mu/2 <= k3 <= (mu-1)/2.
%
%  Inputs:
%     x,y,z location of NU targets on cube [-pi,pi]^3, each length nj
%     f     size (ms,mt,mu) complex Fourier transform value matrix
%           (increasing mode ordering in each dimension; ms fastest and mu
%            slowest).
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)
%     opts.debug: 0 (silent), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts in spreader), 1 (sort, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
%     opts.modeord: 0 (CMCL increasing mode ordering), 1 (FFT mode ordering)
%  Outputs:
%     c     complex double array of nj answers at the targets.
%     ier - 0 if success, else:
%                     1 : eps too small
%	       	      2 : size of arrays to malloc exceed MAX_NF
%                     other codes: as returned by cnufftspread

if nargin<7, o=[]; end
debug=0; if isfield(o,'debug'), debug = o.debug; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
spread_sort=1; if isfield(o,'spread_sort'), spread_sort=o.spread_sort; end
fftw=0; if isfield(o,'fftw'), fftw=o.fftw; end
modeord=0; if isfield(o,'modeord'), modeord=o.modeord; end
nj=numel(x);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
[ms,mt,mu]=size(f);

mex_id_ = 'o int = finufft3d2m(i double, i double[], i double[], i double[], o dcomplex[x], i int, i double, i double, i double, i double, i dcomplex[], i int, i int, i int, i int, i int)';
[ier, c] = finufft(mex_id_, nj, x, y, z, isign, eps, ms, mt, mu, f, debug, nthreads, spread_sort, fftw, modeord, nj);

% ---------------------------------------------------------------------------
