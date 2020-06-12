% FINUFFT3D3   3D complex nonuniform FFT of type 3 (nonuniform to nonuniform).
%
% [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u)
% [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
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
%     opts   optional struct with optional fields controlling the following:
%     opts.debug:   0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.spread_debug: spreader, (no text) 1 (some) or 2 (lots)
%     opts.spread_sort:  0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.spread_kerevalmeth:  0: exp(sqrt()), 1: Horner ppval (faster)
%     opts.spread_kerpad: (iff kerevalmeth=0)  0: don't pad to mult of 4, 1: do
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT)
%     opts.spread_thread:   for ntrans>1 only. 0:auto, 1:seq multi, 2:par, etc
%     opts.maxbatchsize:  for ntrans>1 only. max blocking size, or 0 for auto.
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s,t,u
%     ier   0 if success, else:
%           1 : eps too small (transform still performed at closest eps)
%           2 : size of arrays to malloc exceed MAX_NF
%           3 : spreader: fine grid too small compared to spread (kernel) width
%           4 : spreader: if chkbnds=1, nonuniform pt out of range [-3pi,3pi]^d
%           5 : spreader: array allocation error
%           6 : spreader: illegal direction (should be 1 or 2)
%           7 : upsampfac too small (should be >1.0)
%           8 : upsampfac not a value with known Horner poly eval rule
%           9 : ntrans invalid in "many" (vectorized) or guru interface
%          10 : transform type invalid (guru)
%          11 : general allocation failure
%          12 : dimension invalid (guru)
%
% Notes:
%  * All available threads are used; control how many with maxNumCompThreads.
%  * The above documents the simple (single-transform) interface. To transform
%    ntrans vectors together with the same nonuniform points, add a final
%    dimension of size ntrans>1 to the f and c arrays. See ../docs/matlab.rst
%  * For more details about the opts fields, see ../docs/opts.rst
%  * Full documentation is given in ../finufft-manual.pdf and online at
%    http://finufft.readthedocs.io

% FINUFFT3D3   3D complex nonuniform FFT of type 3 (nonuniform to nonuniform).
%
% [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u)
% [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
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
%     opts   optional struct with optional fields controlling the following:
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%   Outputs:
%     f     size-nk double complex Fourier transform values at target
%            frequencies s,t,u
%     ier   0 if success, else:
%           1 : eps too small (transform still performed at closest eps)
%           2 : size of arrays to malloc exceed MAX_NF
%           3 : spreader: fine grid too small compared to spread (kernel) width
%           4 : spreader: if chkbnds=1, nonuniform pt out of range [-3pi,3pi]^d
%           5 : spreader: array allocation error
%           6 : spreader: illegal direction (should be 1 or 2)
%           7 : upsampfac too small (should be >1.0)
%           8 : upsampfac not a value with known Horner poly eval rule
%           9 : ntrans invalid in "many" (vectorized) or guru interface
%          10 : transform type invalid (guru)
%          11 : general allocation failure
%          12 : dimension invalid (guru)
%
% Notes:
%  * All available threads are used; control how many with maxNumCompThreads.
%  * The above documents the simple (single-transform) interface. To transform
%    ntrans vectors together with the same nonuniform points, add a final
%    dimension of size ntrans>1 to the f and c arrays. See ../docs/matlab.rst
%  * For more details about the opts fields, see ../docs/opts.rst
%  * Full documentation is given in ../finufft-manual.pdf and online at
%    http://finufft.readthedocs.io

function [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u,o)

if nargin<10, o.dummy=1; end
n_transf = round(numel(c)/numel(x));
nj=numel(x);
nk=numel(s);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
if numel(t)~=nk, error('t must have the same number of elements as s'); end
if numel(u)~=nk, error('u must have the same number of elements as s'); end
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(3,3,isign,n_transf,eps,o);
p.finufft_setpts(x,y,z,s,t,u);
[f,ier] = p.finufft_exec(c);
