% FINUFFT1D3   1D complex nonuniform FFT of type 3 (nonuniform to nonuniform).
%
% f = finufft1d3(x,c,isign,eps,s)
% f = finufft1d3(x,c,isign,eps,s,opts)
%
% This computes:
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
%            frequencies s
%
% Notes:
%  * All available threads are used; control how many with maxNumCompThreads.
%  * The above documents the simple (single-transform) interface. To transform
%    ntrans vectors together with the same nonuniform points, add a final
%    dimension of size ntrans>1 to the f and c arrays. See ../docs/matlab.rst
%  * For more details about the opts fields, see ../docs/opts.rst
%  * See ERRHANDLER for list of possible warning/error IDs.
%  * Full documentation is given in ../finufft-manual.pdf and online at
%    http://finufft.readthedocs.io

function f = finufft1d3(x,c,isign,eps,s,o)

if nargin<6, o.dummy=1; end
n_transf = round(numel(c)/numel(x));
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(3,1,isign,n_transf,eps,o);
p.finufft_setpts(x,[],[],s,[],[]);
f = p.finufft_exec(c);
