% FINUFFT2D3   2D complex nonuniform FFT of type 3 (nonuniform to nonuniform).
%
% f = finufft2d3(x,y,c,isign,eps,s,t)
% f = finufft2d3(x,y,c,isign,eps,s,t,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
%
%              nj
%     f[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j])),  for k = 1, ..., nk
%              j=1
%   Inputs:
%     x,y   coordinates of nonuniform sources in R^2, each a length-nj vector.
%     c     length-nj complex vector of source strengths. If numel(c)>nj,
%           expects a stack of vectors (eg, a nj*ntrans matrix) each of which is
%           transformed with the same source and target locations.
%     isign if >=0, uses + sign in exponential, otherwise - sign.
%     eps   relative precision requested (generally between 1e-15 and 1e-1)
%     s,t   frequency coordinates of nonuniform targets in R^2,
%           each a length-nk vector.
%     opts   optional struct with optional fields controlling the following:
%     opts.debug:   0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.spread_debug: spreader: 0 (no text, default), 1 (some), or 2 (lots)
%     opts.spread_sort:  0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.spread_kerevalmeth:  0: exp(sqrt()), 1: Horner ppval (faster)
%     opts.spread_kerpad: (iff kerevalmeth=0)  0: don't pad to mult of 4, 1: do
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT)
%     opts.spread_thread:   for ntrans>1 only. 0:auto, 1:seq multi, 2:par, etc
%     opts.maxbatchsize:  for ntrans>1 only. max blocking size, or 0 for auto.
%     opts.nthreads:   number of threads, or 0: use all available (default)
%   Outputs:
%     f     length-nk complex vector of values at targets, or, if ntrans>1,
%           a matrix of size (nk,ntrans)
%
% Notes:
%  * The vectorized (many vector) interface, ie ntrans>1, can be much faster
%    than repeated calls with the same nonuniform points. Note that here the I/O
%    data ordering is stacked rather than interleaved. See ../docs/matlab.rst
%  * The class of input x (double vs single) controls whether the double or
%    single precision library are called; precisions of all data should match.
%  * For more details about the opts fields, see ../docs/opts.rst
%  * See ERRHANDLER, VALID_* and FINUFFT_PLAN for possible warning/error IDs.
%  * Full documentation is online at http://finufft.readthedocs.io

function f = finufft2d3(x,y,c,isign,eps,s,t,o)

if nargin<8, o.dummy=1; end
valid_setpts(3,2,x,y,[],s,t,[]);
o.floatprec=class(x);                      % should be 'double' or 'single'
n_transf = valid_ntr(x,c);
p = finufft_plan(3,2,isign,n_transf,eps,o);
p.setpts(x,y,[],s,t,[]);
f = p.execute(c);
