% FINUFFT1D1   1D complex nonuniform FFT of type 1 (nonuniform to uniform).
%
% f = finufft1d1(x,c,isign,eps,ms)
% f = finufft1d1(x,c,isign,eps,ms,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
%
%               nj
%     f(k1) =  SUM c[j] exp(+/-i k1 x(j))  for -ms/2 <= k1 <= (ms-1)/2
%              j=1
%   Inputs:
%     x     location of sources on interval [-3pi,3pi], length nj
%     c     size-nj complex array of source strengths
%     isign if >=0, uses + sign in exponential, otherwise - sign.
%     eps   precision requested (>1e-16)
%     ms    number of Fourier modes computed, may be even or odd;
%           in either case, mode range is integers lying in [-ms/2, (ms-1)/2]
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
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default)
%   Outputs:
%     f     size-ms double complex array of Fourier transform values
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

function f = finufft1d1(x,c,isign,eps,ms,o)

if nargin<6, o.dummy=1; end            % make a dummy options struct
M = numel(x);
if M==1, c = c(:).';    % silly case of many M=1 trans: make row vec.
else, if isvector(c), c = c(:); end
end
n_transf = size(c,2);
p = finufft_plan(1,ms,isign,n_transf,eps,o);
p.finufft_setpts(x,[],[]);
f = p.finufft_exec(c);
