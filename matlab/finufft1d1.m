function [f ier] = finufft1d1(x,c,isign,eps,ms,o)
% FINUFFT1D1
%
% [f ier] = finufft1d1(x,c,isign,eps,ms)
% [f ier] = finufft1d1(x,c,isign,eps,ms,opts)
%
% Type-1 1D complex nonuniform FFT.
%
%               nj
%     f(k1) =  SUM c[j] exp(+/-i k1 x(j))  for -ms/2 <= k1 <= (ms-1)/2
%              j=1
%   Inputs:
%     x     location of sources on interval [-3pi,3pi], length nj
%     c     size-nj complex array of source strengths
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps     precision requested (>1e-16)
%     ms     number of Fourier modes computed, may be even or odd;
%            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
%     opts.debug: 0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.nthreads sets requested number of threads (else automatic)
%     opts.spread_sort: 0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.fftw: 0 (use FFTW_ESTIMATE, default), 1 (use FFTW_MEASURE)
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default)
%     opts.upsampfac: either 2.0 (default), or 1.25 (low RAM, smaller FFT size)
%   Outputs:
%     f     size-ms double complex array of Fourier transform values
%     ier - 0 if success, else:
%           1 : eps too small
%           2 : size of arrays to malloc exceed MAX_NF
%           other codes: as returned by cnufftspread


% Alex prototyping how the simple & "auto-detect many" interface could look.
% 5/17/20
opts = nufft_opts();

% problem is now how to get the passed-in o fields into opts! :
if isfield(o,'debug'), opts.set_debug(o.debug); end
if isfield(o,'upsampfac'), opts.set_upsampfac(o.upsampfac); end
% this is repetitive and not very maintainable, to add new opts fields.
% ugh
% Would be fixed by making opts a simple matlab struct, not a matlab obj.

p = nufft_plan();       % would be nice to combine this w/ next line :)
n_modes = [ms];
p.nufft_makeplan(1,n_modes,isign,1,eps,0,opts);   % blsize=0 is default
p.nufft_setpts(x,[],[],[],[],[]);
[f,ier] = p.nufft_excute(c); 
