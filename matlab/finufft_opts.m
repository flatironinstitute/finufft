function opts = finufft_opts(o)
% FINUFFTS_OPTS.  Global set opts for matlab interface to FINUFFT, with defaults

% defaults, used if field absent or empty...
debug=0; if isfield(o,'debug') && ~isempty(o.debug), debug = o.debug; end
nthreads=0; if isfield(o,'nthreads') && ~isempty(o.nthreads), nthreads = o.nthreads; end
spread_sort=2; if isfield(o,'spread_sort') && ~isempty(o.spread_sort), spread_sort=o.spread_sort; end
fftw=0; if isfield(o,'fftw') && ~isempty(o.fftw), fftw=o.fftw; end
modeord=0; if isfield(o,'modeord') && ~isempty(o.modeord), modeord=o.modeord; end
chkbnds=0; if isfield(o,'chkbnds') && ~isempty(o.chkbnds), chkbnds=o.chkbnds; end

% pack up: ordering of opts must match that in finufft_m.cpp:finufft_mex_opts()
% (placement in opts now explicit, catches errors if inputs are not 1x1 sized)
opts = zeros(1,6,'double');
opts(1) = debug;
opts(2) = nthreads;
opts(3) = spread_sort;
opts(4) = fftw;
opts(5) = modeord;
opts(6) = chkbnds;
