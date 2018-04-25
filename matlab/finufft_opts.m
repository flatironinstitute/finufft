function opts = finufft_opts(o)
% FINUFFTS_OPTS.  Global set opts for matlab interface to FINUFFT, with defaults

debug=0; if isfield(o,'debug'), debug = o.debug; end
nthreads=0; if isfield(o,'nthreads'), nthreads = o.nthreads; end
spread_sort=2; if isfield(o,'spread_sort'), spread_sort=o.spread_sort; end
fftw=0; if isfield(o,'fftw'), fftw=o.fftw; end
modeord=0; if isfield(o,'modeord'), modeord=o.modeord; end
chkbnds=1; if isfield(o,'chkbnds'), chkbnds=o.chkbnds; end

% pack up: ordering of opts must match that in finufft_m.cpp:finufft_mex_opts()
opts = double([debug,nthreads,spread_sort,fftw,modeord,chkbnds]);
