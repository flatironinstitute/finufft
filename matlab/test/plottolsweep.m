% MATLAB plot achieved error over a range of tolerances, 3 types, 2 precs.
% Only a single choice of dim for now, and CPU only.
% Successor to obsolete fig_accuracy.m; uses erralltypedim.
% Barnett 12/21/25.
clear % both single & double; just CPU for now...
precdevs = 'sd'; myrand   = @rand; devname  = 'CPU               ';

M = 1e3;            % # NU pts (several secs for >=1e4)
dim = 1; Ntot = 30; % which dimensionality to test, tot #modes
% dim = 2; Ntot = 100; % (note when only 10 modes / dim, fluctuates vs ns)
% dim = 3; Ntot = 1000; % ditto
ntr = 10;           % #transforms to average error over
isign = +1;
o.debug = 0;        % any FINUFFT opts...
o.upsampfac = 2; % 1.25;
o.spread_function = 0;

dims = false(1, 3); dims(dim) = true;  % only test this dim
tolsperdecade = 8;
tolstep = 10 ^ (-1 / tolsperdecade); % multiplicative step in tol, < 1
figure('position', [500 500 1000 500]);
iplot = 1;            % subplot counter

for precdev=precdevs  % ......... loop precisions & devices
                      %  s=single, d=double; sd = CPU, SD = GPU
  if sum (precdev == 'sS'), prec = 'single'; else prec = 'double'; end
  mintol = 0.5 * eps(prec); % stop below eps_mach
  ntols = ceil(log(mintol) / log(tolstep));
  fprintf('%s\tprec=%s\tM=%d, Ntot=%d, ntrans=%d, ntols=%d\n', ...
          devname, prec, M, Ntot, ntr, ntols)
  tols = tolstep.^(0:ntols-1);     % start at tol = 1 &go down
  errs = nan(3, ntols);            % for 3 types (just 1D for now), each tol
  toloks = true(1,ntols);          % whether FINUFFT reported warning for tol
  for t=1:ntols
    tol = tols(t);
    lastwarn('');                  % clean up warnings
    nineerrs = erralltypedim(M,Ntot,ntr,isign,prec,tol,o,myrand,dims);  % 3x3
    errs(:,t) = nineerrs(:,dim);
    [~,id] = lastwarn; toloks(t) = ~strcmp(id, 'FINUFFT:epsTooSmall');
  end
  subplot(1,2,iplot);
  h0 = loglog(tols(toloks), errs(:,toloks), '+'); hold on;
  h1 = plot(tols, tols, 'k-');
  plot(tols(~toloks), errs(:,~toloks),'mo'); % highlight those w / tol warning
  axis tight; xlabel('tol'); ylabel('rel 2-norm err');
  legend([h0; h1], 'type 1', 'type 2', 'type 3', 'tol', 'location', 'nw');
  title(sprintf('%s %dD \\sigma=%g sf=%d M=%d N=%d: rel errs vs tol', ...
                prec, dim, o.upsampfac, o.spread_function, M, Ntot));
  drawnow
  iplot = iplot + 1;
end % ..................
