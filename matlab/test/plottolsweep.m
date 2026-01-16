% MATLAB plot achieved error over a range of tolerances, 3 types, 2 precs.
% Only a single choice of dim for now, and CPU only.
% Successor to obsolete fig_accuracy.m; uses erralltypedim.
% Candidate pass/fail criteria are overly complicated here; see
% tolsweeptest.m for better ones.
% Barnett 12/21/25. Added candidate pass/fail criteria 12/22/25.

addpath(fileparts(mfilename('fullpath')))
clear   % both single & double; just CPU for now...
precdevs = 'sd'; myrand   = @rand; devname  = 'CPU               ';

M = 1e3;              % # NU pts (several secs for >=1e4)
dim = 1; Ntot = 30;   % which dimensionality to test, tot #modes
%dim = 2; Ntot = 480; % (note when only 10 modes / dim, fluctuates vs ns, and misleading faster t12 conv)
%dim = 3; Ntot = 1000; % ditto
ntr = 10;             % #transforms to average error over
isign = +1;
o.debug = 0; o.showwarn=0;        % any FINUFFT opts...
o.upsampfac = 2.0;  %2.0  etc
o.spread_kerformula = 1;
% failure fudge factors...
tolslack = [5.0; 5.0; 15.0];  % factors by which eps can exceed tol (3 types)
floorslack = 20;
w_max=16;            % MAX_NSPREAD
sigma = o.upsampfac; % assume explicitly given
eps_wmax = exp(-pi*w_max*sqrt(1-1/sigma));  % eps floor, conv rate from paper

dims = false(1, 3); dims(dim) = true;    % only test one dim for now
tolsperdecade = 8;
tolstep = 10 ^ (-1 / tolsperdecade); % multiplicative step in tol, < 1
figure('position', [500 500 1000 500]);
iplot = 1;            % subplot counter
warning('off','FINUFFT:epsTooSmall');

for precdev=precdevs  % ......... loop precisions & devices
                      %  s=single, d=double; sd = CPU, SD = GPU
  if sum (precdev == 'sS'), prec = 'single'; else prec = 'double'; end
  mintol = 0.5 * eps(prec);        % stop at eps_mach
  ntols = ceil(log(mintol) / log(tolstep));
  fprintf('%s\tprec=%s\t%dD, M=%d, Ntot=%d, ntrans=%d, ntols=%d\n', ...
          devname, prec, dim, M, Ntot, ntr, ntols)
  tols = tolstep.^(0:ntols-1);     % go down from tol = 1
  errs = nan(3, ntols);            % for 3 types (just 1D for now), each tol
  toloks = true(1,ntols);          % whether FINUFFT reported warning for tol
  for t=1:ntols
    tol = tols(t);
    lastwarn('');                  % clean up warnings
    [nineerrs, info] = erralltypedim(M,Ntot,ntr,isign,prec,tol,o,myrand,dims);
    errs(:,t) = nineerrs(:,dim);   % extract col from 3x3
    [~,id] = lastwarn; toloks(t) = ~strcmp(id, 'FINUFFT:epsTooSmall');
    %o.debug = 0; if max(errs(:,t))>0.1, o.debug = 1; end  % <- for detective
  end
  Nmax = info.Nmax(dims);
  % pass/fail... epsfloor is worst of w_max or cond # limit
  % estim w for each tol using fixed sigma...
  ws_est = min(w_max, ceil(log(1./tols)/(pi*sqrt(1-1/sigma))));
  rdyns = exp(pi*ws_est*(1-0.5/sigma - sqrt(1-1/sigma)));  % rdyn's, from paper
  for type=1:3       % criterion... 3 indep effects
    e = errs(type,:);
    bwmax = floorslack*eps_wmax;          % bnd due to w maxing out
    bCC = 1.0*Nmax*eps(prec)*rdyns;       % rounding err with CC of rdyn,
                                          % t3 seems to be Nmax^2; keep N small
    fails = e>tolslack(type)*tols & e>bwmax & e>bCC;
    failskeep = fails(toloks); tolskeep = tols(toloks);  % discard warned cases
    fprintf('\t\ttype %d: fail=%d\t\t',type,max(failskeep));
    fprintf('%.3g ',tolskeep(failskeep)); fprintf('\n');   % list failed tols
  end
  subplot(1,2,iplot);     % overplot all types...
  h0 = loglog(tols(toloks), errs(:,toloks), '+'); hold on;
  plot(tols(~toloks), errs(:,~toloks),'mo'); % highlight those w / tol warning
  h1 = plot(tols, tols, 'k-');
  h2 = plot(tols, tolslack*tols, 'k--');
  h3 = plot(tols, bwmax + 0*tols, 'm--');
  h4 = plot(tols, bCC, 'c--');
  axis([min(tols), max(tols), eps(prec), max(errs(:))]);
  xlabel('tol'); ylabel('rel 2-norm err');
  legend([h0;h1;h3;h4], 'type 1', 'type 2', 'type 3', 'tol',...
         'b_{wmax}', 'b_{CC}', 'location','nw');
  title(sprintf('%s %dD \\sigma=%g kf=%d M=%d N=%d: rel errs vs tol', ...
                prec, dim, o.upsampfac, o.spread_kerformula, M, Ntot));
  drawnow
  iplot = iplot + 1;
end % ..................
