% Pass/fail test for error over a range of tolerances, 3 types, fixed dim.
% CPU only. Raises error if fails; use for CI. Two upsampfacs for now.
% Simplified from plottolsweep; uses erralltypedim.
% Barnett 12/22/25, add CI vs fig-plot to results 1/20/26; all dims 1/21/26.

addpath(fileparts(mfilename('fullpath')))
clear % both single & double; just CPU for now...
precdevs = 'sd'; myrand   = @rand; devname  = 'CPU               ';
CI = false;  % leave true after debug! not CI: plot figs & don't raise errors

M = 500;            % NU pts in each test
ntr = 20;           % #transforms to average error over (little extra direct cost)
isign = +1;
sigmas = [1.25 2];             % a.k.a. upsampfac, list to test
tolslack = [4.0; 4.0; 5.0];    % factors by which eps can exceed tol (3 types)
warning('off','FINUFFT:epsTooSmall');
o.showwarn = 0; %o.debug=1;
o.spread_kerformula = 0;       % custom FINUFFT opts (should be defaults for CI)
tolsperdecade = 8;             % tol resolution
tolstep = 10 ^ (-1 / tolsperdecade);     % multiplicative step in tol, < 1

for dim = 1:3  % ======== dimensions
  fprintf('tolsweeptest dim=%d: =================================\n',dim)
  if dim==1
    Ntot = 50; % tot #modes (simply N when dim=1).
    % Ntot in 1D is subtle: too small (<50) gives unrealistic fast t1&2 convergence
    % at tol<1e-10, say, for sig=2. But too large (>200) starts to lose digits in
    % single-prec due to condition-num of problem. To keep floors32 around 1e-5
    % need keep N low.
    floors32 = [1e-4 2e-5];        % float: seemingly controlled by Nmax, rdyn
    floors64 = [1e-9 3e-14];       % double: former limited by wmax
  elseif dim==2
    Ntot = 1e3; floors32 = [1e-4 2e-5]; floors64 = [2e-9 3e-14];
  else
    Ntot = 2e3; floors32 = [2e-4 1e-5]; floors64 = [3e-8 3e-14];
  end
  dims = false(1, 3); dims(dim) = true;  % only test this dim

  if ~CI, figure('name','tolsweeptest','position',[100*dim 50*dim 1000 500*numel(sigmas)]); end

  for precdev=precdevs  % ......... loop precisions & devices
                        %  s=single, d=double; sd = CPU, SD = GPU
    if sum (precdev == 'sS'), prec = 'single'; else prec = 'double'; end
    mintol = 0.5 * eps(prec);        % stop at eps_mach
    ntols = ceil(log(mintol) / log(tolstep));
    tols = tolstep.^(0:ntols-1);     % go down from tol = 1
    for j=1:numel(sigmas), o.upsampfac = sigmas(j);         % ------- loop USF
      fprintf('%s\tsigma=%.3g prec=%s M=%d Ntot=%d ntr=%d ntols=%d\n',...
              devname, o.upsampfac, prec, M, Ntot, ntr, ntols)
      errs = nan(3, ntols);          % for 3 types (just 1D for now), each tol
      toloks = true(1,ntols);        % whether FINUFFT reported warning for tol
      for t=1:ntols
        tol = tols(t);
        lastwarn('');                  % clean up warnings
        [nineerrs, info] = erralltypedim(M,Ntot,ntr,isign,prec,tol,o,myrand,dims);
        errs(:,t) = nineerrs(:,dim);   % extract col from 3x3
        [~,id] = lastwarn; toloks(t) = ~strcmp(id, 'FINUFFT:epsTooSmall');
      end
      Nmax = info.Nmax(dims);    % currently unused
      if strcmp(prec,'single'), epsmin=floors32(j); else epsmin=floors64(j); end
      if ~CI, subplot(numel(sigmas),2,2*(j-1)+1+strcmp(prec,'double'));  % plot
        h0 = loglog(tols(toloks), errs(:,toloks), '+'); hold on;
        plot(tols(~toloks), errs(:,~toloks),'mo');    % highlight those w/ warning
        h1 = plot(tols, tols, 'k-');
        h2 = plot(tols, tolslack*tols, '--');
        h3 = plot(tols, 0*tols+epsmin, 'm--');
        axis([min(tols), max(tols), eps(prec), 1.0]);
        title(sprintf('%s %dD \\sigma=%g kf=%d M=%d N=%d',prec,dim,o.upsampfac,...
                      o.spread_kerformula,M,Ntot)); drawnow
      end
      for type=1:3       % simplified pass/fail criterion...
        e = errs(type,:);
        fails = e>tolslack(type)*tols & e>epsmin;
        failskeep = fails(toloks); tolskeep = tols(toloks); % discard warned cases
        ekeep = e(toloks);         % compute worst closeness factor to failure...
        worstfac = max(min(ekeep./(tolslack(type)*tolskeep), ekeep/epsmin));
        if max(failskeep), msg='FAIL'; else msg='pass'; end
        fprintf('\t\ttype %d: worstfac=%.3g\t %s\t\t',type,worstfac,msg);
        fprintf('%.3g ',tolskeep(failskeep)); fprintf('\n');   % list failed tols
        if CI & max(failskeep), error('FINUFFT tolsweeptest failed!'); end
      end
    end                                                    % --------
  end                    % .........
  if ~CI, print('-dpng',sprintf('results/tolsweeptest_%dD.png',dim)); end
end    % ==============
