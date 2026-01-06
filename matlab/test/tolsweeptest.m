% Pass/fail test for error over a range of tolerances, 3 types, 1D fixed N.
% CPU only. Raises error if fails; use for CI. Two upsampfacs for now.
% Simplified from plottolsweep; uses erralltypedim.
% Barnett 12/22/25.

addpath(fileparts(mfilename('fullpath')))
clear % both single & double; just CPU for now...
precdevs = 'sd'; myrand   = @rand; devname  = 'CPU               ';

M = 1e3;            % # NU pts (several secs for >=1e4)
dim = 1; Ntot = 30; % which dimensionality to test, tot #modes
%dim = 2; Ntot = 400;
%dim = 3; Ntot = 1e3;
ntr = 10;           % #transforms to average error over
isign = +1;
sigmas = [1.25 2];             % a.k.a. upsampfac, list to test (v2.4.1 for now)
floors32 = [1e-4 1e-5];        % float: seemingly controlled by rdyn
floors64 = [3e-9 3e-14];       % double: former limited by wmax
tolslack = [5.0; 5.0; 10.0];   % factors by which eps can exceed tol (3 types)
                               %tolslack=dim*tolslack;  % hack
o.showwarn = 0;
warning('off','FINUFFT:epsTooSmall');
o.spread_kerformula = 0;         % any custom FINUFFT opts...
dims = false(1, 3); dims(dim) = true;  % only test this dim
tolsperdecade = 8;
tolstep = 10 ^ (-1 / tolsperdecade); % multiplicative step in tol, < 1

for precdev=precdevs  % ......... loop precisions & devices
                      %  s=single, d=double; sd = CPU, SD = GPU
  if sum (precdev == 'sS'), prec = 'single'; else prec = 'double'; end
  mintol = 0.5 * eps(prec);        % stop at eps_mach
  ntols = ceil(log(mintol) / log(tolstep));
  tols = tolstep.^(0:ntols-1);     % go down from tol = 1
  for j=1:numel(sigmas), o.upsampfac = sigmas(j);         % ------- loop USF
    fprintf('%s\tsigma=%.3g\tprec=%s M=%d Ntot=%d ntr=%d ntols=%d\n',...
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
    Nmax = info.Nmax(dims);
    if strcmp(prec,'single'), epsmin=floors32(j); else epsmin=floors64(j); end
    for type=1:3       % simplified pass/fail criterion...
      e = errs(type,:);
      fails = e>tolslack(type)*tols & e>epsmin;
      failskeep = fails(toloks); tolskeep = tols(toloks); % discard warned cases
      ekeep = e(toloks);         % compute worst closeness factor to failure...
      worstfac = max(min(ekeep./(tolslack(type)*tolskeep), ekeep/epsmin));
      if max(failskeep), msg='FAIL'; else msg='pass'; end
      fprintf('\t\ttype %d: worstfac=%.3g\t %s\t\t',type,worstfac,msg);
      fprintf('%.3g ',tolskeep(failskeep)); fprintf('\n');   % list failed tols
      % cause CI to fail out:
      if max(failskeep), error('FINUFFT tolsweeptest failed!'); end
    end
  end                                                    % --------
end                    % .........
