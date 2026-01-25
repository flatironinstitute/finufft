% Comparison of several kerformulae: error vs widths w, 3 types, fix dim,
% CPU only. One sigma for now.
% This measures the goodness of shape (beta) choice for each type (kf) and w.
% It is unaffected by the logic of choosing w appropriate for a given tol.
% 
% Based on tol sweep from tolsweeptest, using erralltypedim to meas err,
% and extracts w from a spreadinterponly call (ignores tol).
% Barnett 1/16/26. Added err-vs-tol comparison 1/20/26; PSWF 1/23/26.

addpath(fileparts(mfilename('fullpath')))
clear
prec = 'double';  % working precision
myrand = @rand;   % select CPU

M = 1e3;             % # NU pts (several secs for >=1e4)
dim = 1; Ntot = 300; % which dimensionality to test, tot #modes (not too small)
% (weird thing is N small, eg 32, makes KB look better)
%dim = 2; Ntot = 400;  % or try other dims...
                      %dim = 3; Ntot = 1e3;
ntr = 20;           % #transforms to average error over at each tol (was 10)
isign = +1;
sigma = 2;
tolsperdecade = 10;
tolstep = 10 ^ (-1 / tolsperdecade); % multiplicative step in tol, < 1
% following names must match src/finufft_common/kernel.h:
kfnam = {"ES legacy", "ES Beatty", "KB Beatty", "cont-KB Beatty", "cosh-type Bea", "cont cosh Bea", "PSWF Beatty", "PSWF beta-shift", "PSWF Marco"};
kfs = [1 8];       % kernel formulae to test

o.upsampfac = sigma;
%o.debug = 1;
o.showwarn = 0; warning('off','FINUFFT:epsTooSmall');
dims = false(1, 3); dims(dim) = true;  % only test this dim
nkf = numel(kfs);
mintol = 10 * eps(prec);        % stop above eps_mach
ntols = ceil(log(mintol) / log(tolstep));
tols = tolstep.^(0:ntols-1);     % go down from tol = 1
fprintf('%dD sigma=%.3g\tprec=%s M=%d Ntot=%d ntr=%d ntols=%d, kfs:',...
        dim, o.upsampfac, prec, M, Ntot, ntr, ntols);
fprintf(' %d',kfs); fprintf('\n');
errs = nan(nkf, 3, ntols);     % for 3 types (just 1D for now), each tol
toloks = true(1,ntols);        % whether FINUFFT reported warning for tol
ws = zeros(nkf, ntols);        % extracted widths w
for t=1:ntols
  tol = tols(t);
  for i = 1:numel(kfs)  % loop over kernel formulae
    o.spread_kerformula = kfs(i);
    %o.debug = (tol<1e-9 && tol>1e-10);  % *** only to find ns=15 s=1.25 bump :(
    lastwarn('');                  % clean up warnings
    [nineerrs, info] = erralltypedim(M,Ntot,ntr,isign,prec,tol,o,myrand,dims);
    errs(i,:,t) = nineerrs(:,dim);   % extract col from 3x3
    [~,id] = lastwarn; toloks(t) = ~strcmp(id, 'FINUFFT:epsTooSmall');
    % measure w via support of one spread pt to the origin (type-1 only!)...
    oo = o; oo.spreadinterponly = 1;
    if dim==1, du = finufft1d1(0,1,+1,tol,32,oo);   % Nj>=2.ns here
    elseif dim==2, du2 = finufft2d1(0,0,1,+1,tol,32,32,oo); du = sum(du2);
    else, du3 = finufft3d1(0,0,0,1,+1,tol,32,32,32,oo); du = sum(du3,[1 2]); end
    ws(i,t) = sum(du~=0.0);
  end
end

% gather mean errs (ekw) for each kerformula, type, and w...
wmax = max(ws(:)); ekw = nan(nkf,3,wmax); vkw=ekw;
for w = 2:wmax
  for i = 1:numel(kfs)
    tt = find(ws(i,:)==w);   % take mean and var along tol t ind...
    [vkw(i,:,w), ekw(i,:,w)] = var(squeeze(errs(i,:,tt)),0,2);
  end
end

% do the err vs w plot...
figure('name','mean rel err vs w, comparing kernels, for 3 NUFFT types',...
       'position',[200 200 1200 500]);
for y=1:3  % types
  subplot(1,3,y);
  legs = {};
  for i=1:nkf     % kernels. make error bars with +-stddev...
    errorbar(2:wmax, squeeze(ekw(i,y,2:end)), sqrt(squeeze(vkw(i,y,2:end))),'.-','markersize',10);
    set(gca, 'ysc','log'); hold on; xlabel('w'); ylabel('mean rel l2 err');
    legs{i} = sprintf('kf=%d: %s',kfs(i),kfnam{kfs(i)});
  end
  axis([2 wmax min(ekw(:)) max(ekw(:))]);
  legend(legs)
  title(sprintf('%dD type %d %s, N_{tot}=%d, \\sigma=%g',dim,y,prec,Ntot,sigma))
end
print('-dpng',sprintf('results/wsweepkerrcomp_%dD_%s_sig%g.png',dim,prec,sigma))

% print # digits err change between first & 2nd formula (3 cols for types)...
disp('digits changed vs ns & type:'); log10(ekw(2,:,:)./ekw(1,:,:))

% also for kicks plot err vs tol for these kernels...
figure('name','mean rel err vs tol, comparing kernels, for 3 NUFFT types',...
       'position',[200 200 1200 500]);
for y=1:3  % types
  subplot(1,3,y);
  legs = {};
  symb = '+.ox*sd';
  tt = tols(toloks);            % plot only the non-warning tol domain
  for i=1:nkf     % kernels
    loglog(tt, squeeze(errs(i,y,toloks)), symb(i)); % 'markersize',10);
    hold on; xlabel('\epsilon (user tol)'); ylabel('mean rel l2 err');
    legs{i} = sprintf('kf=%d: %s',kfs(i),kfnam{kfs(i)});
  end
  plot(tt,tt,'-'); legs{nkf+1} = '\epsilon';
  axis([min(tt) max(tols) min(tt) 1.0]);
  legend(legs,'location','nw')
  title(sprintf('%dD type %d %s, N_{tot}=%d, \\sigma=%g',dim,y,prec,Ntot,sigma))
end
print('-dpng',sprintf('results/tolsweepkerrcomp_%dD_%s_sig%g.png',dim,prec,sigma))

