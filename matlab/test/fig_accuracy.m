% sweep tolerance and plot actual accuracy, for both precisions, in matlab.
% OBSOLETE, replaced by plottolsweep.m.
% Barnett 6/6/17. Changed to rel 2-norm, 7/22/24, tweak 12/21/25.
clear
M=1e4;
N = 1e2; % keep N small to see close to epsmach(cond # = O(N))
isign = +1; % sign of imaginary unit in exponential
o.debug = 0;      % choose 1 for timing breakdown text output
o.spread_function = 0;    % >0 for experts only
o.upsampfac = 2.0;

% use one of these...
tols = 10.^(0:-0.02:-15); o.upsampfac = 2.0;
%tols = 10.^(0:-0.02:-10); o.upsampfac = 1.25;    % for lowupsampfac
%tols = 10.^(0:-0.02:-11); o.upsampfac = 1.3;     % for lowupsampfac
%tols = 10.^(0:-0.02:-12); o.upsampfac = 1.5;     % intermediate
%tols = 10.^(0:-0.02:-14); o.upsampfac = 1.99;    % v close to 2

errs = nan*tols;
toloks = true(size(tols));
for t=1:numel(tols)
  x = pi*(2*rand(1,M)-1);
  c = randn(1,M)+1i*randn(1,M);
  ns = (ceil(-N/2) : floor((N-1)/2))';  % mode indices, col vec
  lastwarn(''); % clear the warning state
  f = finufft1d1(x, c, isign, tols(t), N, o);
  [~, id] = lastwarn; toloks(t) = ~strcmp(id, 'FINUFFT:epsTooSmall'); % get warn
  fe = exp(1i*isign*ns*x) * c.';         % exact (note mat fill, matvec)
  %errs(t) = max(abs(f(:)-fe(:))) / norm(c,1); % eps as in err analysis...
  %p=2; errs(t) = norm(f(:)-fe(:),p) / norm(c,p);   % ... or p-norm rel to input
  p=2; errs(t) = norm(f(:)-fe(:),p) / norm(fe(:),p); % ... or rel p-norm
end
figure;
loglog(tols(toloks), errs(toloks), '+'); hold on; plot(tols, tols, 'k-');
plot(tols(~toloks), errs(~toloks), 'mo'); % highlight those w / tol warning
axis tight; xlabel('tol'); ylabel('err');
% title(sprintf('1d1: (maxerr)/||c||_1, M=%d, N=%d\n', M, N));
title(sprintf('1d1 \\sigma=%g sf=%d M=%d N=%d: rel 2-norm err in f',...
              o.upsampfac, o.spread_function, M, N));
