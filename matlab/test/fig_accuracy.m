% finufft accuracy test figs, deciding err norm to report. Barnett 6/6/17
% Changed to rel 2-norm, 7/22/24.
clear
%M=1e5; N=1e2;         % M = # NU pts, N = # modes.  Note: keep MN<1e8 for now
M=1e4; N=1e2;         % keel N small to see close to epsmach; cond # = O(N)
%M=1e2; N=1e5; % confusion about N vs M controlling err prefac (it's N)
isign   = +1;     % sign of imaginary unit in exponential
o.debug = 0;      % choose 1 for timing breakdown text output

% use one of these two...
tols = 10.^(-1:-0.02:-15); o.upsampfac = 2.0;
%tols = 10.^(-1:-0.02:-10); o.upsampfac=1.25;    % for lowupsampfac

% other expts...
%tols = 1e-6;
%tols = 10.^(-1:-1:-10); o.upsampfac=1.25;    % for lowupsampfac

errs = nan*tols;
for t=1:numel(tols)
  x = pi*(2*rand(1,M)-1);
  c = randn(1,M)+1i*randn(1,M);
  ns = (ceil(-N/2):floor((N-1)/2))';         % mode indices, col vec
  f = finufft1d1(x,c,isign,tols(t),N,o);
  fe = exp(1i*isign*ns*x) * c.';             % exact (note mat fill, matvec)
  %errs(t) = max(abs(f(:)-fe(:)))/norm(c,1);  % eps as in err analysis...
  %p=2; errs(t) = norm(f(:)-fe(:),p)/norm(c,p);  % ... or p-norm rel to input
  p=2; errs(t) = norm(f(:)-fe(:),p)/norm(fe(:),p);  % ... or rel p-norm
end
figure; loglog(tols,errs,'+'); hold on; plot(tols,tols,'-');
axis tight; xlabel('tol'); ylabel('err');
%title(sprintf('1d1: (maxerr)/||c||_1, M=%d, N=%d\n',M,N));
title(sprintf('1d1: ||\tilde f - f||_2/||f||_2, M=%d, N=%d\n',M,N));

