% finufft accuracy test figs, deciding err norm to report. Barnett 6/6/17
clear
%M=1e5; N=1e2;         % M = # NU pts, N = # modes.  Note: keep MN<1e8 for now
M=1e4; N=1e3;
%M=1e2; N=1e5; % confusion about N vs M controlling err prefac
isign   = +1;     % sign of imaginary unit in exponential
o.debug = 0;      % choose 1 for timing breakdown text output

tols = 10.^(-1:-1:-14);
%tols = 1e-6;
errs = nan*tols;
for t=1:numel(tols)
  x = pi*(2*rand(1,M)-1);
  c = randn(1,M)+1i*randn(1,M);
  f = finufft1d1(x,c,isign,tols(t),N,o);
  ns = (ceil(-N/2):floor((N-1)/2))';         % mode indices, col vec
  fe = exp(1i*isign*ns*x) * c.';             % exact (note mat fill, matvec)
  errs(t) = max(abs(f(:)-fe(:)))/norm(c,1);  % eps as in err analysis...
  %p=2; errs(t) = norm(f(:)-fe(:),p)/norm(c,p);  % ... or rel p-norm
end
figure; loglog(tols,errs,'+'); hold on; plot(tols,tols,'-');
axis tight; xlabel('tol'); ylabel('err');
title(sprintf('1d1: (maxerr)/||c||_1, M=%d, N=%d\n',M,N));

