% compare and design new NUFFT type-2 spreading kernels (windows)
% v.2 cleaned up, no cheb. "badness" is the estimated error.
% needs: cmaes.m, obj.m, gauss.m, badness.m, ft.m
% Barnett 2/13/17

clear
R = 2.0;           % upsampling ratio
ns = 15;            % nspread
i = 0;             % counter over windows to compare

%for ns=2:9, ns      % ================

i=i+1; % optimize a Gaussian's width (be)...
L{i} = ns/2;      % we allow L (half-support) to vary per kernel (to allow jfm)
[be b] = fminbnd(@(be) badness(@(x) exp(-(x/be).^2),L{i},R),0.5,4.0);
fprintf('best Gaussian badness=%.3g @ beta=%.3g\n',b,be);
f{i} = @(x) exp(-(x/be).^2);
nam{i} = 'best-fit gaussian';

i=i+1; % Kaiser-Bessel as Jeremy had it but normalized to 1 at x=0....
fac1 = ns/(ns+mod(ns+1,2));  % jfm's weird truncation
fac2s = [2.2 1.71 1.65 1.45 1.47 1.51 1.48 1.46];   % jfm's list for even ns's
fac2 = fac2s(ceil(ns/2));        % for odd ns, use jfn's setting for ns+1
W = fac1*ns;             % full KB width, jfm's formulae
beta = max(0,pi*sqrt(W^2/4-0.8)*fac2);  % now truncate to W/2...
f{i} = @(x) (abs(x)<W/2).*besseli(0,beta*sqrt(1-(2*x/W).^2))/besseli(0,beta);
L{i} = W/2;    % for ns even, narrower than ns/2, weirdly
bkb = badness(f{i},L{i},R);
fprintf('Kaiser-Bessel badness=%.3g @ fac1=%.3g,fac2=%.3g\n',bkb,fac1,fac2);
nam{i} = 'Kaiser-Bessel jfm';

i=i+1; % Alex exp(sqrt) approx to I0 approx to KB
L{i} = ns/2;
%fes = @(beta,x) exp(beta*sqrt(1-(2*x/ns).^2))/exp(beta)./sqrt(sqrt(1-(2*x/ns).^2));
fes = @(beta,x) exp(beta*sqrt(1-(2*x/ns).^2))/exp(beta)./(1-(2*x/ns).^2).^0;
[beta bes] = fminbnd(@(beta) badness(@(x) fes(beta,x),L{i},R),2.0*ns,2.4*ns);
beta = beta*0.99;    % bit narrower in freq
f{i} = @(x) (abs(x)<ns/2).*fes(beta,x);
bkb = badness(f{i},L{i},R);
fprintf('optim exp(beta*sqrt)/quarter badness=%.3g @ beta=%.3g (beta/ns=%.4g)\n',bkb,beta,beta/ns);
nam{i} = 'exp(sqrt)           ';

%i=i+1; % optimize Kaiser-Bessel with ns/2 width...
%fkb = @(beta,x) besseli(0,beta*sqrt(1-(2*x/ns).^2))/besseli(0,beta);
%L{i} = ns/2;
%nam{i} = 'Kaiser-Bessel optim beta';
%bkb = badness(f{i},L{i},R);
%fprintf('Kaiser-Bessel optim badness=%.3g @ beta=%.3g\n',bkb,beta);

if 0
  i=i+1; % design our exp(poly) or exp(poly(asin())) thing...
nh = 4;   % degree/2. There are nh coeffs to fit, since we fix a_0=0
type =1;   % tell obj what func type
L{i} = ns/2;
ec0 = zeros(nh,1);   % coeffs 2nh,2nh-2,...,4,2
ec0(nh) = -(L{i}/be)^2  % init even coeffs vec (x^2 term only) to best gaussian
fep = @(ec,x) exp(polyval([kron(ec',[1 0]) 0],x/L{i})); %coeffs 2nh,..,4,2 <- ec
%fep = @(ec,x) exp(polyval([kron(ec',[1 0]) 0],(2/pi)*asin(x/L{i}))); %coeffs 2nh,..,4,2 <- ec
fprintf('starting Gaussian est err %.6g (should match above)\n',10^obj(ec0,L{i},R,type))
oo=cmaes; oo.TolFun=1e-3;   % override default opts
[ecb b counteval stopflag out] = cmaes('obj',ec0,0.5,oo,L{i},R,type) % see obj.m
fprintf('nh=%d exp(poly) achieved est err %.6g\n',nh,10^b)
f{i} = @(x) fep(ecb,x);           % best-fit new thingy
nam{i} = sprintf('best fit exp(poly) nh=%d',nh);
end

fprintf('\nsummary table of estimated errors...\n')
clear b; for j=1:numel(f), b{j} = badness(f{j},L{j},R);   % check all badnesses
  fprintf('kernel %d:   %s   \tbadness=%.3g\n',i,nam{j},b{j})
end

 %i=0; end    % =============== loop over ns

%  plot all f's in both x and k
x = linspace(0,ns/2,1e3); k = linspace(0,7*pi,1e3);          % x and k domains
fx = []; for i=1:numel(f), fx = [fx; f{i}(x)]; end           % stack func evals
Fx = []; for i=1:numel(f), Fx = [Fx; ft(f{i},L{i},k)]; end   % stack FT evals
ss = get(0,'screensize'); w=ss(3); h=ss(4);
figure; set(gcf,'position',[.2*w 0 .2*w .9*h]);
subplot(3,1,1); plot(x,fx,'-'); xlabel('x'); axis tight; legend(nam);
subplot(3,1,2); semilogy(x,fx,'-'); xlabel('x'); axis tight; legend(nam,'location','southwest');
subplot(3,1,3); semilogy(k,abs(Fx),'-'); xlabel('k'); axis tight; legend(nam);
vline(pi*[1/R, 2-1/R, 2+1/R 4-1/R, 4+1/R]);

