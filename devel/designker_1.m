function designker
% Design some Cheby expansions with optimal NUFFT type-2 interp errors.
% uses: gauss.m
% Barnett 2/10/17

%test_chebeval; return
%test_chebeveneval; return   % to fix
%test_chebrep; return
%n=10; a = randn(1,n+1); for q=10:10:100, ft_chebser(a,2.0,9,q), end % q conv

if 0  % how long a cheb needed for gaussian, or half a gaussian?
f = @(x) exp(-20*x.^2);
x = linspace(-1,1,1e3); figure; plot(x,f(x),'-'); title('f')
n=50; a=chebrep(n,f);
figure; subplot(2,1,1); semilogy(0:2:n,abs(a(1:2:n+1)),'+');
subplot(2,1,2); plot(x, chebeval(x,a)-f(x), '-');

n=25; a=chebrep(n,@(x) f((x+1)/2));
figure; subplot(2,1,1); semilogy(0:n,abs(a),'+');
x = linspace(0,1,1e3);
subplot(2,1,2); plot(x, chebeval(2*x-1,a)-f(x), '-');
end   % concl: even parity only for Gaussian bit better than half-Gaussian.
% (assuming a Clenshaw-type eval exists for even terms only!)

R = 2.0;
ns = 4; L = ns/2;  % nspread and half-support
% optimize a Gaussian's width...
[be b] = fminbnd(@(be) badness(@(x) exp(-(x/be).^2),L,R),0.5,4.0);
fprintf('best Gaussian b=%.3g @ beta=%.3g\n',b,be);
%chebrep(4,@(x) -(L*x/be).^2), (L/be)^2/2  % debug init cheb coeffs

if 0  % cheby version
  % now optim over cheb coeffs via fminsearch w/ gaussian as initialiation
  nh = 2;   % n/2. There are nh coeffs to fit, since we fix a_0=1
  ec0 = zeros(1,nh);   % coeffs 2,4,...
  ec0(1) = -0.5*(L/be)^2  % init even coeffs vec (x^2 term only) to best gauss
  f = @(ec,x) exp(chebeval(x/L,[1.0 kron(ec,[0 1])]));  % coeffs 2,4,.. <- ec
  obj = @(ec) log10(badness(@(x) f(ec,x),L,R));
  fprintf('starting Gaussian est err %.6g (should match above)\n',obj(ec0))
  eps = 10^-ns;  % acc
  o = optimset('tolfun', eps,'tolx',eps,'maxiter',1e4,'maxfunevals',1e5,'display','iter');
  [ecb bb exitflag] = fminsearch(obj,ec0,o)     % the hard part
  fprintf('nh=%d exp(cheby) achieved est err %.6g\n',nh,10^bb)
  for j=1:4
    nh = nh+1; % add just one extra unknown
    [ecb bb exitflag] = fminsearch(obj,[ecb 0],o)     % the hard part
    fprintf('nh=%d exp(cheby) achieved est err %.6g\n',nh,10^bb)
  end
else   % or plain poly version
  nh = 2;   % degree/2. There are nh coeffs to fit, since we fix a_0=0
  ec0 = zeros(1,nh);   % coeffs 2nh,2nh-2,...,4,2
  ec0(nh) = -(L/be)^2  % init even coeffs vec (x^2 term only) to best gaussian
  f = @(ec,x) exp(polyval([kron(ec,[1 0]) 0],x/L)); % coeffs 2nh,..,4,2 <- ec
  obj = @(ec) log10(badness(@(x) f(ec,x),L,R));
  fprintf('starting Gaussian est err %.6g (should match above)\n',obj(ec0))
  eps = 10^-ns;  % acc
  o = optimset('tolfun', eps,'tolx',eps,'maxiter',1e4,'maxfunevals',1e5,'display','iter');
  [ecb bb exitflag] = fminsearch(obj,ec0,o)     % the hard part
  fprintf('nh=%d exp(poly) achieved est err %.6g\n',nh,10^bb)  
  for j=1:4
    nh = nh+1; % add just one extra unknown
    [ecb bb exitflag] = fminsearch(obj,[0 ecb],o)     % the hard part
    fprintf('nh=%d exp(poly) achieved est err %.6g\n',nh,10^bb)
  end
end
ecb'



%ff = @(x) exp(-(x/be).^2);  % the former Gaussian
ff = @(x) f(ecb,x);           % best-fit new thingy
if 1 %  plot ff in both x and k
  x = linspace(0,L,1e3); k = linspace(0,7*pi,1e3);
  F = ft(ff,L,k);
  %norm(ft(ff,L,k,50)- ft(ff,L,k,30))  % test conv
  figure; subplot(3,1,1);plot(x,ff(x),'-'); xlabel('x'); axis tight;
  subplot(3,1,2);semilogy(x,ff(x),'-'); xlabel('x'); axis tight;
  subplot(3,1,3);semilogy(k,abs(F),'-'); xlabel('k'); axis tight;
  vline(pi*[1/R, 2-1/R, 2+1/R 4-1/R, 4+1/R]);
end



%%%%%%%%

function b = badness(f,L,R)   % inverse figure of merit for func f. small=good
% f should be even-symm and support on [-L,L]
nfreqs = 40;                % pretty much a guess, just dense enough
kuse = linspace(-pi/R,pi/R,nfreqs);   % used freqs, both signs
Fuse = abs(ft(f,L,kuse));
Falias = 0*kuse;
nimg = 4;                      % how far to sum over nearby aliased copies
for m=-nimg:nimg, if m~=0
    Falias = Falias + abs(ft(f,L,kuse+2*pi*m));  % some nearby aliased copies
  end,end
b = max(Falias./Fuse);      % worst case over used freq range
  
%nfreqs = 30;                % pretty much a guess, just dense enough
%kuse = linspace(0,pi/R,nfreqs);   % used freqs
%ktail = linspace(pi*(2-1/R),pi*(2+1/R),nfreqs);
%tail = max(abs(ft(f,L,ktail)));
%tail2 = max(abs(ft(f,L,ktail+2*pi)));
%tail = max(tail,tail2);
%b = tail / max(abs(ft(f,L,kuse)));

%kalias1 = 2*pi-kuse;         % corresponding nearest aliased copies
%kalias2 = 2*pi+kuse;         % corresponding nearest aliased copies
%tail = max(abs([ft(f,L,kalias1);ft(f,L,kalias2)]),[],1); % worst at each freq
%b = max(tail./abs(ft(f,L,kuse)));  % worst-case aliasing error

%b = max(abs(ft(f,L,ktail)./ft(f,L,kuse)));  % worst-case aliasing error

function F = ft(f,L,k,q)
% compute real Fourier transform of even-symm func on [-L,L] at target freqs k.
% q is optional override of even # quadr nodes.
%kmax = 3*pi;            % The largest allowed k
%if sum(abs(k)>kmax)>0, warning('|k| cannot exceed kmax!'); end
kmax = max(abs(k));
if nargin<4
  q = ceil(10 + kmax*L); if mod(q,2), q=q+1; end   % q even
end
[z w] = gauss(q); z = L*z(1:q/2); w = 2*L*w(1:q/2); % even symm quadr on [-L,L]
fj = f(z);      % func evals in one go
F = 0*k;        % same size as k list
for j=1:q/2
  F = F + w(j)*fj(j)*cos(k*z(j));
end

function test_chebrep  % test representation of a func by cheb polys
n=16;                                  % order
f = @(x) exp(x);                       % real-valued smooth test func
a = chebrep(n,f);
x = linspace(-1,1,1e3);                % evaluation test grid
p = chebeval(x,a);
fx = f(x);
norm(p-fx)
figure; plot(x,[fx;p],'-'); hold on; plot(xj,fj,'*');

function a = chebrep(n,f)
% returns Cheby coeffs of n-th order Cheb poly rep of func handle f.
xj = cos(linspace(0,pi,n+1));          % sample f at n+1 cheb pts
fj = f(xj);
fhat = fft([fj fj(end-1:-1:2)])/n;     % periodize, length 2n FFT
a = real(fhat(1:n+1)); a(1)=a(1)/2;    % extract cheb coeffs (a_0 special)

function test_chebeval
x = linspace(-1,1,1e3);
n = 9;                     % order of T_n to plot
a = zeros(1,n+1); a(end) = 1.0;
p = chebeval(x,a); pd = chebevald(x,a);
norm(p-pd)
figure; plot(x,[p;pd],'-');

function test_chebeveneval    % ** Fails
x = linspace(-1,1,1e1);
n = 16; a = randn(1,n+1); a(2:2:end) = 0;   % even series
p = chebeval(x,a); pe = chebeveneval(x,a(1:2:end));   % two ways
p-pe
norm(p-pe)

function p = chebeval(x,a)
% use Clenshaw (Horner's) method for sum_{k=0...n} a_k T_k(x).
% x may be a vector.  Barnett 2/10/17
n = length(a)-1;
b1=0*x; b2=b1;  % init downwards recur, on array of zeros of size x
for k=n:-1:1
  b3=b2;
  b2=b1;
  b1=a(k+1)+2*x.*b2-b3;   % note a_k is in k+1 entry in matlab
end
p = a(1)-b2 + b1.*x;

function p = chebevald(x,a)
% direct slow Cheby eval, to check
n = length(a)-1;
p = 0*x;
for k=0:n
  p = p + a(k+1)*cos(k*acos(x));
end

function p = chebeveneval(x,a)
% *** BROKEN
% use Clenshaw (Horner's) method for sum_{k=0,2,..,n} a_k T_k(x), ie even only.
% x may be a vector.  Barnett 2/11/17
nh = length(a)-1;   % n/2
b1=0*x; b2=b1;  % init downwards recur, on array of zeros of size x
al2 = 4*x.*x;    % alpha^2 = (2x)^2
for kh=nh:-1:1   % recur k/2 down from n/2 to 1
  b3=b2;
  b2=b1;
  b1=a(kh+1) + (al2-1).*b2 - al2.*b3;
end
p = a(1)-b1;

