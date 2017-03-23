% Matlab/octave interface math checker for FINUFFT libraries.
% Barnett 3/22/17

clear % choose params...
isign   = +1;
eps     = 1e-6;   % requested accuracy
o.debug = 0;
%o.nthreads = 1;
M       = 1e6;    % # of NU pts (in all dims)
N       = 1e6;    % # of modes (in all dims)

j = ceil(0.93*M);                               % target pt index to test
k = ceil(0.24*M);                               % freq targ pt index to test

tic; % --------- 1D
x = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);
[f ier] = finufft1d1(x,c,isign,eps,N,o);
nt = ceil(0.37*N);                              % pick a mode index
fe = (1/M)*sum(c.*exp(1i*isign*nt*x));          % exact
fprintf('1D type-1: rel err in F[%d] is %.3g\n',nt,abs((fe-f(nt+N/2+1))/fe))
f = randn(1,N)+1i*randn(1,N);
[c ier] = finufft1d2(x,isign,eps,f,o);
ms=numel(f); mm = ceil(-ms/2):floor((ms-1)/2);  % mode index list
ce = sum(f.*exp(1i*isign*mm*x(j)));             % crucial f, mm same shape
fprintf('1D type-2: rel err in c[%d] is %.3g\n',nt,abs((ce-c(j))/ce))
c = randn(1,M)+1i*randn(1,M);
s = N*(2*rand(1,M)-1);                          % target freqs of size O(N)
[f ier] = finufft1d3(x,c,isign,eps,s,o);
fe = sum(c.*exp(1i*isign*s(k)*x));
fprintf('1D type-3: rel err in f[%d] is %.3g\n',k,abs((fe-f(k))/fe))
fprintf('total 1D time: %.3f s\n',toc)

tic; % --------- 2D
N1=ceil(2*sqrt(N));
N2 = ceil(N/N1);
x = pi*(2*rand(1,M)-1); y = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);
[f ier] = finufft2d1(x,y,c,isign,eps,N1,N2,o);
nt1 = ceil(0.45*N1); nt2 = ceil(-0.35*N2);                % pick mode indices
fe = (1/M)*sum(c.*exp(1i*isign*(nt1*x+nt2*y)));           % exact
fprintf('2D type-1: rel err in F[%d,%d] is %.3g\n',nt1,nt2,abs((fe-f(nt1+N1/2+1,nt2+N2/2+1))/fe))
f = randn(N1,N2)+1i*randn(N1,N2);
[c ier] = finufft2d2(x,y,isign,eps,f,o);
[ms mt]=size(f);
% non-obvious ordering here, to make meshgrid loop over ms fast, mt slow:
[mm2,mm1] = meshgrid(ceil(-mt/2):floor((mt-1)/2),ceil(-ms/2):floor((ms-1)/2));
ce = sum(f(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j))));
fprintf('2D type-2: rel err in c[%d] is %.3g\n',nt,abs((ce-c(j))/ce))
c = randn(1,M)+1i*randn(1,M);
s = N1*(2*rand(1,M)-1);                          % target freqs of size O(N1)
t = N2*(2*rand(1,M)-1);                          % target freqs of size O(N2)
[f ier] = finufft2d3(x,y,c,isign,eps,s,t,o);
fe = sum(c.*exp(1i*isign*(s(k)*x+t(k)*y)));
fprintf('2D type-3: rel err in f[%d] is %.3g\n',k,abs((fe-f(k))/fe))
fprintf('total 2D time: %.3f s\n',toc)