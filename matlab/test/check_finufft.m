% Matlab/octave demo of interfaces for FINUFFT libraries, double-precision.
% Also checks the math.
% Barnett 3/24/17; updated normalization in type-1 6/6/17. upsampfac 6/18/18.
% new interface 6/10/20.
% 7/9/20, Changed rel error normalization to max of outputs.
% 5/14/24, chkbnds deprecated

% Runtime is around 3-10 seconds on a modern machine

clear     % choose params...
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-6;   % requested accuracy
o.debug = 0;      % choose 1 for timing breakdown text output
%o.spread_debug=1;   % see detailed spreader info
FFTW_ESTIMATE = bitshift(1,6); o.fftw = FFTW_ESTIMATE;       % or see fftw3.h
o.upsampfac=0;    % 0 (auto), 2.0 (default), or 1.25 (low-RAM, small-FFT)
M       = 1e6;    % # of NU pts (in all dims)
N       = 1e6;    % # of modes (approx total, used in all dims)

j = ceil(0.93*M);                               % target pt index to test
k = ceil(0.24*M);                               % type-3 targ pt index to test
tt=tic;

tic; % --------- 1D
fprintf('1D: using %d modes...\n',N)
x = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);
f = finufft1d1(x,c,isign,eps,N,o);
nt = floor(0.37*N);                             % pick a mode index
fe = sum(c.*exp(1i*isign*nt*x));                % exact
of1 = floor(N/2)+1;                             % mode index offset
fprintf('1D type-1: rel err in F[%d] is %.3g\n',nt,abs((fe-f(nt+of1))/max(f)))

f = randn(N,1)+1i*randn(N,1);
c = finufft1d2(x,isign,eps,f,o);
ms=numel(f); mm = ceil(-ms/2):floor((ms-1)/2); mm=mm';  % mode index list
ce = sum(f.*exp(1i*isign*mm*x(j)));             % crucial f, mm same shape
fprintf('1D type-2: rel err in c[%d] is %.3g\n',j,abs((ce-c(j))/max(c)))

c = randn(M,1)+1i*randn(M,1);
s = (N/2)*(2*rand(M,1)-1);                      % target freqs of size O(N)
f = finufft1d3(x,c,isign,eps,s,o);
fe = sum(c.*exp(1i*isign*s(k)*x));
fprintf('1D type-3: rel err in f[%d] is %.3g\n',k,abs((fe-f(k))/max(f)))
fprintf('total 1D time: %.3f s\n',toc)

tic; % --------- 2D
N1=ceil(2.0*sqrt(N)); N2=round(N/N1);           % pick Fourier mode ranges
fprintf('2D: using %d*%d modes (total %d)...\n',N1,N2,N1*N2)
x = pi*(2*rand(M,1)-1); y = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);
f = finufft2d1(x,y,c,isign,eps,N1,N2,o);
nt1 = floor(0.45*N1); nt2 = floor(-0.35*N2);              % pick mode indices
fe = sum(c.*exp(1i*isign*(nt1*x+nt2*y)));                 % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1;                 % mode index offsets
fprintf('2D type-1: rel err in F[%d,%d] is %.3g\n',nt1,nt2,abs((fe-f(nt1+of1,nt2+of2))/max(f(:))))

f = randn(N1,N2)+1i*randn(N1,N2);
c = finufft2d2(x,y,isign,eps,f,o);
[ms mt]=size(f);
% ndgrid loops over ms fast, mt slow:
[mm1,mm2] = ndgrid(ceil(-ms/2):floor((ms-1)/2),ceil(-mt/2):floor((mt-1)/2));
ce = sum(f(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j))));
fprintf('2D type-2: rel err in c[%d] is %.3g\n',j,abs((ce-c(j))/max(c(:))))

c = randn(M,1)+1i*randn(M,1);
s = (N1/2)*(2*rand(M,1)-1);                      % target freqs of size O(N1)
t = (N2/2)*(2*rand(M,1)-1);                      % target freqs of size O(N2)
f = finufft2d3(x,y,c,isign,eps,s,t,o);
fe = sum(c.*exp(1i*isign*(s(k)*x+t(k)*y)));
fprintf('2D type-3: rel err in f[%d] is %.3g\n',k,abs((fe-f(k))/max(f(:))))
fprintf('total 2D time: %.3f s\n',toc)

tic; % --------- 3D
N1=ceil(1.4*N^(1/3)); N2=N1; N3=round(N/N1/N2);  % pick Fourier mode ranges
fprintf('3D: using %d*%d*%d modes (total %d)...\n',N1,N2,N3,N1*N2*N3)
x = pi*(2*rand(1,M)-1); y = pi*(2*rand(1,M)-1); z = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);
f = finufft3d1(x,y,z,c,isign,eps,N1,N2,N3,o);
nt1 = floor(0.45*N1); nt2 = floor(-0.35*N2); nt3 = floor(0.17*N3);
fe = sum(c.*exp(1i*isign*(nt1*x+nt2*y+nt3*z)));                 % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1; of3 = floor(N3/2)+1;  % index offsets
fprintf('3D type-1: rel err in F[%d,%d,%d] is %.3g\n',nt1,nt2,nt3,abs((fe-f(nt1+of1,nt2+of2,nt3+of3))/max(f(:))))

f = randn(N1,N2,N3)+1i*randn(N1,N2,N3);
c = finufft3d2(x,y,z,isign,eps,f,o);
[ms mt mu]=size(f);
% ndgrid loops over ms fastest, mu slowest:
[mm1,mm2,mm3] = ndgrid(ceil(-ms/2):floor((ms-1)/2),ceil(-mt/2):floor((mt-1)/2),ceil(-mu/2):floor((mu-1)/2));
ce = sum(f(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j)+mm3(:)*z(j))));
fprintf('3D type-2: rel err in c[%d] is %.3g\n',j,abs((ce-c(j))/max(c(:))))

c = randn(1,M)+1i*randn(1,M);
s = (N1/2)*(2*rand(1,M)-1);                      % target freqs of size O(N1)
t = (N2/2)*(2*rand(1,M)-1);                      % target freqs of size O(N2)
u = (N3/2)*(2*rand(1,M)-1);                      % target freqs of size O(N3)
f = finufft3d3(x,y,z,c,isign,eps,s,t,u,o);
fe = sum(c.*exp(1i*isign*(s(k)*x+t(k)*y+u(k)*z)));
fprintf('3D type-3: rel err in f[%d] is %.3g\n',k,abs((fe-f(k))/max(f(:))))
fprintf('total 3D time: %.3f s\n',toc)

o.many_seq = 0; % 0 simultaneously do nufft on all data (default) or 1 sequentially
tic; % --------- 2Dmanys
N1=ceil(2.0*sqrt(N)); N2=round(N/N1);           % pick Fourier mode ranges
ndata = ceil(1e7/(N1*N2+M));
fprintf('2Dmany: %d data, using %d*%d modes (total %d)...\n',ndata,N1,N2,N1*N2)
x = pi*(2*rand(M,1)-1); y = pi*(2*rand(M,1)-1);
c = randn(M,ndata)+1i*randn(M,ndata);
f = finufft2d1(x,y,c,isign,eps,N1,N2,o);
nt1 = floor(0.45*N1); nt2 = floor(-0.35*N2);            % pick mode indices
fe = c.'*exp(1i*isign*(nt1*x+nt2*y));                   % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1;               % mode index offsets
d = floor(ndata/2)+1;
fprintf('2Dmany type-1: rel err in F[%d,%d,%d] is %.3g\n',nt1,nt2,d, ...
        abs((fe(d)-f(nt1+of1,nt2+of2,d))/max(f(:))))

f = randn(N1,N2,ndata)+1i*randn(N1,N2,ndata);
c = finufft2d2(x,y,isign,eps,f,o);
[ms mt ndata] = size(f);
d = floor(ndata/2)+1;
% ndgrid loops over ms fast, mt slow:
[mm1,mm2] = ndgrid(ceil(-ms/2):floor((ms-1)/2),ceil(-mt/2):floor((mt-1)/2));
fd = f(:,:,d);
ce = sum(fd(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j))));
fprintf('2Dmany type-2: rel err in c[%d,%d] is %.3g\n',j,d,abs((ce-c(j,d))/max(c(:))))
fprintf('total 2Dmany time: %.3f s\n',toc)

fprintf('All-dimensions total time: %.3f s\n',toc(tt))
