% Many-vector (ntrans>1) FINUFFT transforms via guru interface
mip load finufft;
tol = 1e-5;

M = 1000;
isign = +1;
eps = 1e-6;
ndata = 5;

% 1D many type 1
N = 500;
x = pi*(2*rand(M,1)-1);
c = randn(M,ndata)+1i*randn(M,ndata);
f = finufft1d1(x,c,isign,eps,N);
assert(isequal(size(f), [N, ndata]), '1d many type-1 output size')
nt = floor(0.37*N);
of1 = floor(N/2)+1;
d = floor(ndata/2)+1;
fe = c(:,d).'*exp(1i*isign*nt*x);
rel_err = abs((fe - f(nt+of1,d)) / max(abs(f(:))));
assert(rel_err < tol, '1d many type-1 accuracy')

% 2D many type 1
N1 = 50;
N2 = 40;
x = pi*(2*rand(M,1)-1);
y = pi*(2*rand(M,1)-1);
c = randn(M,ndata)+1i*randn(M,ndata);
f = finufft2d1(x,y,c,isign,eps,N1,N2);
assert(isequal(size(f), [N1, N2, ndata]), '2d many type-1 output size')
nt1 = floor(0.45*N1);
nt2 = floor(-0.35*N2);
of1 = floor(N1/2)+1;
of2 = floor(N2/2)+1;
d = floor(ndata/2)+1;
fe = c(:,d).'*exp(1i*isign*(nt1*x+nt2*y));
rel_err = abs((fe - f(nt1+of1,nt2+of2,d)) / max(abs(f(:))));
assert(rel_err < tol, '2d many type-1 accuracy')

% 2D many type 2
f2 = randn(N1,N2,ndata)+1i*randn(N1,N2,ndata);
c2 = finufft2d2(x,y,isign,eps,f2);
assert(isequal(size(c2), [M, ndata]), '2d many type-2 output size')
[ms,mt,~] = size(f2);
[mm1,mm2] = ndgrid(ceil(-ms/2):floor((ms-1)/2), ceil(-mt/2):floor((mt-1)/2));
j = ceil(0.73*M);
d = floor(ndata/2)+1;
fd = f2(:,:,d);
ce = sum(fd(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j))));
rel_err2 = abs((ce - c2(j,d)) / max(abs(c2(:))));
assert(rel_err2 < tol, '2d many type-2 accuracy')

disp('SUCCESS')
