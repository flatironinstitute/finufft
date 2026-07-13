% 2D FINUFFT: types 1, 2, 3
mip load finufft;
tol = 1e-5;

M = 1000;
N = 1000;
isign = +1;
eps = 1e-6;
N1 = ceil(2.0*sqrt(N));
N2 = round(N/N1);

x = pi*(2*rand(M,1)-1);
y = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);

% Type 1: nonuniform to uniform
f = finufft2d1(x,y,c,isign,eps,N1,N2);
assert(isequal(size(f), [N1, N2]), '2d type-1 output size')
nt1 = floor(0.45*N1);
nt2 = floor(-0.35*N2);
fe = sum(c.*exp(1i*isign*(nt1*x+nt2*y)));
of1 = floor(N1/2)+1;
of2 = floor(N2/2)+1;
rel_err = abs((fe - f(nt1+of1,nt2+of2)) / max(abs(f(:))));
assert(rel_err < tol, '2d type-1 accuracy')

% Type 2: uniform to nonuniform
f2 = randn(N1,N2)+1i*randn(N1,N2);
c2 = finufft2d2(x,y,isign,eps,f2);
assert(numel(c2) == M, '2d type-2 output size')
[ms,mt] = size(f2);
[mm1,mm2] = ndgrid(ceil(-ms/2):floor((ms-1)/2), ceil(-mt/2):floor((mt-1)/2));
j = ceil(0.73*M);
ce = sum(f2(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j))));
rel_err2 = abs((ce - c2(j)) / max(abs(c2)));
assert(rel_err2 < tol, '2d type-2 accuracy')

% Type 3: nonuniform to nonuniform
s = (N1/2)*(2*rand(M,1)-1);
t = (N2/2)*(2*rand(M,1)-1);
f3 = finufft2d3(x,y,c,isign,eps,s,t);
assert(numel(f3) == M, '2d type-3 output size')
k = ceil(0.24*M);
fe3 = sum(c.*exp(1i*isign*(s(k)*x+t(k)*y)));
rel_err3 = abs((fe3 - f3(k)) / max(abs(f3)));
assert(rel_err3 < tol, '2d type-3 accuracy')

disp('SUCCESS')
