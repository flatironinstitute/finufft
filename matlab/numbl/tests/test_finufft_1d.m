% 1D FINUFFT: types 1, 2, 3
mip load finufft;
tol = 1e-5;

M = 1000;
N = 500;
isign = +1;
eps = 1e-6;

x = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);

% Type 1: nonuniform to uniform
f = finufft1d1(x,c,isign,eps,N);
assert(numel(f) == N, 'type-1 output size')
nt = floor(0.37*N);
fe = sum(c.*exp(1i*isign*nt*x));
of1 = floor(N/2)+1;
rel_err = abs((fe - f(nt+of1)) / max(abs(f)));
assert(rel_err < tol, 'type-1 accuracy')

% Type 2: uniform to nonuniform
f2 = randn(N,1)+1i*randn(N,1);
c2 = finufft1d2(x,isign,eps,f2);
assert(numel(c2) == M, 'type-2 output size')
ms = numel(f2);
mm = (ceil(-ms/2):floor((ms-1)/2))';
j = ceil(0.73*M);
ce = sum(f2.*exp(1i*isign*mm*x(j)));
rel_err2 = abs((ce - c2(j)) / max(abs(c2)));
assert(rel_err2 < tol, 'type-2 accuracy')

% Type 3: nonuniform to nonuniform
s = (N/2)*(2*rand(M,1)-1);
f3 = finufft1d3(x,c,isign,eps,s);
assert(numel(f3) == M, 'type-3 output size')
k = ceil(0.24*M);
fe3 = sum(c.*exp(1i*isign*s(k)*x));
rel_err3 = abs((fe3 - f3(k)) / max(abs(f3)));
assert(rel_err3 < tol, 'type-3 accuracy')

disp('SUCCESS')
