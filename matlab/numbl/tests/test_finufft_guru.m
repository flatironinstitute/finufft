% Guru interface: plan, setpts, execute
mip load finufft;
tol = 1e-5;

M = 1000;
N = 500;
isign = +1;
eps = 1e-6;

% 1D type 1 via guru
x = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);
p = finufft_plan(1, N, isign, 1, eps);
p.setpts(x);
f = p.execute(c);
assert(numel(f) == N, 'guru 1d type-1 output size')
nt = floor(0.37*N);
of1 = floor(N/2)+1;
fe = sum(c.*exp(1i*isign*nt*x));
rel_err = abs((fe - f(nt+of1)) / max(abs(f)));
assert(rel_err < tol, 'guru 1d type-1 accuracy')
delete(p);

% 2D type 2 via guru
N1 = 50;
N2 = 40;
x = pi*(2*rand(M,1)-1);
y = pi*(2*rand(M,1)-1);
f2 = randn(N1,N2)+1i*randn(N1,N2);
p = finufft_plan(2, [N1;N2], isign, 1, eps);
p.setpts(x,y);
c2 = p.execute(f2);
assert(numel(c2) == M, 'guru 2d type-2 output size')
[ms,mt] = size(f2);
[mm1,mm2] = ndgrid(ceil(-ms/2):floor((ms-1)/2), ceil(-mt/2):floor((mt-1)/2));
j = ceil(0.73*M);
ce = sum(f2(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j))));
rel_err2 = abs((ce - c2(j)) / max(abs(c2)));
assert(rel_err2 < tol, 'guru 2d type-2 accuracy')
delete(p);

% Reuse plan with new points
p = finufft_plan(1, N, isign, 1, eps);
x1 = pi*(2*rand(M,1)-1);
c1 = randn(M,1)+1i*randn(M,1);
p.setpts(x1);
f1 = p.execute(c1);
fe1 = sum(c1.*exp(1i*isign*nt*x1));
rel_err3 = abs((fe1 - f1(nt+of1)) / max(abs(f1)));
assert(rel_err3 < tol, 'guru plan reuse accuracy')
delete(p);

disp('SUCCESS')
