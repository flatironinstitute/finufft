% Adjoint transforms: verify adjoint identity <Ax,y> = <x,A'y>
mip load finufft;
tol = 1e-4;

M = 500;
isign = +1;
eps = 1e-6;

% 1D type 1 adjoint identity: <f1(c), f> = <c, f1_adj(f)>
N = 200;
x = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);
f = randn(N,1)+1i*randn(N,1);

p = finufft_plan(1, N, isign, 1, eps);
p.setpts(x);
Ac = p.execute(c);
Af = p.execute_adjoint(f);

lhs = sum(conj(Ac(:)) .* f(:));
rhs = sum(conj(c(:)) .* Af(:));
rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs));
assert(rel_err < tol, '1d type-1 adjoint identity')
delete(p);

% 2D type 1 adjoint identity
N1 = 30;
N2 = 25;
x = pi*(2*rand(M,1)-1);
y = pi*(2*rand(M,1)-1);
c = randn(M,1)+1i*randn(M,1);
f = randn(N1,N2)+1i*randn(N1,N2);

p = finufft_plan(1, [N1;N2], isign, 1, eps);
p.setpts(x,y);
Ac = p.execute(c);
Af = p.execute_adjoint(f);

lhs = sum(conj(Ac(:)) .* f(:));
rhs = sum(conj(c(:)) .* Af(:));
rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs));
assert(rel_err < tol, '2d type-1 adjoint identity')
delete(p);

% 2D type 2 adjoint identity: <f2(f_in), c_in> = <f_in, f2_adj(c_in)>
f_in = randn(N1,N2)+1i*randn(N1,N2);
c_in = randn(M,1)+1i*randn(M,1);

p = finufft_plan(2, [N1;N2], isign, 1, eps);
p.setpts(x,y);
Af_in = p.execute(f_in);
Ac_in = p.execute_adjoint(c_in);

lhs = sum(conj(Af_in(:)) .* c_in(:));
rhs = sum(conj(f_in(:)) .* Ac_in(:));
rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs));
assert(rel_err < tol, '2d type-2 adjoint identity')
delete(p);

disp('SUCCESS')
