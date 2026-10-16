% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1.
% Lu 5/11/2020. Barnett added timing, tweaked.
% For demo of its adjoint see guru1d1_adjoint.m
clear
% docs-start: guru1d1
M = 3e6; N = 1e6; ntrans = 2;
x = pi*(2*rand(1,M)-1);                         % choose NU points
c = randn(1,M*ntrans)+1i*randn(1,M*ntrans);     % choose stack of strengths
tol = 1e-9;
plan = finufft_plan(1,N,+1,ntrans,tol);
plan.setpts(x);                                 % send in NU pts
f = plan.execute(c);                               % do the transform
delete(plan);
% docs-end: guru1d1

% accuracy check of one mode of one transform, against its exact value...
nt = ceil(0.37*N);                              % pick a mode index
t = ceil(0.7*ntrans);                           % pick a transform in stack
fe = sum(c(M*(t-1)+(1:M)).*exp(1i*nt*x));        % exact
of1 = floor(N/2) + 1 + N*(t-1);                        % mode index offset
assert(all(isfinite(f(:))), 'guru1d1: wrong result, NaN or Inf in f')
Fmax = max(abs(f(:)));
assert(abs(fe-f(nt+of1))/Fmax < 10*tol, 'guru1d1: wrong result, error above 10*tol')
fprintf('rel err in F[%d] is %.3g\n',nt,abs(fe-f(nt+of1))/Fmax)
