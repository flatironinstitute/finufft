% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1,
% performing *adjoint* of planned transform. Compare to guru1d1.m
% Barnett 6/25/25.
clear
% docs-start: guru1d1-adjoint
M = 3e6; N = 1e6; ntrans = 2;
x = pi*(2*rand(1,M)-1);                    % choose NU points
f = randn(N,ntrans)+1i*randn(N,ntrans);    % choose stack of Fourier coeffs
tol = 1e-9;
plan = finufft_plan(1,N,+1,ntrans,tol);
plan.setpts(x);                            % send in NU pts
c = plan.execute_adjoint(f);               % do *adjoint* of planned transform
                                           % (ie, type 2 with flipped isign)
delete(plan);
% docs-end: guru1d1-adjoint

% accuracy check of one output (a strength, since adjoint) vs exact value...
j = ceil(0.77*M);                               % pick a NU target pt
t = ceil(0.7*ntrans);                           % pick a transform in stack
mm = (ceil(-N/2):floor((N-1)/2))';   % mode index list
ce = sum(f(:,t).*exp(-1i*mm*x(j)));        % crucial f, mm same shape
                                                 % note isign flip (by adj)
assert(all(isfinite(c(:))), 'guru1d1_adjoint: wrong result, NaN or Inf in c')
Fmax = max(abs(c(:)));
assert(abs((ce-c(j,t))/Fmax) < 10*tol, 'guru1d1_adjoint: wrong result, error above 10*tol')
fprintf('rel err in c[%d] is %.3g\n',j,abs((ce-c(j,t))/Fmax))
