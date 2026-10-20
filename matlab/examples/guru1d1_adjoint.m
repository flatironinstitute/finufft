% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1,
% performing *adjoint* of planned transform. Compare to guru1d1.m
% Barnett 6/25/25.
clear
type = 1; ntr = 1; o.modeord = 1;   % transform type, #transforms, opts
N = 2e5;                            % how many desired Fourier modes?
plan = finufft_plan(1,N,+1,ntr,1e-10,o);      % plan for N output modes
M = 1e5;                            % number of NU source points
x = 2*pi*rand(M,1);                 % array of NU source points
plan.setpts(x,[],[]);               % pass pointer to this array (M inferred)
f = randn(N,1)+1i*randn(N,1);       % iid random complex Fourier coeffs
c = plan.execute_adjoint(f);        % do *adjoint* of planned transform
delete(plan);                       % (ie, type 2 with flipped isign)

% accuracy check of two strengths vs exact sums (note isign flip), for modeord=1...
k = [0:N/2-1, -N/2:-1]';            % mode indices in modeord=1 order
j = [ceil(0.77*M), ceil(0.31*M)];   % pick NU target pts
ce = exp(-1i*x(j)*k.')*f;           % exact
assert(all(isfinite(c(:))), 'guru1d1_adjoint: wrong result, NaN or Inf in c')
err = max(abs(ce - c(j)))/max(abs(c));
assert(err < 10*1e-10, 'guru1d1_adjoint: wrong result, error above 10*tol')
fprintf('rel err in c[%d], c[%d] is %.3g\n',j(1),j(2),err)
