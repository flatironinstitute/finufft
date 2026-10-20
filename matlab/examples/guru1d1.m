% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1.
% Lu 5/11/2020. Barnett added timing, tweaked.
% For demo of its adjoint see guru1d1_adjoint.m
clear
% docs-start: guru1d1
type = 1; ntr = 1; o.modeord = 1;   % transform type, #transforms, opts
N = 2e5;                            % how many desired Fourier modes?
plan = finufft_plan(1,N,+1,ntr,1e-10,o);      % plan for N output modes
M = 1e5;                            % number of NU source points
x = 2*pi*rand(M,1);                 % array of NU source points
plan.setpts(x,[],[]);               % pass pointer to this array (M inferred)
% (note: the x array should now not be altered until all executes are done!)
c = randn(M,1)+1i*randn(M,1);       % iid random complex data (row or col vec)
f = plan.execute(c);                % do the transform (0.008 sec, ie, faster)
% ...one could now change the points with setpts, and/or do new transforms
% ...with new c data, and/or do adjoint transforms with new data...
delete(plan);                       % don't forget to clean up
% docs-end: guru1d1

% accuracy check of two modes (one k>=0, one k<0) vs exact sums, for modeord=1...
k = [ceil(0.37*N), -ceil(0.21*N)];              % pick mode indices
ie = k + 1 + N*(k<0);                           % their positions in modeord=1 f
fe = exp(1i*k.'*x.')*c;                         % exact
assert(all(isfinite(f(:))), 'guru1d1: wrong result, NaN or Inf in f')
err = max(abs(fe - f(ie)))/max(abs(f));
assert(err < 10*1e-10, 'guru1d1: wrong result, error above 10*tol')
fprintf('rel err in F[%d], F[%d] is %.3g\n',k(1),k(2),err)
