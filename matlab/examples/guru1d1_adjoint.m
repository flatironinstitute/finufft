% MATLAB/octave demo script of guru interface to FINUFFT, 1D type 1,
% performing *adjoint* of planned transform. Compare to guru1d1.m
% Barnett 6/25/25.
clear

% set required parameters...
isign   = +1;     % sign of imaginary unit in exponential, for planned
tol     = 1e-9;   % requested accuracy
M       = 3e6;    % NU pts
N       = 1e6;    % # of modes
type = 1;         % planned type
n_modes = N;      % n_dims inferred from length of this
ntrans = 2;

x = pi*(2*rand(1,M)-1);                    % choose NU points
f = randn(N,ntrans)+1i*randn(N,ntrans);    % choose stack of Fourier coeffs

disp('starting adjoint...'), tic     % just time FINUFFT not the data creation
opts.debug=0;    % set options then plan transform...
plan = finufft_plan(type,n_modes,isign,ntrans,tol,opts);

plan.setpts(x);                            % send in NU pts

c = plan.execute_adjoint(f);               % do *adjoint* of planned transform
                                           % (ie, type 2 with flipped isign)
disp('done.'); toc

% if you do not want to do more transforms of this size, clean up...
delete(plan);

% check the error of one output... because of adjoint it's a strength
j = ceil(0.77*M);                               % pick a NU target pt
t = ceil(0.7*ntrans);                           % pick a transform in stack
ms=size(f,1); mm = (ceil(-ms/2):floor((ms-1)/2))';   % mode index list
ce = sum(f(:,t).*exp(-1i*isign*mm*x(j)));        % crucial f, mm same shape
                                                 % note isign flip (by adj)
fprintf('rel err in c[%d] is %.3g\n',j,abs((ce-c(j,t))/max(c(:))))
