% Example of double-prec spread/interp only CPU tasks, in 2D,
% with basic math tests.
% Warning: this does *not* compute a NUFFT! It spreads/interps with kernel.
% Barnett 2/7/25.
%
% Also see, for the analogous 1D demo from C++:
%   FINUFFT/examples/spreadinterponly1d.cpp

clear
N1 = 500; N2 = 1000;  % size output grid for spread, input grid for interp
M = 1e6;              % number of nonuniform points
isign = +1;
opts.spreadinterponly = 1;     % crucial: engage the spreadinterp only mode
%opts.debug=1;                 % other opts

% the following two now only control the kernel parameters (not grid size):
tol = 1e-9;
opts.upsampfac = 2.0;          % must be one of the legitimate choices

% or, slower direct kernel eval to access nonstandard upsampfacs...
%opts.spread_kerevalmeth=0;
%opts.upsampfac = Inf;          % can be anything in (1,Inf], up to ns<=16

% spread M=1 single unit-strength somewhere (eg, at the origin)...
f = finufft2d1(0.0,0.0,1.0,isign,tol,N1,N2,opts);
kersum = sum(f(:));   % ... to get its mass, and plot it on 0-indexed grid...
figure; surf(0:N1-1,0:N2-1,log10(real(f))'); xlabel('x'); ylabel('y');
hold on; plot3(N1/2,N2/2,0.0,'k.','markersize',20); axis vis3d
colorbar; title('spreadinterponly2d: log_{10} spreading kernel'); drawnow

% spread only demo: ---------
x = 2*pi*rand(M,1); y = 2*pi*rand(M,1);          % NU pts
c = randn(M,1)+1i*randn(M,1);                    % strengths
tic;
f = finufft2d1(x,y,c,isign,tol,N1,N2,opts);      % do it
t = toc;
mass = sum(f(:)); err = abs(mass - kersum*sum(c))/abs(mass);  % relative err
fprintf('2D spread-only: %.3g s (%.3g NU pt/s), mass err %.3g\n',t, M/t, err)

% interp only demo: ---------
f = 0*f+1.0;                                     % unit complex input data
tic;
c = finufft2d2(x,y,isign,tol,f,opts);            % do it
t = toc;
maxerr = max(abs(c-kersum)) / kersum;            % worst-case c err
fprintf('2D interp-only: %.3g s (%.3g NU pt/s), max err %.3g\n', t, M/t, maxerr)
