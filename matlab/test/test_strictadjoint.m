% check t1 and t2 are adjoints to rounding error, not merely to requested tol.
% 1d only for now. Barnett 8/27/18
%clear; addpath ~/numerics/finufft/matlab

M=1e5;  % pts
N=1e4;  % modes
tol = 1e-6;
x = pi*(2*rand(M,1)-1);
% pick two vectors to check (u,F1 v) = (F2 u,v) with...
v = randn(M,1)+1i*randn(M,1);
u = randn(N,1)+1i*randn(N,1);
ip1 = dot(u,finufft1d1(x,v,+1,tol,N));
ip2 = dot(finufft1d2(x,-1,tol,u),v);    % note sign flips to be complex adjoint
fprintf('M=%d,N=%d,tol=%.1g: rel err (u,F1 v) vs (F2 u,v): %.3g\n',M,N,tol,abs(ip1-ip2)/abs(ip1))
clear eps
fprintf('cf estimated rounding err for this prob size; %.3g\n',0.2*eps*N)
