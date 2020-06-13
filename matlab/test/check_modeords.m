% check the two mode-orderings are consistent, via fftshift, 3D type-1 and 2.
% Barnett 10/25/17

clear     % choose params...
isign   = +1;     % sign of imaginary unit in exponential
eps     = 1e-6;   % requested accuracy
o.debug = 0;      % choose 1 for timing breakdown text output
M       = 1e5;    % # of NU pts
N1 = 123; N2=58; N3=24;   % # of modes in each dim (try even & odd)
j = ceil(0.93*M);        % target pt index to test

tic; % --------- 3D
fprintf('3D: using %d*%d*%d modes (total %d)...\n',N1,N2,N3,N1*N2*N3)
x = pi*(2*rand(1,M)-1); y = pi*(2*rand(1,M)-1); z = pi*(2*rand(1,M)-1);
c = randn(1,M)+1i*randn(1,M);
o.modeord = 0;
f = finufft3d1(x,y,z,c,isign,eps,N1,N2,N3,o);
nt1 = ceil(0.45*N1); nt2 = ceil(-0.35*N2); nt3 = ceil(0.17*N3); % mode to check
fe = sum(c.*exp(1i*isign*(nt1*x+nt2*y+nt3*z)));                 % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1; of3 = floor(N3/2)+1;  % index offsets
fprintf('3D type-1 modeord=0: rel err in F[%d,%d,%d] is %.3g\n',nt1,nt2,nt3,abs((fe-f(nt1+of1,nt2+of2,nt3+of3))/fe))
o.modeord = 1;
f1 = finufft3d1(x,y,z,c,isign,eps,N1,N2,N3,o);
f1 = fftshift(f1);  % handles odd dimension lengths Ns same as we do
fprintf('\t modeord=1 vs 0: max error = %g\n',norm(f1(:)-f(:),Inf))
% it's still a mystery how there can be any nonzero difference here - 
%  maybe the compiler is reordering the calcs, so roundoff appears, somehow?

f = randn(N1,N2,N3)+1i*randn(N1,N2,N3);
o.modeord = 0;
c = finufft3d2(x,y,z,isign,eps,f,o);
[ms mt mu]=size(f);
% ndgrid loops over ms fastest, mu slowest:
[mm1,mm2,mm3] = ndgrid(ceil(-ms/2):floor((ms-1)/2),ceil(-mt/2):floor((mt-1)/2),ceil(-mu/2):floor((mu-1)/2));
ce = sum(f(:).*exp(1i*isign*(mm1(:)*x(j)+mm2(:)*y(j)+mm3(:)*z(j))));
fprintf('3D type-2 modeord=0: rel err in c[%d] is %.3g\n',j,abs((ce-c(j))/ce))
o.modeord = 1;
c1 = finufft3d2(x,y,z,isign,eps,ifftshift(f),o);
% note for odd Ns, ifftshift is correct inverse of fftshift
fprintf('\t modeord=1 vs 0: max error = %g\n',norm(c-c1,Inf))
