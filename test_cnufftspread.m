function test_cnufftspread

compile_mex_cnufftspread;

% N=10;
% M=100;
% kx=rand(M,1)*N;
% ky=rand(M,1)*N;
% kz=rand(M,1)*N;
% X=rand(M,1)*2-1;
% eps=0.01;

N=2;
M=1;
kx=0;
ky=0;
kz=0;
X=1;
eps=0.01;

tic;
Y=cnufftspread_type1(N,kx,ky,kz,X,eps);
toc

Y