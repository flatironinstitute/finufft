function test_cnufftspread

compile_mex_cnufftspread;

% N=10;
% M=100;
% kx=rand(M,1)*N;
% ky=rand(M,1)*N;
% kz=rand(M,1)*N;
% X=rand(M,1)*2-1;
% eps=0.01;

N=20;
M=1;
kx=N/2;
ky=N/2;
kz=N/2;
X=1;
eps=0.00001;

tic;
Y=cnufftspread_type1(N,kx,ky,kz,X,eps);
toc

figure; imagesc(Y(:,:,N/2));
