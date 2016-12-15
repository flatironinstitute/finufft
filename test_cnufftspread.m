function test_cnufftspread

compile_mex_cnufftspread;

if 1
    N=100;
    M=1e6;
    kx=rand(M,1)*N/2+N/4;
    ky=rand(M,1)*N/2+N/4;
    kz=rand(M,1)*N/2+N/4;
    X=rand(M,1);
    nspread=10;
    kernel_params=[1;nspread;1;1];
else
    N=20;
    M=1;
    kx=N/2;
    ky=N/2;
    kz=N/2;
    X=1;
    nspread=10;
    kernel_params=[1;nspread;1;1];
end;

tic;
Y=cnufftspread_type1(N,kx,ky,kz,X,kernel_params);
toc

figure; imagesc(Y(:,:,N/2));
