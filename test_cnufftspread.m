%function test_cnufftspread
% Magland, edited Barnett

%compile_mex_cnufftspread; % if needed

if 1
    N=100;
    M=1e6;
    kx=rand(M,1)*(N-1);   % grid pts go from 0 to N-1
    ky=rand(M,1)*(N-1);
    kz=rand(M,1)*(N-1);
  %  kx=(kx-N/2)*0.9+N/2;  % pull in from box edges
  %  ky=(ky-N/2)*0.9+N/2;
  %  kz=(kz-N/2)*0.9+N/2;
    %kz=linspace(0,N,M)';
    
    X=ones(M,1); % randn(M,1); % strengths
    nspread=6; kernel_params=[1;nspread;1;1];         % 2.3 s, 1 core
    %nspread = 16; kernel_params=[1;nspread;0.94;1.46]; % 18 s, 1 core
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

if 0, figure; sc = max(abs(Y(:))); % anim slices
for k=1:N,imagesc(Y(:,:,k));title(sprintf('k=%d',k))
  caxis([-1 1]*sc); colorbar; drawnow; pause(0.01); end
end
