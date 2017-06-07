% validation code for the matlab/octave interfaces
% Barnett 3/21/17

M=1000000;
N1=100; N2=100; N3=100;
isign=1;
tol=1e-6;
nonuniform_locations=rand(M,3)*2*pi-pi;
nonuniform_data=zeros(M,1);

tic; uniform_data=finufft3d1_mex(N1,N2,N3,nonuniform_locations,nonuniform_data,isign,tol,1,24);
toc