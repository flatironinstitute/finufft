function comparisons

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Instructions for installing nfft
% Download current source from here: https://www-user.tu-chemnitz.de/~potts/nfft/download.php
% Extract to comparisons/nfft-3.3.1
% cd to nfft-3.3.1 and run
% > ./configure --with-matlab-arch=glnxa64 --with-matlab=/home/magland/MATLAB/R2017a --enable-all --enable-openmp
%     where /home/magland/MATLAB/R2017a is the appropriate matlab path and set glnxa64 to be the appropriate architecture
% > make
%
% If you want to do single thread as well, repeat the above exercise and
% omit the following flags in the configuration:
%      --enable-all and --enable-openmp
%
% Then put it into the comparisons/nfft-3.3.1-single-thread
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1);

NN=128;
N1=NN; N2=NN; N3=NN;
%M=8e4;
M=1e6;
multithreaded=1;
max_nthreads=8;
spread_sort=2;
radial_sampling=1;

if (radial_sampling)
    theta=rand(1,M)*2*pi;
    phi=rand(1,M)*2*pi;
    rad=rand(1,M)*pi;
    x=rad.*cos(theta).*cos(phi);
    y=rad.*sin(theta).*cos(phi);
    z=rad.*sin(phi);
else
    x=rand(1,M)*2*pi-pi;
    y=rand(1,M)*2*pi-pi;
    z=rand(1,M)*2*pi-pi;
end;
data_in=randn(1,M);
if multithreaded
    finufft_nthreads=max_nthreads;
    nfft_single_thread=0;
else
    finufft_nthreads=1;
    nfft_single_thread=1;
end;

nfft_algopts.fftw_measure=1;
nfft_algopts.reshape=0;
nfft_algopts.precompute_phi=0;
nfft_algopts.precompute_psi=0;

title0=sprintf('Type 1, %dx%dx%d, %g sources, %d threads',NN,NN,NN,M,finufft_nthreads);

ALGS={};

%truth
eps=1e-14;
ALG=struct;
ALG.algtype=0;
ALG.name=sprintf('truth',eps);
ALG.algopts.eps=eps;
ALG.algopts.opts.nthreads=max_nthreads; ALG.algopts.opts.spread_sort=2; ALG.algopts.isign=1;
ALG.init=@dummy_init; ALG.run=@run_finufft3d1;
ALGS{end+1}=ALG;

%finufft
addpath('..');
epsilons=10.^(-(2:12));
finufft_algopts.opts.nthreads=finufft_nthreads;
finufft_algopts.opts.spread_sort=spread_sort;
finufft_algopts.isign=1;
finufft.algopts.opts.debug=1;
for ieps=1:length(epsilons)
    eps=epsilons(ieps);
    ALG=struct;
    ALG.algtype=1;
    ALG.name=sprintf('finufft(%g)',eps);
    ALG.algopts=finufft_algopts;
    ALG.algopts.eps=eps;
    ALG.init=@dummy_init; ALG.run=@run_finufft3d1;
    ALGS{end+1}=ALG;
end;

%nfft
if nfft_single_thread
    rmpath('nfft-3.3.1/matlab/nfft');
    addpath('nfft-3.3.1-single-thread/matlab/nfft');
else
    addpath('nfft-3.3.1/matlab/nfft');
    rmpath('nfft-3.3.1-single-thread/matlab/nfft');
end;
m_s=1:6;
if 1
    % create nfft dummy run to initialize with fftw_measure
    ALG=struct;
    ALG.algtype=0;
    ALG.name=sprintf('nfft(1) - dummy');
    ALG.algopts=nfft_algopts;
    ALG.algopts.m=1;
    ALG.init=@init_nfft; ALG.run=@run_nfft;
    ALGS{end+1}=ALG;
end;
for im=1:length(m_s)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ALG=struct;
    ALG.algtype=2;
    ALG.name=sprintf('nfft(%d)',m_s(im));
    ALG.algopts=nfft_algopts;
    ALG.algopts.m=m_s(im);
    ALG.init=@init_nfft; ALG.run=@run_nfft;
    ALGS{end+1}=ALG;
end;


results=run_algs(ALGS,x,y,z,data_in,N1,N2,N3);
%print_accuracy_comparison_and_timings(ALGS,results);

errors=[];
init_times=[];
run_times=[];
total_times=[];
algtypes=[];
X0=results{1}.data_out;
fprintf('\n\n%15s %15s %15s %15s %15s\n','name','init_time(s)','run_time(s)','tot_time(s)','err');
for j=1:length(ALGS)
    X1=results{j}.data_out;
    %m0=(max(abs(X0(:)))+max(abs(X0(:))))/2;
    %max_diff0=max(abs(X0(:)-X1(:)))/m0;
    %avg_diff0=mean(abs(X0(:)-X1(:)))/m0;
    numer0=norm(X0(:)-X1(:));
    denom0=norm(X0(:));
    errors(j)=numer0/denom0;
    init_times(j)=results{j}.init_time;
    run_times(j)=results{j}.run_time;
    total_times(j)=results{j}.tot_time;
    algtypes(j)=ALGS{j}.algtype;
    
    fprintf('%15s %15g %15g %15g %15g\n',ALGS{j}.name,init_times(j),run_times(j),total_times(j),errors(j));
end;


ii1=find(algtypes==1);
ii2=find(algtypes==2);
figure;
semilogx(errors(ii1),run_times(ii1),'b.-');
hold on;
semilogx(errors(ii2),run_times(ii2),'r.-');
xlabel('Error');
ylabel('Run time (s)');
set(gca,'xlim',[1e-12,1e0]);
legend({'finufft','nfft'});
title(title0);

function results=run_algs(ALGS,x,y,z,data_in,N1,N2,N3)
results={};
for j=1:length(ALGS)
    ALG=ALGS{j};
    RESULT=struct;
    
    fprintf('(%d/%d) Initializing %s...\n',j,length(ALGS),ALG.name);
    tA=tic;
    init_data=ALG.init(ALG.algopts,x,y,z,N1,N2,N3);
    RESULT.init_time=toc(tA);
    
    fprintf('Running %s...\n',ALG.name);
    tA=tic;
    RESULT.data_out=ALG.run(ALG.algopts,x,y,z,data_in,N1,N2,N3,init_data);
    RESULT.run_time=toc(tA);
    
    RESULT.tot_time=RESULT.init_time+RESULT.run_time;
    
    fprintf('init_time=%.3f, run_time=%.3f, tot_time=%.3f\n',RESULT.init_time,RESULT.run_time,RESULT.tot_time);
    
    results{j}=RESULT;
end;

function print_accuracy_comparison_and_timings(ALGS,results)
MM=length(ALGS);

max_diffs=zeros(MM,MM);
avg_diffs=zeros(MM,MM);
for j1=1:MM
    X1=results{j1}.data_out;
    for j2=1:MM
        X2=results{j2}.data_out;
        m0=(max(abs(X1(:)))+max(abs(X2(:))))/2;
        max_diffs(j1,j2)=max(abs(X1(:)-X2(:)))/m0;
        avg_diffs(j1,j2)=mean(abs(X1(:)-X2(:)))/m0;
    end;
end;

fprintf('Max. differences:\n');
fprintf('%15s ','');
for j2=1:MM
    fprintf('%15s ',ALGS{j2}.name);
end;
fprintf('\n');
for j1=1:MM
    fprintf('%15s ',ALGS{j1}.name);
    for j2=1:MM
        fprintf('%15g ',max_diffs(j1,j2));
    end;
    fprintf('\n');
end;
fprintf('\n');
fprintf('Avg. differences:\n');
fprintf('%15s ','');
for j2=1:MM
    fprintf('%15s ',ALGS{j2}.name);
end;
fprintf('\n');
for j1=1:MM
    fprintf('%15s ',ALGS{j1}.name);
    for j2=1:MM
        fprintf('%15g ',avg_diffs(j1,j2));
    end;
    fprintf('\n');
end;
fprintf('\n');
%fprintf('%15s %15s %15s %15s\n','Algorithm','Init time (s)','Run time (s)','RAM (GB)');
fprintf('%15s %15s %15s %15s\n','Algorithm','Init time (s)','Run time (s)','Tot time (s)');
for j1=1:MM
    %fprintf('%15s %15.3f %15.3f %15.3f\n',ALGS{j1}.name,results{j1}.init_time,results{j1}.run_time,results{j1}.memory_used);
    fprintf('%15s %15.3f %15.3f %15.3f\n',ALGS{j1}.name,results{j1}.init_time,results{j1}.run_time,results{j1}.tot_time);
end;
fprintf('\n');

function init_data=dummy_init(algopts,x,y,z,N1,N2,N3)
init_data=struct;

function data_out=run_finufft3d1(algopts,x,y,z,data_in,N1,N2,N3,init_data)
data_out=finufft3d1(x,y,z,data_in,algopts.isign,algopts.eps,N1,N2,N3,algopts.opts);


function init_data=init_nfft(algopts,x,y,z,N1,N2,N3)

M=length(x);
N=[N1;N2;N3];
%plan=nfft(3,N,M); 
n=2^(ceil(log(max(N))/log(2))+1);

ticA=tic;
flags=NFFT_OMP_BLOCKWISE_ADJOINT;
if (algopts.precompute_psi)
    flags=bitor(PRE_PSI,flags);
end;
if (algopts.precompute_phi)
    flags=bitor(PRE_PHI_HUT,flags);
end;
fftw_flag=FFTW_ESTIMATE;
if (algopts.fftw_measure)
    fftw_flag=FFTW_MEASURE;
end;
plan=nfft(3,N,M,n,n,n,algopts.m,flags,fftw_flag); % use of nfft_init_guru

% if (algopts.fftw_measure)
%     plan=nfft(3,N,M,n,n,n,algopts.m,bitor(PRE_PHI_HUT,bitor(PRE_PSI,NFFT_OMP_BLOCKWISE_ADJOINT)),FFTW_MEASURE); % use of nfft_init_guru
% else
%     plan=nfft(3,N,M,n,n,n,algopts.m,bitor(PRE_PHI_HUT,bitor(PRE_PSI,NFFT_OMP_BLOCKWISE_ADJOINT)),FFTW_ESTIMATE); % use of nfft_init_guru
% end;

fprintf('  Creating nfft plan: %g s\n',toc(ticA));

ticA=tic;
xyz=cat(1,x,y,z)';
fprintf('  Concatenating locations: %g s\n',toc(ticA));

ticA=tic;
plan.x=(xyz)/(2*pi); % set nodes in plan
fprintf('  Setting nodes in plan: %g s\n',toc(ticA));

ticA=tic;
nfft_precompute_psi(plan); % precomputations
fprintf('  time for nfft_precompute_psi: %g s\n',toc(ticA));

init_data.plan=plan;


function X=run_nfft(algopts,x,y,z,d,N1,N2,N3,init_data)

plan=init_data.plan;

M=length(x);
plan.f=d';
nfft_adjoint(plan);
if algopts.reshape
    X=reshape(plan.fhat,[N1,N2,N3])/M;
else
    X=plan.fhat/M;
end;
