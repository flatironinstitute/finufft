function addnfftpath(multithreaded)
% ADDNFFTPATH.  adds NFFT to matlab path. Must match your NFFT installation.

% Barnett 11/7/17

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Instructions for installing nfft, by J Magland.
% Download current source from here:
%     https://www-user.tu-chemnitz.de/~potts/nfft/download.php
% Extract to nfft-3.3.2
% cd to nfft-3.3.2 and run
% > ./configure --with-matlab-arch=glnxa64 --with-matlab=/home/magland/MATLAB/R2017a --enable-all --enable-openmp
%     where /home/magland/MATLAB/R2017a is the path to your matlab installation, and set glnxa64 to be the appropriate architecture
% > make
%
% If you want to do single thread as well, repeat the above exercise and
% omit the following flags in the configuration:
%      --enable-all and --enable-openmp
%
% Then change that directory name to nfft-3.3.2-single-thread
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

base = '~/SCDA/nufft';     % dir that nfft is installed from
mul = [base '/nfft-3.3.2/matlab/nfft'];
sng = [base '/nfft-3.3.2-single-thread/matlab/nfft'];

if multithreaded
  addpath(mul)
  rmpath(sng)
else
  addpath(nsg)
  rmpath(mul)
end
