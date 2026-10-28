% MATLAB CPU/GPU FINUFFT library math test, both precisions, both devices,
% many-vector option (ntr). Tests only one typical tol per precision.
% (For sweep over tols see plottolsweep.m)
% Short runtime (~1 sec per device).
% Barnett 4/2/25, broken out erralltypedim 12/21/25,
% added zero-size checks, tweaked formatting 10/21/26

addpath(fileparts(mfilename('fullpath')))
clear
precdevs = '';
if exist('finufft')==3                                        % CPU .mex exist?
  precdevs = [precdevs 'sd'];                                 % add CPU
end
if exist('cufinufft')==3 && exist('canUseGPU') && canUseGPU() % GPU .mex exist?
  precdevs = [precdevs 'SD'];                                 % add GPU
end
if isempty(precdevs)
  warning('Found neither CPU nor GPU MEX files; testing nothing!');
end
disp('fullmathtest...')

for precdev=precdevs  % ......... loop precisions & devices
                      %  s=single, d=double; sd = CPU, SD = GPU
  if sum(precdev=='sd')
    devname = 'CPU               ';
    myrand = @rand;
    edgecases = 1;
  else
    dev = gpuDevice(); devname = dev.Name;
    myrand = @gpuArray.rand;
    edgecases = 0;	% until #947 fixed, don't test zero-length GPU
  end
  if sum(precdev=='sS')
    prec = 'single';
    tol = 1e-3;   % requested relative accuracy
  else
    prec = 'double';
    tol = 1e-9;
  end
  errcheck = 10*tol;    % acceptable rel l2 error norm (leave due to GPU)

  % choose small problems suitable for direct NUDFT computation...
  M       = 1e3;    % # of NU pts (in all dims, and for type 3 targs too)
  Ntot    = 1e3;    % # of modes (approx total, used in all dims, causes
                    % round-off esp in 1D, hence not big).
  ntr     = 20;     % # transforms (averages err)
  isign   = +1;     % sign of imaginary unit in exponential
  % various opts
  o.debug = 0;      % choose 1 for timing breakdown text output
  %o.spread_debug=0;   % spread-specific debug info (not for GPU)
  %o.spread_kerformula=0;     % (not for GPU)
  o.upsampfac=0;    % 0 (auto), 2.0 (default), or 1.25 (low-RAM, small-FFT)
  fprintf('%s\tprec=%s\ttol=%.3g, M=%d, Ntot=%d, ntrans=%d\n',...
          devname, prec, tol, M, Ntot, ntr)
  
  % meas the errors, with text reporting
  err = erralltypedim(M,Ntot,ntr,isign,prec,tol,o,myrand,[],errcheck);
  % raise error if needed (breaks script, and CI)
  if max(err(:))>errcheck || sum(isnan(err(:)))>0
    error('  one or more math test error failed!');
  end
  disp('  all math tests passed.')
  

  if edgecases
    disp('  Checking zero-length edge cases:')
    err = erralltypedim(0,Ntot,ntr,isign,prec,tol,o,myrand,[]);
    if max(err(:))>0 || sum(isnan(err(:)))>0
      error('  one or more M=0 tests failed!');
    end
    disp('  all M=0 (empty NU pts array) tests passed.')
    err = erralltypedim(M,0,ntr,isign,prec,tol,o,myrand,[]);
    if max(err(:))>0 || sum(isnan(err(:)))>0
      error('  one or more Ntot=0 tests failed!');
    end
    disp('  all Ntot=0 (empty mode, or type-3 targets) tests passed.')
  end

end     % ..........................
