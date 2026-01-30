function errhandler(ier)
% ERRHANDLER  translate FINUFFT's ier status into MATLAB warnings/error throws.

% Barnett 6/13/20, updated 1/29/26.

% Note that there are other matlab-only error types defined in valid_*.m

switch ier
  % These should match the ERR_ #defines in ../include/finufft_errors.h
  % All of those should be caught here, else a C++ error will crash MATLAB entirely
  % and make for unhappy users:
 case 1
  warning('FINUFFT:epsTooSmall','FINUFFT eps tolerance too small to achieve');
 case 2
  error('FINUFFT:mallocGtMaxNf','FINUFFT malloc size requested greater than MAXNF');
 case 3
  error('FINUFFT:spreadinterp:fineGridSmall','FINUFFT spreader fine grid too small compared to kernel width');
 case 4
  error('FINUFFT:spreadinterp:NUrange','[DEPRECATED]');
 case 5
  error('FINUFFT:spreadinterp:malloc','FINUFFT spreader malloc error');
 case 6
  error('FINUFFT:spreadinterp:badDir','FINUFFT spreader illegal direction (must be 1 or 2)');
 case 7
  error('FINUFFT:upsampfacSmall','FINUFFT opts.upsampfac not > 1.0');
 case 8
  error('FINUFFT:GPU:upsampfacNotHorner','FINUFFT GPU unsupported sigma for Horner evaluation method');
 case 9
  error('FINUFFT:badNtrans','FINUFFT number of transforms ntrans invalid');
 case 10
  error('FINUFFT:badType','FINUFFT transform type invalid');
 case 11
  error('FINUFFT:malloc','FINUFFT general malloc failure');
 case 12
  error('FINUFFT:badDim','FINUFFT number of dimensions dim invalid');
 case 13
  error('FINUFFT:spreadinterp:badNthr','FINUFFT spread threads invalid');
 case 14
  error('FINUFFT:GPU:badDim','FINUFFT GPU number of dimensions dim invalid');
 case 15
  error('FINUFFT:GPU:failure','FINUFFT GPU general failure');
 case 16
  error('FINUFFT:GPU:badPlan','FINUFFT GPU plan not valid');
 case 17
  error('FINUFFT:GPU:badMethod','FINUFFT GPU method not valid');
 case 18
  error('FINUFFT:GPU:badBinSize','FINUFFT GPU internal bin size not valid');
 case 19
  error('FINUFFT:GPU:mallocShMem','FINUFFT GPU insufficient shared memory');
 case 20
  error('FINUFFT:badNNU','FINUFFT invalid number of nonuniform points (M or N)');
 case 21
  error('FINUFFT:GPU:badArg','FINUFFT GPU invalid argument (eg to setpts)');
 case 22
  error('FINUFFT:badLockFun','FINUFFT invalid lock functions for FFTW');
 case 23
  error('FINUFFT:badNthr','FINUFFT number of threads invalid');
 case 24
  error('FINUFFT:badKerFormula','FINUFFT opts.spread_kerformula invalid');
end
