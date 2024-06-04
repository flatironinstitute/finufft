function errhandler(ier)
% ERRHANDLER  translate FINUFFT's ier status into MATLAB warnings/error throws.

% Barnett 6/13/20

% Note that there are other matlab-only error types defined in valid_*.m

switch ier
 % These are the ERR_ #defines in ../include/finufft_errors.h:
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
  error('FINUFFT:upsampfacNotHorner','FINUFFT opts.upsampfac not a value with known Horner polynomial rule');
 case 9
  error('FINUFFT:badNtrans','FINUFFT number of transforms ntrans invalid');
 case 10
  error('FINUFFT:badType','FINUFFT transform type invalid');
 case 11
  error('FINUFFT:malloc','FINUFFT general malloc failure');
 case 12
  error('FINUFFT:badDim','FINUFFT number of dimensions dim invalid');
end
