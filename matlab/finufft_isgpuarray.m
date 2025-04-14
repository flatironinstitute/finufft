function is_gpuarray = finufft_isgpuarray(input)
% FINUFFT_ISGPUARRAY    check if an array is an gpuArray.
%
% Note: this is currently unused since GPU codes have distinct cufinufft*
% names.

% check if the isgpuarray function is available
    if exist('isgpuarray') == 0
      % return 0, since no parallel computing toolbox, can not use gpuarray
      is_gpuarray = logical(0);
    else
      % call the parallel computing toolbox isgpuarray function
      is_gpuarray = isgpuarray(input);
    end
end
