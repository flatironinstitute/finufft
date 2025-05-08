function is_gpuarray = finufft_isgpuarray(input)
%FINUFFT_ISGPUARRAY   Check if an array is a gpuArray.

try
    % Try calling MATLAB's built-in isgpuarray routine. If isgpuarray does
    % not exist (e.g., because we're in Octave) then this will throw an
    % error, which we catch below.
    is_gpuarray = isgpuarray(input);
catch
    is_gpuarray = false;
end

% % check if the isgpuarray function is available
% if exist('isgpuarray') == 0
%   % return 0, since no parallel computing toolbox, can not use gpuarray
%   is_gpuarray = logical(0);
% else
%   % call the parallel computing toolbox isgpuarray function
%   is_gpuarray = isgpuarray(input);
% end

end
