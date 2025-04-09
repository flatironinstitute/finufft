% FINUFFT1D3   GPU 1D complex nonuniform FFT, type 3 (nonuniform to nonuniform).
%
% See CUFINUFFT1D3
function f = finufft1d3(varargin)
f = cufinufft1d3(varargin{:});
