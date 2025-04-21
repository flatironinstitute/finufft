% FINUFFT2D3   GPU 2D complex nonuniform FFT, type 3 (nonuniform to nonuniform).
%
% See CUFINUFFT2D3
function f = finufft2d3(varargin)
f = cufinufft2d3(varargin{:});
