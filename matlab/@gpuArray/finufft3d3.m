% FINUFFT3D3   GPU 3D complex nonuniform FFT, type 3 (nonuniform to nonuniform).
%
% See CUFINUFFT3D3
function f = finufft3d3(varargin)
f = cufinufft3d3(varargin{:});
