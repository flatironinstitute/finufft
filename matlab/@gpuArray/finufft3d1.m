% FINUFFT3D1   GPU 3D complex nonuniform FFT, type 1 (nonuniform to uniform).
%
% See CUFINUFFT3D1
function f = finufft3d1(varargin)
f = cufinufft3d1(varargin{:});
