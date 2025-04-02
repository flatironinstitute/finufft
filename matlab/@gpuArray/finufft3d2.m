% FINUFFT3D2   GPU 3D complex nonuniform FFT, type 2 (uniform to nonuniform).
%
% See CUFINUFFT3D2
function c = finufft3d2(varargin)
c = cufinufft3d2(varargin{:});
