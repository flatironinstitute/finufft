% FINUFFT2D1   GPU 2D complex nonuniform FFT, type 1 (nonuniform to uniform).
%
% See CUFINUFFT2D1
function f = finufft2d1(varargin)
f = cufinufft2d1(varargin{:});
