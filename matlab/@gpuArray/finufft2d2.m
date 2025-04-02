% FINUFFT2D2   GPU 2D complex nonuniform FFT, type 2 (uniform to nonuniform).
%
% See CUFINUFFT2D2
function c = finufft2d2(varargin)
c = cufinufft2d2(varargin{:});
