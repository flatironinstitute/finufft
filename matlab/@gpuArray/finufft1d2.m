% FINUFFT1D2   GPU 1D complex nonuniform FFT, type 2 (uniform to nonuniform).
%
% See CUFINUFFT1D2
function c = finufft1d2(varargin)
c = cufinufft1d2(varargin{:});
