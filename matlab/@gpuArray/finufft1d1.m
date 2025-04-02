% FINUFFT1D1   GPU 1D complex nonuniform FFT, type 1 (nonuniform to uniform).
%
% See CUFINUFFT1D1
function f = finufft1d1(varargin)
f = cufinufft1d1(varargin{:});
