% FINUFFT1D1   GPU 1D complex nonuniform FFT of type 1 (nonuniform to uniform).
%
% See also CUFINUFFT1D1
function f = finufft1d1(x,c,isign,eps,ms,o)
f = cufinufft1d1(x,c,isign,eps,ms,o)