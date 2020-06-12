function [c ier] = finufft2d2(x,y,isign,eps,f,o)
% FINUFFT2D2   2D complex nonuniform FFT of type 2 (uniform to nonuniform).
%
% [c ier] = finufft2d2(x,y,isign,eps,f)
% [c ier] = finufft2d2(x,y,isign,eps,f,opts)
%
% This computes, to relative precision eps, via a fast algorithm:
%
%    c[j] =  SUM   f[k1,k2] exp(+/-i (k1 x[j] + k2 y[j]))  for j = 1,..,nj
%           k1,k2
%     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,
%
%  Inputs:
%     x,y   location of NU targets on the square [-3pi,3pi]^2, each length nj
%     f     size (ms,mt) complex Fourier transform value matrix
%           (mode ordering given by opts.modeord in each dimension)
%     isign  if >=0, uses + sign in exponential, otherwise - sign.
%     eps    precision requested (>1e-16)

if nargin<6, o.dummy=1; end
[ms,mt,n_transf] = size(f);
if numel(y)~=numel(x), error('y must have the same number of elements as x'); end
if ms==1, warning('f must be a column vector for n_transf=1, n_transf should be the last dimension of f.'); end
p = finufft_plan(2,[ms;mt],isign,n_transf,eps,o);
p.finufft_setpts(x,y,[],[],[],[]);
[c,ier] = p.finufft_exec(f);
