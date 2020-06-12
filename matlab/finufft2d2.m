function [c ier] = finufft2d2(x,y,isign,eps,f,o)

if nargin<6, o.dummy=1; end
[ms,mt,n_transf] = size(f);
if numel(y)~=numel(x), error('y must have the same number of elements as x'); end
if ms==1, warning('f must be a column vector for n_transf=1, n_transf should be the last dimension of f.'); end
p = finufft_plan(2,[ms;mt],isign,n_transf,eps,o);
p.finufft_setpts(x,y,[],[],[],[]);
[c,ier] = p.finufft_exec(f);
