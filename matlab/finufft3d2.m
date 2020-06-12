function [c ier] = finufft3d2(x,y,z,isign,eps,f,o)

if nargin<7, o.dummy=1; end
[ms,mt,mu,n_transf] = size(f);
nj=numel(x);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
if ms==1, warning('f must be a column vector for n_transf=1, n_transf should be the last dimension of f.'); end
p = finufft_plan(2,[ms;mt;mu],isign,n_transf,eps,o);
p.finufft_setpts(x,y,z,[],[],[]);
[c,ier] = p.finufft_exec(f);
