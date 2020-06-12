function [f ier] = finufft3d1(x,y,z,c,isign,eps,ms,mt,mu,o)

if nargin<10, o.dummy=1; end
nj=numel(x);
n_transf = round(numel(c)/numel(x));
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
if n_transf*nj~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(1,[ms;mt;mu],isign,n_transf,eps,o);
p.finufft_setpts(x,y,z,[],[],[]);
[f,ier] = p.finufft_exec(c);
