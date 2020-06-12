function [f ier] = finufft3d3(x,y,z,c,isign,eps,s,t,u,o)

if nargin<10, o.dummy=1; end
n_transf = round(numel(c)/numel(x));
nj=numel(x);
nk=numel(s);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(z)~=nj, error('z must have the same number of elements as x'); end
if numel(t)~=nk, error('t must have the same number of elements as s'); end
if numel(u)~=nk, error('u must have the same number of elements as s'); end
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(3,3,isign,n_transf,eps,o);
p.finufft_setpts(x,y,z,s,t,u);
[f,ier] = p.finufft_exec(c);
