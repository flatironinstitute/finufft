function [f ier] = finufft2d3(x,y,c,isign,eps,s,t,o)
% FINUFFT2D3

if nargin<8, o.dummy=1; end
n_transf = round(numel(c)/numel(x));
nj=numel(x);
nk=numel(s);
if numel(y)~=nj, error('y must have the same number of elements as x'); end
if numel(t)~=nk, error('t must have the same number of elements as s'); end
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(3,2,isign,n_transf,eps,o);
p.finufft_setpts(x,y,[],s,t,[]);
[f,ier] = p.finufft_exec(c);
