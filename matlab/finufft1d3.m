function [f ier] = finufft1d3(x,c,isign,eps,s,o)

if nargin<6, o.dummy=1; end
n_transf = round(numel(c)/numel(x));
if n_transf*numel(x)~=numel(c), error('the number of elements of c must be divisible by the number of elements of x'); end
p = finufft_plan(3,1,isign,n_transf,eps,o);
p.finufft_setpts(x,[],[],s,[],[]);
[f,ier] = p.finufft_exec(c);
