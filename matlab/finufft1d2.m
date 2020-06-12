function [c ier] = finufft1d2(x,isign,eps,f,o)

if nargin<5, o.dummy=1; end
[ms,n_transf]=size(f);
if ms==1, warning('f must be a column vector for n_transf=1, n_transf should be the last dimension of f.'); end
p = finufft_plan(2,ms,isign,n_transf,eps,o);
p.finufft_setpts(x,[],[],[],[],[]);
[c,ier] = p.finufft_exec(f);
