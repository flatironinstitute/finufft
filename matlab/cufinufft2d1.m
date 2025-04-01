function f = cufinufft2d1(x,y,c,isign,eps,ms,mt,o)

valid_setpts(true,1,2,x,y);
o.floatprec = underlyingType(x);              % should be 'double' or 'single'
n_transf = valid_ntr(x,c);
p = cufinufft_plan(1,[ms;mt],isign,n_transf,eps,o);
p.setpts(x,y);
f = p.execute(c);
