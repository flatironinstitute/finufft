function f = obj(ec,L,R,type);
% ec is col vec
if type==1   % exp(poly)
  % ec is col vec of even poly coeffs {a_2nh, ..., a_4, a_2}
  fep = @(x) exp(polyval([kron(ec',[1 0]) 0],x/L)); % coeffs 2nh,..,4,2 <- ec
end
f = log10(badness(fep,L,R));
