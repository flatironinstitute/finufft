function AHAf = applyAHA(f,x,tol)         % use pair of NUFFTs to apply A^* A
  Af = finufft1d2(x,+1,tol,f);         % apply A
  AHAf = finufft1d1(x,Af,-1,tol,length(f));    % apply A^*
end
