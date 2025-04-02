% MATLAB GPU FINUFFT library test, both precisions.
% Runtime is around *** seconds on an A6000 GPU.
% Barnett 4/2/25.
clear
isign   = +1;     % sign of imaginary unit in exponential
if canUseGPU(), precdevs='sdSD'; else precdevs='sd'; end
for precdev=precdevs  % ......... loop precisions & devices
                      %  s=single, d=double; sd = CPU, SD = GPU
  if sum(precdev=='sd')
    devname = 'CPU               ';
    myrand = @rand;
  else
    dev = gpuDevice(); devname = dev.Name;
    myrand = @gpuArray.rand;
  end
  if sum(precdev=='sS')
    prec = 'single';
    tol = 1e-3;   % requested relative accuracy
  else
    prec = 'double';
    tol = 1e-9;
  end
  errcheck = 10*tol;    % acceptable rel l2 error norm
  
  % choose small problems suitable for direct NUDFT computation...
  M       = 1e3;    % # of NU pts (in all dims)
  Ntot    = 1e3;    % # of modes (approx total, used in all dims)
  ntr = 2;          % # transforms
  % various opts
  o.debug = 0;      % choose 1 for timing breakdown text output
  o.upsampfac=0;    % 0 (auto), 2.0 (default), or 1.25 (low-RAM, small-FFT)  
  fprintf('%s\tprec=%s\ttol=%.3g, M=%d, Ntot=%d, ntrans=%d\n',...
          devname, prec, tol, M, Ntot, ntr)

  x = 2*pi*myrand(M,1,prec);      % random NU pts on whichever device, all dims
  y = 2*pi*myrand(M,1,prec);      % (col vecs)
  z = 2*pi*myrand(M,1,prec);
  % complex strengths, possibly stacked in M*ntr array...
  c = (2*myrand(M,ntr,prec)-1) + 1i*(2*myrand(M,ntr,prec)-1);
  
  N = Ntot; % ----------- 1D

  k = (ceil(-N/2):floor((N-1)/2))';            % mode list
  f = finufft1d1(x,c,isign,tol,N,o);
  A = exp(1i*isign*k*x');      % NUDFT matrix
  fe = A*c;                    % exact direct (also for ntr>1)
  err = norm(f(:)-fe(:))/norm(fe(:)); 
  fprintf('Rel l2 errs:\t1D type 1:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  C = finufft1d2(x,isign,tol,f,o);
  Ce = A.'*f;                    % exact direct via non-conj transpose
  err = norm(C(:)-Ce(:))/norm(Ce(:)); 
  fprintf('Rel l2 errs:\t1D type 2:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end
  
end     % ..........................
