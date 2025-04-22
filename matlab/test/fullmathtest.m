% MATLAB CPU/GPU FINUFFT library math test, both precisions, both devices,
% many-vector option (ntr).
% For type 3 the space-bandwidth product chosen so FFT size roughly same as
% that in the type 1 and 2 cases.
% Short runtime (~1 sec per device).
% Barnett 4/2/25.
clear

precdevs = '';
if exist('finufft')==3                                        % CPU .mex exist?
  precdevs = [precdevs 'sd'];                                 % add CPU
end
if exist('cufinufft')==3 && exist('canUseGPU') && canUseGPU() % GPU .mex exist?
  precdevs = [precdevs 'SD'];                                 % add GPU
end
if isempty(precdevs)
  warning('Found neither CPU nor GPU MEX files; testing nothing!');
end

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
  M       = 1e3;    % # of NU pts (in all dims, and for type 3 targs too)
  Ntot    = 1e3;    % # of modes (approx total, used in all dims)
  ntr     = 3;      % # transforms
  isign   = +1;     % sign of imaginary unit in exponential
  % various opts
  o.debug = 0;      % choose 1 for timing breakdown text output
  o.spread_debug=0; % spread-specific debug info
  o.upsampfac=0;    % 0 (auto), 2.0 (default), or 1.25 (low-RAM, small-FFT)
  fprintf('%s\tprec=%s\ttol=%.3g, M=%d, Ntot=%d, ntrans=%d\n',...
          devname, prec, tol, M, Ntot, ntr)

  x = 2*pi*myrand(M,1,prec);      % random NU pts on whichever device, all dims
  y = 2*pi*myrand(M,1,prec);      % (col vecs)
  z = 2*pi*myrand(M,1,prec);
  % complex strengths, possibly stacked in M*ntr array...
  c = (2*myrand(M,ntr,prec)-1) + 1i*(2*myrand(M,ntr,prec)-1);

  % ----------------------------------------------- 1D -----------------
  N = Ntot;

  k = (ceil(-N/2):floor((N-1)/2))';            % mode list
  f = finufft1d1(x,c,isign,tol,N,o);
  A = exp(1i*isign*k*x');      % NUDFT matrix (via outer prod)
  fe = A*c;                    % exact direct (also for ntr>1)
  err = norm(f(:)-fe(:))/norm(fe(:));
  fprintf('Rel l2 errs:\t1D type 1:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  C = finufft1d2(x,isign,tol,f,o);
  Ce = A.'*f;                    % exact direct via non-conj transpose
  err = norm(C-Ce)/norm(Ce);
  fprintf('\t\t1D type 2:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  s = N*myrand(M,1,prec);            % M target freqs of space-bandwidth O(N)
  f = finufft1d3(x,c,isign,tol,s,o);
  fe = exp(1i*isign*s*x')*c;         % type 3 NUDFT mat (via outer prod)
  err = norm(f-fe)/norm(fe);
  fprintf('\t\t1D type 3:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  % ------------------------------------------------- 2D --------------
  N1 = round(sqrt(2*Ntot)); N2 = round(Ntot/N1);   % pick sizes prod ~ Ntot

  kx = (ceil(-N1/2):floor((N1-1)/2))';         % modes in each dim
  ky = (ceil(-N2/2):floor((N2-1)/2))';
  [kx ky] = ndgrid(kx,ky);                     % mode index lists
  f = finufft2d1(x,y,c,isign,tol,N1,N2,o);
  A = exp(1i*isign*(kx(:)*x'+ky(:)*y'));       % NUDFT matrix (via outer prods)
  fe = A*c;                    % exact direct (also for ntr>1)
  err = norm(f(:)-fe(:))/norm(fe(:));
  fprintf('\t\t2D type 1:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  C = finufft2d2(x,y,isign,tol,f,o);
  Ce = A.' * reshape(f,[N1*N2, ntr]);    % exact direct via non-conj transpose
  err = norm(C-Ce)/norm(Ce);
  fprintf('\t\t2D type 2:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  s = N1*myrand(M,1,prec); t = N2*myrand(M,1,prec);   % M target freqs
  f = finufft2d3(x,y,c,isign,tol,s,t,o);
  fe = exp(1i*isign*(s*x'+t*y'))*c;    % type 3 NUDFT matrix (via outer prods)
  err = norm(f-fe)/norm(fe);
  fprintf('\t\t2D type 3:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  % ------------------------------------------------- 3D --------------
  N1 = round((2*Ntot)^(1/3)); N2 = round(Ntot^(1/3)); N3 = round(Ntot/N1/N2);

  kx = (ceil(-N1/2):floor((N1-1)/2))';         % modes in each dim
  ky = (ceil(-N2/2):floor((N2-1)/2))';
  kz = (ceil(-N3/2):floor((N3-1)/2))';
  [kx ky kz] = ndgrid(kx,ky,kz);               % mode index lists
  f = finufft3d1(x,y,z,c,isign,tol,N1,N2,N3,o);
  A = exp(1i*isign*(kx(:)*x'+ky(:)*y'+kz(:)*z'));   % NUDFT matrix
  fe = A*c;                    % exact direct (also for ntr>1)
  err = norm(f(:)-fe(:))/norm(fe(:));
  fprintf('\t\t3D type 1:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  C = finufft3d2(x,y,z,isign,tol,f,o);
  Ce = A.' * reshape(f,[N1*N2*N3, ntr]);  % exact direct via non-conj transpose
  err = norm(C-Ce)/norm(Ce);
  fprintf('\t\t3D type 2:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

  s = N1*myrand(M,1,prec); t = N2*myrand(M,1,prec); u = N3*myrand(M,1,prec);
  f = finufft3d3(x,y,z,c,isign,tol,s,t,u,o);
  fe = exp(1i*isign*(s*x'+t*y'+u*z'))*c;    % type 3 NUDFT matrix
  err = norm(f-fe)/norm(fe);
  fprintf('\t\t3D type 3:\t%.3g\n',err)
  if err>errcheck, error('error fail'); end

end     % ..........................

disp('fullmathtest passed.')
