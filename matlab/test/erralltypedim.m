% ERRALLTYPEDIM   Measure FINUFFT errors for all types and dims, at one tol.
%
% err = erralltypedim(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,errcheck)
% returns 3*3 array of relative 2-norm errors where the i'th row is
% for type i, and the j'th column is for dimension j (untested dims give
% NaNs). Random data is generated and used.
% For type 3 the space-bandwidth product is chosen so FFT size roughly same
% as that in the type 1 and 2 cases.
% Can use CPU or GPU transforms (via myrand). Can report text output.
%
% Input arguments:
%  M      = # NU pts
%  Ntot   = # modes (approx, total)
%  ntr    = # transform vectors to test per type & dim (direct: for free)
%  isign  = +-1 passed to FINUFFT
%  prec   = `single` or `double`
%  tol    = tolerance parameter passed to FINUFFT
%  o      = [optional] opts struct passed to FINUFFT
%  myrand = [optional] @rand (for CPU) or @gpuArray.rand (for GPU transforms).
%  dims   = [optional] bool length-3 vec giving which dims to test (empty=all)
%  errcheck = [optional] make text output of errors as the tests done, and
%             report fail if error>errcheck.
%
% [err, info] = erralltypedim(...) returns info struct with at least fields:
%   info.Nmax = max N used in each dim, 1x3 vector
%
% Barnett 12/21/25. Helper used by fullmathtest and plottolsweep.
%
function [err, info] = erralltypedim(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,errcheck)
err = nan(3,3);
info.Nmax = nan(1,3);
% defaults...
if nargin<7, o=struct(); end
if nargin<8, myrand=@rand; end          % use CPU
if nargin<9 || isempty(dims), dims = true(3,1); end      % test all dims
if nargin<10, errcheck = -1; end        % don't output text

x = 2*pi*myrand(M,1,prec);      % random NU pts on whichever device, all dims
y = 2*pi*myrand(M,1,prec);      % (col vecs)
z = 2*pi*myrand(M,1,prec);
% complex strengths, possibly stacked in M*ntr array...
c = (2*myrand(M,ntr,prec)-1) + 1i*(2*myrand(M,ntr,prec)-1);

if dims(1) % ----------------------------------------------- 1D ----------
  N = Ntot; info.Nmax(1) = N;

  k = (ceil(-N/2):floor((N-1)/2))';            % mode list
  f = finufft1d1(x,c,isign,tol,N,o);
  A = exp(1i*isign*k*x');      % NUDFT matrix (via outer prod)
  fe = A*c;                    % exact direct (also for ntr>1)
  err(1,1) = norm(f(:)-fe(:))/norm(fe(:));
  if errcheck>0, fprintf('Rel l2 errs:\t1D type 1:\t%.3g   \t',err(1,1));
    if err(1,1)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end

  C = finufft1d2(x,isign,tol,f,o);
  Ce = A.'*f;                    % exact direct via non-conj transpose
  err(2,1) = norm(C-Ce)/norm(Ce);
  if errcheck>0, fprintf('\t\t1D type 2:\t%.3g   \t',err(2,1));
    if err(2,1)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end

  s = N*myrand(M,1,prec);            % M target freqs of space-bandwidth O(N)
  f = finufft1d3(x,c,isign,tol,s,o);
  fe = exp(1i*isign*s*x')*c;         % type 3 NUDFT mat (via outer prod)
  err(3,1) = norm(f-fe)/norm(fe);
  if errcheck>0, fprintf('\t\t1D type 3:\t%.3g   \t',err(3,1));
    if err(3,1)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end
end
if dims(2) % ------------------------------------------------- 2D ---------
  N1 = round(sqrt(2*Ntot)); N2 = round(Ntot/N1);   % pick sizes prod ~ Ntot
  info.Nmax(2) = max([N1,N2]);

  kx = (ceil(-N1/2):floor((N1-1)/2))';         % modes in each dim
  ky = (ceil(-N2/2):floor((N2-1)/2))';
  [kx ky] = ndgrid(kx,ky);                     % mode index lists
  f = finufft2d1(x,y,c,isign,tol,N1,N2,o);
  A = exp(1i*isign*(kx(:)*x'+ky(:)*y'));       % NUDFT matrix (via outer prods)
  fe = A*c;                    % exact direct (also for ntr>1)
  err(1,2) = norm(f(:)-fe(:))/norm(fe(:));
  if errcheck>0, fprintf('\t\t2D type 1:\t%.3g   \t',err(1,2));
    if err(1,2)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end

  C = finufft2d2(x,y,isign,tol,f,o);
  Ce = A.' * reshape(f,[N1*N2, ntr]);    % exact direct via non-conj transpose
  err(2,2) = norm(C-Ce)/norm(Ce);
  if errcheck>0, fprintf('\t\t2D type 2:\t%.3g   \t',err(2,2));
    if err(2,2)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end

  s = N1*myrand(M,1,prec); t = N2*myrand(M,1,prec);   % M target freqs
  f = finufft2d3(x,y,c,isign,tol,s,t,o);
  fe = exp(1i*isign*(s*x'+t*y'))*c;    % type 3 NUDFT matrix (via outer prods)
  err(3,2) = norm(f-fe)/norm(fe);
  if errcheck>0, fprintf('\t\t2D type 3:\t%.3g   \t',err(3,2));
    if err(3,2)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end
end

if dims(3) % ------------------------------------------------- 3D ---------
  N1 = round((2*Ntot)^(1/3)); N2 = round(Ntot^(1/3)); N3 = round(Ntot/N1/N2);
  info.Nmax(3) = max([N1,N2,N3]);

  kx = (ceil(-N1/2):floor((N1-1)/2))';         % modes in each dim
  ky = (ceil(-N2/2):floor((N2-1)/2))';
  kz = (ceil(-N3/2):floor((N3-1)/2))';
  [kx ky kz] = ndgrid(kx,ky,kz);               % mode index lists
  f = finufft3d1(x,y,z,c,isign,tol,N1,N2,N3,o);
  A = exp(1i*isign*(kx(:)*x'+ky(:)*y'+kz(:)*z'));   % NUDFT matrix
  fe = A*c;                    % exact direct (also for ntr>1)
  err(1,3) = norm(f(:)-fe(:))/norm(fe(:));
  if errcheck>0, fprintf('\t\t3D type 1:\t%.3g   \t',err(1,3));
    if err(1,3)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end

  C = finufft3d2(x,y,z,isign,tol,f,o);
  Ce = A.' * reshape(f,[N1*N2*N3, ntr]);  % exact direct via non-conj transpose
  err(2,3) = norm(C-Ce)/norm(Ce);
  if errcheck>0, fprintf('\t\t3D type 2:\t%.3g   \t',err(2,3));
    if err(2,3)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end

  s = N1*myrand(M,1,prec); t = N2*myrand(M,1,prec); u = N3*myrand(M,1,prec);
  f = finufft3d3(x,y,z,c,isign,tol,s,t,u,o);
  fe = exp(1i*isign*(s*x'+t*y'+u*z'))*c;    % type 3 NUDFT matrix
  err(3,3) = norm(f-fe)/norm(fe);
  if errcheck>0, fprintf('\t\t3D type 3:\t%.3g   \t',err(3,3));
    if err(3,3)>errcheck, fprintf('FAIL!\n'); else fprintf('pass\n'); end
  end
end
