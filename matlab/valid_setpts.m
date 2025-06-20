function [nj, nk] = valid_setpts(is_gpuarray,type,dim,x,y,z,s,t,u)
% VALID_SETPTS   validate guru finufft_setpts input sizes for a type and dim
%                is_gpuarray should be 1 if data is expected on gpu, or 0 if on cpu
%
% [nj nk] = valid_setpts(is_gpuarray,type,dim,x,y,z,s,t,u) raises errors if there are
%  incompatible input sizes for transform type (1, 2 or 3) and dimension dim,
%  and returns nj (aka M), and, for type 3, also nk (aka N). It also checks if the arrays
%  are on the expected device, is_gpuarray 0 for cpu, 1 for gpu. The returned
%  values are int64 (not the usual double class of numel).
%
% nj = valid_setpts(is_gpuarray,type,dim,x,y,z) is also allowed for types 1, 2.

% Barnett 6/19/20, split out from guru so simple ints can check before plan.
% s,t,u are only checked for type 3.
% note that isvector([]) is false, isa([], 'gpuArray') is false, isa(gpuArray([]), 'gpuArray') is true.
if ~isvector(x), error('FINUFFT:badXshape','FINUFFT x must be a vector'); end
if isa(x, 'gpuArray') ~= is_gpuarray, error('FINUFFT:badDataDevice', 'input data must be on the specified device'); end
nj = numel(x);
if type==3
  nk = numel(s);
  if ~isvector(s), error('FINUFFT:badSshape','FINUFFT s must be a vector'); end
  if isa(s, 'gpuArray') ~= is_gpuarray, error('FINUFFT:badDataDevice', 'input data must be on the specified device'); end
else
  nk = 0;   % dummy output
end
if dim>1
  if ~isvector(y), error('FINUFFT:badYshape','FINUFFT y must be a vector'); end
  if isa(y, 'gpuArray') ~= is_gpuarray, error('FINUFFT:badDataDevice', 'input data must be on the specified device'); end
  if numel(y)~=nj, error('FINUFFT:badYlen','FINUFFT y must have same length as x'); end
  if type==3
    if ~isvector(t), error('FINUFFT:badTshape','FINUFFT t must be a vector'); end
    if isa(t, 'gpuArray') ~= is_gpuarray, error('FINUFFT:badDataDevice', 'input data must be on the specified device'); end
    if numel(t)~=nk, error('FINUFFT:badTlen','FINUFFT t must have same length as s'); end
  end
end
if dim>2
  if ~isvector(z), error('FINUFFT:badZshape','FINUFFT z must be a vector'); end
  if isa(z, 'gpuArray') ~= is_gpuarray, error('FINUFFT:badDataDevice', 'input data must be on the specified device'); end
  if numel(z)~=nj, error('FINUFFT:badZlen','FINUFFT z must have same length as x'); end
  if type==3
    if ~isvector(u), error('FINUFFT:badUshape','FINUFFT u must be a vector'); end
    if isa(u, 'gpuArray') ~= is_gpuarray, error('FINUFFT:badDataDevice', 'input data must be on the specified device'); end
    if numel(u)~=nk, error('FINUFFT:badUlen','FINUFFT u must have same length as s'); end
  end
end
