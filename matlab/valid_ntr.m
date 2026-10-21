function n_transf = valid_ntr(x,c)
% VALID_NTR   deduce n_transforms and validate the size of c, for types 1 and 3.
%             also check for array device consistency.

if isa(x, 'gpuArray') ~= isa(c, 'gpuArray')
  error('FINUFFT:mixedDevice','FINUFFT: x and c must be both on GPU or CPU');
end

if ~isempty(x)
  n_transf = round(numel(c)/numel(x));    % this allows general row/col vec, matrix, input shapes
else
  s = size(c);                            % for empty NU pts array, can only infer n_transf from
  n_transf = s(end);                      % outer dim of c (eg, 0-by-n_transf in 1D). This is a bit
                                          % silly, but is the correct size (eg for erralltypedim)
end
if n_transf*numel(x)~=numel(c)            % raises no error in the empty case
  error('FINUFFT:badCsize','FINUFFT numel(c) must be divisible by numel(x)');
end
