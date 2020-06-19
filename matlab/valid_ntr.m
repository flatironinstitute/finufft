function n_transf = valid_ntr(x,c);
% VALID_NTR   deduce n_transforms and validate the size of c, for types 1 and 3.

n_transf = round(numel(c)/numel(x));    % this allows general row/col vec, matrix, input shapes
if n_transf*numel(x)~=numel(c)
  error('FINUFFT:badCsize','FINUFFT numel(c) must be divisible by numel(x)');
end
