function [ok, varargout] = safe_call(fn)
% SAFE_CALL  Call fn(), catching FINUFFT:epsTooSmall.
%   [ok, ...] = safe_call(@() somefun(args))
%   Returns ok=true on success, ok=false if eps tolerance was unachievable.
%   All other errors are rethrown.
  try
    [varargout{1:nargout-1}] = fn();
    ok = true;
  catch ME
    if strcmp(ME.identifier, 'FINUFFT:epsTooSmall')
      ok = false;
    else
      rethrow(ME);
    end
  end
end
