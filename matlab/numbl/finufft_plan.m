% FINUFFT_PLAN  Guru interface to FINUFFT via WebAssembly.
%
% Rewritten for numbl: calls JS/WASM functions instead of MEX.
% See original finufft_plan.m for full documentation.
%
classdef finufft_plan < handle

  properties
    plan_handle     % integer handle to WASM-side plan
    floatprec
    type
    dim
    n_modes         % 3-element array, 1's in the unused dims
    n_trans
    nj              % number of NU pts (type 1,2), or input NU pts (type 3)
    nk              % number of output NU pts (type 3)
    xj
    yj
    zj
  end

  methods

    function plan = finufft_plan(type, n_modes_or_dim, iflag, n_trans, tol, opts)
    % FINUFFT_PLAN   create guru plan object for one/many general nonuniform FFTs.

      plan.floatprec = 'double';
      if nargin < 6
        opts = struct();
      end

      n_modes = ones(3, 1);
      if type == 3
        if length(n_modes_or_dim) ~= 1
          error('FINUFFT type 3 plan n_modes_or_dim must be one number, the dimension');
        end
        dim = n_modes_or_dim;
      else
        dim = length(n_modes_or_dim);
        n_modes(1:dim) = n_modes_or_dim;
      end

      plan.dim = dim;
      plan.type = type;
      plan.n_modes = n_modes;
      plan.n_trans = n_trans;

      plan.plan_handle = finufft_makeplan(type, dim, n_modes, iflag, n_trans, tol);
    end

    function setpts(plan, xj, yj, zj, s, t, u)
    % SETPTS   process nonuniform points for general FINUFFT transform(s).

      emp = [];
      if nargin < 3, yj = emp; end
      if nargin < 4, zj = emp; end
      if nargin < 5, s = emp; end
      if nargin < 6, t = emp; end
      if nargin < 7, u = emp; end

      [nj, nk] = valid_setpts(0, plan.type, plan.dim, xj, yj, zj, s, t, u);
      plan.nj = nj;
      plan.nk = nk;
      plan.xj = xj;
      plan.yj = yj;
      plan.zj = zj;

      finufft_setpts(plan.plan_handle, nj, xj, yj, zj, nk, s, t, u);
    end

    function result = execute(plan, data_in)
    % EXECUTE   execute single or many-vector FINUFFT transforms in a plan.

      ms = plan.n_modes(1); mt = plan.n_modes(2); mu = plan.n_modes(3);
      nj = plan.nj; nk = plan.nk; n_trans = plan.n_trans;

      if plan.type == 1 || plan.type == 2
        ncoeffs = ms * mt * mu * n_trans;
      end

      if plan.type == 1
        n_in = n_trans * nj;
        n_out = ncoeffs;
      elseif plan.type == 2
        n_in = ncoeffs;
        n_out = n_trans * nj;
      else
        n_in = n_trans * nj;
        n_out = n_trans * nk;
      end

      result = finufft_execute(plan.plan_handle, data_in, n_in, n_out);

      % Reshape output to match MATLAB conventions
      if plan.type == 1
        result = squeeze(reshape(result, [ms mt mu n_trans]));
      elseif plan.type == 2
        result = reshape(result, [nj, n_trans]);
      elseif plan.type == 3
        result = reshape(result, [nk, n_trans]);
      end
    end

    function result = execute_adjoint(plan, data_in)
    % EXECUTE_ADJOINT   execute adjoint of the planned FINUFFT transform(s).

      ms = plan.n_modes(1); mt = plan.n_modes(2); mu = plan.n_modes(3);
      nj = plan.nj; nk = plan.nk; n_trans = plan.n_trans;

      if plan.type == 1 || plan.type == 2
        ncoeffs = ms * mt * mu * n_trans;
      end

      if plan.type == 1
        % Adjoint of type 1: input is coefficients, output is values at NU pts
        n_in = ncoeffs;
        n_out = n_trans * nj;
      elseif plan.type == 2
        % Adjoint of type 2: input is values at NU pts, output is coefficients
        n_in = n_trans * nj;
        n_out = ncoeffs;
      else
        % Adjoint of type 3: reversed data flow
        n_in = n_trans * nk;
        n_out = n_trans * nj;
      end

      result = finufft_execute_adjoint(plan.plan_handle, data_in, n_in, n_out);

      % Reshape output to match MATLAB conventions
      if plan.type == 1
        result = reshape(result, [nj, n_trans]);
      elseif plan.type == 2
        result = squeeze(reshape(result, [ms mt mu n_trans]));
      elseif plan.type == 3
        result = reshape(result, [nj, n_trans]);
      end
    end

    function delete(plan)
    % DELETE   clean up the WASM-side plan.
      if ~isempty(plan.plan_handle)
        finufft_destroy(plan.plan_handle);
        plan.plan_handle = [];
      end
    end

  end
end
