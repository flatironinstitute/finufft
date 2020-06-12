
classdef finufft_plan < handle

  properties
% this is a dummy property to tell MWrap to treat this in OO way...
% (mwptr = MWrap-pointer, not MathWorks!)
    mwptr;
  end

  methods

    function [plan, ier] = finufft_plan(type, n_modes_or_dim, iflag, n_transf, tol, opts)
    % FINUFFT_PLAN
      mex_id_ = 'finufft_mex_setup()';
finufft(mex_id_);
      mex_id_ = 'o finufft_plan* = new()';
[p] = finufft(mex_id_);
      plan.mwptr = p;                        % alex asks: is this needed? 
      assert(type==1 || type==2 || type==3);
      n_modes = ones(3,1);    % dummy for type 3
      if type==3
        assert(length(n_modes_or_dim)==1);
        dim = n_modes_or_dim;      % interpret as dim
      else
        dim = length(n_modes_or_dim);
        n_modes(1:dim) = n_modes_or_dim;
      end
      assert(dim==1 || dim==2 || dim==3);
      if nargin<6
        mex_id_ = 'o nufft_opts* = new()';
[o] = finufft(mex_id_);
        mex_id_ = 'finufft_default_opts(i nufft_opts*)';
finufft(mex_id_, o);
        mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i finufft_plan*, i nufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_transf, tol, plan, o, 3);
        mex_id_ = 'delete(i nufft_opts*)';
finufft(mex_id_, o);
      else
        % first set o with default value and then replace with fields in opts
        mex_id_ = 'o nufft_opts* = new()';
[o] = finufft(mex_id_);
        mex_id_ = 'finufft_default_opts(i nufft_opts*)';
finufft(mex_id_, o);
        mex_id_ = 'copy_nufft_opts(i mxArray, i nufft_opts*)';
finufft(mex_id_, opts, o);
        mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i finufft_plan*, i nufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_transf, tol, plan, o, 3);
        mex_id_ = 'delete(i nufft_opts*)';
finufft(mex_id_, o);
      end
    end

    function delete(plan)
    % NODOCS
      mex_id_ = 'finufft_destroy(i finufft_plan*)';
finufft(mex_id_, plan);
    end

    function finufft_destroy(plan)
    % FINUFFT_DESTROY
      mex_id_ = 'finufft_destroy(i finufft_plan*)';
finufft(mex_id_, plan);
    end

    function [ier] = finufft_setpts(plan, xj, yj, zj, s, t, u)
    % FINUFFT_SETPTS
      nj = numel(xj);   % note the matlab way is to extract sizes like this
      nk = numel(s);
      mex_id_ = 'o int = finufft_setpts(i finufft_plan*, i int64_t, i double[], i double[], i double[], i int64_t, i double[], i double[], i double[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
    end

    function [result, ier] = finufft_exec(plan, data_in)
    % FINUFFT_EXEC
                                                                                                                                    
      mex_id_ = 'o int = get_type(i finufft_plan*)';
[type] = finufft(mex_id_, plan);
      mex_id_ = 'o int = get_ntransf(i finufft_plan*)';
[n_transf] = finufft(mex_id_, plan);

      if type==1
        mex_id_ = 'get_nmodes(i finufft_plan*, o int64_t&, o int64_t&, o int64_t&)';
[ms, mt, mu] = finufft(mex_id_, plan);
        outsize = ms*mt*mu*n_transf;
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, outsize);
        result = reshape(result, [ms mt mu n_transf]);
      elseif type==2
        mex_id_ = 'o int64_t = get_nj(i finufft_plan*)';
[nj] = finufft(mex_id_, plan);
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, o dcomplex[xx], i dcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_transf);
      elseif type==3
        mex_id_ = 'o int64_t = get_nk(i finufft_plan*)';
[nk] = finufft(mex_id_, plan);
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[xx])';
[ier, result] = finufft(mex_id_, plan, data_in, nk, n_transf);
      else
        result = [];       % why was it 4?
        ier = 1;
      end
    end

  end
end
