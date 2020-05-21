
classdef nufft_plan < handle

  properties
    mwptr;
    nthreads_user;
    nthreads_omp;
  end

  methods

    function [plan] = nufft_plan(type, n_modes, iflag, n_transf, tol, blksize, opts)
      mex_id_ = 'o finufft_plan* = new()';
[p] = nufft_plan_mex(mex_id_);
      plan.mwptr = p;
      assert(type==1 || type==2 || type==3);
      if type==3
        assert(length(n_modes)==1);
        n_dims = n_modes;
      else
        n_dims = length(n_modes);
      end
      assert(n_dims==1 || n_dims==2 || n_dims==3);
      % if n_modes had length <3, pad with 1's (also overwrites whatever there):
      if n_dims<2, n_modes(2)=1; end
      if n_dims<3, n_modes(3)=1; end
      if nargin<7
        mex_id_ = 'o nufft_opts* = new()';
[o] = nufft_plan_mex(mex_id_);
        mex_id_ = 'finufft_default_opts(i nufft_opts*)';
nufft_plan_mex(mex_id_, o);
        mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i int, i finufft_plan*, i nufft_opts*)';
[ier] = nufft_plan_mex(mex_id_, type, n_dims, n_modes, iflag, n_transf, tol, blksize, plan, o, 3);
        mex_id_ = 'delete(i nufft_opts*)';
nufft_plan_mex(mex_id_, o);
      else
        % first set o with default value and then replace with fields in opts
        mex_id_ = 'o nufft_opts* = new()';
[o] = nufft_plan_mex(mex_id_);
        mex_id_ = 'finufft_default_opts(i nufft_opts*)';
nufft_plan_mex(mex_id_, o);
        mex_id_ = 'copy_nufft_opts(i mxArray, i nufft_opts*)';
nufft_plan_mex(mex_id_, opts, o);
        mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i int, i finufft_plan*, i nufft_opts*)';
[ier] = nufft_plan_mex(mex_id_, type, n_dims, n_modes, iflag, n_transf, tol, blksize, plan, o, 3);
        mex_id_ = 'delete(i nufft_opts*)';
nufft_plan_mex(mex_id_, o);
      end
    end

    function delete(plan)
      mex_id_ = 'finufft_destroy(i finufft_plan*)';
nufft_plan_mex(mex_id_, plan);
    end

    function nufft_destroy(plan)
      mex_id_ = 'finufft_destroy(i finufft_plan*)';
nufft_plan_mex(mex_id_, plan);
    end

    function [ier] = nufft_setpts(plan, xj, yj, zj, s, t, u)
      nj = numel(xj);   % note the matlab way is to extract sizes like this
      nk = numel(s);
      mex_id_ = 'o int = finufft_setpts(i finufft_plan*, i int64_t, i double[], i double[], i double[], i int64_t, i double[], i double[], i double[])';
[ier] = nufft_plan_mex(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
    end

    function [result, ier] = nufft_excute(plan, data_in)
                                                                                                                                    
      mex_id_ = 'o int = get_type(i finufft_plan*)';
[type] = nufft_plan_mex(mex_id_, plan);
      mex_id_ = 'o int = get_ntransf(i finufft_plan*)';
[n_transf] = nufft_plan_mex(mex_id_, plan);

      if type==1
        mex_id_ = 'get_nmodes(i finufft_plan*, o int64_t&, o int64_t&, o int64_t&)';
[ms, mt, mu] = nufft_plan_mex(mex_id_, plan);
        outsize = ms*mt*mu*n_transf;
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[x])';
[ier, result] = nufft_plan_mex(mex_id_, plan, data_in, outsize);
        reshape(result, ms, mt, mu, n_transf);
      elseif type==2
        mex_id_ = 'o int64_t = get_nj(i finufft_plan*)';
[nj] = nufft_plan_mex(mex_id_, plan);
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, o dcomplex[xx], i dcomplex[])';
[ier, result] = nufft_plan_mex(mex_id_, plan, data_in, nj, n_transf);
      elseif type==3
        mex_id_ = 'o int64_t = get_nk(i finufft_plan*)';
[nk] = nufft_plan_mex(mex_id_, plan);
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[xx])';
[ier, result] = nufft_plan_mex(mex_id_, plan, data_in, nk, n_transf);
      else
        result = 4;
        ier = 1;
      end
    end

  end
end
