
classdef finufft_plan < handle

  properties
% this is a special property that MWrap uses as an opaque pointer to C++ object
% (mwptr = MWrap-pointer, not MathWorks!)
    mwptr
  end

  methods

    function plan = finufft_plan(type, n_modes_or_dim, iflag, n_transf, tol, opts)
    % FINUFFT_PLAN   create guru plan object for one/many general nonuniform FFTs.

      mex_id_ = 'finufft_mex_setup()';
finufft(mex_id_);
      mex_id_ = 'o finufft_plan* = new()';
[p] = finufft(mex_id_);
      plan.mwptr = p;       % crucial: save the ptr as in p.12 of MWrap manual
      n_modes = ones(3,1);  % dummy for type 3
      if type==3
        if length(n_modes_or_dim)~=1
          error('FINUFFT:badT3dim', 'FINUFFT type 3 plan n_modes_or_dim must be one number, the dim');
        end
        dim = n_modes_or_dim;      % interpret as dim
      else
        dim = length(n_modes_or_dim);
        n_modes(1:dim) = n_modes_or_dim;
      end
      % (checks of type and dim will occur in the C++ library so omit them above)

      mex_id_ = 'o nufft_opts* = new()';
[o] = finufft(mex_id_);
      mex_id_ = 'finufft_default_opts(i nufft_opts*)';
finufft(mex_id_, o);
      if nargin==6                 % replace whichever fields are in opts
        mex_id_ = 'copy_nufft_opts(i mxArray, i nufft_opts*)';
finufft(mex_id_, opts, o);
      end
      mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i finufft_plan*, i nufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_transf, tol, plan, o, 3);
      mex_id_ = 'delete(i nufft_opts*)';
finufft(mex_id_, o);
      errhandler(ier);             % convert C++ codes to matlab-style errors
    end

    function delete(plan)
      mex_id_ = 'finufft_destroy(i finufft_plan*)';
finufft(mex_id_, plan);
    end

    function finufft_destroy(plan)
    % FINUFFT_DESTROY   deallocate (delete) a nonuniform FFT plan.
      mex_id_ = 'finufft_destroy(i finufft_plan*)';
finufft(mex_id_, plan);
    end

    function finufft_setpts(plan, xj, yj, zj, s, t, u)
    % FINUFFT_SETPTS   process nonuniform points for general NUFFT transform(s).
      if nargin<5, s=[]; t=[]; u=[]; end
      
                                          
      mex_id_ = 'o int = get_dim(i finufft_plan*)';
[dim] = finufft(mex_id_, plan);
      mex_id_ = 'o int = get_type(i finufft_plan*)';
[type] = finufft(mex_id_, plan);
      % get number(s) of NU pts (also validates the NU pt array sizes)...
      [nj, nk] = valid_setpts(type, dim, xj, yj, zj, s, t, u);
      mex_id_ = 'o int = finufft_setpts(i finufft_plan*, i int64_t, i double[], i double[], i double[], i int64_t, i double[], i double[], i double[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      errhandler(ier);
    end

    function result = finufft_exec(plan, data_in)
    % FINUFFT_EXEC   execute single or many-vector NUFFT transforms in a plan.

                                                                                                
      mex_id_ = 'o int = get_type(i finufft_plan*)';
[type] = finufft(mex_id_, plan);
      mex_id_ = 'o int = get_ntransf(i finufft_plan*)';
[n_transf] = finufft(mex_id_, plan);
      mex_id_ = 'o int64_t = get_nj(i finufft_plan*)';
[nj] = finufft(mex_id_, plan);
      % check data input length...
      if type==1 || type==2
        mex_id_ = 'get_nmodes(i finufft_plan*, o int64_t&, o int64_t&, o int64_t&)';
[ms, mt, mu] = finufft(mex_id_, plan);
        ncoeffs = ms*mt*mu*n_transf;                  % total Fourier coeffs (out t1, or in t2)
      end
      if type==2, ninputs = ncoeffs; else, ninputs = n_transf*nj; end     % supposed input numel
      if numel(data_in)~=ninputs
        error('FINUFFT:badDataInSize','FINUFFT numel(data_in) must be n_trans times number of NU pts (type 1, 3) or Fourier modes (type 2)');
      end
      if type==1
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, ncoeffs);
        % make output correct matlab shape; for d<3, squeeze removes unused dims...
        result = squeeze(reshape(result, [ms mt mu n_transf]));
      elseif type==2
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, o dcomplex[xx], i dcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_transf);
      elseif type==3
        mex_id_ = 'o int64_t = get_nk(i finufft_plan*)';
[nk] = finufft(mex_id_, plan);
        mex_id_ = 'o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[xx])';
[ier, result] = finufft(mex_id_, plan, data_in, nk, n_transf);
      else
        ier = 10;        % type must have been corrupted since plan stage - the horror
      end
      errhandler(ier);
    end

  end
end
