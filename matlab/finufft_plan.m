
classdef finufft_plan < handle

  properties
    % this is a special property that MWrap uses as an opaque pointer to C++
    % object (mwptr = MWrap-pointer, not MathWorks! see MWrap manual)...
    mwptr
    % tracks what prec C++library is being called ('single' or 'double')...
    floatprec
  end

  methods

    function plan = finufft_plan(type, n_modes_or_dim, iflag, n_trans, tol, opts)
    % FINUFFT_PLAN   create guru plan object for one/many general nonuniform FFTs.

      plan.floatprec='double';                      % set precision: default
      if nargin<6, opts = []; end
      if isfield(opts,'floatprec')                  % a matlab-only option
        if ~strcmp(opts.floatprec,'single') && ~strcmp(opts.floatprec,'double')
          error('FINUFFT:badFloatPrec','FINUFFT plan opts.floatprec must be single or double');
        else
          plan.floatprec = opts.floatprec;
        end
      end
      
      n_modes = ones(3,1);  % dummy for type 3
      if type==3
        if length(n_modes_or_dim)~=1
          error('FINUFFT:badT3dim', 'FINUFFT type 3 plan n_modes_or_dim must be one number, the dimension');
        end
        dim = n_modes_or_dim;      % interpret as dim
      else
        dim = length(n_modes_or_dim);    % allows any ms,mt,mu to be 1 (weird..)
        n_modes(1:dim) = n_modes_or_dim;
      end
      % (checks of type, dim will occur in the C++ library, so omit them above)

      mex_id_ = 'finufft_mex_setup()';
finufft(mex_id_);
      mex_id_ = 'o nufft_opts* = new()';
[o] = finufft(mex_id_);
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o finufft_plan* = new()';
[p] = finufft(mex_id_);
        mex_id_ = 'finufft_default_opts(i nufft_opts*)';
finufft(mex_id_, o);
      else
        mex_id_ = 'o finufftf_plan* = new()';
[p] = finufft(mex_id_);
        mex_id_ = 'finufftf_default_opts(i nufft_opts*)';
finufft(mex_id_, o);
      end
      plan.mwptr = p;   % crucial: save the opaque ptr (p.12 of MWrap manual)

      % replace in nufft_opts struct whichever fields are in incoming opts...
      mex_id_ = 'copy_nufft_opts(i mxArray, i nufft_opts*)';
finufft(mex_id_, opts, o);
      if strcmp(plan.floatprec,'double')
        tol = double(tol);   % scalar match for mwrap>=0.33.11
        mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i finufft_plan*, i nufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      else
        tol = single(tol);   % scalar match for mwrap>=0.33.11
        mex_id_ = 'o int = finufftf_makeplan(i int, i int, i int64_t[x], i int, i int, i float, i finufftf_plan*, i nufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      end
      mex_id_ = 'delete(i nufft_opts*)';
finufft(mex_id_, o);
      errhandler(ier);             % convert C++ codes to matlab-style errors
    end

    function delete(plan)
    % This does clean-up (deallocation) of the C++ struct before the matlab
    % object deletes.
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'finufft_destroy(i finufft_plan)';
finufft(mex_id_, plan);
      else
        mex_id_ = 'finufftf_destroy(i finufftf_plan)';
finufft(mex_id_, plan);
      end
    end

    function setpts(plan, xj, yj, zj, s, t, u)
    % SETPTS   process nonuniform points for general FINUFFT transform(s).

                                                                              
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = get_dim(i finufft_plan)';
[dim] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_type(i finufft_plan)';
[type] = finufft(mex_id_, plan);
        if nargin<5, s=double.empty; t=double.empty; u=double.empty; end
      else
        mex_id_ = 'o int = get_dimf(i finufftf_plan)';
[dim] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_typef(i finufftf_plan)';
[type] = finufft(mex_id_, plan);
        if nargin<5, s=single.empty; t=single.empty; u=single.empty; end
      end
      % get number(s) of NU pts (also validates the NU pt array sizes)...
      [nj, nk] = valid_setpts(type, dim, xj, yj, zj, s, t, u);
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = finufft_setpts(i finufft_plan, i int64_t, i double[], i double[], i double[], i int64_t, i double[], i double[], i double[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      else
        mex_id_ = 'o int = finufftf_setpts(i finufftf_plan, i int64_t, i float[], i float[], i float[], i int64_t, i float[], i float[], i float[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      end
      errhandler(ier);
    end

    function result = exec(plan, data_in)
    % EXEC   execute single or many-vector FINUFFT transforms in a plan.

                                                                                                            
                                                                                                
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = get_type(i finufft_plan)';
[type] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_ntrans(i finufft_plan)';
[n_trans] = finufft(mex_id_, plan);
        mex_id_ = 'o int64_t = get_nj(i finufft_plan)';
[nj] = finufft(mex_id_, plan);
      else
        mex_id_ = 'o int = get_typef(i finufftf_plan)';
[type] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_ntransf(i finufftf_plan)';
[n_trans] = finufft(mex_id_, plan);
        mex_id_ = 'o int64_t = get_njf(i finufftf_plan)';
[nj] = finufft(mex_id_, plan);
      end
      % check data input length...
      if type==1 || type==2
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'get_nmodes(i finufft_plan, o int64_t&, o int64_t&, o int64_t&)';
[ms, mt, mu] = finufft(mex_id_, plan);
        else
          mex_id_ = 'get_nmodesf(i finufftf_plan, o int64_t&, o int64_t&, o int64_t&)';
[ms, mt, mu] = finufft(mex_id_, plan);
        end
        ncoeffs = ms*mt*mu*n_trans;    % total Fourier coeffs (out t1, or in t2)
      end
      if type==2, ninputs = ncoeffs; else, ninputs = n_trans*nj; end     % supposed input numel
      if numel(data_in)~=ninputs
        error('FINUFFT:badDataInSize','FINUFFT numel(data_in) must be n_trans times number of NU pts (type 1, 3) or Fourier modes (type 2)');
      end
      if type==1
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int = finufft_exec(i finufft_plan, i dcomplex[], o dcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, ncoeffs);
        else
          mex_id_ = 'o int = finufftf_exec(i finufftf_plan, i fcomplex[], o fcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, ncoeffs);
        end
        % make modes output correct shape; when d<3 squeeze removes unused dims...
        result = squeeze(reshape(result, [ms mt mu n_trans]));
      elseif type==2
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int = finufft_exec(i finufft_plan, o dcomplex[xx], i dcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        else
          mex_id_ = 'o int = finufftf_exec(i finufftf_plan, o fcomplex[xx], i fcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        end
      elseif type==3
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int64_t = get_nk(i finufft_plan)';
[nk] = finufft(mex_id_, plan);
          mex_id_ = 'o int = finufft_exec(i finufft_plan, i dcomplex[], o dcomplex[xx])';
[ier, result] = finufft(mex_id_, plan, data_in, nk, n_trans);
        else
          mex_id_ = 'o int64_t = get_nkf(i finufftf_plan)';
[nk] = finufft(mex_id_, plan);
          mex_id_ = 'o int = finufftf_exec(i finufftf_plan, i fcomplex[], o fcomplex[xx])';
[ier, result] = finufft(mex_id_, plan, data_in, nk, n_trans);
        end
      else
        ier = 10;         % type was corrupted since plan stage - the horror
      end
      errhandler(ier);
    end

  end
end
