% FINUFFT_PLAN   is a class which wraps the guru interface to FINUFFT.
%
%  Full documentation is given in ../finufft-manual.pdf and online at
%  http://finufft.readthedocs.io
%  Also see examples in the matlab/examples and matlab/test directories.
%
% PROPERTIES
%   mwptr - opaque pointer to a C++ finufft_plan object (see MWrap manual),
%           whose properties cannot be accessed directly
%   floatprec - either 'double' or 'single', tracks what precision of C++
%           library is being called
%   type, dim, n_modes, n_trans, nj, nk - other plan parameters
%  Note: the user should never alter these plan properties directly! Rather,
%  the below methods should be used to create, use, and destroy plans.
%
% METHODS
%   finufft_plan - create guru plan object for one/many general nonuniform FFTs.
%   setpts       - process nonuniform points for general FINUFFT transform(s).
%   execute      - execute single or many-vector FINUFFT transforms in a plan.
%
% General notes:
%  * use delete(plan) to remove a plan after use.
%  * See ERRHANDLER, VALID_*, and this code for warning/error IDs.
%
%
%
% =========== Detailed description of guru methods ==========================
%
% 1) FINUFFT_PLAN create guru plan object for one/many general nonuniform FFTs.
%
% plan = finufft_plan(type, n_modes_or_dim, isign, ntrans, eps)
% plan = finufft_plan(type, n_modes_or_dim, isign, ntrans, eps, opts)
%
% Creates a finufft_plan MATLAB object in the guru interface to FINUFFT, of
%  type 1, 2 or 3, and with given numbers of Fourier modes (unless type 3).
%
% Inputs: 
%     type            transform type: 1, 2, or 3
%     n_modes_or_dim  if type is 1 or 2, the number of Fourier modes in each
%                     dimension: [ms] in 1D, [ms mt] in 2D, or [ms mt mu] in 3D.
%                     Its length sets the dimension, which must be 1, 2 or 3.
%                     If type is 3, in contrast, its *value* fixes the dimension
%     isign if >=0, uses + sign in exponential, otherwise - sign.
%     eps   relative precision requested (generally between 1e-15 and 1e-1)
%     opts   optional struct with optional fields controlling the following:
%     opts.debug:   0 (silent, default), 1 (timing breakdown), 2 (debug info).
%     opts.spread_debug: spreader: 0 (no text, default), 1 (some), or 2 (lots)
%     opts.spread_sort:  0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.spread_kerevalmeth:  0: exp(sqrt()), 1: Horner ppval (faster)
%     opts.spread_kerpad: (iff kerevalmeth=0)  0: don't pad to mult of 4, 1: do
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT)
%     opts.spread_thread:   for ntrans>1 only. 0:auto, 1:seq multi, 2:par, etc
%     opts.maxbatchsize:  for ntrans>1 only. max blocking size, or 0 for auto.
%     opts.nthreads:   number of threads, or 0: use all available (default)
%     opts.floatprec: library precision to use, 'double' (default) or 'single'.
%     for type 1 and 2 only, the following opts fields are also relevant:
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: [DEPRECATED] has no effect
% Outputs:
%     plan            finufft_plan object (opaque pointer)
%
% Notes:
%  * For type 1 and 2, this does the FFTW planning and kernel-FT precomputation.
%  * For type 3, this does very little, since the FFT sizes are not yet known.
%  * Be default all threads are planned; control how many with opts.nthreads.
%  * The vectorized (many vector) plan, ie ntrans>1, can be much faster
%    than repeated calls with the same nonuniform points. Note that here the I/O
%    data ordering is stacked rather than interleaved. See ../docs/matlab.rst
%  * For more details about the opts fields, see ../docs/opts.rst
%
%
% 2) SETPTS   process nonuniform points for general FINUFFT transform(s).
%
% plan.setpts(xj)
% plan.setpts(xj, yj)
% plan.setpts(xj, yj, zj)
% plan.setpts(xj, [], [], s)
% plan.setpts(xj, yj, [], s, t)
% plan.setpts(xj, yj, zj, s, t, u)
%
%  When plan is a finufft_plan MATLAB object, brings in nonuniform point
%  coordinates (xj,yj,zj), and additionally in the type 3 case, nonuniform
%  frequency target points (s,t,u). Empty arrays may be passed in the case of
%  unused dimensions. For all types, sorting is done to internally store a
%  reindexing of points, and for type 3 the spreading and FFTs are planned.
%  The nonuniform points may be used for multiple transforms.
%
% Inputs:
%     xj     vector of x-coords of all nonuniform points
%     yj     empty (if dim<2), or vector of y-coords of all nonuniform points
%     zj     empty (if dim<3), or vector of z-coords of all nonuniform points
%     s      vector of x-coords of all nonuniform frequency targets
%     t      empty (if dim<2), or vector of y-coords of all frequency targets
%     u      empty (if dim<3), or vector of z-coords of all frequency targets
% Input/Outputs:
%     plan   finufft_plan object
%
% Notes:
%  * The values in xj (and if nonempty, yj and zj) are real-valued, and
%    invariant under translations by multiples of 2pi. For type 1
%    they are "sources", whereas for type 2 they are "targets".
%    For type 3 there is no periodicity, and no restrictions other
%    than the resulting size of the internal fine grids.
%  * s (and t and u) are only relevant for type 3, and may be omitted otherwise
%  * The matlab vectors xj,... and s,... should not be changed before calling
%    future execute calls, because the plan stores only pointers to the
%    arrays (they are not duplicated internally).
%  * The precision (double/single) of all inputs must match that chosen at the
%    plan stage using opts.floatprec, otherwise an error is raised.
%
%
% 3) EXECUTE   execute single or many-vector FINUFFT transforms in a plan.
%
% result = plan.execute(data_in);
%
%  For plan a previously created finufft_plan object also containing all
%  needed nonuniform point coordinates, do a single (or if ntrans>1 in the
%  plan stage, multiple) NUFFT transform(s), with the strengths or Fourier
%  coefficient inputs vector(s) from data_in. The result of the transform(s)
%  is returned as a (possibly multidimensional) array.
%
% Inputs:
%     plan     finufft_plan object
%     data_in  strengths (types 1 or 3) or Fourier coefficients (type 2)
%              vector, matrix, or array of appropriate size. For type 1 and 3,
%              this is either a length-M vector (where M is the length of xj),
%              or an (M,ntrans) matrix when ntrans>1. For type 2, in 1D this is
%              length-ms, in 2D size (ms,mt), or in 3D size (ms,mt,mu), or
%              each of these with an extra last dimension ntrans if ntrans>1.
% Outputs:
%     result   vector of output strengths at targets (types 2 or 3), or array
%              of Fourier coefficients (type 1), or, if ntrans>1, a stack of
%              such vectors or arrays, of appropriate size.
%              Specifically, if ntrans=1, for type 1, in 1D
%              this is a length-ms column vector, in 2D a matrix of size
%              (ms,mt), or in 3D an array of size (ms,mt,mu); for types 2 and 3
%              it is a column vector of length M (the length of xj in type 2),
%              or nk (the length of s in type 3). If ntrans>1 its is a stack
%              of such objects, ie, it has an extra last dimension ntrans.
%
% Notes:
%  * The precision (double/single) of all inputs must match that chosen at the
%    plan stage using opts.floatprec, otherwise an error is raised.
%
%
% 4) To deallocate (delete) a nonuniform FFT plan, use delete(plan)
%
% This deallocates all stored FFTW plans, nonuniform point sorting arrays,
%  kernel Fourier transforms arrays, etc.
%
%

classdef finufft_plan < handle

  properties
    % this is a special property that MWrap uses as an opaque pointer to C++
    % object (mwptr = MWrap-pointer, not MathWorks! see MWrap manual)...
    mwptr
    % track what prec C++ library is being called ('single' or 'double')...
    floatprec
    % track other properties we'd rather not have to query the C++ plan for...
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

      plan.floatprec='double';                      % set precision: default
      if nargin<6, opts = []; end
      if isfield(opts,'floatprec')                  % a matlab-only option
        if ~strcmp(opts.floatprec,'single') && ~strcmp(opts.floatprec,'double')
          error('FINUFFT:badFloatPrec','FINUFFT plan opts.floatprec must be single or double');
        else
          plan.floatprec = opts.floatprec;
        end
      end
      
      n_modes = ones(3,1);         % is dummy for type 3
      if type==3
        if length(n_modes_or_dim)~=1
          error('FINUFFT:badT3dim', 'FINUFFT type 3 plan n_modes_or_dim must be one number, the dimension');
        end
        dim = n_modes_or_dim;      % interpret as dim
      else
        dim = length(n_modes_or_dim);    % allows any ms,mt,mu to be 1 (weird..)
        n_modes(1:dim) = n_modes_or_dim;   % unused dims left as 1
      end
      % (checks of type, dim will occur in the C++ library, so omit them above)

      mex_id_ = 'finufft_mex_setup()';
finufft(mex_id_);
      mex_id_ = 'o finufft_opts* = new()';
[o] = finufft(mex_id_);
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o finufft_plan* = new()';
[p] = finufft(mex_id_);
        mex_id_ = 'finufft_default_opts(i finufft_opts*)';
finufft(mex_id_, o);
      else
        mex_id_ = 'o finufftf_plan* = new()';
[p] = finufft(mex_id_);
        mex_id_ = 'finufftf_default_opts(i finufft_opts*)';
finufft(mex_id_, o);
      end
      plan.mwptr = p;   % crucial: save the opaque ptr (p.12 of MWrap manual)
      plan.dim = dim;   % save other stuff to avoid having to access via C++...
      plan.type = type;
      plan.n_modes = n_modes;
      plan.n_trans = n_trans;
      % Note the peculiarity that mwrap only accepts a double for n_trans, even
      % though it's declared int. It complains, also with int64 for nj, etc :(
      
      % replace in finufft_opts struct whichever fields are in incoming opts...
      mex_id_ = 'copy_finufft_opts(i mxArray, i finufft_opts*)';
finufft(mex_id_, opts, o);
      if strcmp(plan.floatprec,'double')
        tol = double(tol);   % scalar type must match for mwrap>=0.33.11
        mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i finufft_plan*, i finufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      else
        tol = single(tol);   % ditto
        mex_id_ = 'o int = finufftf_makeplan(i int, i int, i int64_t[x], i int, i int, i float, i finufftf_plan*, i finufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      end
      mex_id_ = 'delete(i finufft_opts*)';
finufft(mex_id_, o);
      errhandler(ier);             % convert C++ codes to matlab-style errors
    end

    function setpts(plan, xj, yj, zj, s, t, u)
    % SETPTS   process nonuniform points for general FINUFFT transform(s).

      % fill missing inputs with empties of correct type
      if strcmp(plan.floatprec,'double')
        emp = double([]);
      else
        emp = single([]);
      end
      if nargin<3, yj=emp; end
      if nargin<4, zj=emp; end
      if nargin<5, s=emp; end
      if nargin<6, t=emp; end
      if nargin<7, u=emp; end
      % get number(s) of NU pts (also validates the NU pt array sizes)...
      [nj, nk] = valid_setpts(plan.type, plan.dim, xj, yj, zj, s, t, u);
      plan.nj = nj;            % save to avoid having to query the C++ plan
      plan.nk = nk;            % "
      % Force MATLAB to preserve the memory of xj/yj/zj by storing them as class
      % properties (see issue #185). Ideally, we would pass plan.xj/yj/zj to the
      % MWrap call below, but MWrap fails to parse the "." syntax. However,
      % simply storing xj/yj/zj ensures that the memory will be preserved.
      plan.xj = xj;
      plan.yj = yj;
      plan.zj = zj;
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = finufft_setpts(i finufft_plan, i int64_t, i double[], i double[], i double[], i int64_t, i double[], i double[], i double[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      else
        mex_id_ = 'o int = finufftf_setpts(i finufftf_plan, i int64_t, i float[], i float[], i float[], i int64_t, i float[], i float[], i float[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      end
      errhandler(ier);
    end

    function result = execute(plan, data_in)
    % EXECUTE   execute single or many-vector FINUFFT transforms in a plan.

      % get shape info from the matlab-side plan (since can't pass "dot"
      % variables like a.b as mwrap sizes, too)...
      ms = plan.n_modes(1); mt = plan.n_modes(2); mu = plan.n_modes(3);
      nj = plan.nj; nk = plan.nk; n_trans = plan.n_trans;

      % check data input length...
      if plan.type==1 || plan.type==2
        ncoeffs = ms*mt*mu*n_trans;    % total # Fourier coeffs
      end
      if plan.type==2
        ninputs = ncoeffs;
      else
        ninputs = n_trans*nj;
      end
      if numel(data_in)~=ninputs
        error('FINUFFT:badDataInSize','FINUFFT numel(data_in) must be n_trans times number of NU pts (type 1, 3) or Fourier modes (type 2)');
      end
      if plan.type == 1
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int = finufft_execute(i finufft_plan, i dcomplex[], o dcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, ncoeffs);
        else
          mex_id_ = 'o int = finufftf_execute(i finufftf_plan, i fcomplex[], o fcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, ncoeffs);
        end
        % make modes output correct shape; when d<3 squeeze removes unused dims...
        result = squeeze(reshape(result, [ms mt mu n_trans]));
      elseif plan.type == 2
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int = finufft_execute(i finufft_plan, o dcomplex[xx], i dcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        else
          mex_id_ = 'o int = finufftf_execute(i finufftf_plan, o fcomplex[xx], i fcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        end
      elseif plan.type == 3
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int = finufft_execute(i finufft_plan, i dcomplex[], o dcomplex[xx])';
[ier, result] = finufft(mex_id_, plan, data_in, nk, n_trans);
        else
          mex_id_ = 'o int = finufftf_execute(i finufftf_plan, i fcomplex[], o fcomplex[xx])';
[ier, result] = finufft(mex_id_, plan, data_in, nk, n_trans);
        end
      end
      errhandler(ier);
    end

    function delete(plan)
    % This does clean-up (deallocation) of the C++ struct before the matlab
    % object deletes. It is automatically called by MATLAB and octave if the
    % plan goes out of scope.
      if ~isempty(plan.mwptr)    % catch octave's allowance of >1 deletings!
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'finufft_destroy(i finufft_plan)';
finufft(mex_id_, plan);
        else
          mex_id_ = 'finufftf_destroy(i finufftf_plan)';
finufft(mex_id_, plan);
        end
        plan.mwptr = '';         % we use to mean "destroyed on the C++ side"
      end
    end

  end
end
