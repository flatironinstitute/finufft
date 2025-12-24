% CUFINUFFT_PLAN   is a class which wraps the guru GPU interface to FINUFFT.
%
%  Full documentation is online at http://finufft.readthedocs.io
%  Also see examples in matlab/cuda/examples and matlab/cuda/test
%
% PROPERTIES
%   mwptr - opaque pointer to a C++ finufft_plan object (see MWrap manual),
%           whose properties cannot be accessed directly
%   floatprec - either 'double' or 'single', tracks what precision of C++
%           library is being called
%   type, dim, n_modes, n_trans, nj, nk, xj, yj, zj - other plan parameters.
%  Note: the user should never alter these plan properties directly! Rather,
%  the below methods should be used to create, use, and destroy plans.
%
% METHODS
%   cufinufft_plan - create plan object for one/many general nonuniform FFTs.
%   setpts       - process nonuniform points for general FINUFFT transform(s).
%   execute      - execute single or many-vector FINUFFT transforms in a plan.
%
% General notes:
%  * All array inputs and outputs are MATLAB gpuArrays of the same precision.
%  * Use delete(plan) to remove a plan after use.
%  * See ERRHANDLER, VALID_*, and this code for warning/error IDs.
%
%
%
% =========== Detailed description of guru methods ==========================
%
% 1) CUFINUFFT_PLAN create plan object for one/many general nonuniform FFTs.
%
% plan = cufinufft_plan(type, n_modes_or_dim, isign, ntrans, eps)
% plan = cufinufft_plan(type, n_modes_or_dim, isign, ntrans, eps, opts)
%
% Creates a cufinufft_plan MATLAB object in the interface to GPU FINUFFT, of
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
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT).
%     opts.gpu_method:  0 (auto, default), 1 (GM or GM-sort), 2 (SM).
%     opts.gpu_sort:  0 (do not sort NU pts), 1 (sort when GM method, default).
%     opts.gpu_kerevalmeth:  0 (slow reference). 1 (Horner ppoly, default).
%     opts.gpu_maxsubprobsize:  max # NU pts per subprob (gpu_method=2 only).
%     opts.gpu_binsize{x,y,z}:  various binsizes in GM-sort/SM (for experts).
%     opts.gpu_maxbatchsize:   0 (auto, default), or many-vector batch size.
%     opts.gpu_device_id:  sets the GPU device ID (experts only).
%     opts.floatprec: library precision to use, 'double' (default) or 'single'.
%     for type 1 and 2 only, the following opts fields are also relevant:
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.gpu_spreadinterponly: 0 (do NUFFT, default), 1 (only spread/interp)
% Outputs:
%     plan            cufinufft_plan object (opaque pointer)
%
% Notes:
%  * For type 1 and 2, this does the cuFFT planning and kernel-FT precompute.
%  * For type 3, this does very little, since cuFFT sizes are not yet known.
%  * The vectorized (many vector) plan, ie ntrans>1, can be faster
%    than repeated calls with the same nonuniform points. Note that here the
%    I/O data ordering is stacked not interleaved. See ../docs/matlab_gpu.rst
%  * For more details about the opts fields, see ../docs/c_gpu.rst
%
%
% 2) SETPTS   process nonuniform points for general GPU FINUFFT transform(s).
%
% plan.setpts(xj)
% plan.setpts(xj, yj)
% plan.setpts(xj, yj, zj)
% plan.setpts(xj, [], [], s)
% plan.setpts(xj, yj, [], s, t)
% plan.setpts(xj, yj, zj, s, t, u)
%
%  When plan is a cufinufft_plan MATLAB object, brings in nonuniform point
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
%     plan   cufinufft_plan object
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
% 3) EXECUTE   execute single or many-vector GPU FINUFFT transforms in a plan.
%
% result = plan.execute(data_in);
%
%  For plan a previously created cufinufft_plan object also containing all
%  needed nonuniform point coordinates, do a single (or if ntrans>1 in the
%  plan stage, multiple) NUFFT transform(s), with the strengths or Fourier
%  coefficient inputs vector(s) from data_in. The result of the transform(s)
%  is returned as a (possibly multidimensional) gpuArray.
%
% Inputs:
%     plan     cufinufft_plan object
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
%  * The precision (double/single) of all gpuArrays must match that chosen at
%    the plan stage using opts.floatprec, otherwise an error is raised.
%
%
% 4) To deallocate (delete) a GPU nonuniform FFT plan, use delete(plan)
%
% This deallocates all stored cuFFT plans, nonuniform point sorting arrays,
%  kernel Fourier transforms arrays, etc.
%
%

classdef cufinufft_plan < handle

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

    function plan = cufinufft_plan(type, n_modes_or_dim, iflag, n_trans, tol, opts)
    % CUFINUFFT_PLAN   create guru plan object for one/many general nonuniform FFTs.

      plan.floatprec='double';                      % set precision: default
      if nargin<6, opts = struct(); end
      if isfield(opts,'floatprec')                  % a matlab-only option
        if ~strcmp(opts.floatprec,'single') && ~strcmp(opts.floatprec,'double')
          error('FINUFFT:badFloatPrec','CUFINUFFT plan opts.floatprec must be single or double');
        else
          plan.floatprec = opts.floatprec;
        end
      end

      n_modes = ones(3,1);         % is dummy for type 3
      if type==3
        if length(n_modes_or_dim)~=1
          error('FINUFFT:badT3dim', 'CUFINUFFT type 3 plan n_modes_or_dim must be one number, the dimension');
        end
        dim = n_modes_or_dim;      % interpret as dim
      else
        dim = length(n_modes_or_dim);    % allows any ms,mt,mu to be 1 (weird..)
        n_modes(1:dim) = n_modes_or_dim;   % unused dims left as 1
      end
      % (checks of type, dim will occur in the C++ library, so omit them above)

      mex_id_ = 'c o cufinufft_opts* = new()';
[o] = cufinufft(mex_id_);
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'c o cufinufft_plan* = new()';
[p] = cufinufft(mex_id_);
        mex_id_ = 'cufinufft_default_opts(c i cufinufft_opts*)';
cufinufft(mex_id_, o);
      else
        mex_id_ = 'c o cufinufftf_plan* = new()';
[p] = cufinufft(mex_id_);
        mex_id_ = 'cufinufft_default_opts(c i cufinufft_opts*)';
cufinufft(mex_id_, o);
      end
      plan.mwptr = p;   % crucial: save the opaque ptr (p.12 of MWrap manual)
      plan.dim = dim;   % save other stuff to avoid having to access via C++...
      plan.type = type;
      plan.n_modes = n_modes;
      plan.n_trans = n_trans;
      % Note the peculiarity that mwrap only accepts a double for n_trans, even
      % though it's declared int. It complains, also with int64 for nj, etc :(

      % replace in cufinufft_opts struct whichever fields are in incoming opts...
      mex_id_ = 'copy_cufinufft_opts(c i mxArray, c i cufinufft_opts*)';
cufinufft(mex_id_, opts, o);
      if strcmp(plan.floatprec,'double')
        tol = double(tol);   % scalar type must match for mwrap>=0.33.11
        mex_id_ = 'c o int = cufinufft_makeplan(c i int, c i int, c i int64_t[x], c i int, c i int, c i double, c i cufinufft_plan*, c i cufinufft_opts*)';
[ier] = cufinufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      else
        tol = single(tol);   % ditto
        mex_id_ = 'c o int = cufinufftf_makeplan(c i int, c i int, c i int64_t[x], c i int, c i int, c i float, c i cufinufftf_plan*, c i cufinufft_opts*)';
[ier] = cufinufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      end
      mex_id_ = 'delete(c i cufinufft_opts*)';
cufinufft(mex_id_, o);
      errhandler(ier);             % convert C++ codes to matlab-style errors
    end

    function setpts(plan, xj, yj, zj, s, t, u)
    % SETPTS   process nonuniform points for general GPU FINUFFT transform(s).

      % fill missing inputs with empties of correct type
      if strcmp(plan.floatprec,'double')
        emp = gpuArray(double([]));
      else
        emp = gpuArray(single([]));
      end
      if nargin<3 || numel(yj)==0, yj=emp; end
      if nargin<4 || numel(zj)==0, zj=emp; end
      if nargin<5 || numel(s)==0, s=emp; end
      if nargin<6 || numel(t)==0, t=emp; end
      if nargin<7 || numel(u)==0, u=emp; end
      % get number(s) of NU pts (also validates the NU pt array sizes)...
      [nj, nk] = valid_setpts(1,plan.type, plan.dim, xj, yj, zj, s, t, u);
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
        mex_id_ = 'c o int = cufinufft_setpts(c i cufinufft_plan, c i int, g i double[], g i double[], g i double[], c i int, g i double[], g i double[], g i double[])';
[ier] = cufinufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      else
        mex_id_ = 'c o int = cufinufftf_setpts(c i cufinufftf_plan, c i int, g i float[], g i float[], g i float[], c i int, g i float[], g i float[], g i float[])';
[ier] = cufinufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      end
      errhandler(ier);
    end

    function result = execute(plan, data_in)
    % EXECUTE   execute single or many-vector GPU FINUFFT transforms in a plan.

      % check if data_in is gpuArray
      if ~isa(data_in, 'gpuArray')
        error('FINUFFT:badDataDevice','input data must be a gpuArray');
      end

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
        error('FINUFFT:badDataInSize','CUFINUFFT numel(data_in) must be n_trans times number of NU pts (type 1, 3) or Fourier modes (type 2)');
      end
      if plan.type == 1
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'c o int = cufinufft_execute(c i cufinufft_plan, g i dcomplex[], g o dcomplex[x])';
[ier, result] = cufinufft(mex_id_, plan, data_in, ncoeffs);
        else
          mex_id_ = 'c o int = cufinufftf_execute(c i cufinufftf_plan, g i fcomplex[], g o fcomplex[x])';
[ier, result] = cufinufft(mex_id_, plan, data_in, ncoeffs);
        end
        % make modes output correct shape; when d<3 squeeze removes unused dims...
        result = squeeze(reshape(result, [ms mt mu n_trans]));
      elseif plan.type == 2
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'c o int = cufinufft_execute(c i cufinufft_plan, g o dcomplex[xx], g i dcomplex[])';
[ier, result] = cufinufft(mex_id_, plan, data_in, nj, n_trans);
        else
          mex_id_ = 'c o int = cufinufftf_execute(c i cufinufftf_plan, g o fcomplex[xx], g i fcomplex[])';
[ier, result] = cufinufft(mex_id_, plan, data_in, nj, n_trans);
        end
      elseif plan.type == 3
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'c o int = cufinufft_execute(c i cufinufft_plan, g i dcomplex[], g o dcomplex[xx])';
[ier, result] = cufinufft(mex_id_, plan, data_in, nk, n_trans);
        else
          mex_id_ = 'c o int = cufinufftf_execute(c i cufinufftf_plan, g i fcomplex[], g o fcomplex[xx])';
[ier, result] = cufinufft(mex_id_, plan, data_in, nk, n_trans);
        end
      end
      errhandler(ier);
    end

    function delete(plan)
    % This does clean-up (deallocation) of the C++ struct before the matlab
    % object deletes. It is automatically called by MATLAB if the
    % plan goes out of scope.
      if ~isempty(plan.mwptr)    % catch octave's allowance of >1 deletings!
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'cufinufft_destroy(c i cufinufft_plan)';
cufinufft(mex_id_, plan);
        else
          mex_id_ = 'cufinufftf_destroy(c i cufinufftf_plan)';
cufinufft(mex_id_, plan);
        end
        plan.mwptr = '';         % we use to mean "destroyed on the C++ side"
      end
    end

  end
end
