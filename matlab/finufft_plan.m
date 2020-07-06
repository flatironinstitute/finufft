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
%
% METHODS
%   finufft_plan - create guru plan object for one/many general nonuniform FFTs.
%   finufft_setpts  - process nonuniform points for general NUFFT transform(s).
%   finufft_exec - execute single or many-vector NUFFT transforms in a plan.
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
%     opts.spread_debug: spreader, (no text) 1 (some) or 2 (lots)
%     opts.spread_sort:  0 (don't sort NU pts), 1 (do), 2 (auto, default)
%     opts.spread_kerevalmeth:  0: exp(sqrt()), 1: Horner ppval (faster)
%     opts.spread_kerpad: (iff kerevalmeth=0)  0: don't pad to mult of 4, 1: do
%     opts.fftw: FFTW plan mode, 64=FFTW_ESTIMATE (default), 0=FFTW_MEASURE, etc
%     opts.upsampfac:   sigma.  2.0 (default), or 1.25 (low RAM, smaller FFT)
%     opts.spread_thread:   for ntrans>1 only. 0:auto, 1:seq multi, 2:par, etc
%     opts.maxbatchsize:  for ntrans>1 only. max blocking size, or 0 for auto.
%     opts.floatprec: library precision to use, 'double' (default) or 'single'.
%     for type 1 and 2 only, the following opts fields are also relevant:
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default)
% Outputs:
%     plan            finufft_plan object (opaque pointer)
%
% Notes:
%  * For type 1 and 2, this does the FFTW planning and kernel-FT precomputation.
%  * For type 3, this does very little, since the FFT sizes are not yet known.
%  * All available threads are planned; control how many with maxNumCompThreads.
%  * The vectorized (many vector) plan, ie ntrans>1, can be much faster
%    than repeated calls with the same nonuniform points. Note that here the I/O
%    data ordering is stacked rather than interleaved. See ../docs/matlab.rst
%  * For more details about the opts fields, see ../docs/opts.rst
%
%
% 2) FINUFFT_SETPTS   process nonuniform points for general NUFFT transform(s).
%
% finufft_setpts(plan, xj, yj, zj)
% finufft_setpts(plan, xj, yj, zj, s, t, u)
%  or
% plan.finufft_setpts(xj, yj, zj)
% plan.finufft_setpts(xj, yj, zj, s, t, u)
%
% Inputs nonuniform point coordinates (xj,yj,zj) in the case of all types, and
%  also nonuniform frequency target points (s,t,u) for type 3.
%  For all types, sorting is done to internally store a reindexing of points,
%  and for type 3 the spreading and FFTs are planned. The nonuniform points may
%  be used for multiple transforms.
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
%  * For type 1 and 2, the values in xj (and if nonempty, yj and zj) must
%    lie in the interval [-3pi,3pi]. For type 1 they are "sources", but for
%    type 2, "targets". In contrast, for type 3 there are no restrictions other
%    than the resulting size of the internal fine grids.
%  * s (and t and u) are only relevant for type 3, and may be omitted otherwise
%  * The matlab vectors xj,... and s,... should not be changed before calling
%    future finufft_exec calls, because the plan stores only pointers to the
%    arrays (they are not duplicated internally).
%  * If the data precision (double/single) does not match opts.floatprec used
%    in finufft_plan, warning is raised and copies used of the plan precision.
%
%
% 3) FINUFFT_EXEC   execute single or many-vector NUFFT transforms in a plan.
%
% result = finufft_exec(plan, data_in);
% or
% result = plan.finufft_exec(data_in); 
%
% Execute a single (or if ntrans>1 in the plan stage, multiple) NUFFT transforms
%  according to the previously defined plan, using the nonuniform points chosen
%  previously with finufft_setpts, and with the strengths or Fourier
%  coefficient inputs vector(s) from data_in, creating result, a new array of
%  the output vector(s).
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
%  * If the data precision (double/single) does not match opts.floatprec used
%    in finufft_plan, warning is raised and copies used of the plan precision.
%
%
% 4) To deallocate (delete) a nonuniform FFT plan, use delete(plan)
%
% This deallocates all stored FFTW plans, nonuniform point sorting arrays,
%  kernel Fourier transforms arrays, etc.
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
      plan.mwptr = p;     % crucial: save the opaque ptr (p.12 of MWrap manual)

      % replace in nufft_opts struct whichever fields are in incoming opts...
      mex_id_ = 'copy_nufft_opts(i mxArray, i nufft_opts*)';
finufft(mex_id_, opts, o);
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = finufft_makeplan(i int, i int, i int64_t[x], i int, i int, i double, i finufft_plan*, i nufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      else
        mex_id_ = 'o int = finufftf_makeplan(i int, i int, i int64_t[x], i int, i int, i float, i finufftf_plan*, i nufft_opts*)';
[ier] = finufft(mex_id_, type, dim, n_modes, iflag, n_trans, tol, plan, o, 3);
      end
      mex_id_ = 'delete(i nufft_opts*)';
finufft(mex_id_, o);
      errhandler(ier);             % convert C++ codes to matlab-style errors
    end

    function delete(plan)
    % This does clean-up (deallocation) of the C++ ptr before the obj deletes.
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'finufft_destroy(i finufft_plan*)';
finufft(mex_id_, plan);
      else
        mex_id_ = 'finufftf_destroy(i finufftf_plan*)';
finufft(mex_id_, plan);
      end
    end

    function finufft_setpts(plan, xj, yj, zj, s, t, u)
    % FINUFFT_SETPTS   process nonuniform points for general NUFFT transform(s).

                                                                              
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = get_dim(i finufft_plan*)';
[dim] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_type(i finufft_plan*)';
[type] = finufft(mex_id_, plan);
        if nargin<5, s=double.empty; t=double.empty; u=double.empty; end
      else
        mex_id_ = 'o int = get_dimf(i finufftf_plan*)';
[dim] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_typef(i finufftf_plan*)';
[type] = finufft(mex_id_, plan);
        if nargin<5, s=single.empty; t=single.empty; u=single.empty; end
      end
      % get number(s) of NU pts (also validates the NU pt array sizes)...
      [nj, nk] = valid_setpts(type, dim, xj, yj, zj, s, t, u);
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = finufft_setpts(i finufft_plan*, i int64_t, i double[], i double[], i double[], i int64_t, i double[], i double[], i double[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      else
        mex_id_ = 'o int = finufftf_setpts(i finufftf_plan*, i int64_t, i float[], i float[], i float[], i int64_t, i float[], i float[], i float[])';
[ier] = finufft(mex_id_, plan, nj, xj, yj, zj, nk, s, t, u);
      end
      errhandler(ier);
    end

    function result = finufft_exec(plan, data_in)
    % FINUFFT_EXEC   execute single or many-vector NUFFT transforms in a plan.

                                                                                                            
                                                                                                
      if strcmp(plan.floatprec,'double')
        mex_id_ = 'o int = get_type(i finufft_plan*)';
[type] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_ntrans(i finufft_plan*)';
[n_trans] = finufft(mex_id_, plan);
        mex_id_ = 'o int64_t = get_nj(i finufft_plan*)';
[nj] = finufft(mex_id_, plan);
      else
        mex_id_ = 'o int = get_typef(i finufftf_plan*)';
[type] = finufft(mex_id_, plan);
        mex_id_ = 'o int = get_ntransf(i finufftf_plan*)';
[n_trans] = finufft(mex_id_, plan);
        mex_id_ = 'o int64_t = get_njf(i finufftf_plan*)';
[nj] = finufft(mex_id_, plan);
      end
      % check data input length...
      if type==1 || type==2
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'get_nmodes(i finufft_plan*, o int64_t&, o int64_t&, o int64_t&)';
[ms, mt, mu] = finufft(mex_id_, plan);
        else
          mex_id_ = 'get_nmodesf(i finufftf_plan*, o int64_t&, o int64_t&, o int64_t&)';
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
          mex_id_ = 'o int = finufft_exec(i finufft_plan*, i dcomplex[], o dcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, ncoeffs);
        else
          mex_id_ = 'o int = finufftf_exec(i finufftf_plan*, i fcomplex[], o fcomplex[x])';
[ier, result] = finufft(mex_id_, plan, data_in, ncoeffs);
        end
        % make modes output correct shape; when d<3 squeeze removes unused dims...
        result = squeeze(reshape(result, [ms mt mu n_trans]));
      elseif type==2
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int = finufft_exec(i finufft_plan*, o dcomplex[xx], i dcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        else
          mex_id_ = 'o int = finufftf_exec(i finufftf_plan*, o fcomplex[xx], i fcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        end
      elseif type==3
        if strcmp(plan.floatprec,'double')
          mex_id_ = 'o int64_t = get_nk(i finufft_plan*)';
[nk] = finufft(mex_id_, plan);
          mex_id_ = 'o int = finufft_exec(i finufft_plan*, o dcomplex[xx], i dcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        else
          mex_id_ = 'o int64_t = get_nkf(i finufftf_plan*)';
[nk] = finufft(mex_id_, plan);
          mex_id_ = 'o int = finufftf_exec(i finufftf_plan*, o fcomplex[xx], i fcomplex[])';
[ier, result] = finufft(mex_id_, plan, data_in, nj, n_trans);
        end
      else
        ier = 10;         % type was corrupted since plan stage - the horror
      end
      errhandler(ier);
    end

  end
end
