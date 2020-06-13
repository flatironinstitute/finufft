% FINUFFT_PLAN   is a class which wraps the guru interface to FINUFFT.
%
%  Full documentation is given in ../finufft-manual.pdf and online at
%  http://finufft.readthedocs.io
%  Also see examples in the matlab/examples and matlab/test directories.
%
% PROPERTIES
%   mwptr - opaque pointer to a C++ finufft_plan object (see MWrap manual),
%           whose properties cannot be accessed directly
%
% METHODS
%   finufft_plan - create guru plan object for one/many general nonuniform FFTs.
%   finufft_setpts  - process nonuniform points for general NUFFT transform(s).
%   finufft_exec - execute single or many-vector NUFFT transforms in a plan.
%   finufft_destroy - deallocate (delete) a nonuniform FFT plan.
%
% General notes:
%  * See ERRHANDLER for all possible warning/error IDs, and ../docs/error.rst
%
%
%
% =========== Detailed description of methods =================================
%
% 1) FINUFFT_PLAN create guru plan object for one/many general nonuniform FFTs.
%
% plan = finufft_plan(type, n_modes_or_dim, iflag, ntrans, eps)
% plan = finufft_plan(type, n_modes_or_dim, iflag, ntrans, eps, opts)
%
% Creates a finufft_plan MATLAB object in the guru interface to FINUFFT, of
%  type 1, 2 or 3, and with given numbers of Fourier modes (unless type 3).
%
% Inputs: 
%     type            transform type, 1, 2, or 3
%     n_modes_or_dim  if type is 1 or 2, the number of Fourier modes in each
%                     dimension: [ms] in 1D, [ms mt] in 2D, or [ms mt mu] in 3D.
%                     Its length sets the dimension, which must be 1, 2 or 3.
%                     If type is 3, in contrast, its value equals the dimension.
%     iflag           if >=0, uses + sign in exponential, otherwise - sign
%     eps             precision requested (>1e-16)
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
%     for type 1 and 2 only, the following opts fields are also relevant:
%     opts.modeord: 0 (CMCL increasing mode ordering, default), 1 (FFT ordering)
%     opts.chkbnds: 0 (don't check NU points valid), 1 (do, default)
% Outputs:
%     plan            finufft_plan object
%
% Notes:
%  * For type 1 and 2, this does the FFTW planning and kernel-FT precomputation.
%  * For type 3, this does very little, since the FFT sizes are not yet known.
%  * All available threads are planned; control how many with maxNumCompThreads.
%  * For more details about the opts fields, see ../docs/opts.rst
%
%
% 2) FINUFFT_SETPTS   process nonuniform points for general NUFFT transform(s).
%
% finufft_setpts(plan, xj, yj, zj, s, t, u)
%  or
% plan.finufft_setpts(xj, yj, zj, s, t, u)
%
% Inputs nonuniform spatial points (xj,yj,zj) in the case of all types, and
%  also nonuniform frequency target points (s,t,u) for type 3. These points
%  are not duplicated, but are bin-sorted. For type 3 the spreading and
%  FFTs are planned. These nonuniform points may be used for multiple
%  transforms.
%
% Inputs:
%     xj     vector of x-coords of all nonuniform points
%     yj     empty (if dim<2), or vector of y-coords of all nonuniform points
%     zj     empty (if dim<3), or vector of z-coords of all nonuniform points
%     s      vector of x-coords of all nonuniform frequency targets
%     t      empty (if dim<2), or vector of y-coords of all NU freq targets
%     u      empty (if dim<3), or vector of z-coords of all NU freq targets
% Input/Outputs:
%     plan   finufft_plan object
%
% Notes:
%  * For type 1 and 2, the values in xj (and if nonempty, yj and zj) must
%    lie in the interval [-3pi,3pi]. For type 1 they are "sources", but for
%    type 2, "targets". In contrast, for type 3 there are no restrictions other
%    than the resulting size of the internal fine grids.
%  * s (and t and u) are only relevant for type 3, and may be omitted otherwise
% 
%
% 3) FINUFFT_EXEC   execute single or many-vector NUFFT transforms in a plan.
%
% result = finufft_exec(plan, data_in);
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
%              vector or matrix of appropriate size
% Outputs:
%     result   vector of output strengths at targets (types 2 or 3), or array
%              of Fourier coefficients (type 1), or, if ntrans>1, a stack of
%              such vectors or arrays.
%
% *** stuff about row vecs vs cols, and ntrans>1 case, for I and O.
%
%
% 4) FINUFFT_DESTROY   deallocate (delete) a nonuniform FFT plan.
%
% Usage: p.finufft_destroy; where p is a finufft_plan object.
%
% Note: since this is a handle class, one may instead clean up with: clear p;


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
      plan.mwptr = p;       % crucial: copies p.12 of mwrap doc; I don't get it
      n_modes = ones(3,1);  % dummy for type 3
      if type==3
        assert(length(n_modes_or_dim)==1);
        dim = n_modes_or_dim;      % interpret as dim
      else
        dim = length(n_modes_or_dim);
        n_modes(1:dim) = n_modes_or_dim;
      end
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
      errhandler(ier);
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
      nj = numel(xj);   % note the matlab way is to extract sizes like this
      nk = numel(s);
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
        ier = 10;        % something went horribly wrong with type
      end
      errhandler(ier);
    end

  end
end
