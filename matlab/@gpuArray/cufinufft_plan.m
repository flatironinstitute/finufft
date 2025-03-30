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
%     opts.gpu_maxsubproblemsize:  max # NU pts per subprob (gpu_method=2 only).
%     opts.gpu_binsize{x,y,z}:  various binsizes in GM-sort/SM (for experts).
%     opts.gpu_maxbatchsize:   0 (auto, default), or many-vector batch size. 
%     opts.gpu_stream:  CUDA stream, allows H2D/D2H while computing (experts).
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
%    than repeated calls with the same nonuniform points. Note that here the I/O
%    data ordering is stacked not interleaved. See ../docs/matlab_gpu.rst
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

