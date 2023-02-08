C interface (GPU)
=================

There are four steps needed to call cuFINUFFT from C: 1) making a plan, 2) setting the nonuniform points in this plan, 3) executing one or more transforms using the current nonuniform points, then, when all is done, 4) destroying the plan.
The simplest case is to call them in order, i.e., 1234.
However, it is possible to repeat 3 with new strength or coefficient data, or to repeat 2 to choose new nonuniform points in order to do one or more step 3's again, before destroying.
For instance, 123334 and 1232334 are allowed.
If non-standard algorithm options are desired, an extra function is needed before making the plan (see bottom of this page).

This API matches very closely that of the plan interface to FINUFFT (in turn modeled on those of FFTW and NFFT).
Here is the full documentation for these functions.
They have double (``cufinufft``) and single (``cufinufftf``) precision versions, which we document together.
The mathematical transforms are exactly as performed by FINUFFT, to which we refer for their definitions.

You will also want to read the examples in ``examples/cuda`` and ``test/cuda/cufinufft*.cu``, and may want to read extra documentation in the source at ``src/cuda/cufinufft.cu``.


Plan
----

Given the user's desired dimension, number of Fourier modes in each direction, sign in the exponential, number of transforms, tolerance, and desired batch size, and (possibly) an options struct, this creates a new plan object.

.. code-block:: c

    int cufinufft_makeplan(int type, int dim, int* nmodes, int iflag, int ntransf, double tol,
            int maxbatchsize, cufinufft_plan *plan, cufinufft_opts *opts)

    int cufinufftf_makeplan(int type, int dim, int* nmodes, int iflag, int ntransf, float tol,
            int maxbatchsize, cufinufftf_plan *plan, cufinufft_opts *opts)

    Inputs:

    type            type of the transform, 1 or 2 (note: 3 is not implemented yet)
    dim             overall dimension of the transform, 2 or 3 (note: 1 is not implemented
                    yet)
    nmodes          a length-dim integer array: nmodes[d] is the number of Fourier modes in
                    (zero-indexed) direction d. Specifically,
                    in 2D: nmodes[0]=N1, nmodes[1]=N2,
                    in 3D: nmodes[0]=N1, nmodes[1]=N2, nmodes[2]=N3.
    iflag           if >=0, uses + sign in complex exponential, otherwise - sign
    ntransf         number of transforms to performed in the execute stage (>=1). This
                    controls the number of input/output data expected for c and fk.
    tol             relative tolerance requested
                    (must be >1e-16 for double precision, >1e-8 for single precision)
    maxbatchsize    when ntransf>1, size of batch of data vectors to perform
                    cuFFT on. (default is 0, which chooses a heuristic). Ignored if
                    ntransf=1.
    opts            optional pointer to options-setting struct. If NULL, uses defaults.
                    See cufinufft_default_opts below for the non-NULL case.

    Input/Output:

    plan            a pointer to an instance of a cufinufft_plan (in double precision)
                    or cufinufftf_plan (in single precision).

    Returns:

    status          zero if success, otherwise an error occurred


Note: under the hood, in double precision, a ``cufinufft_plan`` object is simply a pointer to a ``cufinufft_plan_s`` struct (or in single precision, a ``cufinufftf_plan`` is a pointer to a ``cufinufftf_plan_s`` struct).
The struct contains the actual planning arrays, some of which live on the GPU.
This extra level of indirection leads to a simpler interface, and follows the approach of FFTW and FINUFFT.
See definitions in ``include/cufinufft_eitherprec.h``

Set nonuniform points
---------------------

This tells cuFINUFFT where to look for the coordinates of nonuniform points, and, if appropriate, creates an internal sorting index array to choose a good order to sweep through these points.
For type 1 these points are "sources", but for type 2, "targets".

.. code-block:: c

    int cufinufft_setpts(int M, double* x, double* y, double* z, int N, double* s,
        double* t, double *u, cufinufft_plan plan)

    int cufinufftf_setpts(int M, float* x, float* y, float* z, int N, float* s,
        float* t, float *u, cufinufftf_plan plan)

    Input:

    M           number of nonuniform points
    x, y, z     length-M GPU arrays of x,y (in 2D) or x,y,z (in 3D) coordinates of
                nonuniform points. In each dimension they refer to a periodic domain
                [-pi,pi), but values out to [-3pi, 3pi) will be folded back correctly
                into this domain. Beyond that, they will not, and may result in crash.
                In dimension 2, z is ignored.
    N, s, t, u  (unused for types 1 or 2 transforms; reserved for future type 3)

    Input/Output:

    plan        the cufinufft plan object from the above plan stage

    Returns:

    status      zero if success, otherwise an error occurred

Note: The user must not change the contents of the GPU arrays ``x``, ``y``, or ``z`` (in 3D case) between this step and the below execution step. They are read in the execution step also.

Note: The actual plan (not its pointer is passed in); new GPU arrays are allocated and filled in the internal plan struct that the plan points to.

Execute
-------

This reads the strength (for type 1) or coefficient (for type 2) data and carries out one or more transforms (as specified in the plan stage), using the current nonuniform points chosen in the previous step.
Multiple transforms use the same set of nonuniform points.
The result is written into whichever array was not the input (the roles of these two swap for type 1 vs type 2 transforms).

.. code-block:: c

    int cufinufft_execute(cuDoubleComplex* c, cuDoubleComplex* f, cufinufft_plan plan)

    int cufinufftf_execute(cuFloatComplex* c, cuFloatComplex* f, cufinufftf_plan plan)

    Input/Output:

    c        If type 1, the input strengths at the nonuniform point sources
             (size M*ntransf complex array).
             If type 2, the output values at the nonuniform point targets
             (size M*ntransf complex array).
    f        If type 1, the output Fourier mode coefficients (size N1*N2*ntransf
             or N1*N2*N3*ntransf complex array, when dim = 2 or 3 respectively).
             If type 2, the input Fourier mode coefficients (size N1*N2*ntransf
             or N1*N2*N3*ntransf complex array, when dim = 2 or 3 respectively).
    plan     the cufinufft plan object

    Returns:

    status   zero if success, otherwise an error occurred

Note: The contents of the arrays ``x``, ``y``, and ``z`` (if relevant) must not have changed since the ``cufinufft_setpts`` call that read them.
The execution rereads them (this way of doing business saves RAM).

Note: ``f`` and ``c`` are contiguous Fortran-style (row-major) arrays with the transform number being the "slowest" (outer) dimension, if ``ntransf>1``. For the ``f`` array, ``x`` is "fastest", then ``y``, then (if relevant) ``z`` is "slowest".

Destroy
-------

.. code-block:: c

    int cufinufft_destroy(cufinufft_plan plan)

    int cufinufftf_destroy(cufinufftf_plan plan)

    Input:

    plan     the cufinufft plan object

    Returns:

    status   zero if success, otherwise an error occurred

This deallocates all arrays inside the ``plan`` struct, freeing all internal memory used in the above three stages.
Note: the plan (being just a pointer to the plan struct) is not actually "destroyed"; rather, its internal struct is destroyed.
There is no need for further deallocation of the plan.

Non-standard options
--------------------

The last argument in the above plan stage accepts a pointer to an options structure, which is the same in both single and double precision.
To create such a structure, use:

.. code-block:: c

    cufinufft_opts opts;
    ier = cufinufft_default_opts(type, dim, &opts);

where ``type`` and ``dim`` are as above in ``cufinufft_makeplan``.
As before, the returned value is zero if success, otherwise an error occurred.
Then you may change fields of ``opts`` by hand, finally pass ``&opts`` in as the last argument to ``cufinufft_makeplan`` or ``cufinufftf_makeplan``.
The options fields are currently only documented in the ``include/cufinufft_opts.h``.

For examples of this advanced usage, see ``test/cuda/cufinufft*.cu```
