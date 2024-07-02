C interface (GPU)
=================

There are four steps needed to call cuFINUFFT from C: 1) making a plan, 2) setting the nonuniform points in this plan, 3) executing one or more transforms using the current nonuniform points, then, when all is done, 4) destroying the plan.
The simplest case is to call them in order, i.e., 1234.
However, it is possible to repeat 3 with new strength or coefficient data, or to repeat 2 to choose new nonuniform points in order to do one or more step 3's again, before destroying.
For instance, 123334 and 1232334 are allowed.
If non-standard algorithm options are desired, an extra function is needed before making the plan; see bottom of this page for options.

This API matches very closely that of the plan interface to FINUFFT (in turn modeled on those of FFTW and NFFT).
Here is the full documentation for these functions.
They have double (``cufinufft``) and single (``cufinufftf``) precision versions, which we document together.
The mathematical transforms are exactly as performed by FINUFFT, to which we refer for their definitions.

You will also want to read the examples in ``examples/cuda`` and ``test/cuda/cufinufft*.cu``, and may want to read extra documentation in the source at ``src/cuda/cufinufft.cu``.

*Note*: The interface to cuFINUFFT has changed between versions 1.3 and 2.2.
Please see :ref:`Migration to cuFINUFFT v2.2<cufinufft_migration>` for details.

Getting started
---------------

Let us consider applying a 1D type-1 transform in single precision.
First we need to include some headers

.. code-block:: c

    #include <cufinufft.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <math.h>
    #include <complex.h>
    #include <cuComplex.h>

Inside our ``main`` function, we first define the problem parameters and some pointers for the data arrays:

.. code-block:: c

    const int M = 100000, N = 10000;

    int64_t modes[1] = {N};

    float *x;
    float _Complex *c;
    float _Complex *f;

Here ``M`` is the number of nonuniform points, ``N`` is the size of our 1D grid. cuFINUFFT expets the grid size to be given as an array, so we also define the ``modes`` integer array. Finally we have the nonuniform points ``x``, the coefficients (or strengths) ``c`` and the output values ``f`` on the grid.

Note that we are using the ``float _Complex`` type here to remain compatible with both C and C++.
For the former, we can use ``float complex`` while the latter would accept ``std::complex<float>`` instead.

We also define the corresponding data pointers on the device (GPU) as well as the cuFINUFFT plan:

.. code-block:: c

    float *d_x;
    cuFloatComplex *d_c, *d_f;

    cufinufftf_plan plan;

Finally, we'll need some variables to compute the NUDFT at some arbitrary point to check the accuracy of the cuFINUFFT call:

.. code-block:: c

    int idx;
    float _Complex f0;

Now the actual work can begin. First, we allocate the host (CPU) arrays and fill the ``x`` and ``c`` arrays with appropriate values (``f`` will hold the output of the cuFINUFFT call). The frequencies in ``x`` are interpreted with periodicity :math:`2\pi`, while the coefficients ``c`` can be any value. Here we draw the frequencies and coefficients from the uniform distributions on :math:`[-\pi, \pi]` and :math:`[-1, 1]^2` respectively.

.. code-block:: c

    x = (float *) malloc(M * sizeof(float));
    c = (float _Complex *) malloc(M * sizeof(float _Complex));
    f = (float _Complex *) malloc(N * sizeof(float _Complex));

    srand(0);

    for(int j = 0; j < M; ++j) {
        x[j] = 2 * M_PI * (((float) rand()) / RAND_MAX - 1);
        c[j] = (2 * ((float) rand()) / RAND_MAX - 1)
               + I * (2 * ((float) rand()) / RAND_MAX - 1);
    }

Now that the data is generated, we must transfer it to the device. For this, we first allocate the necessary arrays using ``cudaMalloc`` and then transfer the data using ``cudaMemcpy``.

.. code-block:: c

    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_c, M * sizeof(float _Complex));
    cudaMalloc(&d_f, N * sizeof(float _Complex));

    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, M * sizeof(float _Complex), cudaMemcpyHostToDevice);

It's finally time to put cuFINUFFT to work. First, we create a plan using ``cufinufftf_makeplan`` (the prefix ``cufinufftf_`` is replaced with ``cufinufft_`` when working in double precision).

.. code-block:: c

    cufinufftf_makeplan(1, 1, modes, 1, 1, 1e-6, &plan, NULL);

The first argument gives the type, while the second gives the number of dimensions. After this, we have the grid size as an integer array, followed by the sign in the complex exponential (here positive) and the number of transforms to compute simultaneously (here just one). Then there's the tolerance (six digits) and finally there's a pointer to the plan and a non-mandatory options structure.

Once the plan is created, we set the points and execute the plan.

.. code-block:: c

    cufinufftf_setpts(plan, M, d_x, NULL, NULL, 0, NULL, NULL, NULL);

    cufinufftf_execute(plan, d_c, d_f);

Once the results are calculated, we transfer the data back onto the host, destroy the plan, and free the device arrays.

.. code-block:: c

    cudaMemcpy(f, d_f, N * sizeof(float _Complex), cudaMemcpyDeviceToHost);

    cufinufftf_destroy(plan);

    cudaFree(d_x);
    cudaFree(d_c);
    cudaFree(d_f);

The result is now in the host array ``f`` and we can print out its value at a particular index.

.. code-block:: c

    idx = 4 * N / 7;

    printf("f[%d] = %lf + %lfi\n", idx, crealf(f[idx]), cimagf(f[idx]));

If we want, we can complare this to the value obtained using the type-1 NUDFT formula.

.. code-block:: c

    f0 = 0;

    for(int j = 0; j < M; ++j) {
        f0 += c[j] * cexp(I * x[j] * (idx - N / 2));
    }

    printf("f0[%d] = %lf + %lfi\n", idx, crealf(f0), cimagf(f0));

Finally, we'll want to deallocate the arrays once we're done with them.

.. code-block:: c

    free(x);
    free(c);
    free(f);

The complete listing can be found in ``examples/cuda/getting_started.cpp``.

Full documentation
------------------

Plan
~~~~

Given the user's desired dimension, number of Fourier modes in each direction, sign in the exponential, number of transforms, tolerance, and desired batch size, and (possibly) an options struct, this creates a new plan object.

.. code-block:: c

    int cufinufft_makeplan(int type, int dim, int64_t* nmodes, int iflag,
            int ntr, double eps, cufinufft_plan *plan, cufinufft_opts *opts)

    int cufinufftf_makeplan(int type, int dim, int64_t* nmodes, int iflag,
            int ntr, float tol, cufinufftf_plan *plan, cufinufft_opts *opts)

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
See definitions in ``include/cufinufft.h``

Set nonuniform points
~~~~~~~~~~~~~~~~~~~~~

This tells cuFINUFFT where to look for the coordinates of nonuniform points, and, if appropriate, creates an internal sorting index array to choose a good order to sweep through these points.
For type 1 these points are "sources", but for type 2, "targets".

.. code-block:: c

    int cufinufft_setpts(cufinufft_plan plan, int M, double* x, double* y,
            double* z, int N, double* s, double* t, double *u)

    int cufinufftf_setpts(cufinufftf_plan plan, int M, float* x, float* y,
            float* z, int N, float* s, float* t, float *u)

    Input:

    M           number of nonuniform points
    x, y, z     length-M GPU arrays of x (in 1D), x, y (in 2D), or x, y, z (in 3D) coordinates of
                nonuniform points. In each dimension they refer to a periodic domain
                [-pi,pi), but values outside will be folded back correctly
                into this domain. In dimension 1, y and z are ignored. In dimension 2, z is
                ignored.
    N, s, t, u  (unused for types 1 or 2 transforms; reserved for future type 3)

    Input/Output:

    plan        the plan object from the above plan stage

    Returns:

    status      zero if success, otherwise an error occurred

Note: The user must not change the contents of the GPU arrays ``x``, ``y``, or ``z`` between this step and the below execution step. They are read in the execution step also.

Note: Here we pass in the actual plan handle, not its pointer as in ``cufinufft_makeplan``. The same goes for the other functions below.

Execute
~~~~~~~

This reads the strength (for type 1) or coefficient (for type 2) data and carries out one or more transforms (as specified in the plan stage), using the current nonuniform points chosen in the previous step.
Multiple transforms use the same set of nonuniform points.
The result is written into whichever array was not the input (the roles of these two swap for type 1 vs type 2 transforms).

.. code-block:: c

    int cufinufft_execute(cufinufft_plan plan, cuDoubleComplex* c, cuDoubleComplex* f)

    int cufinufftf_execute(cufinufftf_plan plan, cuFloatComplex* c, cuFloatComplex* f)

    Input/Output:

    plan     the plan object
    c        If type 1, the input strengths at the nonuniform point sources
             (size M*ntransf complex array).
             If type 2, the output values at the nonuniform point targets
             (size M*ntransf complex array).
    f        If type 1, the output Fourier mode coefficients (size N1*N2*ntransf
             or N1*N2*N3*ntransf complex array, when dim = 2 or 3 respectively).
             If type 2, the input Fourier mode coefficients (size N1*N2*ntransf
             or N1*N2*N3*ntransf complex array, when dim = 2 or 3 respectively).

    Returns:

    status   zero if success, otherwise an error occurred

Note: The contents of the arrays ``x``, ``y``, and ``z`` must not have changed since the ``cufinufft_setpts`` call that read them.
The execution rereads them (this way of doing business saves RAM).

Note: ``f`` and ``c`` are contiguous Fortran-style (row-major) arrays with the transform number being the "slowest" (outer) dimension, if ``ntr>1``. For the ``f`` array, ``x`` is "fastest", then ``y``, then (if relevant) ``z`` is "slowest".

Destroy
~~~~~~~

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

Options for GPU code
--------------------

The last argument in the above plan stage accepts a pointer to an options structure, which is the same in both single and double precision.
To create such a structure, use:

.. code-block:: c

    cufinufft_opts opts;
    cufinufft_default_opts(&opts);

Then you may change fields of ``opts`` by hand, finally pass ``&opts`` in as the last argument to ``cufinufft_makeplan`` or ``cufinufftf_makeplan``. Here are the options, with the important user-controllable ones documented. For their default values, see below.

Data handling options
~~~~~~~~~~~~~~~~~~~~~

**modeord**: Fourier coefficient frequency index ordering; see the CPU option of the same name :ref:`modeord<modeord>`.
As a reminder, ``modeord=0`` selects increasing frequencies (negative through positive) in each dimension,
while ``modeord=1`` selects FFT-style ordering starting at zero and wrapping over to negative frequencies half way through.

**gpu_device_id**: Sets the GPU device ID. Leave at default unless you know what you're doing. [To be documented]

Diagnostic options
~~~~~~~~~~~~~~~~~~

**gpu_spreadinterponly**: if ``0`` do the NUFFT as intended. If ``1``, omit the FFT and kernel FT deconvolution steps and return garbage answers.
Nonzero value is *only* to be used to aid timing tests (although currently there are no timing codes that exploit this option), and will give wrong or undefined answers for the NUFFT transforms!


Algorithm performance options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**gpu_method**: Spreader/interpolator algorithm.

* ``gpu_method=0`` : makes an automatic choice of one of the below methods, based on our heuristics.

* ``gpu_method=1`` : uses a nonuniform points-driven method, either unsorted which is referred to as GM in our paper, or sorted which is called GM-sort in our paper, depending on option ``gpu_sort`` below
  
* ``gpu_method=2`` : for spreading only, ie, type 1 transforms, uses a shared memory output-block driven method, referred to as SM in our paper. Has no effect for interpolation (type 2 transforms)

* ``gpu_method>2`` : (various upsupported experimental methods due to Melody Shih, not for regular users. Eg ``3`` tests an idea of Paul Springer's to group NU points when spreading, ``4`` is a block gather method of possible interest.)

**gpu_sort**: ``0`` do not sort nonuniform points, ``1`` do sort nonuniform points. Only has an effect when ``gpu_method=1`` (or if this method has been internally chosen when ``gpu_method=0``). Unlike the CPU code, there is no auto-choice since in our experience sorting is fast and always helps. It is possible for structured NU point inputs that ``gpu_sort=0`` may be the faster.

**gpu_kerevalmeth**: ``0`` use direct (reference) kernel evaluation, which is not recommended for speed (however, it allows nonstandard ``opts.upsampfac`` to be used). ``1`` use Horner piecewise polynomial evaluation (recommended, and enforces ``upsampfac=2.0``)

**upsampfac**: set upsampling factor. For the recommended ``kerevalmeth=1`` you must choose the standard ``upsampfac=2.0``. If you are willing to risk a slower kernel evaluation, you may set any ``upsampfac>1.0``, but this is experimental and unsupported.

**gpu_maxsubprobsize**: maximum number of NU points to be handled in a single subproblem in the spreading SM method (``gpu_method=2`` only)

**gpu_{o}binsize{x,y,z}**: various bisizes for sorting (GM-sort) or SM subproblem methods. Values of ``-1`` trigger the heuristically set default values. Leave at default unless you know what you're doing. [To be documented]

**gpu_maxbatchsize**: ``0`` use heuristically defined batch size for vectorized (many-transforms with same NU points) interface, else set this batch size.

**gpu_stream**: CUDA stream to use. Leave at default unless you know what you're doing. [To be documented]


For all GPU option default values we refer to the source code in
``src/cuda/cufinufft.cu:cufinufft_default_opts``):

.. literalinclude:: ../src/cuda/cufinufft.cu
   :start-after: @gpu_defopts_start
   :end-before: @gpu_defopts_end

For examples of advanced options-switching usage, see ``test/cuda/cufinufft*.cu`` and ``perftest/cuda/cuperftest.cu``.

You may notice a lack of debugging/timing options in the GPU code. This is to avoid CUDA writing to stdout. Please help us out by adding some of these.
