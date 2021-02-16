# cuFINUFFT documentation for C++/CUDA usage

There are four steps needed to call cuFINUFFT from C++:
1) making a plan, 2) setting the nonuniform points in this plan, 3) executing one or more transforms using the current nonuniform points,
then when all is done 4) destroying the plan.
The simplest case is to call them in order, ie, 1234. However, it is possible to repeat 3 with
new strength/coefficient data, or to repeat 2 to choose new nonuniform points
in order to do one or more step 3's again,
before destroying. For instance, 123334 and 1232334 are allowed.
If non-standard algorithm options are desired, an extra function is needed
before making the plan (see bottom of this page).

This API matches
very closely that of the plan interface to FINUFFT (in turn modeled on those of,
eg, FFTW or NFFT). 
Here is the full documentation for these functions. They have double
(`cufinufft`)
and single (`cufinufftf`) precision versions, which we document together.

You will also want to read the examples in `examples/` and `test/cufinufft*.cu`,
and may want to read extra documentation in the source
at [`src/cufinufft.cu`](https://github.com/flatironinstitute/cufinufft/blob/master/src/cufinufft.cu)


## 1. PLAN

Given the user's desired dimension, number of Fourier modes in each direction,
sign in the exponential, number of transforms, tolerance, and desired batch size,
and (possibly) an options struct, this creates a new plan object.

```c++
int cufinufft_makeplan(int type, int dim, int* nmodes, int iflag, int ntransf, double tol,
        int maxbatchsize, cufinufft_plan *plan, cufinufft_opts *opts)
int cufinufftf_makeplan(int type, int dim, int* nmodes, int iflag, int ntransf, single tol,
        int maxbatchsize, cufinufftf_plan *plan, cufinufft_opts *opts)
```
    
```
	Input:
        
	type    type of the transform, 1 or 2 (note 3 is not implemented yet)
	dim     overall dimension of the transform, 2 or 3 (note 1 is not implemented
                yet)
	nmodes  a length-3 integer array: nmodes[d] is the number of modes in
	        each zero-indexed direction, ie, d=0,1 (2D case), or 0,1,2 (3D case)
	iflag   if >=0, uses + sign in complex exponential, otherwise - sign
        ntransf number of transforms to performed in the execute stage (>=1). This
                controls the amount of input/output data expected for c and fk.
	tol     relative tolerance requested
                (must be >1e-16 for double precision, >1e-8 for single precision)
	maxbatchsize	when ntransf>1, size of batch of data vectors to perform
			cuFFT on. (default is 0, which chooses a heuristic). Ignored
                        if ntransf=1.
	opts	optional pointer to options-setting struct. If NULL, uses defaults.
		See cufinufft_default_opts below for the non-NULL case.

	Input/Output:
        
	plan    a pointer to an instance of a cufinufft_plan (in double precision)
                or cufinufftf_plan (in single precision).

        Returned value:
        
        status flag, 0 if success, otherwise an error occurred
```

Note: under the hood, a plan is simply a pointer to a `cufinufft_plan_s` struct (in double
precision, or `cufinufftf_plan_s` struct in single). This struct contains the actual
planning arrays. This extra level of indirection leads to a simpler interface, and
follows the approach of FFTW and FINUFFT.
See definitions in `include/cufinufft_eitherprec.h`



## 2. SET NONUNIFORM POINTS

This tells cuFINUFFT where to look for the coordinates of nonuniform points, and,
if appropriate, creates an internal sorting index array to choose a good order to sweep
through these points.

```c++
int cufinufft_setpts(int M, double* x, double* y, double* z, int N, double* s,
	double* t, double *u, cufinufft_plan plan)
int cufinufftf_setpts(int M, single* x, single* y, single* z, int N, single* s,
	single* t, single *u, cufinufftf_plan plan)
```

```
	Input:
        
	M        number of nonuniform points
	x, y, z  length-M GPU arrays of x,y (in 2D) or x,y,z (in 3D) coordinates of
                 nonuniform points. In each dimension they refer to a periodic domain
                 [-pi,pi), but values out to [-3pi, 3pi) will be folded back correctly
                 into this domain. Beyond that, they will not, and may result in crash.
                 In dimension 2, z is ignored.
	N, s, t, u  (unused for types 1 or 2 transforms; reserved for future type 3)

	Input/Output:
        
        plan     cufinufft_plan object (if double prec), or cufinufftf_plan (if single)

        Returned value:
        
        status flag, 0 if success, otherwise an error occurred
```

Note: The user must not change the contents of the GPU arrays `x`, `y`, or `z` (in 3D case)
between this step and the below execution step. They are needed in the execution step also.

Note: new GPU arrays are allocated and filled in the internal
plan struct that the plan points to.



## 3. EXECUTE ONE OR MORE TRANSFORMS

This reads the strength or coefficient data and carries out one or more transforms
(as determined by `ntransf` in the plan stage), using the current nonuniform points inputted
in the previous step.



