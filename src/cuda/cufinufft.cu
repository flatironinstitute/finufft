#include <complex>
#include <cufft.h>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <math.h>

#include <cufinufft.h>

#include <cufinufft/common.h>
#include <cufinufft/cudeconvolve.h>
#include <cufinufft/defs.h>
#include <cufinufft/impl.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

using namespace cufinufft::common;
using namespace cufinufft::memtransfer;
using namespace cufinufft::spreadinterp;
using namespace cufinufft::utils;
using std::min;

void SETUP_BINSIZE(int type, int dim, cufinufft_opts *opts) {
    switch (dim) {
    case 1: {
        opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 1024 : opts->gpu_binsizex;
        opts->gpu_binsizey = 1;
        opts->gpu_binsizez = 1;
    } break;
    case 2: {
        opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 32 : opts->gpu_binsizex;
        opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 32 : opts->gpu_binsizey;
        opts->gpu_binsizez = 1;
    } break;
    case 3: {
        switch (opts->gpu_method) {
        case 1:
        case 2: {
            opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 16 : opts->gpu_binsizex;
            opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 16 : opts->gpu_binsizey;
            opts->gpu_binsizez = (opts->gpu_binsizez < 0) ? 2 : opts->gpu_binsizez;
        } break;
        case 4: {
            opts->gpu_obinsizex = (opts->gpu_obinsizex < 0) ? 8 : opts->gpu_obinsizex;
            opts->gpu_obinsizey = (opts->gpu_obinsizey < 0) ? 8 : opts->gpu_obinsizey;
            opts->gpu_obinsizez = (opts->gpu_obinsizez < 0) ? 8 : opts->gpu_obinsizez;
            opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 4 : opts->gpu_binsizex;
            opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 4 : opts->gpu_binsizey;
            opts->gpu_binsizez = (opts->gpu_binsizez < 0) ? 4 : opts->gpu_binsizez;
        } break;
        }
    } break;
    }
}

template <typename T>
int cufinufft_makeplan_impl(int type, int dim, int *nmodes, int iflag, int ntransf, T tol, int maxbatchsize,
                            cufinufft_plan_t<T> **d_plan_ptr, cufinufft_opts *opts) {
    /*
        "plan" stage (in single or double precision).
            See ../docs/cppdoc.md for main user-facing documentation.
            Note that *d_plan_ptr in the args list was called simply *plan there.
            This is the remaining dev-facing doc:

    This performs:
            (0) creating a new plan struct (d_plan), a pointer to which is passed
                back by writing that pointer into *d_plan_ptr.
            (1) set up the spread option, d_plan.spopts.
            (2) calculate the correction factor on cpu, copy the value from cpu to
                gpu
            (3) allocate gpu arrays with size determined by number of fourier modes
                and method related options that had been set in d_plan.opts
            (4) call cufftPlanMany and save the cufft plan inside cufinufft plan
            Variables and arrays inside the plan struct are set and allocated.

        Melody Shih 07/25/19. Use-facing moved to markdown, Barnett 2/16/21.
    */
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    if (opts == NULL) {
        // options might not be supplied to this function => assume device
        // 0 by default
        cudaSetDevice(0);
    } else {
        cudaSetDevice(opts->gpu_device_id);
    }

    int ier;

    /* allocate the plan structure, assign address to user pointer. */
    cufinufft_plan_t<T> *d_plan = new cufinufft_plan_t<T>;
    *d_plan_ptr = d_plan;
    // Zero out your struct, (sets all pointers to NULL)
    memset(d_plan, 0, sizeof(*d_plan));

    /* If a user has not supplied their own options, assign defaults for them. */
    if (opts == NULL) { // use default opts
        ier = cufinufft_default_opts(type, dim, &(d_plan->opts));
        if (ier != 0) {
            printf("error: cufinufft_default_opts returned error %d.\n", ier);
            return ier;
        }
    } else {                  // or read from what's passed in
        d_plan->opts = *opts; // keep a deep copy; changing *opts now has no effect
    }

    /* Setup Spreader */
    ier = setup_spreader_for_nufft(d_plan->spopts, tol, d_plan->opts);
    if (ier > 1) // proceed if success or warning
        return ier;

    d_plan->dim = dim;
    d_plan->ms = nmodes[0];
    d_plan->mt = nmodes[1];
    d_plan->mu = nmodes[2];

    SETUP_BINSIZE(type, dim, &d_plan->opts);
    CUFINUFFT_BIGINT nf1 = 1, nf2 = 1, nf3 = 1;
    set_nf_type12(d_plan->ms, d_plan->opts, d_plan->spopts, &nf1, d_plan->opts.gpu_obinsizex);
    if (dim > 1)
        set_nf_type12(d_plan->mt, d_plan->opts, d_plan->spopts, &nf2, d_plan->opts.gpu_obinsizey);
    if (dim > 2)
        set_nf_type12(d_plan->mu, d_plan->opts, d_plan->spopts, &nf3, d_plan->opts.gpu_obinsizez);
    int fftsign = (iflag >= 0) ? 1 : -1;

    d_plan->nf1 = nf1;
    d_plan->nf2 = nf2;
    d_plan->nf3 = nf3;
    d_plan->iflag = fftsign;
    d_plan->ntransf = ntransf;
    if (maxbatchsize == 0)              // implies: use a heuristic.
        maxbatchsize = min(ntransf, 8); // heuristic from test codes
    d_plan->maxbatchsize = maxbatchsize;
    d_plan->type = type;

    if (d_plan->type == 1)
        d_plan->spopts.spread_direction = 1;
    if (d_plan->type == 2)
        d_plan->spopts.spread_direction = 2;

    switch (d_plan->dim) {
    case 1: {
        ier = allocgpumem1d_plan<T>(d_plan);
    } break;
    case 2: {
        ier = allocgpumem2d_plan<T>(d_plan);
    } break;
    case 3: {
        ier = allocgpumem3d_plan<T>(d_plan);
    } break;
    }

    cufftHandle fftplan;
    switch (d_plan->dim) {
    case 1: {
        int n[] = {(int)nf1};
        int inembed[] = {(int)nf1};

        cufftPlanMany(&fftplan, 1, n, inembed, 1, inembed[0], inembed, 1, inembed[0], cufft_type<T>(), maxbatchsize);
    } break;
    case 2: {
        int n[] = {(int)nf2, (int)nf1};
        int inembed[] = {(int)nf2, (int)nf1};

        cufftPlanMany(&fftplan, 2, n, inembed, 1, inembed[0] * inembed[1], inembed, 1, inembed[0] * inembed[1],
                      cufft_type<T>(), maxbatchsize);
    } break;
    case 3: {
        int n[] = {(int)nf3, (int)nf2, (int)nf1};
        int inembed[] = {(int)nf3, (int)nf2, (int)nf1};

        cufftPlanMany(&fftplan, 3, n, inembed, 1, inembed[0] * inembed[1] * inembed[2], inembed, 1,
                      inembed[0] * inembed[1] * inembed[2], cufft_type<T>(), maxbatchsize);
    } break;
    }
    d_plan->fftplan = fftplan;

    std::complex<double> a[3 * MAX_NQUAD];
    T f[3 * MAX_NQUAD];
    onedim_fseries_kernel_precomp(nf1, f, a, d_plan->spopts);
    if (dim > 1) {
        onedim_fseries_kernel_precomp(nf2, f + MAX_NQUAD, a + MAX_NQUAD, d_plan->spopts);
    }
    if (dim > 2) {
        onedim_fseries_kernel_precomp(nf3, f + 2 * MAX_NQUAD, a + 2 * MAX_NQUAD, d_plan->spopts);
    }

    cuDoubleComplex *d_a;
    T *d_f;
    checkCudaErrors(cudaMalloc(&d_a, dim * MAX_NQUAD * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_f, dim * MAX_NQUAD * sizeof(T)));
    checkCudaErrors(cudaMemcpy(d_a, a, dim * MAX_NQUAD * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_f, f, dim * MAX_NQUAD * sizeof(T), cudaMemcpyHostToDevice));
    ier = cufserieskernelcompute(d_plan->dim, nf1, nf2, nf3, d_f, d_a, d_plan->fwkerhalf1, d_plan->fwkerhalf2,
                                 d_plan->fwkerhalf3, d_plan->spopts.nspread);

    cudaFree(d_a);
    cudaFree(d_f);
    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);

    return ier;
}

template <typename T>
int cufinufft_setpts_impl(int M, T *d_kx, T *d_ky, T *d_kz, int N, T *d_s, T *d_t, T *d_u, cufinufft_plan_t<T> *d_plan)
/*
    "setNUpts" stage (in single or double precision).

    In this stage, we
        (1) set the number and locations of nonuniform points
        (2) allocate gpu arrays with size determined by number of nupts
        (3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled
            coordinates are stored)
        (4) determine the spread/interp properties that only relates to the
            locations of nupts (see 2d/spread2d_wrapper.cu,
            3d/spread3d_wrapper.cu for what have been done in
            function spread<dim>d_<method>_prop() )

        See ../docs/cppdoc.md for main user-facing documentation.
        Here is the old developer docs, which are useful only to translate
        the argument names from the user-facing ones:

    Input:
    M                 number of nonuniform points
    d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M
                      T array) in [-pi, pi). set h_kz to "NULL" if dimension
                      is less than 3. same for h_ky for dimension 1.
    N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

    Input/Output:
    d_plan            pointer to a CUFINUFFT_PLAN_S. Variables and arrays inside
                      the plan are set and allocated.

        Returned value:
        a status flag: 0 if success, otherwise an error occurred

Notes: the type T means either single or double, matching the
    precision of the library version called.

    Melody Shih 07/25/19; Barnett 2/16/21 moved out docs.
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int nf3 = d_plan->nf3;
    int dim = d_plan->dim;

    d_plan->M = M;

    int ier;
    switch (d_plan->dim) {
    case 1: {
        ier = allocgpumem1d_nupts<T>(d_plan);
    } break;
    case 2: {
        ier = allocgpumem2d_nupts<T>(d_plan);
    } break;
    case 3: {
        ier = allocgpumem3d_nupts<T>(d_plan);
    } break;
    }

    d_plan->kx = d_kx;
    if (dim > 1)
        d_plan->ky = d_ky;
    if (dim > 2)
        d_plan->kz = d_kz;

    switch (d_plan->dim) {
    case 1: {
        if (d_plan->opts.gpu_method == 1) {
            ier = cuspread1d_nuptsdriven_prop<T>(nf1, M, d_plan);
            if (ier != 0) {
                printf("error: cuspread1d_nupts_prop, method(%d)\n", d_plan->opts.gpu_method);

                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);

                return 1;
            }
        }
        if (d_plan->opts.gpu_method == 2) {
            ier = cuspread1d_subprob_prop<T>(nf1, M, d_plan);
            if (ier != 0) {
                printf("error: cuspread1d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);

                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);

                return 1;
            }
        }
    } break;
    case 2: {
        if (d_plan->opts.gpu_method == 1) {
            ier = cuspread2d_nuptsdriven_prop<T>(nf1, nf2, M, d_plan);
            if (ier != 0) {
                printf("error: cuspread2d_nupts_prop, method(%d)\n", d_plan->opts.gpu_method);

                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);

                return 1;
            }
        }
        if (d_plan->opts.gpu_method == 2) {
            ier = cuspread2d_subprob_prop<T>(nf1, nf2, M, d_plan);
            if (ier != 0) {
                printf("error: cuspread2d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);

                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);

                return 1;
            }
        }
    } break;
    case 3: {
        if (d_plan->opts.gpu_method == 4) {
            int ier = cuspread3d_blockgather_prop<T>(nf1, nf2, nf3, M, d_plan);
            if (ier != 0) {
                printf("error: cuspread3d_blockgather_prop, method(%d)\n", d_plan->opts.gpu_method);

                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);

                return ier;
            }
        }
        if (d_plan->opts.gpu_method == 1) {
            ier = cuspread3d_nuptsdriven_prop<T>(nf1, nf2, nf3, M, d_plan);
            if (ier != 0) {
                printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n", d_plan->opts.gpu_method);

                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);

                return ier;
            }
        }
        if (d_plan->opts.gpu_method == 2) {
            int ier = cuspread3d_subprob_prop<T>(nf1, nf2, nf3, M, d_plan);
            if (ier != 0) {
                printf("error: cuspread3d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);

                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);

                return ier;
            }
        }
    } break;
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);

    return 0;
}

template <typename T>
int cufinufft_execute_impl(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan)
/*
    "exec" stage (single and double precision versions).

    The actual transformation is done here. Type and dimension of the
    transformation are defined in d_plan in previous stages.

        See ../docs/cppdoc.md for main user-facing documentation.

    Input/Output:
    d_c   a size d_plan->M CUFINUFFT_CPX array on gpu (input for Type 1; output for Type
          2)
    d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CUFINUFFT_CPX array on gpu ((input for
          Type 2; output for Type 1)

    Notes:
        i) Here CUFINUFFT_CPX is a defined type meaning either complex<float> or complex<double>
        to match the precision of the library called.
        ii) All operations are done on the GPU device (hence the d_* names)

    Melody Shih 07/25/19; Barnett 2/16/21.
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int ier;
    int type = d_plan->type;
    switch (d_plan->dim) {
    case 1: {
        if (type == 1)
            ier = cufinufft1d1_exec<T>(d_c, d_fk, d_plan);
        if (type == 2)
            ier = cufinufft1d2_exec<T>(d_c, d_fk, d_plan);
        if (type == 3) {
            std::cerr << "Not Implemented yet" << std::endl;
            ier = 1;
        }
    } break;
    case 2: {
        if (type == 1)
            ier = cufinufft2d1_exec<T>(d_c, d_fk, d_plan);
        if (type == 2)
            ier = cufinufft2d2_exec<T>(d_c, d_fk, d_plan);
        if (type == 3) {
            std::cerr << "Not Implemented yet" << std::endl;
            ier = 1;
        }
    } break;
    case 3: {
        if (type == 1)
            ier = cufinufft3d1_exec<T>(d_c, d_fk, d_plan);
        if (type == 2)
            ier = cufinufft3d2_exec<T>(d_c, d_fk, d_plan);
        if (type == 3) {
            std::cerr << "Not Implemented yet" << std::endl;
            ier = 1;
        }
    } break;
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);

    return ier;
}

template <typename T>
int cufinufft_destroy_impl(cufinufft_plan_t<T> *d_plan)
/*
    "destroy" stage (single and double precision versions).

    In this stage, we
        (1) free all the memories that have been allocated on gpu
        (2) delete the cuFFT plan

        Also see ../docs/cppdoc.md for main user-facing documentation.
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    // Can't destroy a Null pointer.
    if (!d_plan) {
        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
        return 1;
    }

    if (d_plan->fftplan)
        cufftDestroy(d_plan->fftplan);

    switch (d_plan->dim) {
    case 1: {
        freegpumemory1d<T>(d_plan);
    } break;
    case 2: {
        freegpumemory2d<T>(d_plan);
    } break;
    case 3: {
        freegpumemory3d<T>(d_plan);
    } break;
    }

    /* free/destruct the plan */
    delete d_plan;
    /* set pointer to NULL now that we've hopefully free'd the memory. */
    d_plan = NULL;

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);
    return 0;
}

extern "C" {
int cufinufft_makeplanf(int type, int dim, int *nmodes, int iflag, int ntransf, float tol, int maxbatchsize,
                        cufinufftf_plan *d_plan_ptr, cufinufft_opts *opts) {
    return cufinufft_makeplan_impl(type, dim, nmodes, iflag, ntransf, tol, maxbatchsize,
                                   (cufinufft_plan_t<float> **)d_plan_ptr, opts);
}
int cufinufft_makeplan(int type, int dim, int *nmodes, int iflag, int ntransf, double tol, int maxbatchsize,
                       cufinufft_plan *d_plan_ptr, cufinufft_opts *opts) {
    return cufinufft_makeplan_impl(type, dim, nmodes, iflag, ntransf, tol, maxbatchsize,
                                   (cufinufft_plan_t<double> **)d_plan_ptr, opts);
}

int cufinufft_setptsf(int M, float *d_kx, float *d_ky, float *d_kz, int N, float *d_s, float *d_t, float *d_u,
                      cufinufftf_plan d_plan) {
    return cufinufft_setpts_impl(M, d_kx, d_ky, d_kz, N, d_s, d_t, d_u, (cufinufft_plan_t<float> *)d_plan);
}
int cufinufft_setpts(int M, double *d_kx, double *d_ky, double *d_kz, int N, double *d_s, double *d_t, double *d_u,
                     cufinufft_plan d_plan) {
    return cufinufft_setpts_impl(M, d_kx, d_ky, d_kz, N, d_s, d_t, d_u, (cufinufft_plan_t<double> *)d_plan);
}

int cufinufft_executef(cuFloatComplex *d_c, cuFloatComplex *d_fk, cufinufftf_plan d_plan) {
    return cufinufft_execute_impl<float>(d_c, d_fk, (cufinufft_plan_t<float> *)d_plan);
}
int cufinufft_execute(cuDoubleComplex *d_c, cuda_complex<double> *d_fk, cufinufft_plan d_plan) {
    return cufinufft_execute_impl<double>(d_c, d_fk, (cufinufft_plan_t<double> *)d_plan);
}

int cufinufft_destroyf(cufinufftf_plan *d_plan) {
    return cufinufft_destroy_impl<float>((cufinufft_plan_t<float> *)d_plan);
}
int cufinufft_destroy(cufinufft_plan *d_plan) {
    return cufinufft_destroy_impl<double>((cufinufft_plan_t<double> *)d_plan);
}

int cufinufft_default_opts(int type, int dim, cufinufft_opts *opts)
/*
    Sets the default options in cufinufft_opts. This must be called
    before the user changes any options from default values.
    The resulting struct may then be passed (instead of NULL) to the last
    argument of cufinufft_plan().

    Options with prefix "gpu_" are used for gpu code.

    Notes:
    Values set in this function for different type and dimensions are preferable
    based on experiments. User can experiement with different settings by
    replacing them after calling this function.

    Melody Shih 07/25/19; Barnett 2/5/21.
*/
{
    int ier;
    opts->upsampfac = 2.0;

    /* following options are for gpu */
    opts->gpu_nstreams = 0;
    opts->gpu_sort = 1; // access nupts in an ordered way for nupts driven method

    opts->gpu_maxsubprobsize = 1024;
    opts->gpu_obinsizex = -1;
    opts->gpu_obinsizey = -1;
    opts->gpu_obinsizez = -1;

    opts->gpu_binsizex = -1;
    opts->gpu_binsizey = -1;
    opts->gpu_binsizez = -1;

    opts->gpu_spreadinterponly = 0; // default to do the whole nufft

    switch (dim) {
    case 1: {
        opts->gpu_kerevalmeth = 0; // using exp(sqrt())
        if (type == 1) {
            opts->gpu_method = 2;
        }
        if (type == 2) {
            opts->gpu_method = 1;
        }
        if (type == 3) {
            std::cerr << "Not Implemented yet" << std::endl;
            ier = 1;
            return ier;
        }
    } break;
    case 2: {
        opts->gpu_kerevalmeth = 0; // using exp(sqrt())
        if (type == 1) {
            opts->gpu_method = 2;
        }
        if (type == 2) {
            opts->gpu_method = 1;
        }
        if (type == 3) {
            std::cerr << "Not Implemented yet" << std::endl;
            ier = 1;
            return ier;
        }
    } break;
    case 3: {
        opts->gpu_kerevalmeth = 0; // using exp(sqrt())
        if (type == 1) {
            opts->gpu_method = 2;
        }
        if (type == 2) {
            opts->gpu_method = 1;
        }
        if (type == 3) {
            std::cerr << "Not Implemented yet" << std::endl;
            ier = 1;
            return ier;
        }
    } break;
    }

    // By default, only use device 0
    opts->gpu_device_id = 0;

    return 0;
}
}

template int cufinufft_makeplan_impl(int type, int dim, int *nmodes, int iflag, int ntransf, float tol,
                                     int maxbatchsize, cufinufft_plan_t<float> **d_plan_ptr, cufinufft_opts *opts);
template int cufinufft_makeplan_impl(int type, int dim, int *nmodes, int iflag, int ntransf, double tol,
                                     int maxbatchsize, cufinufft_plan_t<double> **d_plan_ptr, cufinufft_opts *opts);

template int cufinufft_setpts_impl(int M, float *d_kx, float *d_ky, float *d_kz, int N, float *d_s, float *d_t,
                                   float *d_u, cufinufft_plan_t<float> *d_plan);
template int cufinufft_setpts_impl(int M, double *d_kx, double *d_ky, double *d_kz, int N, double *d_s, double *d_t,
                                   double *d_u, cufinufft_plan_t<double> *d_plan);

template int cufinufft_destroy_impl<float>(cufinufft_plan_t<float> *d_plan_ptr);
template int cufinufft_destroy_impl<double>(cufinufft_plan_t<double> *d_plan_ptr);
