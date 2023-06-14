#include <cufinufft/types.h>
#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/memtransfer.h>
#include <helper_cuda.h>

namespace cufinufft {
namespace memtransfer {

template <typename T>
int allocgpumem1d_plan(cufinufft_plan_template<T> d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 11/21/21
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int nf1 = d_plan->nf1;
    int maxbatchsize = d_plan->maxbatchsize;

    d_plan->byte_now = 0;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            int numbins = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
            checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins * sizeof(int)));
        }
    } break;
    case 2: {
        int numbins = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
        checkCudaErrors(cudaMalloc(&d_plan->numsubprob, numbins * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts, (numbins + 1) * sizeof(int)));
    } break;
    default:
        std::cerr << "err: invalid method " << std::endl;
    }

    if (!d_plan->opts.gpu_spreadinterponly) {
        checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize * nf1 * sizeof(cuda_complex<T>)));
        checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1, (nf1 / 2 + 1) * sizeof(T)));
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);
    return 0;
}

template <typename T>
int allocgpumem1d_nupts(cufinufft_plan_template<T> d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 11/21/21
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int M = d_plan->M;

    if (d_plan->sortidx)
        checkCudaErrors(cudaFree(d_plan->sortidx));
    if (d_plan->idxnupts)
        checkCudaErrors(cudaFree(d_plan->idxnupts));

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort)
            checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)));
    } break;
    case 2:
    case 3: {
        checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)));
    } break;
    default:
        std::cerr << "err: invalid method" << std::endl;
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);

    return 0;
}

template <typename T>
void freegpumemory1d(cufinufft_plan_template<T> d_plan)
/*
    wrapper for freeing gpu memory.

    Melody Shih 11/21/21
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    if (!d_plan->opts.gpu_spreadinterponly) {
        checkCudaErrors(cudaFree(d_plan->fw));
        checkCudaErrors(cudaFree(d_plan->fwkerhalf1));
    }
    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            checkCudaErrors(cudaFree(d_plan->idxnupts));
            checkCudaErrors(cudaFree(d_plan->sortidx));
            checkCudaErrors(cudaFree(d_plan->binsize));
            checkCudaErrors(cudaFree(d_plan->binstartpts));
        } else {
            checkCudaErrors(cudaFree(d_plan->idxnupts));
        }
    } break;
    case 2: {
        checkCudaErrors(cudaFree(d_plan->idxnupts));
        checkCudaErrors(cudaFree(d_plan->sortidx));
        checkCudaErrors(cudaFree(d_plan->numsubprob));
        checkCudaErrors(cudaFree(d_plan->binsize));
        checkCudaErrors(cudaFree(d_plan->binstartpts));
        checkCudaErrors(cudaFree(d_plan->subprobstartpts));
        checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
    } break;
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);
}

template <typename T>
int allocgpumem2d_plan(cufinufft_plan_template<T> d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int maxbatchsize = d_plan->maxbatchsize;

    d_plan->byte_now = 0;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            int numbins[2];
            numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
            numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
            checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins[0] * numbins[1] * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0] * numbins[1] * sizeof(int)));
        }
    } break;
    case 2: {
        int numbins[2];
        numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
        numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
        checkCudaErrors(cudaMalloc(&d_plan->numsubprob, numbins[0] * numbins[1] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins[0] * numbins[1] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0] * numbins[1] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts, (numbins[0] * numbins[1] + 1) * sizeof(int)));
    } break;
    default:
        std::cerr << "err: invalid method " << std::endl;
    }

    if (!d_plan->opts.gpu_spreadinterponly) {
        checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize * nf1 * nf2 * sizeof(CUCPX)));
        checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1, (nf1 / 2 + 1) * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2, (nf2 / 2 + 1) * sizeof(T)));
    }

    cudaStream_t *streams = (cudaStream_t *)malloc(d_plan->opts.gpu_nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < d_plan->opts.gpu_nstreams; i++)
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    d_plan->streams = streams;

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);
    return 0;
}

template <typename T>
int allocgpumem2d_nupts(cufinufft_plan_template<T> d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int M = d_plan->M;

    if (d_plan->sortidx)
        checkCudaErrors(cudaFree(d_plan->sortidx));
    if (d_plan->idxnupts)
        checkCudaErrors(cudaFree(d_plan->idxnupts));

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort)
            checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)));
    } break;
    case 2: {
        checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)));
    } break;
    default:
        std::cerr << "err: invalid method" << std::endl;
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);

    return 0;
}

template <typename T>
void freegpumemory2d(cufinufft_plan_template<T> d_plan)
/*
    wrapper for freeing gpu memory.

    Melody Shih 07/25/19
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    if (!d_plan->opts.gpu_spreadinterponly) {
        checkCudaErrors(cudaFree(d_plan->fw));
        checkCudaErrors(cudaFree(d_plan->fwkerhalf1));
        checkCudaErrors(cudaFree(d_plan->fwkerhalf2));
    }
    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            checkCudaErrors(cudaFree(d_plan->idxnupts));
            checkCudaErrors(cudaFree(d_plan->sortidx));
            checkCudaErrors(cudaFree(d_plan->binsize));
            checkCudaErrors(cudaFree(d_plan->binstartpts));
        } else {
            checkCudaErrors(cudaFree(d_plan->idxnupts));
        }
    } break;
    case 2: {
        checkCudaErrors(cudaFree(d_plan->idxnupts));
        checkCudaErrors(cudaFree(d_plan->sortidx));
        checkCudaErrors(cudaFree(d_plan->numsubprob));
        checkCudaErrors(cudaFree(d_plan->binsize));
        checkCudaErrors(cudaFree(d_plan->binstartpts));
        checkCudaErrors(cudaFree(d_plan->subprobstartpts));
        checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
    } break;
    }

    for (int i = 0; i < d_plan->opts.gpu_nstreams; i++)
        checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);
}

template <typename T>
int allocgpumem3d_plan(cufinufft_plan_template<T> d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int nf3 = d_plan->nf3;
    int maxbatchsize = d_plan->maxbatchsize;

    d_plan->byte_now = 0;

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            int numbins[3];
            numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
            numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
            numbins[2] = ceil((T)nf3 / d_plan->opts.gpu_binsizez);
            checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
        }
    } break;
    case 2: {
        int numbins[3];
        numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
        numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
        numbins[2] = ceil((T)nf3 / d_plan->opts.gpu_binsizez);
        checkCudaErrors(cudaMalloc(&d_plan->numsubprob, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts, (numbins[0] * numbins[1] * numbins[2] + 1) * sizeof(int)));
    } break;
    case 4: {
        int numobins[3], numbins[3];
        int binsperobins[3];
        numobins[0] = ceil((T)nf1 / d_plan->opts.gpu_obinsizex);
        numobins[1] = ceil((T)nf2 / d_plan->opts.gpu_obinsizey);
        numobins[2] = ceil((T)nf3 / d_plan->opts.gpu_obinsizez);

        binsperobins[0] = d_plan->opts.gpu_obinsizex / d_plan->opts.gpu_binsizex;
        binsperobins[1] = d_plan->opts.gpu_obinsizey / d_plan->opts.gpu_binsizey;
        binsperobins[2] = d_plan->opts.gpu_obinsizez / d_plan->opts.gpu_binsizez;

        numbins[0] = numobins[0] * (binsperobins[0] + 2);
        numbins[1] = numobins[1] * (binsperobins[1] + 2);
        numbins[2] = numobins[2] * (binsperobins[2] + 2);

        checkCudaErrors(cudaMalloc(&d_plan->numsubprob, numobins[0] * numobins[1] * numobins[2] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts, (numbins[0] * numbins[1] * numbins[2] + 1) * sizeof(int)));
        checkCudaErrors(
            cudaMalloc(&d_plan->subprobstartpts, (numobins[0] * numobins[1] * numobins[2] + 1) * sizeof(int)));
    } break;
    default:
        std::cerr << "err: invalid method" << std::endl;
    }

    if (!d_plan->opts.gpu_spreadinterponly) {
        checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize * nf1 * nf2 * nf3 * sizeof(cuda_complex<T>)));
        checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1, (nf1 / 2 + 1) * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2, (nf2 / 2 + 1) * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf3, (nf3 / 2 + 1) * sizeof(T)));
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);

    return 0;
}

template <typename T>
int allocgpumem3d_nupts(cufinufft_plan_template<T> d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    int M = d_plan->M;

    d_plan->byte_now = 0;

    if (d_plan->sortidx)
        checkCudaErrors(cudaFree(d_plan->sortidx));
    if (d_plan->idxnupts)
        checkCudaErrors(cudaFree(d_plan->idxnupts));

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort)
            checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)));
    } break;
    case 2: {
        checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)));
    } break;
    case 4: {
        checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)));
    } break;
    default:
        std::cerr << "err: invalid method" << std::endl;
    }

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);

    return 0;
}

template <typename T>
void freegpumemory3d(cufinufft_plan_template<T> d_plan)
/*
    wrapper for freeing gpu memory.

    Melody Shih 07/25/19
*/
{
    // Mult-GPU support: set the CUDA Device ID:
    int orig_gpu_device_id;
    cudaGetDevice(&orig_gpu_device_id);
    cudaSetDevice(d_plan->opts.gpu_device_id);

    if (!d_plan->opts.gpu_spreadinterponly) {
        cudaFree(d_plan->fw);
        cudaFree(d_plan->fwkerhalf1);
        cudaFree(d_plan->fwkerhalf2);
        cudaFree(d_plan->fwkerhalf3);
    }

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            checkCudaErrors(cudaFree(d_plan->idxnupts));
            checkCudaErrors(cudaFree(d_plan->sortidx));
            checkCudaErrors(cudaFree(d_plan->binsize));
            checkCudaErrors(cudaFree(d_plan->binstartpts));
        } else {
            checkCudaErrors(cudaFree(d_plan->idxnupts));
        }
    } break;
    case 2: {
        checkCudaErrors(cudaFree(d_plan->idxnupts));
        checkCudaErrors(cudaFree(d_plan->sortidx));
        checkCudaErrors(cudaFree(d_plan->numsubprob));
        checkCudaErrors(cudaFree(d_plan->binsize));
        checkCudaErrors(cudaFree(d_plan->binstartpts));
        checkCudaErrors(cudaFree(d_plan->subprobstartpts));
        checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
    } break;
    case 4: {
        checkCudaErrors(cudaFree(d_plan->idxnupts));
        checkCudaErrors(cudaFree(d_plan->sortidx));
        checkCudaErrors(cudaFree(d_plan->numsubprob));
        checkCudaErrors(cudaFree(d_plan->binsize));
        checkCudaErrors(cudaFree(d_plan->binstartpts));
        checkCudaErrors(cudaFree(d_plan->subprobstartpts));
        checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
    } break;
    }

    for (int i = 0; i < d_plan->opts.gpu_nstreams; i++)
        checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));

    // Multi-GPU support: reset the device ID
    cudaSetDevice(orig_gpu_device_id);
}

template int allocgpumem1d_plan<float>(cufinufft_plan_template<float> d_plan);
template int allocgpumem1d_plan<double>(cufinufft_plan_template<double> d_plan);
template int allocgpumem1d_nupts<float>(cufinufft_plan_template<float> d_plan);
template int allocgpumem1d_nupts<double>(cufinufft_plan_template<double> d_plan);
template void freegpumemory1d<float>(cufinufft_plan_template<float> d_plan);
template void freegpumemory1d<double>(cufinufft_plan_template<double> d_plan);

template int allocgpumem2d_plan<float>(cufinufft_plan_template<float> d_plan);
template int allocgpumem2d_plan<double>(cufinufft_plan_template<double> d_plan);
template int allocgpumem2d_nupts<float>(cufinufft_plan_template<float> d_plan);
template int allocgpumem2d_nupts<double>(cufinufft_plan_template<double> d_plan);
template void freegpumemory2d<float>(cufinufft_plan_template<float> d_plan);
template void freegpumemory2d<double>(cufinufft_plan_template<double> d_plan);

template int allocgpumem3d_plan<float>(cufinufft_plan_template<float> d_plan);
template int allocgpumem3d_plan<double>(cufinufft_plan_template<double> d_plan);
template int allocgpumem3d_nupts<float>(cufinufft_plan_template<float> d_plan);
template int allocgpumem3d_nupts<double>(cufinufft_plan_template<double> d_plan);
template void freegpumemory3d<float>(cufinufft_plan_template<float> d_plan);
template void freegpumemory3d<double>(cufinufft_plan_template<double> d_plan);

} // namespace mem
} // namespace cufinufft
