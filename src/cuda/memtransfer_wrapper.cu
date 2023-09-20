#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

#include <cufinufft/contrib/helper_cuda.h>

namespace cufinufft {
namespace memtransfer {

template <typename T>
int allocgpumem1d_plan(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 11/21/21
*/
{
    utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);

    int ier;
    int nf1 = d_plan->nf1;
    int maxbatchsize = d_plan->maxbatchsize;

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            int numbins = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
            if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins * sizeof(int)))))
                goto finalize;
            if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins * sizeof(int)))))
                goto finalize;
        }
    } break;
    case 2: {
        int numbins = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->numsubprob, numbins * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts, (numbins + 1) * sizeof(int)))))
            goto finalize;
    } break;
    default:
        std::cerr << "err: invalid method " << std::endl;
    }

    if (!d_plan->opts.gpu_spreadinterponly) {
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize * nf1 * sizeof(cuda_complex<T>)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1, (nf1 / 2 + 1) * sizeof(T)))))
            goto finalize;
    }

finalize:
    if (ier)
        freegpumemory(d_plan);

    return ier;
}

template <typename T>
int allocgpumem1d_nupts(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 11/21/21
*/
{
    utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
    int ier;

    int M = d_plan->M;
    CUDA_FREE_AND_NULL(d_plan->sortidx);
    CUDA_FREE_AND_NULL(d_plan->idxnupts);

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort && (ier = checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)))))
            goto finalize;
    } break;
    case 2: {
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)))))
            goto finalize;
    } break;
    default:
        std::cerr << "[allocgpumem1d_nupts] error: invalid method\n";
    }

finalize:
    if (ier)
        freegpumemory(d_plan);

    return ier;
}

template <typename T>
int allocgpumem2d_plan(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
    utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
    int ier;

    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int maxbatchsize = d_plan->maxbatchsize;

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            int numbins[2];
            numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
            numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
            if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins[0] * numbins[1] * sizeof(int)))))
                goto finalize;
            if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0] * numbins[1] * sizeof(int)))))
                goto finalize;
        }
    } break;
    case 2: {
        int64_t numbins[2];
        numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
        numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->numsubprob, numbins[0] * numbins[1] * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins[0] * numbins[1] * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0] * numbins[1] * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts, (numbins[0] * numbins[1] + 1) * sizeof(int)))))
            goto finalize;
    } break;
    default:
        std::cerr << "[allocgpumem2d_plan] error: invalid method\n";
    }

    if (!d_plan->opts.gpu_spreadinterponly) {
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize * nf1 * nf2 * sizeof(cuda_complex<T>)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1, (nf1 / 2 + 1) * sizeof(T)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2, (nf2 / 2 + 1) * sizeof(T)))))
            goto finalize;
    }

finalize:
    if (ier)
        freegpumemory(d_plan);

    return ier;
}

template <typename T>
int allocgpumem2d_nupts(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
    utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
    int ier;

    const int M = d_plan->M;

    CUDA_FREE_AND_NULL(d_plan->sortidx);
    CUDA_FREE_AND_NULL(d_plan->idxnupts);

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort && (ier = checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)))))
            goto finalize;
    } break;
    case 2: {
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)))))
            goto finalize;
    } break;
    default:
        std::cerr << "[allocgpumem2d_nupts] error: invalid method\n";
    }

finalize:
    if (ier)
        freegpumemory(d_plan);

    return ier;
}

template <typename T>
int allocgpumem3d_plan(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
    utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
    int ier;

    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int nf3 = d_plan->nf3;
    int maxbatchsize = d_plan->maxbatchsize;

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort) {
            const int64_t nbins_tot = ceil((T)nf1 / d_plan->opts.gpu_binsizex) *
                                      ceil((T)nf2 / d_plan->opts.gpu_binsizey) *
                                      ceil((T)nf3 / d_plan->opts.gpu_binsizez);
            if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binsize, nbins_tot * sizeof(int)))))
                goto finalize;
            if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binstartpts, nbins_tot * sizeof(int)))))
                goto finalize;
        }
    } break;
    case 2: {
        const int64_t nbins_tot = ceil((T)nf1 / d_plan->opts.gpu_binsizex) * ceil((T)nf2 / d_plan->opts.gpu_binsizey) *
                                  ceil((T)nf3 / d_plan->opts.gpu_binsizez);

        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->numsubprob, nbins_tot * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binsize, nbins_tot * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binstartpts, nbins_tot * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts, (nbins_tot + 1) * sizeof(int)))))
            goto finalize;
    } break;
    case 4: {
        const int numobins[3] = {(int)ceil((T)nf1 / d_plan->opts.gpu_obinsizex),
                                 (int)ceil((T)nf2 / d_plan->opts.gpu_obinsizey),
                                 (int)ceil((T)nf3 / d_plan->opts.gpu_obinsizez)};

        const int binsperobins[3] = {d_plan->opts.gpu_obinsizex / d_plan->opts.gpu_binsizex,
                                     d_plan->opts.gpu_obinsizey / d_plan->opts.gpu_binsizey,
                                     d_plan->opts.gpu_obinsizez / d_plan->opts.gpu_binsizez};

        const int numbins[3] = {numobins[0] * (binsperobins[0] + 2), numobins[1] * (binsperobins[1] + 2),
                                numobins[2] * (binsperobins[2] + 2)};

        const int64_t numobins_tot = numobins[0] * numobins[1] * numobins[2];
        const int64_t numbins_tot = numbins[0] * numbins[1] * numbins[2];

        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->numsubprob, numobins_tot * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binsize, numbins_tot * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->binstartpts, (numbins_tot + 1) * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts, (numobins_tot + 1) * sizeof(int)))))
            goto finalize;
    } break;
    default:
        std::cerr << "[allocgpumem3d_plan] error: invalid method\n";
    }

    if (!d_plan->opts.gpu_spreadinterponly) {
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize * nf1 * nf2 * nf3 * sizeof(cuda_complex<T>)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1, (nf1 / 2 + 1) * sizeof(T)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2, (nf2 / 2 + 1) * sizeof(T)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf3, (nf3 / 2 + 1) * sizeof(T)))))
            goto finalize;
    }

finalize:
    if (ier)
        freegpumemory(d_plan);

    return ier;
}

template <typename T>
int allocgpumem3d_nupts(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
    utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
    int ier;
    int M = d_plan->M;

    CUDA_FREE_AND_NULL(d_plan->sortidx);
    CUDA_FREE_AND_NULL(d_plan->idxnupts)

    switch (d_plan->opts.gpu_method) {
    case 1: {
        if (d_plan->opts.gpu_sort && ((ier = checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int))))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)))))
            goto finalize;
    } break;
    case 2: {
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->idxnupts, M * sizeof(int)))))
            goto finalize;
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)))))
            goto finalize;
    } break;
    case 4: {
        if ((ier = checkCudaErrors(cudaMalloc(&d_plan->sortidx, M * sizeof(int)))))
            goto finalize;
    } break;
    default:
        std::cerr << "[allocgpumem3d_nupts] error: invalid method\n";
    }

finalize:
    if (ier)
        freegpumemory(d_plan);

    return ier;
}

template <typename T>
void freegpumemory(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for freeing gpu memory.

    Melody Shih 11/21/21
*/
{
    utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);

    CUDA_FREE_AND_NULL(d_plan->fw);
    CUDA_FREE_AND_NULL(d_plan->fw);
    CUDA_FREE_AND_NULL(d_plan->fwkerhalf1);
    CUDA_FREE_AND_NULL(d_plan->fwkerhalf2);
    CUDA_FREE_AND_NULL(d_plan->fwkerhalf3);

    CUDA_FREE_AND_NULL(d_plan->idxnupts);
    CUDA_FREE_AND_NULL(d_plan->sortidx);
    CUDA_FREE_AND_NULL(d_plan->numsubprob);
    CUDA_FREE_AND_NULL(d_plan->binsize);
    CUDA_FREE_AND_NULL(d_plan->binstartpts);
    CUDA_FREE_AND_NULL(d_plan->subprob_to_bin);
    CUDA_FREE_AND_NULL(d_plan->subprobstartpts);

    CUDA_FREE_AND_NULL(d_plan->numnupts);
    CUDA_FREE_AND_NULL(d_plan->numsubprob);
}

template int allocgpumem1d_plan<float>(cufinufft_plan_t<float> *d_plan);
template int allocgpumem1d_plan<double>(cufinufft_plan_t<double> *d_plan);
template int allocgpumem1d_nupts<float>(cufinufft_plan_t<float> *d_plan);
template int allocgpumem1d_nupts<double>(cufinufft_plan_t<double> *d_plan);

template void freegpumemory<float>(cufinufft_plan_t<float> *d_plan);
template void freegpumemory<double>(cufinufft_plan_t<double> *d_plan);

template int allocgpumem2d_plan<float>(cufinufft_plan_t<float> *d_plan);
template int allocgpumem2d_plan<double>(cufinufft_plan_t<double> *d_plan);
template int allocgpumem2d_nupts<float>(cufinufft_plan_t<float> *d_plan);
template int allocgpumem2d_nupts<double>(cufinufft_plan_t<double> *d_plan);

template int allocgpumem3d_plan<float>(cufinufft_plan_t<float> *d_plan);
template int allocgpumem3d_plan<double>(cufinufft_plan_t<double> *d_plan);
template int allocgpumem3d_nupts<float>(cufinufft_plan_t<float> *d_plan);
template int allocgpumem3d_nupts<double>(cufinufft_plan_t<double> *d_plan);

} // namespace memtransfer
} // namespace cufinufft
