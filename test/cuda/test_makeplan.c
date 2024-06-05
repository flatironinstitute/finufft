#include "finufft_errors.h"
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufinufft.h>

typedef struct {
  char *p[2];
} wasteful_pointers;

// hackish way to make allocation failures happen
wasteful_pointers alloc_remaining_device_mem() {
  wasteful_pointers a = {NULL, NULL};
  for (int i = 0; i < 2; ++i) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);

    int ier  = 1;
    int iter = 0;
    while (ier && (iter < 60)) {
      ier = cudaMalloc((void **)&a.p[i], free - (1 << iter));
      iter++;
    }
  }

  return a;
}

void free_wasteful_pointers(wasteful_pointers a) {
  cudaFree(a.p[0]);
  cudaFree(a.p[1]);
}

int main() {
  cufinufftf_plan plan;
  // defaults. tests should shadow them to override
  const int dim      = 1;
  const int type     = 1;
  const int iflag    = 1;
  const float tol    = 1e-5;
  const int ntransf  = 1;
  const int64_t N[3] = {10, 20, 15};

  // Dimension failure
  {
    const int dim = 0;
    assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
           FINUFFT_ERR_DIM_NOTVALID);
    cudaDeviceSynchronize();
  }

  // 1D failure modes
  {
    const int dim     = 1;
    const int type    = 1;
    const int ntransf = 1;

    // nice input should succeed
    assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) == 0);
    cufinufftf_destroy(plan);
    cudaDeviceSynchronize();

    // Ignore higher dims, even if invalid
    {
      int64_t N[3] = {10, 0, 15};
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) == 0);
      cufinufftf_destroy(plan);
      cudaDeviceSynchronize();
    }

    {
      int64_t N[3] = {0, 20, 15};
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NDATA_NOTVALID);
      cudaDeviceSynchronize();
    }

    // cufinufft can't handle arrays bigger than INT32_MAX (cufft limitation)
    {
      int64_t N[3] = {(int64_t)INT32_MAX + 1, 1, 1};
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NDATA_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      const int ntransf = 0;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NTRANS_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      const int type = 4;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_TYPE_NOTVALID);
      cudaDeviceSynchronize();
    }

    /* { */
    /*     wasteful_pointers p = alloc_remaining_device_mem(); */
    /*     int64_t N[3] = {INT32_MAX, 1, 1}; */
    /*     assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL)
     * == FINUFFT_ERR_CUDA_FAILURE);
     */
    /*     free_wasteful_pointers(p); */
    /* } */
  }

  {
    const int dim = 2;
    assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) == 0);
    cudaDeviceSynchronize();

    cufinufftf_destroy(plan);

    {
      int64_t N[3] = {10, 0, 1};
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NDATA_NOTVALID);
      cudaDeviceSynchronize();
    }

    // FIXME: nf calculation overflows -- need to handle upsampling mode calculation
    // properly
    /* { */
    /*     int64_t N[3] = {INT32_MAX / 2, 2, 1}; */
    /*     assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL)
     * == 0); */
    /* } */

    {
      int64_t N[3] = {INT32_MAX, 2, 1};
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NDATA_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      const int type = 4;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_TYPE_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      const int ntransf = 0;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NTRANS_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      cufinufft_opts opts;
      cufinufft_default_opts(&opts);
      opts.upsampfac       = 0.9;
      opts.gpu_kerevalmeth = 1;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, &opts) ==
             FINUFFT_ERR_HORNER_WRONG_BETA);

      opts.gpu_kerevalmeth = 0;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, &opts) ==
             FINUFFT_ERR_UPSAMPFAC_TOO_SMALL);

      // Should produce a warning, not an error
      opts.upsampfac = 4.5;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, &opts) == 0);
      cufinufftf_destroy(plan);
      cudaDeviceSynchronize();
    }

    // This technique to cause cuda failures works most of the time, but sometimes
    // would break following calls and could cause issues with other contexts using
    // the same GPU
    /* { */
    /*     wasteful_pointers p = alloc_remaining_device_mem(); */
    /*     int64_t N[3] = {sqrt(INT32_MAX - 1), sqrt(INT32_MAX) - 1, 1}; */
    /*     assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL)
     * == FINUFFT_ERR_CUDA_FAILURE);
     */
    /*     free_wasteful_pointers(p); */
    /* } */
  }

  {
    const int dim = 3;
    assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) == 0);
    cufinufftf_destroy(plan);
    cudaDeviceSynchronize();

    {
      int64_t N[3] = {10, 15, 0};
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NDATA_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      int64_t N[3] = {INT32_MAX / 2, 2, 2};
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NDATA_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      const int type = 4;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_TYPE_NOTVALID);
      cudaDeviceSynchronize();
    }

    {
      const int ntransf = 0;
      assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL) ==
             FINUFFT_ERR_NTRANS_NOTVALID);
      cudaDeviceSynchronize();
    }

    /* { */
    /*     wasteful_pointers p = alloc_remaining_device_mem(); */
    /*     int64_t N[3] = {pow(INT32_MAX - 1, 1.0 / 3), pow(INT32_MAX - 1, 1.0 / 3),
     * pow(INT32_MAX - 1, 1.0 / 3)};
     */
    /*     assert(cufinufftf_makeplan(type, dim, N, iflag, ntransf, tol, &plan, NULL)
     * == FINUFFT_ERR_CUDA_FAILURE);
     */
    /*     free_wasteful_pointers(p); */
    /* } */
  }
}
