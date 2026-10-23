#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufinufft.h>
#include <finufft_errors.h>

// assert.h goes last: a later include re-evaluates NDEBUG and turns the
// asserts back into no-ops in a Release build
#undef NDEBUG
#include <assert.h>

// Ways to exceed MAX_NF; each must reach the makeplan/setpts guards and return
// FINUFFT_ERR_MAXNALLOC with no large allocation (issue #890):
// - a single dimension whose upsampled nf itself overflows CUFINUFFT_BIGINT (int32);
// - a 3D grid whose per-dim nf each fit int32 but whose product overflows int64
//   (2^22 * 2^22 * 2^20 = 2^64);
// - a type-3 grid whose nf comes from widely separated points (the setpts guard).
int main() {
  const int type       = 1;
  const int iflag      = 1;
  const int ntransf    = 1;
  const int64_t N1[3]  = {1200000000, 1, 1}; // upsampfac*N1 alone overflows int32
  const int64_t N3[3]  = {700, 700, 700};    // ms*mt*mu fits int32; nf1*nf2*nf3 does not
  // nf = upsampfac*ms = 2^22, 2^22, 2^20: each nf fits int32, their product overflows
  // int64. The mode counts stay small because the C API caps ms*mt*mu at int32.
  const int64_t N64[3] = {1024, 1024, 256};

  {
    cufinufftf_plan plan;
    assert(cufinufftf_makeplan(type, 1, N1, iflag, ntransf, 1e-5f, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufft_plan plan;
    assert(cufinufft_makeplan(type, 1, N1, iflag, ntransf, 1e-5, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufftf_plan plan;
    assert(cufinufftf_makeplan(type, 3, N3, iflag, ntransf, 1e-5f, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufft_plan plan;
    assert(cufinufft_makeplan(type, 3, N3, iflag, ntransf, 1e-5, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.upsampfac       = 4096.0; // nf = upsampfac*ms exactly, so the product is 2^64
    opts.gpu_kerevalmeth = 0;      // Horner coefficients cover upsampfac 1.25 and 2.0
    cufinufftf_plan plan;
    assert(cufinufftf_makeplan(type, 3, N64, iflag, ntransf, 1e-5f, &plan, &opts) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.upsampfac       = 4096.0;
    opts.gpu_kerevalmeth = 0;
    cufinufft_plan plan;
    assert(cufinufft_makeplan(type, 3, N64, iflag, ntransf, 1e-5, &plan, &opts) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    // type 3: X = S = 1e6 gives nf past MAX_NF; M = N = 2 allocates nothing large
    const int64_t Nt3[3] = {1, 1, 1}; // type 3 ignores the mode counts
    cufinufftf_plan plan;
    assert(cufinufftf_makeplan(3, 1, Nt3, iflag, ntransf, 1e-5f, &plan, NULL) == 0);
    const float xh[2] = {0.0f, 2e6f}, sh[2] = {0.0f, 2e6f};
    float *d_x, *d_s;
    assert(cudaMalloc((void **)&d_x, sizeof xh) == cudaSuccess);
    assert(cudaMalloc((void **)&d_s, sizeof sh) == cudaSuccess);
    assert(cudaMemcpy(d_x, xh, sizeof xh, cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_s, sh, sizeof sh, cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cufinufftf_setpts(plan, 2, d_x, NULL, NULL, 2, d_s, NULL, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
    cudaDeviceSynchronize();
    cudaFree(d_x);
    cudaFree(d_s);
    cufinufftf_destroy(plan);
  }
  {
    const int64_t Nt3[3] = {1, 1, 1};
    cufinufft_plan plan;
    assert(cufinufft_makeplan(3, 1, Nt3, iflag, ntransf, 1e-5, &plan, NULL) == 0);
    const double xh[2] = {0.0, 2e6}, sh[2] = {0.0, 2e6};
    double *d_x, *d_s;
    assert(cudaMalloc((void **)&d_x, sizeof xh) == cudaSuccess);
    assert(cudaMalloc((void **)&d_s, sizeof sh) == cudaSuccess);
    assert(cudaMemcpy(d_x, xh, sizeof xh, cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_s, sh, sizeof sh, cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cufinufft_setpts(plan, 2, d_x, NULL, NULL, 2, d_s, NULL, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
    cudaDeviceSynchronize();
    cudaFree(d_x);
    cudaFree(d_s);
    cufinufft_destroy(plan);
  }
  {
    // spread/interp-only plans allocate no fwBatch, so batchsize*nf may exceed MAX_NF:
    // each transform reads and writes the user array on its own (issue #890)
    const int64_t Nio[3] = {1073741824, 1, 1}; // 2^30 grid; 2 * nf > MAX_NF
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_spreadinterponly = 1;
    opts.gpu_sort             = 0;
    opts.gpu_maxbatchsize     = 2;
    cufinufft_plan plan;
    const int ier = cufinufft_makeplan(2, 1, Nio, iflag, 2, 1e-5, &plan, &opts);
    assert(ier != FINUFFT_ERR_MAXNALLOC);
    if (ier == 0) cufinufft_destroy(plan);
  }
  return 0;
}
