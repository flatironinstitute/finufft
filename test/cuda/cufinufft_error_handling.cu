// Small CUDA C test to exercise deterministic CUFINUFFT error paths.
// Prints concise PASS on success and non-zero exit on mismatch.
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cufinufft.h>
#include <finufft_errors.h>

static int check_rc(const char *label, int got, int expected, int code) {
  if (got != expected) {
    fprintf(stderr, "%s: expected %d got %d\n", label, expected, got);
    return code;
  }
  return 0;
}

static int check_cuda(const char *label, cudaError_t err, int code) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: cuda error %s\n", label, cudaGetErrorString(err));
    return code;
  }
  return 0;
}

int main(void) {
  const int64_t N[3] = {16, 16, 16};
  int rc;

  // Invalid dim -> expect FINUFFT_ERR_DIM_NOTVALID
  cufinufftf_plan fplan = NULL;
  rc                    = cufinufftf_makeplan(1, 0, N, 1, 1, 1e-5f, &fplan, NULL);
  rc                    = check_rc("dim invalid", rc, FINUFFT_ERR_DIM_NOTVALID, 2);
  if (rc) return rc;

  // Invalid type -> expect FINUFFT_ERR_TYPE_NOTVALID
  rc = cufinufftf_makeplan(4, 1, N, 1, 1, 1e-5f, &fplan, NULL);
  rc = check_rc("type invalid", rc, FINUFFT_ERR_TYPE_NOTVALID, 3);
  if (rc) return rc;

  // Invalid ntrans -> expect FINUFFT_ERR_NTRANS_NOTVALID
  rc = cufinufftf_makeplan(1, 1, N, 1, 0, 1e-5f, &fplan, NULL);
  rc = check_rc("ntrans invalid", rc, FINUFFT_ERR_NTRANS_NOTVALID, 4);
  if (rc) return rc;

  // Invalid nmodes -> expect FINUFFT_ERR_NDATA_NOTVALID
  const int64_t Nbad[3] = {0, 1, 1};
  rc                    = cufinufftf_makeplan(1, 1, Nbad, 1, 1, 1e-5f, &fplan, NULL);
  rc                    = check_rc("nmodes invalid", rc, FINUFFT_ERR_NDATA_NOTVALID, 5);
  if (rc) return rc;

  // Upsamp too small -> expect FINUFFT_ERR_UPSAMPFAC_TOO_SMALL
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.upsampfac       = 0.9;
  opts.gpu_kerevalmeth = 0;
  rc                   = cufinufftf_makeplan(1, 2, N, 1, 1, 1e-5f, &fplan, &opts);
  rc = check_rc("upsampfac too small", rc, FINUFFT_ERR_UPSAMPFAC_TOO_SMALL, 6);
  if (rc) return rc;

  // Horner wrong beta -> expect FINUFFT_ERR_HORNER_WRONG_BETA
  cufinufft_default_opts(&opts);
  opts.upsampfac       = 0.9;
  opts.gpu_kerevalmeth = 1;
  rc                   = cufinufftf_makeplan(1, 2, N, 1, 1, 1e-5f, &fplan, &opts);
  rc = check_rc("horner wrong beta", rc, FINUFFT_ERR_HORNER_WRONG_BETA, 7);
  if (rc) return rc;

  // Insufficient shared memory in 3D: huge binsizes -> expect
  // FINUFFT_ERR_INSUFFICIENT_SHMEM
  cufinufft_default_opts(&opts);
  opts.gpu_method   = 2; // subproblem method uses shared memory
  opts.gpu_binsizex = 1024;
  opts.gpu_binsizey = 1024;
  opts.gpu_binsizez = 1024;
  rc                = cufinufftf_makeplan(1, 3, N, 1, 1, 1e-5f, &fplan, &opts);
  if (rc) {
    fprintf(stderr, "makeplan insufficient shmem setup failed: rc=%d\n", rc);
    return 8;
  }
  float *d_x            = NULL;
  float *d_y            = NULL;
  float *d_z            = NULL;
  cuFloatComplex *d_c   = NULL;
  cuFloatComplex *d_fk  = NULL;
  const size_t fk_count = (size_t)N[0] * (size_t)N[1] * (size_t)N[2];
  rc = check_cuda("cudaMalloc d_x", cudaMalloc((void **)&d_x, sizeof(float)), 9);
  if (rc) return rc;
  rc = check_cuda("cudaMalloc d_y", cudaMalloc((void **)&d_y, sizeof(float)), 10);
  if (rc) return rc;
  rc = check_cuda("cudaMalloc d_z", cudaMalloc((void **)&d_z, sizeof(float)), 11);
  if (rc) return rc;
  rc =
      check_cuda("cudaMalloc d_c", cudaMalloc((void **)&d_c, sizeof(cuFloatComplex)), 12);
  if (rc) return rc;
  rc = check_cuda("cudaMalloc d_fk",
                  cudaMalloc((void **)&d_fk, sizeof(cuFloatComplex) * fk_count), 13);
  if (rc) return rc;
  rc = check_cuda("cudaMemset d_x", cudaMemset(d_x, 0, sizeof(float)), 14);
  if (rc) return rc;
  rc = check_cuda("cudaMemset d_y", cudaMemset(d_y, 0, sizeof(float)), 15);
  if (rc) return rc;
  rc = check_cuda("cudaMemset d_z", cudaMemset(d_z, 0, sizeof(float)), 16);
  if (rc) return rc;
  rc = check_cuda("cudaMemset d_c", cudaMemset(d_c, 0, sizeof(cuFloatComplex)), 17);
  if (rc) return rc;
  rc = check_cuda("cudaMemset d_fk",
                  cudaMemset(d_fk, 0, sizeof(cuFloatComplex) * fk_count), 18);
  if (rc) return rc;
  rc = cufinufftf_setpts(fplan, 1, d_x, d_y, d_z, 0, NULL, NULL, NULL);
  if (rc) {
    fprintf(stderr, "setpts for insufficient shmem failed: rc=%d\n", rc);
    return 19;
  }
  rc = cufinufftf_execute(fplan, d_c, d_fk);
  rc = check_rc("insufficient shmem", rc, FINUFFT_ERR_INSUFFICIENT_SHMEM, 20);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_c);
  cudaFree(d_fk);
  cufinufftf_destroy(fplan);
  if (rc) return rc;

  // Destroy null -> expect FINUFFT_ERR_PLAN_NOTVALID
  rc = cufinufft_destroy(NULL);
  rc = check_rc("destroy null", rc, FINUFFT_ERR_PLAN_NOTVALID, 21);
  if (rc) return rc;

  // Type 3 invalid argument -> expect FINUFFT_ERR_INVALID_ARGUMENT
  rc = cufinufftf_makeplan(3, 1, N, 1, 1, 1e-5f, &fplan, NULL);
  if (rc) {
    fprintf(stderr, "makeplan type3 failed: rc=%d\n", rc);
    return 22;
  }
  rc = cufinufftf_setpts(fplan, 1, NULL, NULL, NULL, 1, NULL, NULL, NULL);
  rc = check_rc("type3 invalid argument", rc, FINUFFT_ERR_INVALID_ARGUMENT, 23);
  cufinufftf_destroy(fplan);
  if (rc) return rc;

  // Type 3 invalid N -> expect FINUFFT_ERR_NUM_NU_PTS_INVALID
  rc = cufinufftf_makeplan(3, 1, N, 1, 1, 1e-5f, &fplan, NULL);
  if (rc) {
    fprintf(stderr, "makeplan type3 (N invalid) failed: rc=%d\n", rc);
    return 24;
  }
  rc = cufinufftf_setpts(fplan, 1, NULL, NULL, NULL, -1, NULL, NULL, NULL);
  rc = check_rc("type3 N invalid", rc, FINUFFT_ERR_NUM_NU_PTS_INVALID, 25);
  cufinufftf_destroy(fplan);
  if (rc) return rc;

  printf("cufinufft_error_handling: PASS\n");
  return 0;
}
